"""Real-time MuJoCo world, measured motor hardware and Rhizome sensor publishers.

No agent or navigation policy is implemented here. Control checks use the
external motor protocol; stationary observations keep the scene servos.
"""
from contextlib import nullcontext
import gc
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from sim_harness.adapters import stepping
from sim_harness.scene import _read_spec, load_scene
from sim_harness.export import export_mujoco
from .agriculture import compose_robot
from .agriculture_drive import camera_rig
from .navigation import add_cameras
from .rhizome_camera import ShmCamera, combo_frame

STATE = Path('/dev/shm/alamin_state.npz')
CONTROL_STATS = Path('/dev/shm/sensors.json')
PUBLISHER_STATS = Path('/dev/shm/publisher.json')

def run(config):
    scene = Path(config['scene'])
    output = Path(config['output'])
    generation = json.loads((scene/'generation.json').read_text())
    profile = json.loads((scene/'robot_profile.json').read_text())
    manifest = json.loads((scene/'manifest.json').read_text())
    robot = _read_spec(scene/'robot.mjz')
    profile = add_cameras(robot, profile, {'cameras': ['front', 'rear']})
    spec, model, data = compose_robot(_read_spec(scene/'environment.mjz'), robot,
                                      profile, generation['spawn'], manifest['terrain']['bounds_m'])
    model.vis.quality.offsamples = 0
    model.vis.quality.shadowsize = 512
    export_mujoco(spec, data, output/'world.mjz')
    (output/'camera_profile.json').write_text(json.dumps(profile))
    hardware = None
    if config.get('control_test'):
        from .rhizome_hardware import RhizomeMotorAdapter, motor_bindings
        hardware = RhizomeMotorAdapter(model, data, motor_bindings(model, profile, config))
    with stepping(model, data, hardware) as step:
        log = Path('/dev/shm/physics.jsonl').open('w', buffering=1) if hardware else nullcontext()
        with log as trajectory:
            _run_physics(model, data, output, step, hardware, trajectory)


def _run_physics(model, data, output, step, hardware, trajectory):
    from pyzome import core

    status_path = CONTROL_STATS if hardware else output/'sensors.json'
    stats = {'state_samples': 0, 'max_physics_lag_s': 0.0, 'motion_connected': bool(hardware),
             'max_snapshot_write_seconds': 0.0, 'max_step_batch_seconds': 0.0,
             'physics_overruns': 0}
    # The static scene graph lives for this process; keep it out of periodic GC scans.
    gc.freeze()
    start = time.monotonic()
    clock_started = hardware is None
    next_publish = start
    next_status = start
    while True:
        now = time.monotonic()
        if not clock_started:
            start = now-data.time
            clock_started = (output/'clock-start').exists()
        lag = now-start-data.time
        stats['max_physics_lag_s'] = max(stats['max_physics_lag_s'], lag)
        if lag > .5:
            if hardware:
                stats['physics_overruns'] += 1
                status_path.write_text(json.dumps(stats,indent=2)+'\n')
                raise RuntimeError(f'Actuated physics fell {lag:.3f}s behind wall time')
            # Stationary observation mode can catch up; no commands are applied.
            stats['physics_overruns'] += 1
            print(f'Physics scheduling overrun: {lag:.3f}s; navigation timing not validated',flush=True)
        step_begin = time.monotonic()
        if hardware:
            hardware.poll()
        while data.time < now-start:
            step()
        stats['max_step_batch_seconds'] = max(stats['max_step_batch_seconds'],time.monotonic()-step_begin)
        if now >= next_publish:
            publish_begin = time.monotonic()
            snapshot_path = STATE.with_suffix('.tmp')
            with snapshot_path.open('wb') as f:
                np.savez(f,qpos=data.qpos,qvel=data.qvel,time=data.time,
                         root_velocity=data.qvel[int(model.joint('element/root').dofadr[0]):][:3],
                         timestamp_ns=round(core.now().seconds()*1e9))
            snapshot_path.replace(STATE)
            stats['state_samples'] += 1
            if trajectory:
                trajectory.write(json.dumps(dict(time=float(data.time),qpos=data.qpos.tolist(),qvel=data.qvel.tolist(),xyz=data.xpos[model.body('element/chassis').id].tolist(),
                    motors={f'{pid}/{k}':dict(motors.feedback(k),reference=motors.motors[k].reference,parameters=motors.motors[k].params,zeroed=motors.motors[k].zeroed) for pid,motors in hardware.devices.items() for k in motors.motors}))+'\n')
            stats['max_snapshot_write_seconds'] = max(stats['max_snapshot_write_seconds'],time.monotonic()-publish_begin)
            next_publish = now+.02
        if now >= next_status:
            stats.update(simulated_seconds=float(data.time), wall_seconds=now-start,
                         warnings=[int(w.number) for w in data.warning])
            temporary = status_path.with_suffix('.tmp')
            temporary.write_text(json.dumps(stats,indent=2)+'\n')
            temporary.replace(status_path)
            next_status = now+1
        if hardware:
            hardware.servers[0].poll(.0005)
        else:
            time.sleep(.001)


def render(config):
    from pyzome import core
    output = Path(config['output'])
    model, data = load_scene(output/'world.mjz')
    writers = {}
    stats = {'frames': {'front': 0, 'rear': 0}, 'max_capture_age_s': 0.0}
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        try:
            writers = {p: ShmCamera(f'/tmp/combo/nav/{p}/shm') for p in ('front','rear')}
            option = mujoco.MjvOption()
            option.geomgroup[3:] = 0
            while True:
                begin = time.monotonic()
                for position, writer in writers.items():
                    with np.load(STATE) as snapshot:
                        data.qpos[:] = snapshot['qpos']
                        data.qvel[:] = snapshot['qvel']
                        data.time = float(snapshot['time'])
                        ns = int(snapshot['timestamp_ns'])
                    mujoco.mj_forward(model,data)
                    renderer.update_scene(data,camera='element/nav_'+position,scene_option=option)
                    rgb = renderer.render().copy()
                    renderer.enable_depth_rendering()
                    depth = renderer.render().copy()
                    renderer.disable_depth_rendering()
                    writer.push(combo_frame(rgb,depth,ns))
                    Image.fromarray(rgb).save(output/f'nav_{position}.png')
                    stats['frames'][position] += 1
                    stats['max_capture_age_s'] = max(stats['max_capture_age_s'],core.now().seconds()-ns/1e9)
                temp = output/'render.tmp'
                temp.write_text(json.dumps(stats,indent=2)+'\n')
                temp.replace(output/'render.json')
                time.sleep(max(0,1/config['camera_hz']-(time.monotonic()-begin)))
        finally:
            for writer in writers.values():
                writer.close()



def publish(config):
    from pyzome import core
    from robot_description.resolve import resolve
    import yaml
    root = Path(config['root'])
    output = Path(config['output'])
    profile = json.loads((output/'camera_profile.json').read_text())
    model,data = load_scene(output/'world.mjz')
    rig = camera_rig(profile, 640, 480)
    chassis = model.body('element/chassis').id
    description = resolve(yaml.safe_load((root/'src/robot_description/robot_cad.yml').read_text()),
                          yaml.safe_load((root/'src/base_control/tools.yml').read_text()),
                          config['chassis_configuration'], config['payload_configuration']).message()
    node = core.BaseNode(core.NodeConfig('alamin_sensors'))
    description_msg = core.RobotDescriptionMsg(json.dumps(description))
    next_calibration = 0
    previous = None
    stats = {'odom_messages': 0, 'stale_samples_dropped': 0, 'max_snapshot_age_s': 0.0}
    status_path = PUBLISHER_STATS if config.get('control_test') else output/'publisher.json'
    gc.freeze()
    while True:
        begin = time.monotonic()
        with np.load(STATE) as snapshot:
            ns = int(snapshot['timestamp_ns'])
            age = core.now().seconds()-ns/1e9
            stats['max_snapshot_age_s'] = max(stats['max_snapshot_age_s'],age)
            if age > .25:
                # Do not turn an old physical observation into fresh odometry.
                stats['stale_samples_dropped'] += 1
                time.sleep(.05)
                continue
            data.qpos[:] = snapshot['qpos']
            data.qvel[:] = snapshot['qvel']
        if ns != previous:
            mujoco.mj_forward(model,data)
            now = time.monotonic()
            xyz = data.xpos[chassis].copy()
            quat = data.xquat[chassis].copy()
            pose = dict(zip(('x','y','z','qw','qx','qy','qz'), [*xyz,*quat]))
            pose['frame'] = 'odom'
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,chassis,velocity,1)
            node.Publish('/odom', core.OdometryMsg(dict(pose=pose,twist=dict(v=float(velocity[3]),w=float(velocity[2])))))
            inverse = Rotation.from_quat([*quat[1:],quat[0]]).inv()
            oq = inverse.as_quat()
            op = inverse.apply(-xyz)
            node.Publish('/frames/odom',core.CoordinateFrame(dict(origin=dict(zip(('x','y','z','qx','qy','qz','qw'),[*op,*oq]),frame='robot'))))
            stats['odom_messages'] += 1
            if now >= next_calibration:
                if config.get('weeding_test'):
                    for key in ('chassis_configuration','payload_configuration'):
                        node.Publish('/settings/'+key,core.StringMsg(dict(str=config[key])))
                else:
                    node.Publish(core.RobotDescription.topic, description_msg)
                rot = data.xmat[chassis].reshape(3,3)
                for position in ('front','rear'):
                    name = 'nav_'+position
                    cid = model.camera('element/'+name).id
                    q = Rotation.from_matrix(rot.T@data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1])).as_quat()
                    p = rot.T@(data.cam_xpos[cid]-xyz)
                    origin = dict(zip(('x','y','z','qx','qy','qz','qw'), [*p,*q]), frame='robot')
                    node.Publish('/frames/'+name, core.CoordinateFrame(dict(origin=origin)))
                    k = rig[name]['K']
                    node.Publish(f'/camera/nav/{position}/intrinsics/color', core.CamIntrinsicsMsg(dict(
                        params=dict(fx=k[0][0],fy=k[1][1],cx=k[0][2],cy=k[1][2],
                                    frame_width=640,frame_height=480,scale=.0001))))
                next_calibration = now+1
                temp = status_path.with_suffix('.tmp')
                temp.write_text(json.dumps(stats,indent=2)+'\n')
                temp.replace(status_path)
            previous = ns
        time.sleep(max(0,.02-(time.monotonic()-begin)))
