"""Run Rhizome's native e2e striking services on the four-arm soybean plot."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from .agriculture import sha256
from .rhizome_build import verify
from .rhizome_e2e_camera import CropCameras, stage_crop, start_crop
from .rhizome_e2e_world import PlotWorld
from .rhizome_task import Task


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def pose(position, rotation, parent):
    return dict(zip(('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw'), [*position, *rotation.as_quat()]),
                frame=parent)


def robot_description(root, world, cameras):
    from robot_description.resolve import resolve
    chassis, payload = 'gen-3.0', 'raptor_4x_30in_gen3'
    overlay = dict(chassis_configuration=chassis, payload_configuration=payload, payloads={})
    for i, arm in enumerate(world.profile['arms']):
        device, slot = i // 2 + 1, i % 2
        entries = overlay['payloads'].setdefault(str(device), {})
        _, optical = cameras.calibration(device, slot)
        camera = optical['origin']
        camera_xyz = np.array([camera[k] for k in ('x', 'y', 'z')])
        robot_to_camera = Rotation.from_quat([camera[k] for k in ('qx', 'qy', 'qz', 'qw')]).inv()
        arm_rotation = Rotation.from_quat(np.roll(arm['quaternion_chassis_wxyz'], -1))
        entries[f'crop_{slot}'] = camera
        entries[f'arm_{slot}'] = pose(robot_to_camera.apply(arm['position_chassis_m'] - camera_xyz),
                                     robot_to_camera * arm_rotation, f'crop_{slot}')
    resolved = resolve(yaml.safe_load((root / 'src/robot_description/robot_cad.yml').read_text()),
                       yaml.safe_load((root / 'src/base_control/tools.yml').read_text()), chassis, payload, overlay)
    if resolved.errors or resolved.overlay is None:
        raise ValueError(f'Scene calibration failed: {resolved.errors}')
    return resolved


def node_config(directory, task, name):
    node = task.node(name)
    if node['missing_includes']:
        raise ValueError(f"{name} has missing includes: {node['missing_includes']}")
    meta = dict(node['meta'])
    meta.pop('includes', None)
    save(directory / 'nodes' / f'{name}.json', dict(meta=meta, config=node['config']))


def start_services(args, output, task, sockets, ram, stack):
    from tools.weeding_replay.runtime import NodeRuntime, WireLog, stop_runtimes

    buses = {}
    runtimes = {}
    for device in ('rover', 1, 2):
        cfg = output / f'config-{device}'
        (cfg / 'nodes').mkdir(parents=True)
        (cfg / 'payloads.yml').write_text(yaml.safe_dump({'payloads': [
            dict(hostname=f'payload{d}', ip=f'169.254.0.{65+d}') for d in (1, 2)]}))
        if device == 'rover':
            node_config(cfg, task, 'arm_coordinator')
        else:
            for name in ('trajectory', 'trajectory_0', 'trajectory_1'):
                node_config(cfg, task, name)
            stage_crop(task, cfg, model_package=args.model, field_package=args.field,
                       library_dir=args.libraries, socket_root=sockets[device], field_dir=ram / str(device))
        runtime = NodeRuntime(args.host, output / str(device), config=cfg, source=args.root / 'src',
                              simulation_feedback=False)
        runtimes[device] = stack.enter_context(runtime)
        buses[device] = stack.enter_context(WireLog(runtime, raw_spool=True, live_prefixes=(
            '/trajectory/field/', '/arm_planner/plan/')))
        stack.callback(runtime.stop)
    stack.callback(stop_runtimes, runtimes)
    bridge = dict(delay_ms=0, report=str(output / 'relay-report.json'), buses={
        str(k): dict(publish=f'ipc://{r.ipc_root}/build/host/aigen-proxy-sub',
                     subscribe=f'ipc://{r.ipc_root}/build/host/aigen-proxy-pub') for k, r in runtimes.items()})
    save(output / 'relay.json', bridge)
    runtimes['rover'].start('relay', [sys.executable, str(args.root / 'tools/weeding_replay/multi_bridge.py'),
                                      str(output / 'relay.json')])
    runtimes['rover'].start('arm_coordinator')
    for d in (1, 2):
        runtimes[d].start('trajectory')
        start_crop(runtimes[d], args.crop_binary, ram / str(d))
    time.sleep(.25)
    return runtimes, buses


def publish_configuration(model_dir, resolved, cameras, task, buses):
    from pyzome import core as p
    from robot_description.resolve import publications
    from tools.weeding_replay.multi_replay import local_publication, native_body

    stamp = time.time_ns()
    static = []
    for topic, body in publications(resolved):
        cls = p.CoordinateFrame if topic.startswith('/frames/') else p.RobotDescriptionMsg
        static.append(native_body(cls, topic, stamp, **body))
    crops = yaml.safe_load((model_dir / 'config.yml').read_text())['inference']['crop_conditioning']['crops']
    crop = 'soybean' if 'soybean' in crops else 'all'
    if crop not in crops:
        raise ValueError('Model package does not support soybean or all-crops conditioning')
    for d, bus in buses.items():
        for message in static:
            if d == 'rover':
                msg = dict(message)
                if msg['header']['topic'].startswith('/frames/') and '@' in msg['header']['topic']:
                    origin = dict(msg['origin'])
                    if origin['frame'].startswith(('crop_', 'arm_')):
                        origin['frame'] += '@' + msg['header']['topic'].rsplit('@', 1)[1]
                    msg['origin'] = origin
            else:
                msg = local_publication(message, d)
            bus.publish(msg)
        arms = [dict(payload_index=k, local_arm_index=s, target_priority='SIZE',
                     physics=dict(weeding_mode='STRIKE', soil_penetration_depth_m=task.node(
                         f'trajectory_{s}')['config']['penetration_m'])) for k in (1, 2) for s in (0, 1)]
        bus.publish(native_body(p.WeedingSettings, '/settings/weeding', stamp, model={'crop': crop}, arms=arms))
        if d != 'rover':
            for slot in (0, 1):
                intrinsics, _ = cameras.calibration(d, slot)
                bus.publish(native_body(p.CamIntrinsicsMsg, f'/camera/crop/{slot}/intrinsics/depth',
                                         stamp, **intrinsics))
            bus.publish(native_body(p.MuxSelectMsg, '/arm_mux/state', stamp, requested='AUTONOMY'))
    return crop


def drive(args, output, report, world, cameras, runtimes, buses, stack):
    from pyzome import core as p
    from tools.weeding_replay.multi_replay import native_body

    chassis = world.model.body('element/chassis').id
    arm_names = [a['id'] for a in world.profile['arms']]
    commands = {k: {a: v[a] for a in ('pitch', 'yaw')} for k, v in world.feedback().items()}
    counts = {name: Counter() for name in arm_names}
    latest = {name: {} for name in arm_names}
    sent = {name: 0 for name in arm_names}
    start = time.monotonic()
    start_ns = next_camera = camera_capture_ns = None
    end_step = round(args.seconds / world.model.opt.timestep)
    command_stamps = {name: None for name in arm_names}
    trace = stack.enter_context((output / 'world.jsonl').open('w', buffering=1))
    report['arms'] = counts
    dt = world.model.opt.timestep

    def advance_to(tick):
        count = tick - round(world.data.time / dt)
        if count > 0:
            position = world.data.xpos[chassis].tolist()
            before_s, before_joints = world.data.time, world.feedback()
            applied = world.advance(count * dt, forward_mps=float(np.clip(1.3 - position[0], 0, .18)),
                                    yaw_rate_rps=0., commands=commands)
            trace.write(json.dumps(dict(time_s=before_s, end_time_s=world.data.time, position_m=position,
                                        frame_ns=ns, camera_capture_ns=camera_capture_ns, camera_frames=sent,
                                        command_ns=command_stamps, commands=commands, applied_commands=applied,
                                        measured=before_joints, end_measured=world.feedback()))+'\n')

    while start_ns is None or world.data.time + 1e-8 < args.seconds:
        now = time.monotonic()
        if start_ns is None and now - start > args.warmup_seconds:
            raise RuntimeError(f'Native warmup timed out. Worker log files are under {output}')
        for r in runtimes.values():
            r.check()
        ns = start_ns + round(world.data.time * 1e9) if start_ns is not None else time.time_ns()
        xyz = world.data.xpos[chassis].copy()
        rot = Rotation.from_matrix(world.data.xmat[chassis].reshape(3, 3))
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(world.model, world.data, mujoco.mjtObj.mjOBJ_XBODY, chassis, velocity, 1)
        odom = native_body(p.OdometryMsg, '/odom', ns, pose=pose(xyz, rot, 'odom'),
                           twist=dict(v=float(velocity[3]), w=float(velocity[2])))
        frame = native_body(p.CoordinateFrame, '/frames/odom', ns,
                            origin=pose(rot.inv().apply(-xyz), rot.inv(), 'robot'))
        feedback = world.feedback()
        events = []
        for device, bus in buses.items():
            bus.publish(odom)
            bus.publish(frame)
            if device != 'rover':
                for slot in (0, 1):
                    name = arm_names[(device-1)*2+slot]
                    value = feedback[name]
                    for topic, suffix in [('angle', ''), ('velocity', '_velocity')]:
                        bus.publish(native_body(p.ArmMotorMsg, f'/motors/arm/{topic}/{slot}', ns,
                                                pitch=value['pitch'+suffix], yaw=value['yaw'+suffix]))
            for msg in bus.drain():
                if device == 'rover':
                    continue
                topic = msg['header']['topic']
                prefix, _, slot = topic.rpartition('/')
                if slot not in ('0', '1'):
                    continue
                name = arm_names[(device-1)*2+int(slot)]
                counts[name][prefix] += 1
                timestamp = msg['header']['timestamp']
                msg_ns = timestamp['secs'] * 10**9 + timestamp['nsecs']
                latest[name][prefix] = msg.get('capture_ns', msg_ns)
                if prefix == '/arm_planner/plan':
                    if any(msg[a]['mode'] != 'POSITION' for a in ('pitch', 'yaw')):
                        raise ValueError(f'Unexpected native arm command: {msg}')
                    events.append((msg_ns, name, {a: msg[a]['position_ref'] for a in ('pitch', 'yaw')}))
        wall_ns = time.time_ns()
        if next_camera is None or now >= next_camera:
            for i, name in enumerate(arm_names):
                cameras.publish(i//2+1, i%2, ns)
                sent[name] += 1
            camera_capture_ns = ns
            next_camera = now + 1/15
        fresh = all(wall_ns - values.get('/trajectory/field', 0) < 600_000_000 and
                    wall_ns - values.get('/arm_planner/plan', 0) < 300_000_000 for values in latest.values())
        if start_ns is None and fresh:
            start_ns = wall_ns
            report.update(warmup_wall_s=time.monotonic()-start, start_ns=start_ns)
        if start_ns is None:
            for msg_ns, name, value in sorted(events):
                commands[name] = value
                command_stamps[name] = msg_ns
            time.sleep(.005)
            continue
        if not fresh:
            raise RuntimeError('Stale field or arm command during row pass')

        # Native workers use wall time. Replay their timestamped position commands
        # on the first matching physics tick, including intervening motor feedback.
        dt_ns = round(dt * 1e9)
        target = min(end_step, max(0, (wall_ns - start_ns) // dt_ns))
        for msg_ns, name, value in sorted(events):
            tick = max(0, (msg_ns - start_ns + dt_ns - 1) // dt_ns)
            if tick > end_step:
                break
            advance_to(min(tick, target))
            commands[name] = value
            command_stamps[name] = msg_ns
        advance_to(target)
        report.update(simulated_seconds=world.data.time, contacts=world.contact_report(),
                      camera_frames=sent, chassis_position_m=world.data.xpos[chassis].tolist())
        time.sleep(max(0., .01 - (time.monotonic() - now)))
    report['end_ns'] = start_ns + round(args.seconds * 1e9)


def run(args, output, report):
    from pyzome import core as p

    if not Path(p.__file__).resolve().is_relative_to(args.host):
        raise ValueError('pyzome.core was imported from a different host build')
    if any(Path('/opt/aigen/task').glob('task.*')):
        raise ValueError('Run inside the documented private mount namespace to isolate task overrides')
    task = Task(args.root, 'e2e')
    services = task.services()
    if 'arm_coordinator' not in services['rover'] or not {'crop_ie', 'trajectory'} <= set(services['weeder']):
        raise ValueError('Selected checkout does not have the native e2e services')
    report['host'] = verify(args.host, args.root)
    if not {'proxy', 'trajectory', 'arm_coordinator'} <= report['host'].get('binaries_sha256', {}).keys():
        raise ValueError('Host manifest must cover proxy, arm_coordinator and trajectory')
    report['scene_sha256'] = sha256(args.scene / 'scene.mjz')
    report['runtime_sha256'] = {str(f): sha256(f) for f in (
        args.crop_binary, args.libraries / 'libe2e_pipeline.so', args.libraries / 'libe2e_trt.so',
        args.model / 'core.plan', args.model / 'grid_post.plan', args.field / 'field.plan')}
    world = PlotWorld(args.scene)
    arm_names = list(world.arms)
    with ExitStack() as stack:
        ipc = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='alamin-crop-')))
        ram = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix='alamin-field-', dir='/dev/shm')))
        sockets = {d: ipc / str(d) for d in (1, 2)}
        cameras = stack.enter_context(CropCameras(world, task, sockets))
        resolved = robot_description(args.root, world, cameras)
        description = resolved.message()
        save(output / 'description.json', description)
        report['arm_limits'] = dict(source='description.json (payload arm config.extents)',
                                    arms=world.bind_arm_extents(json.loads(description['description'])['payloads']))
        runtimes, buses = start_services(args, output, task, sockets, ram, stack)
        report['model_conditioning'] = publish_configuration(args.model, resolved, cameras, task, buses)
        drive(args, output, report, world, cameras, runtimes, buses, stack)
    from .rhizome_e2e_trace import analyze
    report['trace'] = analyze(output, arm_names, report['start_ns'], report['end_ns'])
    if not report['trace']['all_arms_traced']:
        raise RuntimeError('Incomplete capture-to-command trace on one or more arms')
    report['completed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'host', 'crop-binary', 'libraries', 'model', 'field', 'scene', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=20.)
    parser.add_argument('--warmup-seconds', type=float, default=90.)
    args = parser.parse_args()
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds, args.warmup_seconds)) or not math.isclose(
            round(args.seconds * 100), args.seconds * 100):
        parser.error('Use positive finite durations and a row pass in whole 10 ms intervals')
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(completed=False, command=shlex.join([sys.executable,'-m',__spec__.name,*sys.argv[1:]]),
                  revision=subprocess.check_output(['git','-C',args.root,'rev-parse','HEAD'],text=True).strip())
    try:
        run(args, args.output, report)
    except Exception as error:
        report['error'] = str(error)
        raise
    finally:
        save(args.output / 'report.json', report)


if __name__ == '__main__':
    main()
