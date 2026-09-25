"""Wheel-actuated agricultural rollouts and synchronized multi-camera capture."""
import math
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec
from .agriculture import compose_robot, robot_footprint, sha256
from .contracts import PipelineError, read_json, write_json
from .dataset import Recorder, inspect


def drive_config(config, seconds=None):
    if config is not None and not isinstance(config,dict):
        raise PipelineError('DRIVE_CONFIG','Expected a JSON object')
    config = dict(config or {})
    if 'controller' in config:
        from .rhizome import validate_config
        if 'commands' in config:
            raise PipelineError('DRIVE_CONFIG','Use controller or timed commands, not both')
        config['controller']=validate_config(config['controller'])
    duration = float(seconds if seconds is not None else config.get('seconds', 12))
    if not math.isfinite(duration) or duration < .01 or duration > 3600:
        raise PipelineError('DRIVE_CONFIG', 'seconds must be between .01 and 3600')
    if not math.isclose(duration/.002, round(duration/.002), abs_tol=1e-7):
        raise PipelineError('DRIVE_CONFIG', 'Duration must lie on the 2 ms physics clock')
    commands = config.get('commands', [dict(time=0,forward_mps=0,yaw_rate_rps=0),
                                        dict(time=1,forward_mps=.25,yaw_rate_rps=0),
                                        dict(time=max(1.01,duration-1),forward_mps=0,yaw_rate_rps=0)])
    normalized=[]
    if not isinstance(commands,list):raise PipelineError('DRIVE_CONFIG','commands must be a list')
    for command in commands:
        if not isinstance(command,dict) or not {'time','forward_mps','yaw_rate_rps'}<=command.keys():
            raise PipelineError('DRIVE_CONFIG','Each command needs time, forward_mps and yaw_rate_rps')
        values = {k:float(command[k]) for k in ('time','forward_mps','yaw_rate_rps')}
        if not all(math.isfinite(v) for v in values.values()) or values['time'] < 0:
            raise PipelineError('DRIVE_CONFIG','Commands must contain finite values and nonnegative times')
        if not math.isclose(values['time']/.01,round(values['time']/.01),abs_tol=1e-7):
            raise PipelineError('DRIVE_CONFIG','Command times must lie on the 100 Hz control clock')
        normalized.append(values)
    if not normalized or normalized[0]['time'] != 0 or any(b['time']<=a['time'] for a,b in zip(normalized,normalized[1:])):
        raise PipelineError('DRIVE_CONFIG','Commands must start at zero and have strictly increasing times')
    camera = config.get('camera',{})
    width,height = camera.get('width',640),camera.get('height',360)
    if (not isinstance(width,int) or not isinstance(height,int) or width<64 or height<36
            or width>1280 or height>720 or width*9 != height*16):
        raise PipelineError('DRIVE_CONFIG','Camera resolution must be 16:9, from 64x36 through 1280x720')
    result={**config,'seconds':duration,'camera':dict(width=width,height=height)}
    if 'names' in camera:
        names=camera['names']
        if (not isinstance(names,list) or not names or not all(isinstance(n,str) for n in names)
                or len(names)!=len(set(names))):
            raise PipelineError('DRIVE_CONFIG','camera.names must be a nonempty list of unique camera names')
        result['camera']['names']=names
    if 'controller' not in config:result['commands']=normalized
    return result


def wheel_targets(profile, forward, yaw_rate):
    v = float(np.clip(forward,-profile['max_forward_mps'],profile['max_forward_mps']))
    w = float(np.clip(yaw_rate,-profile['max_yaw_rate_rps'],profile['max_yaw_rate_rps']))
    return np.array([wheel.get('drive_sign',1)*(v-w*wheel['center_m'][1])/wheel['radius_m'] for wheel in profile['wheels']]), [v,w]


def arm_schedule(config, profile):
    """Validate independent payload commands in Rhizome motor radians."""
    commands = config.get('arm_commands', {})
    arms = {arm['id']: arm for arm in profile.get('arms', [])}
    if not isinstance(commands, dict) or set(commands)-arms.keys():
        raise PipelineError('DRIVE_CONFIG', 'arm_commands must name arms in the robot profile')
    for name, sequence in commands.items():
        if not isinstance(sequence, list) or not sequence:
            raise PipelineError('DRIVE_CONFIG', 'Each arm command schedule must be a nonempty list')
        previous = -1.
        for command in sequence:
            if not isinstance(command, dict) or set(command) != {'time', 'yaw_rad', 'pitch_rad'}:
                raise PipelineError('DRIVE_CONFIG', 'Arm commands require time, yaw_rad and pitch_rad')
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in command.values()):
                raise PipelineError('DRIVE_CONFIG', 'Arm commands must be finite numbers')
            t = command['time']
            if t < 0 or t <= previous or not math.isclose(t/.01, round(t/.01), abs_tol=1e-7):
                raise PipelineError('DRIVE_CONFIG', 'Arm times must increase on the 100 Hz clock')
            previous = t
            for axis, limits in zip(('yaw_rad', 'pitch_rad'), arms[name]['motor_limits_rad']):
                if not limits[0] <= command[axis] <= limits[1]:
                    raise PipelineError('DRIVE_CONFIG', f'{name}.{axis} outside estimated joint limits')
    return commands


def shadow_row_profile(model,data,chassis,mount,quaternion=(1,0,0,0)):
    """Sample terrain truth in the configured arm frame, for oracle mode only."""
    from .rhizome_perception import quaternion_rotation
    chassis_rotation=data.xmat[chassis].reshape(3,3)
    rotation=chassis_rotation@quaternion_rotation(quaternion)
    origin=data.xpos[chassis]+chassis_rotation@np.asarray(mount)
    points=[]
    for lateral in np.linspace(-.3,.3,7):
        point=origin+rotation@np.array([0.,lateral,0.])
        start=np.array([point[0],point[1],max(origin[2],point[2])+2.])
        distance=mujoco.mj_rayHfield(model,data,model.geom('terrain').id,start,np.array([0.,0.,-1.]))
        if distance<0:raise PipelineError('RHIZOME_TERRAIN','Shadow arm row profile missed terrain')
        hit=start-np.array([0,0,distance])
        x,y,z=rotation.T@(hit-origin)
        points.append(dict(x=x,y=y,z=z,frame='arm_shadow'))
    return points


def label_tables(model, manifest):
    ids={name:i+1 for i,name in enumerate(sorted(manifest['instances']))}
    ids['element']=len(ids)+1
    instance=np.zeros(model.ngeom+1,dtype=np.int32)
    semantic=np.zeros(model.ngeom+1,dtype=np.uint8)
    for geom in range(model.ngeom):
        name=model.geom(geom).name
        if name.startswith('element/'):
            instance[geom]=ids['element'];semantic[geom]=4
        elif name in manifest['geom_labels']:
            label=manifest['geom_labels'][name]
            instance[geom]=ids.get(label['instance'],0);semantic[geom]=label['class_id']
    return ids,instance,semantic


def camera_rig(profile, width, height):
    rig={}
    for name,camera in profile['cameras'].items():
        optics=camera['optics']
        fx=optics['lens_mm']/optics['sensor_width_mm']*width
        fy=optics['lens_mm']/optics['sensor_height_mm']*height
        rig[name]={**camera,'width':width,'height':height,'rate_hz':10,
                   'K':[[fx,0,width/2],[0,fy,height/2],[0,0,1]],
                   'depth_convention':'positive optical-axis distance, metres; ideal MuJoCo Z buffer',
                   'stream':'camera_'+name,'observation_model':'ideal RGB and depth; no stereo holes or noise'}
    return rig


def trajectory_image(path, points, bounds, rows):
    image=Image.new('RGB',(800,500),'#f5f1e5');draw=ImageDraw.Draw(image)
    low,high=np.asarray(bounds)
    def pixel(point):
        x,y=(np.asarray(point)[:2]-low)/(high-low)
        return (30+float(x)*740,470-float(y)*440)
    for row in rows:
        draw.line([pixel(p) for p in row['endpoints_m']],fill='#608c42',width=3)
    if len(points)>1:draw.line([pixel(p) for p in points],fill='#2546c2',width=3)
    draw.text((12,8),'Recorded chassis trajectory (blue), crop rows (green)',fill='black')
    image.save(path)


def run(scene, output, *, config=None, seconds=None, tier='full', seed=0, video=True, timeout=900):
    scene=Path(scene);output=Path(output)
    if not math.isfinite(timeout) or timeout<=0:raise PipelineError('DRIVE_CONFIG','timeout must be positive and finite')
    if output.exists():raise PipelineError('OUTPUT_EXISTS',str(output))
    config=drive_config(read_json(config) if isinstance(config,(str,Path)) else config,seconds)
    generation=read_json(scene/'generation.json')
    if generation.get('domain')!='agriculture':
        raise PipelineError('AGRICULTURE_SCENE','agriculture-drive requires an agricultural scene')
    profile=read_json(scene/'robot_profile.json');manifest=read_json(scene/'manifest.json')
    arm_commands=arm_schedule(config,profile)
    bounds=np.asarray(manifest['terrain']['bounds_m'])
    spawn=config.get('spawn',generation['spawn'])
    spec,model,data=compose_robot(_read_spec(scene/'environment.mjz'),_read_spec(scene/'robot.mjz'),profile,spawn,bounds)
    # Match the replay renderer's shadow budget; 4K shadow maps dominate
    # software capture time without improving depth or segmentation.
    spec.visual.quality.shadowsize=1024
    model.vis.quality.shadowsize=1024
    # Multisampling blends integer segmentation colours at leaf boundaries,
    # creating wrong labels (and sometimes out-of-range IDs). Keep all aligned
    # capture passes single-sampled rather than silently discarding bad labels.
    spec.visual.quality.offsamples=0
    model.vis.quality.offsamples=0
    output.mkdir(parents=True);started=time.perf_counter()
    export_mujoco(spec,data,output/'rollout_scene.mjz')
    width,height=config['camera']['width'],config['camera']['height']
    cameras=camera_rig(profile,width,height)
    if 'names' in config['camera']:
        if set(config['camera']['names'])-cameras.keys():raise PipelineError('DRIVE_CONFIG','Unknown capture camera')
        cameras={name:cameras[name] for name in config['camera']['names']}
    controller_config=config.get('controller')
    physical_arm=None;physical_arms={};multi=bool(controller_config and controller_config.get('physical_arms'))
    if controller_config:
        from .rhizome import bind_physical_arm,bind_physical_arms
        if multi:
            controller_config,physical_arms=bind_physical_arms(controller_config,profile)
            owned={a['camera'] for a in physical_arms.values()}
            if 'names' not in config['camera']:cameras={name:cameras[name] for name in cameras if name in owned}
            if set(cameras)!=owned:raise PipelineError('RHIZOME_CONFIG','Capture cameras must match the selected physical arms')
        else:
            controller_config,physical_arm=bind_physical_arm(controller_config,profile)
            if physical_arm:physical_arms={physical_arm['id']:physical_arm}
        config['controller']=controller_config
        if set(physical_arms)&arm_commands.keys():
            raise PipelineError('DRIVE_CONFIG','A physical arm cannot have both native and timed commands')
    if controller_config:
        if controller_config.get('perception','none')!='none' and tier!='full':
            raise PipelineError('RHIZOME_CONFIG','Visible perception requires --tier full')
        if controller_config.get('shadow_arm') and controller_config['shadow_arm']['camera'] not in cameras:
            raise PipelineError('RHIZOME_CONFIG','shadow_arm.camera is not in the robot rig')
        if controller_config.get('perception')=='neural' and set(controller_config['neural']['target_positions_chassis_m'])!=set(profile['cameras']):
            raise PipelineError('RHIZOME_CONFIG','Neural target mounts must cover exactly the simulated cameras')
    benchmark=manifest.get('weeding_benchmark')
    if benchmark:
        if set(physical_arms)!=set(benchmark.get('arms',[benchmark.get('arm')])):
            raise PipelineError('WEEDING_CONFIG','The benchmark requires its selected physical Rhizome arm')
        controller_config['targeting_diagnostics']=True
    record_tools=bool(physical_arms and controller_config.get('targeting_diagnostics'))
    if record_tools:
        tool_geoms={name:model.geom('element/'+arm['pitch_joint']+'_tool_collision').id for name,arm in physical_arms.items()}
    instance_ids,instances,classes=label_tables(model,manifest)
    rig=dict(cameras=cameras,rates_hz=dict(state=100,imu=100,commands=100,camera=10),
             clock='simulation_seconds',profile=profile,spawn=spawn,
             rendering=dict(shadow_map_size=1024,offscreen_samples=0),
             imu=dict(frame='chassis',gyro_sigma_rad_s=.002,accelerometer_sigma_m_s2=.03,calibrated=False))
    write_json(output/'rig.json',rig);write_json(output/'drive_config.json',config)
    write_json(output/'manifest.json',manifest)
    metadata=dict(flow='agriculture-drive',rig=rig,semantics=dict(instance_ids=instance_ids,class_names=manifest['class_names']),
                  seed=seed,tier=tier,clock='simulation_seconds',ground_truth='state, poses, masks and plant metadata are simulator truth',
                  source_sha256=sha256(scene/'scene.mjz'),provenance=generation['provenance'])
    if controller_config:
        metadata['controller']=controller_config
        metadata['arm_feedback']='MuJoCo measured joints' if physical_arms else 'Rhizome software motor model; no MuJoCo feedback'
    wheel_names=['element/'+w['id'] for w in profile['wheels']]
    controls=np.array([model.actuator(name).id for name in wheel_names])
    chassis=model.body('element/chassis').id
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    rng=np.random.default_rng(seed)
    frames={};neural_frames={};positions=[];failure=None;obstacle_contacts=set();max_contacts=0
    warnings=np.zeros(len(data.warning),dtype=int)
    recorder=Recorder(output/'data.h5',metadata)
    renderer=None;controller=None;native_info=None
    oracle_count=0;neural_count=0;tracked_count=0;shadow_states=set();shadow_targets=set()
    arm_states={name:set() for name in physical_arms};arm_targets={name:set() for name in physical_arms}
    camera_counts={name:dict(oracle_detections=0,neural_detections=0,tracked_detections=0) for name in cameras}
    command_index=0;requested=[0.,0.];applied=[0.,0.]
    steps=round(config['seconds']/.002)
    try:
        if controller_config:
            from .rhizome import RhizomeController,RhizomeMultiController,visible_detections
            controller=(RhizomeMultiController if multi else RhizomeController)(controller_config,output,
                **({'arms':physical_arms} if multi else {}),timeout=max(.001,timeout-(time.perf_counter()-started)))
            native_info=controller.info
        if tier=='full':renderer=mujoco.Renderer(model,height=height,width=width)
        for tick in range(steps+1):
            if time.perf_counter()-started>=timeout:
                failure='Wall-clock budget exceeded';break
            mujoco.mj_forward(model,data)
            t=float(data.time)
            if not math.isclose(t,tick*.002,abs_tol=1e-6):
                failure='MuJoCo reset the simulation clock';break
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                failure='Non-finite physics state';break
            if record_tools:
                for name,tool_geom in tool_geoms.items():
                    recorder.append('weeding_tool_'+name if multi else 'weeding_tool',t,position_world=data.geom_xpos[tool_geom].copy(),
                                    rotation_world=data.geom_xmat[tool_geom].reshape(3,3).copy(),
                                    half_size_m=model.geom_size[tool_geom].copy())
            if tick%5==0:
                if controller:
                    velocity=np.zeros(6)
                    mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,chassis,velocity,1)
                    measured={name:np.asarray(payload['motor_to_joint_sign'])*[
                        data.joint('element/'+payload[key]).qpos[0] for key in ('yaw_joint','pitch_joint')]
                        for name,payload in physical_arms.items()}
                    native_result=controller.step(t,data.xpos[chassis],data.xquat[chassis],velocity[3],
                        **({'arm_measured':measured if multi else measured[physical_arm['id']]} if physical_arms else {}))
                    requested=native_result['twist']
                    results=native_result.get('arms',{}) if multi else ({physical_arm['id'] if physical_arm else 'shadow':native_result['arm']} if 'arm' in native_result else {})
                    for name,arm in results.items():
                        shadow_states.add(arm['state'])
                        if arm['target_id']>=0:shadow_targets.add(arm['target_id'])
                        payload=physical_arms.get(name)
                        if payload:
                            arm_states[name].add(arm['state'])
                            if arm['target_id']>=0:arm_targets[name].add(arm['target_id'])
                            command=np.asarray(arm['commanded'])
                            if command.shape!=(2,) or not np.isfinite(command).all():raise PipelineError('RHIZOME_COMMAND','Invalid native arm command')
                            for axis,key in enumerate(('yaw_joint','pitch_joint')):
                                aid=model.actuator('element/'+payload[key]).id
                                data.ctrl[aid]=np.clip(command[axis]*payload['motor_to_joint_sign'][axis],*model.actuator_ctrlrange[aid])
                        stream='rhizome_'+name if multi else ('rhizome_physical_arm' if payload else 'rhizome_shadow_arm')
                        recorder.append(stream,t,commanded_yaw_pitch=np.asarray(arm['commanded']),
                                        measured_yaw_pitch=np.asarray(arm['measured']),target_id=arm['target_id'],
                                        detection_count=arm['detections'])
                else:
                    while command_index+1<len(config['commands']) and config['commands'][command_index+1]['time']<=t+1e-9:
                        command_index+=1
                    command=config['commands'][command_index]
                    requested=[command['forward_mps'],command['yaw_rate_rps']]
                targets,applied=wheel_targets(profile,*requested)
                data.ctrl[controls]=targets
                for arm in profile.get('arms', []):
                    names=['element/'+arm[key] for key in ('yaw_joint','pitch_joint')]
                    indices=[model.actuator(name).id for name in names]
                    for command in reversed(arm_commands.get(arm['id'], [])):
                        if command['time'] <= t+1e-9:
                            data.ctrl[indices]=np.asarray(arm['motor_to_joint_sign'])*[command['yaw_rad'],command['pitch_rad']]
                            break
                    signs=np.asarray(arm['motor_to_joint_sign'])
                    recorder.append(arm['id'],t,
                        commanded_motor_rad=data.ctrl[indices]*signs,
                        measured_motor_rad=np.array([data.joint(name).qpos[0] for name in names])*signs,
                        velocity_motor_rad_s=np.array([data.joint(name).qvel[0] for name in names])*signs,
                        actuator_torque_nm=data.actuator_force[indices].copy(),
                        tip_position_world=data.site('element/'+arm['tool_site']).xpos.copy())
                if profile.get('suspension'):
                    recorder.append('suspension',t,
                        position_rad=np.array([data.joint('element/'+s['joint']).qpos[0] for s in profile['suspension']]),
                        velocity_rad_s=np.array([data.joint('element/'+s['joint']).qvel[0] for s in profile['suspension']]))
                mujoco.mj_forward(model,data)
                positions.append(data.xpos[chassis].copy())
                wheel_pos=np.array([data.joint(name).qpos[0] for name in wheel_names])
                wheel_vel=np.array([data.joint(name).qvel[0] for name in wheel_names])
                recorder.append('state',t,qpos=data.qpos.copy(),qvel=data.qvel.copy(),ctrl=data.ctrl.copy(),
                                chassis_position=data.xpos[chassis].copy(),chassis_quaternion=data.xquat[chassis].copy(),
                                wheel_position=wheel_pos,wheel_velocity=wheel_vel,actuator_force=data.actuator_force[controls].copy(),
                                contact_count=data.ncon)
                gyro=data.sensor('element/gyro').data.copy();accel=data.sensor('element/accelerometer').data.copy()
                recorder.append('imu',t,gyro_truth=gyro,gyro=gyro+rng.normal(0,.002,3),
                                accelerometer_truth=accel,accelerometer=accel+rng.normal(0,.03,3))
                recorder.append('commands',t,requested_twist=np.asarray(requested),applied_twist=np.asarray(applied),wheel_target=targets)
            if renderer and tick%50==0:
                for name in cameras:
                    renderer.update_scene(data,'element/'+name,scene_option=option)
                    rgb=renderer.render().copy()
                    renderer.enable_depth_rendering();depth=renderer.render().copy();renderer.disable_depth_rendering()
                    renderer.enable_segmentation_rendering()
                    try:raw=renderer.render().copy()
                    except IndexError as exc:
                        raise PipelineError('RENDER_SEGMENTATION',f'Invalid segmentation ID for {name} at {t:.3f}s: {exc}') from exc
                    finally:renderer.disable_segmentation_rendering()
                    geom=raw[:,:,0];valid=(raw[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))&(geom>=0)&(geom<model.ngeom)
                    indices=np.where(valid,geom,model.ngeom)
                    camera_id=model.camera('element/'+name).id
                    recorder.append('camera_'+name,t,rgb=rgb,depth=depth,instance=instances[indices],semantic=classes[indices],
                                    position_world=data.cam_xpos[camera_id].copy(),rotation_world=data.cam_xmat[camera_id].reshape(3,3).copy())
                    frames[name]=rgb
                    if controller and controller_config.get('perception')=='oracle_visible':
                        detections=visible_detections(instances[indices],instance_ids,manifest)
                        shadow=next((a for a in physical_arms.values() if a['camera']==name),None) if multi else controller_config.get('shadow_arm')
                        row=shadow_row_profile(model,data,chassis,shadow['position_chassis_m'],shadow.get('quaternion_chassis_wxyz',[1,0,0,0])) if shadow and name==shadow['camera'] else []
                        from .rhizome import visible_row_geometry
                        foliage=visible_row_geometry(instances[indices],instance_ids,manifest)
                        observation=controller.observe(t,name,detections,row,foliage)
                        oracle_count+=len(detections);tracked_count+=len(observation['tracks']['detections'])
                        camera_counts[name]['oracle_detections']+=len(detections)
                        camera_counts[name]['tracked_detections']+=len(observation['tracks']['detections'])
                        recorder.append('rhizome_'+name,t,oracle_detection_count=len(detections),
                                        tracked_detection_count=len(observation['tracks']['detections']))
                    elif controller and controller_config.get('perception')=='neural':
                        observation=controller.infer(t,name,rgb=rgb,depth=depth,K=np.asarray(cameras[name]['K']),
                            camera_position_world=data.cam_xpos[camera_id].copy(),
                            camera_rotation_world=data.cam_xmat[camera_id].reshape(3,3).copy(),
                            chassis_position_world=data.xpos[chassis].copy(),
                            chassis_rotation_world=data.xmat[chassis].reshape(3,3).copy())
                        count=len(observation['prediction']['detections'])
                        neural_count+=count;tracked_count+=len(observation['tracks']['detections'])
                        camera_counts[name]['neural_detections']+=count
                        camera_counts[name]['tracked_detections']+=len(observation['tracks']['detections'])
                        recorder.append('rhizome_'+name,t,neural_detection_count=count,
                            tracked_detection_count=len(observation['tracks']['detections']),
                            row_profile_count=len((observation['row_profile'] or {}).get('profile',[])),
                            inference_wall_seconds=observation['inference_wall_seconds'])
                        from .rhizome import neural_overlay
                        neural_frames[name]=neural_overlay(rgb,observation['prediction'],cameras[name]['K'],
                            data.cam_xmat[camera_id].reshape(3,3),data.cam_xpos[camera_id],
                            data.xmat[chassis].reshape(3,3),data.xpos[chassis],
                            controller_config['neural']['target_positions_chassis_m'][name],
                            controller_config['neural'].get('target_quaternions_chassis_wxyz',{}).get(name,[1,0,0,0]))
            max_contacts=max(max_contacts,data.ncon)
            for contact in data.contact:
                names=[model.geom(int(g)).name for g in contact.geom]
                if any(n.startswith('element/') for n in names) and not all(n=='terrain' or n.startswith('element/') for n in names):
                    obstacle_contacts.add(tuple(names))
            up=data.xmat[chassis].reshape(3,3)[2,2]
            xy=data.xpos[chassis,:2]
            rotation=data.xmat[chassis].reshape(3,3)[:2,:2]
            corners=robot_footprint(profile)@rotation.T+xy
            if (corners<bounds[0]).any() or (corners>bounds[1]).any():failure='Robot footprint left the field'
            if up<.4:failure='Robot tipped over'
            if obstacle_contacts:failure='Robot contacted an obstacle'
            warnings=np.maximum(warnings,np.array([w.number for w in data.warning]))
            if warnings.any():failure='MuJoCo reported a simulation warning'
            if failure or tick==steps:break
            mujoco.mj_step(model,data)
    except PipelineError as exc:
        failure=f'{exc.code}: {exc}'
    finally:
        data.ctrl[:]=0
        if controller:controller.close()
        recorder.close()
        if renderer:renderer.close()
    data.ctrl[:]=0
    for name,frame in frames.items():Image.fromarray(frame).save(output/(name+'.png'))
    for name,frame in neural_frames.items():frame.save(output/(name+'_neural.png'))
    positions=np.asarray(positions)
    trajectory_image(output/'trajectory.png',positions,bounds,manifest['rows'])
    report=dict(flow='agriculture-drive',passed=failure is None,failure=failure,simulated_seconds=float(data.time),
                success_criteria='Completed duration without simulation warnings, field exit, tipping or obstacle contact; motion tracking is not calibrated',
                wall_seconds=time.perf_counter()-started,warnings=warnings.tolist(),max_contacts=max_contacts,
                obstacle_contacts=sorted(obstacle_contacts),streams=inspect(output/'data.h5'),
                displacement_m=float(np.linalg.norm(positions[-1,:2]-positions[0,:2])) if len(positions) else 0.,
                calibrated=False,source_revision=generation['provenance'].get('inicio_revision'),
                limitations=profile['limitations']+(native_info['limitations'] if native_info else ['Timed commands; no autonomous controller']))
    if controller_config:
        final_goal=np.asarray(controller_config['waypoints'][-1]['xy'])
        goal_distance=float(np.linalg.norm(data.xpos[chassis,:2]-final_goal))
        arrival=(native_info or {}).get('local_control_parameters',{}).get('arrival_distance',.25)
        report['rhizome']=dict(provenance=native_info,goal_distance_m=goal_distance,goal_reached=goal_distance<arrival,
                               oracle_detections=oracle_count,tracked_detections=tracked_count,
                               neural_detections=neural_count,
                               arm_states=sorted(shadow_states),arm_target_ids=[] if multi else sorted(shadow_targets),
                               shadow_states=[] if physical_arms else sorted(shadow_states),
                               shadow_target_ids=[] if physical_arms else sorted(shadow_targets),
                               physical_arm=physical_arm['id'] if physical_arm else None,
                               arms={name:dict(camera=a['camera'],states=sorted(arm_states[name]),target_ids=sorted(arm_targets[name])) for name,a in physical_arms.items()},
                               cameras=camera_counts,arm_feedback='MuJoCo' if physical_arms else 'software motor model',
                               physical_weeding=False)
    if benchmark:
        from .weeding import evaluate,evaluate_multi
        report['simulation_passed']=report['passed']
        report['weeding']=(evaluate_multi(output,manifest,physical_arms,simulation_passed=report['passed']) if multi else
            evaluate(output,manifest,physical_arm,simulation_passed=report['passed']))
        report['passed']=report['passed'] and report['weeding']['passed']
        report['success_criteria']='Completed simulation and ordered weed detection, tracking, selection and geometric tool contact, with no crop protection contact'
    write_json(output/'report.json',report)
    if video and shutil.which('ffmpeg') and len(positions)>1:
        from .replay import video as replay
        replay(output,model=model)
    from .inspection import scene_page
    scene_page(output)
    return report
