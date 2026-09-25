"""Private JSON-lines worker. Only imports Rhizome after selecting its build.

Native stdout is redirected to stderr before constructing nodes. stdout is
reserved for the request/reply protocol. No live proxy or hardware node runs.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


def timestamp(t):
    ns=round((t+1)*1_000_000_000)  # Keep zero's special 'unset' meaning out of capture timestamps.
    return dict(secs=ns//1_000_000_000,nsecs=ns%1_000_000_000)


class Runtime:
    def __init__(self,config,output):
        import yaml
        self.config=config
        self.root=Path(config['root']).expanduser().resolve()
        host=Path(config.get('host_build',self.root/'build/host')).expanduser().resolve()
        if (host/'build_manifest.json').exists():
            if __package__:
                from .rhizome_build import verify
            else:
                from rhizome_build import verify
            verify(host)
        build=host/'py'
        sys.path.insert(0,str(build))
        from pyzome import core, nav, node_test
        self.p,self.nt=core,node_test
        conf=Path(output)/'conf';(conf/'nodes').mkdir(parents=True)
        self.sources={}
        def documents(relative):
            path=self.root/relative
            self.sources[relative]=hashlib.sha256(path.read_bytes()).hexdigest()
            return list(yaml.safe_load_all(path.read_text()))
        self.documents=documents
        meta,params=documents('src/nav/local_control.node.yml')
        params.update(auto_speed_max_ms=.25,turn_speed=.2,boost_speed_max_ms=0.,goals_stale_sec=0.)
        params.update(config.get('parameters',{}))
        # Every fixture tick runs once. Freshness is enforced by the parent RPC
        # deadline, not production wall-clock goals expiry or speed boosting.
        (conf/'nodes/local_control.json').write_text(json.dumps(dict(meta=meta,config=params)))
        (conf/'payloads.json').write_text('{"payloads":[]}')
        os.environ['AIGEN_CONFIG_PATH']=str(conf)
        self.navigation_enabled=config.get('navigation_enabled',True)
        if self.navigation_enabled:
            self.command=core.DriveCtrlMsg()
            self.fixture,self.node=self.create_node('local_control',nav.LocalControl,{'/nav/drive':self.command})
            for key,value in params.items():
                actual=self.node.config_.get(key)
                same=math.isclose(actual,value,rel_tol=1e-6,abs_tol=1e-8) if isinstance(value,(int,float)) else actual==value
                if not same:
                    raise ValueError(f'Native local_control config differs at {key}; rebuild the host bindings or remove installed task overrides')
        self.goals=core.WaypointsMsg({'waypoints':[
            dict(point=dict(x=w['xy'][0],y=w['xy'][1],frame='odom'),
                 dir=w.get('direction','FORWARD'),type=w.get('type','NORMAL')) for w in config['waypoints']]})
        self.trackers={};self.conf=conf;self.arm=None;self.last_time=None
        self.row_profile=None
        self.output=Path(output)
        self.tracker_config=documents('src/vision/tracker_0.node.yml') if config.get('perception','none')!='none' else None
        self.last_command=[0.,0.]
        if config.get('shadow_arm'):
            self.init_arm(config['shadow_arm'])
        self.neural=None
        if config.get('perception')=='neural':
            sys.path.insert(0,str(self.root/'src'))
            from rhizome_perception import NeuralPerception
            self.neural=NeuralPerception({**config['neural'],'require_geometry':self.arm is not None},core)
            for relative in ('src/crop_inference_engine/model/model.py','src/crop_inference_engine/model/runtime.py',
                             'src/crop_inference_engine/model/package.py','src/crop_inference_engine/model/species.py',
                             'src/crop_inference_engine/msg/publisher.py','src/crop_inference_engine/msg/geometry.py'):
                self.sources[relative]=hashlib.sha256((self.root/relative).read_bytes()).hexdigest()
        modules={name:hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((build/'pyzome').glob('*.so')) if (name:=path.name)}
        revision=subprocess.run(['git','-C',str(self.root),'rev-parse','HEAD'],capture_output=True,text=True).stdout.strip()
        self.info=dict(root=str(self.root),revision=revision,host_build=str(host),
                       build_manifest=json.loads((host/'build_manifest.json').read_text()) if (host/'build_manifest.json').exists() else None,
                       modules_sha256=modules,source_sha256=self.sources,
                       navigation_enabled=self.navigation_enabled,local_control_parameters=params,adapter='native NodeTestFixture; inproc bus',
                       clocks='MuJoCo schedules steps, geometry age and registry expiry; ODOM and planner detection/camera timestamps use simulation seconds + 1; node and row-profile headers use host clock',
                       perception=config.get('perception','none'),shadow_arm=config.get('shadow_arm'),
                       limitations=['LocalControl only; route scheduling, drive_mux and hardware drive_control are not running',
                                    'Truth odometry; no wheel odometry/RTK estimation',
                                    'Native build hashes identify binaries independently of checkout revision',
                                    'Oracle detections are not neural inference; fixed 2 cm keypoint size',
                                    'Shadow arm uses Rhizome software motor model; no MuJoCo arm or soil contact'])
        if self.arm:
            (Path(output)/'shadow_arm_config.json').write_text(json.dumps(self.arm_snapshot,indent=2))
        if config.get('physical_arm'):
            self.info['physical_arm']=config['physical_arm']
            self.info['limitations'][-1]='Selected arm receives MuJoCo motor feedback; native planner tool geometry and dynamics remain estimates; no plant removal'
        if self.neural:
            self.info['neural']=self.neural.info
            self.info['limitations'][3]='Neural perception on ideal rendered RGB/depth; inference targets use configured payload frames'

    def create_node(self,name,kind,subscriptions):
        os.environ['NODE_NAME']=name
        with self.nt.suppress_stdout():
            fixture=self.nt.NodeTestFixture()
            node=kind()
            fixture.add_node(node,subscriptions)
        os.environ.pop('NODE_NAME',None)
        return fixture,node

    def init_arm(self,config):
        from pyzome import physics
        if not hasattr(self.p,'FoliageScene') or not hasattr(physics.PhysicsEnvironment,'get_foliage_state'):
            raise ValueError('Incompatible Rhizome host build: separate crop-row/foliage geometry is required. '
                             'Rebuild current host bindings and select controller.host_build or RHIZOME_HOST_BUILD.')
        self.ph=physics
        hardware=self.documents('test/data/physics/arm_1.yml')[0]
        tools=self.documents('src/base_control/arms.yml')[0]
        params=self.documents('src/physics/physics_0.node.yml')[1]
        self.arm_snapshot=dict(hardware=hardware,tools=tools,physics=params,mount=config,
                               note='Provisional stock test arm, chassis-aligned; geometry, joint limits and mount are not calibrated to RAPTOR_30')
        self.arm_config=physics.ArmConfig()
        arm=self.arm_config
        arm.tube_length=hardware['tube']['length'];arm.tube_radius=hardware['tube']['radius']
        for axis in ('pitch','yaw'):
            for limit in ('min','max'):setattr(arm,axis+'_'+limit,hardware['extents'][axis][limit])
        if self.config.get('physical_arm'):
            for axis,limits in zip(('yaw','pitch'),config['motor_limits_rad']):
                setattr(arm,axis+'_min',limits[0]);setattr(arm,axis+'_max',limits[1])
            self.arm_snapshot['effective_motor_limits_rad']=config['motor_limits_rad']
            self.arm_snapshot['note']='Gen 3 mount and estimated limits; stock native tool shape is approximate; measured angles come from MuJoCo'
        arm.tool_type=hardware['tool']['type']
        arm.set_tool_points([self.p.Point(p) for p in tools['tool'][arm.tool_type]])
        arm.set_blade_point(self.p.Point(tools['blade']['tip'][arm.tool_type]))
        arm.set_blade_leading_edge_indices(tools['blade']['leading_edge'][arm.tool_type])
        volumes=[]
        for value in hardware.get('volumes',[]):
            volume=physics.VolumeConfig()
            for key in ('name','min_pitch','max_pitch','x1','y1','x2','y2'):
                if key in value:setattr(volume,key,value[key])
            volume.type=physics.VolumeType.LINEAR if value['type']=='linear' else physics.VolumeType.DISCRETE
            volumes.append(volume)
        arm.volumes=volumes
        self.params=physics.PhysicsParams(params)
        self.arm=physics.PhysicsEnvironment()
        environment=physics.EnvironmentConfig()
        for key in ('bounds_min_x','bounds_max_x','bounds_min_y','bounds_max_y'):
            if key in params.get('environment',{}):setattr(environment,key,params['environment'][key])
        self.detection_timeout=params.get('environment',{}).get('detection_timeout',5.)
        # The native registry ages detections with robot_time::now(). Rendering
        # and CPU inference must not consume simulated track lifetime. Retain
        # its configured TTL, but expire through the bound native remove API.
        environment.detection_timeout_seconds=0.
        self.detection_seen={}
        self.arm_snapshot['detection_expiry']=dict(clock='simulation_seconds',timeout_seconds=self.detection_timeout,
            native_wall_timeout_disabled=True)
        self.arm.set_config(environment)
        self.arm.initialize_robot(arm,self.params,.02,.04)
        self.graph=self.p.TransformationGraph()
        x,y,z=config['position_chassis_m']
        qw,qx,qy,qz=config.get('quaternion_chassis_wxyz',[1,0,0,0])
        self.graph.update(self.p.CoordinateFrame(dict(header=dict(topic='/frames/arm_shadow',timestamp=timestamp(0)),
                            origin=dict(x=x,y=y,z=z,qw=qw,qx=qx,qy=qy,qz=qz,frame='robot'))))
        self.arm.set_transform_graph(self.graph,'arm_shadow')
        self.motors=physics.SimMotorController(self.arm)
        self.motors.set_arm_state(self.arm.arm_state_manager)
        self.motors.set_mode(physics.ControllerMode.PLAYBACK if self.config.get('physical_arm') else physics.ControllerMode.SIMULATION)

    def step(self,msg):
        t=msg['time']
        if self.last_time is not None and not math.isclose(t-self.last_time,.01,abs_tol=1e-6):
            raise ValueError('Expected monotonically increasing 100 Hz simulation steps')
        self.last_time=t
        x,y,z=msg['position'];qw,qx,qy,qz=msg['quaternion']
        pose=dict(x=x,y=y,z=z,qw=qw,qx=qx,qy=qy,qz=qz,frame='odom')
        if self.navigation_enabled and round(t*100)%10==0:
            odom=self.p.OdometryMsg(dict(pose=pose,twist=dict(v=msg['forward_mps'],w=0)))
            self.fixture.test_single({'/odom':odom,'/nav/goals':self.goals},lambda:None)
            self.last_command=[self.command.lin,self.command.ang]
        result=dict(twist=self.last_command)
        if self.arm or self.neural:
            # Rhizome roots the graph at ROBOT: /frames/odom contains the
            # ODOM origin expressed in ROBOT, the inverse of MuJoCo's pose.
            r=((1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)),
               (2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)),
               (2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)))
            inv=[-sum(r[j][i]*msg['position'][j] for j in range(3)) for i in range(3)]
            self.odom_frame=self.p.CoordinateFrame(dict(header=dict(topic='/frames/odom',timestamp=timestamp(t)),
                              origin=dict(x=inv[0],y=inv[1],z=inv[2],qw=qw,qx=-qx,qy=-qy,qz=-qz,frame='robot')))
        if self.arm:
            self.graph.update(self.odom_frame)
            _,_,qy,qz=self.config['shadow_arm'].get('quaternion_chassis_wxyz',[1,0,0,0])
            self.arm.set_robot_speed(msg['forward_mps']*(1-2*(qy*qy+qz*qz)))
            if self.row_profile is not None:
                # The native row-profile freshness guard uses host time. The
                # synchronous simulator holds the latest capture until the next
                # camera tick, as Rhizome's offline replay adapter does.
                self.row_profile.header.timestamp=self.p.now()
                self.row_profile.camera_timestamp=self.p.now()
                self.arm.update_row_profile(self.row_profile)
            if self.config.get('physical_arm'):
                yaw,pitch=msg['arm_measured']
                if not all(math.isfinite(v) for v in (yaw,pitch)):raise ValueError('Nonfinite MuJoCo arm feedback')
                self.motors.set_measured_motor_positions(pitch,yaw)
            self.arm.set_measured_motor_angles(self.motors.measured_yaw,self.motors.measured_pitch)
            if self.detection_timeout>0:
                for track_id,seen in list(self.detection_seen.items()):
                    if t-seen>self.detection_timeout:
                        self.arm.remove_detection(track_id);del self.detection_seen[track_id]
            self.arm.step(.01)
            self.motors.step(.01)
            state=self.arm.get_targeting_state()
            result['arm']=dict(commanded=[self.motors.commanded_yaw,self.motors.commanded_pitch],
                               measured=[self.motors.measured_yaw,self.motors.measured_pitch],
                               target_id=self.arm.weeding_mode_controller.locked_target_id,
                               state=str(state.state),detections=len(self.arm.get_detection_states()))
            if self.config.get('targeting_diagnostics'):
                fields=('track_id','x','y','z','size','in_reach_circle','in_reach_triangle',
                        'strike_ready','is_locked_target','is_eliminated','is_valid_candidate',
                        'arm_protection_radius','tool_protection_radius','has_pitch_down_marker')
                diagnostics=[]
                for detection in self.arm.get_detection_states():
                    entry={key:getattr(detection,key) for key in fields}
                    entry['class_label']=str(detection.detection_class).rsplit('.',1)[-1]
                    for key in ('strike_marker_pos','followthrough_marker_pos'):
                        point=getattr(detection,key)
                        entry[key]=[point.x,point.y]
                    diagnostics.append(entry)
                result['arm']['diagnostics']=diagnostics
                geometry=self.arm.get_arm_state()
                result['arm']['workspace']=dict(reach_radius_m=geometry.reach_radius,
                    projected_tube_length_m=geometry.projected_length)
                result['arm']['foliage']=dict(self.arm.get_foliage_state())
                if self.config.get('planner_debug') and round(t*100)%100==0:
                    from PIL import Image
                    folder=self.output/'planner';folder.mkdir(exist_ok=True)
                    Image.fromarray(self.ph.debug_render(self.ph.DebugView(),self.arm)[:,:,::-1]).save(folder/f'{t:06.2f}.png')
        if not all(math.isfinite(v) for v in result['twist']):raise ValueError('Nonfinite native drive command')
        return result

    def observe(self,msg,inf=None,frames=()):
        from pyzome import tracker
        camera=msg['camera']
        if camera not in self.trackers:
            meta,params=self.tracker_config
            (self.conf/'nodes/tracker_0.json').write_text(json.dumps(dict(meta=meta,config=params)))
            output=self.p.TrackerMsg()
            fixture,node=self.create_node('tracker_0',tracker.TrackerNode,{'/vision/crop_trks/0':output})
            if node.configs.min_hits!=params['min_hits']:
                raise ValueError('Native tracker did not load the selected configuration')
            self.trackers[camera]=(fixture,node,output)
        fixture,node,output=self.trackers[camera]
        # Oracle detections already live in ODOM. Seed the history at capture
        # time so the tracker's timestamp lookup has a valid snapshot.
        if inf is None:
            node.transforms.update(self.p.CoordinateFrame(dict(header=dict(topic='/frames/odom',timestamp=timestamp(msg['time'])),
                                                             origin=dict(qw=1,frame='robot'))))
            foliage=msg.get('foliage')
            if foliage is None:
                if any(d['class_label']=='CROP' for d in msg['detections']):
                    raise ValueError('Oracle crop detections require an explicit row geometry snapshot')
                foliage=dict(present=True,rows=[],polygons=[])
            inf=self.p.TrackerMsg(dict(camera_timestamp=timestamp(msg['time']),camera_latency=0.,
                                      inference_latency=0.,detections=msg['detections'],foliage=foliage))
        else:
            for frame in frames:node.transforms.update(frame)
        fixture.test_single({'/vision/crop_infs/0':inf},lambda:None)
        if frames and output.camera_timestamp.nanoseconds()!=inf.camera_timestamp.nanoseconds():
            raise ValueError('Tracker did not publish for this neural capture; check its frame history')
        tracked=json.loads(str(output))
        if self.arm and camera==self.config['shadow_arm']['camera']:
            if any(d['keypoint']['kp']['frame']!='odom' for d in tracked['detections']):
                raise ValueError('Planner requires ODOM tracks')
            # Foliage and obstacle age is measured against the ODOM timestamp;
            # both capture clocks must stay on the simulation timeline.
            arm_message=self.p.TrackerMsg(tracked)
            arm_message.header.timestamp=output.camera_timestamp
            self.arm.update_detections(arm_message)
            for detection in tracked['detections']:
                self.detection_seen[detection['track_id']]=msg['time']
            profile=self.p.RowProfileMsg(dict(profile=msg['row_profile']))
            profile.header.timestamp=self.p.now();profile.camera_timestamp=self.p.now()
            self.arm.update_row_profile(profile)
            self.row_profile=profile
        return dict(tracks=tracked)

    def infer(self,msg):
        import numpy as np
        camera=msg['camera']
        if self.neural is None:raise ValueError('Neural perception is not configured')
        with np.load(msg['frame_file'],allow_pickle=False) as arrays:
            inf,row,density,diagnostics=self.neural.infer(camera,round((msg['time']+1)*1e9),arrays)
        x,y,z=self.config['neural']['target_positions_chassis_m'][camera]
        qw,qx,qy,qz=self.config['neural'].get('target_quaternions_chassis_wxyz',{}).get(camera,[1,0,0,0])
        target=self.p.CoordinateFrame(dict(header=dict(topic='/frames/inference_'+camera,timestamp=timestamp(msg['time'])),
                                          origin=dict(x=x,y=y,z=z,qw=qw,qx=qx,qy=qy,qz=qz,frame='robot')))
        if self.arm:
            self.graph.update(target);self.graph.update(self.odom_frame)
        # Feed only the model's row profile. Empty predictions remain empty.
        row_points=json.loads(str(row))['profile'] if row is not None else []
        result=self.observe(dict(time=msg['time'],camera=camera,row_profile=row_points),inf=inf,frames=(target,self.odom_frame))
        if self.arm and camera==self.config['shadow_arm']['camera'] and density is not None:
            values=diagnostics['density']['densities']
            self.arm.update_density_metrics(values)
            weeds=sum(values.get(key,0) for key in ('WEED','WEED_STEM','GRASS','GRASS_STEM'))
            total=weeds+sum(values.get(key,0) for key in ('CROP','CROP_STEM','BACKGROUND'))
            if total>0:self.arm.set_weed_pressure(weeds/total)
        return {**result,**diagnostics}


def main():
    protocol=os.fdopen(os.dup(1),'w',buffering=1)
    os.dup2(2,1)
    runtime=None
    for line in sys.stdin:
        try:
            msg=json.loads(line)
            if msg['op']=='init':
                runtime=Runtime(msg['config'],msg['output']);result=runtime.info
            elif msg['op']=='step':result=runtime.step(msg)
            elif msg['op']=='observe':result=runtime.observe(msg)
            elif msg['op']=='infer':result=runtime.infer(msg)
            else:raise ValueError('Unknown worker operation')
            protocol.write(json.dumps(result,allow_nan=False)+'\n')
        except Exception as exc:
            traceback.print_exc()
            protocol.write(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'))+'\n')
            return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
