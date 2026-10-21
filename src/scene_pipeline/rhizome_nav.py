"""Run Rhizome's unmodified NavModel tick with simulation-owned inputs/time."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation


class Navigation:
    def __init__(self,runtime,config):
        from nav_model import nav_model as nm
        from nav_model.nav_model_package import NavModelPackage
        self.r=runtime;self.config=config;self.nm=nm;self.time=1.;self.messages={};self.invariants={}
        # Shadow: nav_model only watches the runtime's goals (e.g. the ai_agent's)
        # for obstacles, as on the rover where navigation or the agent drives.
        self.shadow=config.get('drive',True) is False
        self.services=[];self.obstacle={}
        self.grid_capable=hasattr(nm.NavModel,'_configure_model')
        if (config.get('depth') or config.get('obstacle_detection')) and not self.grid_capable:
            raise ValueError('Navigation depth/obstacles require the Rhizome obstacle-grid implementation')
        core=runtime.p
        # Only this module sees the clock facade; native node scheduling and
        # the other simulation workers retain their own clocks.
        class Clock:
            def __getattr__(_,name):return getattr(core,name)
            def now(_):return core.nanoseconds(round(self.time*1e9))
        nm.pyzome=Clock()
        task=runtime.documents('src/tasks/nav-model-drive.yml')[0]
        self.goal_timeout=float(task['nodes']['local_control']['goals_stale_sec'])
        if not np.isfinite(self.goal_timeout) or self.goal_timeout<=0:
            raise ValueError('nav-model-drive requires a positive model-goal timeout')
        meta,params=runtime.documents('src/nav_model/nav_model.node.yml')
        meta.pop('includes',None)
        params.update(task['nodes']['nav_model'])
        params.update(config.get('params',{}))  # node.yml overrides, e.g. obstacle_max_height_m
        params.update(drive=not self.shadow,use_map=False,model_dir=config['package'],model_file=config['model_file'],
                      cameras=config['cameras'],arrangement=config['arrangement'],steer_source=config['steer_source'],
                      camera=dict(nav=dict(color=dict(format=dict(shm='RGBA')))))
        if self.grid_capable:
            dimensions=config.get('robot_dimensions')
            if hasattr(core,'RobotDescription'):
                # Rhizome reads the footprint from /robot/description: robot_cad.yml's
                # default robot, resized to the simulated chassis when given.
                core.RobotDescription.set_test_description(sim_description(core,runtime,dimensions))
            else:
                params.update(dimensions=dimensions or runtime.documents('tools/provisioning/robot/config/dimensions.yml')[0])
            params.update(obstacle_incident_dir=str((runtime.output/'obstacle_incidents').resolve()))
            params['camera']['nav']['combo']=dict(format=dict(shm='RGBA'))
        (runtime.conf/'nodes/nav_model.json').write_text(json.dumps(dict(meta=meta,config=params)))
        adapter=self
        class SimNavModel(nm.NavModel):
            def Publish(self,topic,message):
                message.header.timestamp=core.nanoseconds(round(adapter.time*1e9))
                message.header.topic=topic
                adapter.messages[topic]=json.loads(str(message))
            def assert_invariant(self,condition,name):adapter.invariants[name]=bool(condition)
            def _call_service(self,name,message):
                # Retain light/snapshot requests as evidence; no hardware services.
                adapter.services.append(dict(service=name,message=json.loads(str(message))))
        package=NavModelPackage(config['package'],device=config['device'],model_file=config['model_file'],num_threads=config.get('threads',2))
        warm=package.warmup(batch=len(config['cameras']))
        error=package.selftest()
        if error is not None and (not np.isfinite(error) or error>1e-3):
            raise ValueError(f'Navigation package selftest failed: {error}')
        self.fixture,self.node=runtime.create_node('nav_model',lambda:SimNavModel(frame_sources={},package_loader=lambda _:package),{})
        self.node.model=package;self.node._loaded_source=config['package'];self.node._debug_factory=None
        if self.grid_capable:
            package.reset()
            self.node._configure_model()
        self.node.mux_state.requested=core.ControlSelect.AUTONOMY
        self.last_capture=None;self.goals=core.WaypointsMsg();self.result={}
        root=Path(config['package'])
        self.info=dict(task='nav-model-drive',node='Rhizome NavModel.Tick → native LocalControl',
            package=str(root),device=config['device'],selftest_max_error=error,warmup_ms=warm,
            parameters=params,goal_watchdog_seconds=self.goal_timeout,files_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file()},
            input='RGB, camera calibration and truth odometry; no plant metadata, row truth, depth or masks',
            clocks='Simulation time for NavModel and 0.5 s model-goal watchdog; synchronous inference pauses simulation',
            limitations=['Mapless mode; navigation/geofence/drive_mux services are not running',
                         'Nominal optical extrinsics and estimated mounting height/intrinsics; not sensor-calibrated'])
        if self.grid_capable:
            self.info.update(obstacle_detection=bool(config.get('obstacle_detection')),depth=bool(config.get('depth')),
                input='RGB, ideal rendered depth when enabled, camera calibration, robot dimensions and truth odometry; no scene labels or obstacle coordinates',
                depth_units='GRAY16 0.1 mm; zero for invalid/out-of-uint16-range rays',
                stopping='Rhizome NavModel OBSTACLE gate withholds goals; native LocalControl plus simulation goal watchdog stops wheels',
                services='Light and snapshot requests recorded locally; no external service calls')
            self.info['limitations'].append('Ideal depth lacks stereo noise/holes; drive_control directional clip and hardware braking are not exercised')

    def infer(self,t,arrays):
        p=self.r.p;node=self.node;self.time=t+1
        capture=round(self.time*1e9);stamp=p.nanoseconds(capture)
        self.services=[]
        if self.grid_capable:
            node.nav_settings.header.timestamp=stamp
            node.nav_settings.obstacle_detection.enabled=bool(self.config.get('obstacle_detection'))
            node.drive_cmd.lin,node.drive_cmd.ang=self.r.last_command
        node.odom=p.OdometryMsg(dict(header=dict(timestamp=dict(secs=capture//10**9,nsecs=capture%10**9)),
            pose=self.r.pose,twist=dict(v=self.r.forward_mps,w=0)))
        node._on_odom(node.odom)
        # TransformationGraph.at(t) selects strictly BEFORE t. Publish the
        # synchronous capture pose 1 ns earlier so its snapshot uses this pose,
        # including the first frame, rather than the preceding camera tick.
        transform_ns=capture-1
        transform_time=dict(secs=transform_ns//10**9,nsecs=transform_ns%10**9)
        odom_frame=p.CoordinateFrame(json.loads(str(self.r.odom_frame)))
        odom_frame.header.timestamp=p.nanoseconds(transform_ns)
        node.transforms.update(odom_frame)
        node.frames={};self.messages={};self.invariants={}
        for position in self.config['cameras']:
            key=position+'_'
            if key+'rgb' not in arrays:continue
            rgb=arrays[key+'rgb'];K=arrays[key+'K']
            rotation=arrays[key+'rotation_robot'];xyz=arrays[key+'position_robot']
            q=Rotation.from_matrix(rotation).as_quat()[[3,0,1,2]]
            node.transforms.update(p.CoordinateFrame(dict(header=dict(topic='/frames/nav_'+position,
                timestamp=transform_time),origin=dict(
                    x=xyz[0],y=xyz[1],z=xyz[2],qw=q[0],qx=q[1],qy=q[2],qz=q[3],frame='robot'))))
            intr=p.CamIntrinsicsMsg();intr.header.timestamp=stamp
            intr.params.fx=float(K[0,0]);intr.params.fy=float(K[1,1])
            intr.params.cx=float(K[0,2]);intr.params.cy=float(K[1,2])
            intr.params.frame_width=rgb.shape[1];intr.params.frame_height=rgb.shape[0]
            intr.params.cam_height=float(arrays[key+'height_m'])
            node.intrinsics[position]=intr
            rgba=np.concatenate([rgb,np.full((*rgb.shape[:2],1),255,np.uint8)],axis=2)
            if self.config.get('depth'):
                depth=arrays.get(key+'depth')
                if depth is None or depth.shape!=rgb.shape[:2] or depth.dtype!=np.uint16:
                    raise ValueError('Navigation depth requires an aligned uint16 GRAY16 plane per camera')
                node.frames[position]=self.nm.Frame(rgba,capture,depth)
            else:node.frames[position]=self.nm.Frame(rgba,capture)
        node._camera_errors={pos:'Missing simulated camera frame' for pos in self.config['cameras'] if pos not in node.frames}
        if self.shadow:node.goals=self.r.goals
        start=time.perf_counter();node.Tick()
        prediction=self.messages.get('/nav_model/prediction',{})
        healthy=all(self.invariants.values())
        goals=self.messages.get('/nav_model/goals',{}) if healthy else {}
        self.goals=p.WaypointsMsg(goals);self.last_capture=t
        self.result=dict(simulation_time=t,prediction=prediction,goals=json.loads(str(self.goals)),
            nav_goal=self.messages.get('/nav/goal'),invariants=self.invariants.copy(),
            driving=healthy and bool(prediction.get('driving')) and bool(self.goals.waypoints),
            inference_wall_seconds=time.perf_counter()-start)
        if self.grid_capable:
            self.obstacle=self.messages.get('/vision/obstacle',self.obstacle)
            self.result.update(obstacle=self.obstacle,obstacle_health=self.messages.get('nav_obstacle/health'),
                               service_requests=self.services.copy(),seen=self.messages.pop('/nav_model/seen',None))
        return self.result

    def command_goals(self,t):
        if self.shadow:return self.r.goals
        fresh=self.last_capture is not None and 0<=t-self.last_capture<=self.goal_timeout
        return self.goals if fresh and self.result.get('driving') else self.r.p.WaypointsMsg()


def sim_description(core,runtime,dimensions):
    """RobotDescriptionMsg of robot_cad.yml's default robot, its chassis
    resized to `dimensions` if given (dimensions.yml layout: robot, drive, wheel)."""
    from robot_description.resolve import resolve
    resolved=resolve(runtime.documents('src/robot_description/robot_cad.yml')[0],
                     runtime.documents('src/base_control/tools.yml')[0],'','')
    message=resolved.message()
    if not dimensions:
        return core.RobotDescriptionMsg(json.dumps(message))
    description=json.loads(message['description'])
    chassis=description['chassis']
    chassis['body'].update(length=dimensions['robot']['length'],width=dimensions['robot']['width'])
    chassis['drive']['trackwidth']=dimensions['drive']['trackwidth']
    chassis['wheel']['radius']=dimensions['wheel']['radius']
    description['wheelbase']=dimensions['drive']['wheelbase']
    message['description']=json.dumps(description)
    return core.RobotDescriptionMsg(json.dumps(message))
