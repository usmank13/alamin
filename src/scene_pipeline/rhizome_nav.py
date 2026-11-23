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
        # Arbitrated: nav_model drives only while /nav/source is NAV_MODEL
        # (goals_topic /nav/goals); otherwise it watches the runtime's goals.
        self.arbitrated=bool(config.get('arbitrated'))
        self.source='NONE'
        # Surveyed field boundaries ([[xmin, ymin], [xmax, ymax]], world m). The
        # route of the field nearest the robot is the loaded one, as if an
        # operator loaded each field's route on arrival; world == MAP.
        self.fields=config.get('fields') or [];self.field=None
        self.services=[];self.obstacle={}
        self.previous_grid = None
        self.first_violation_tick = None
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
        # dual: nav_model picks RTK plan vs its own goals (nav_source) and
        # LocalControl follows the model's while they are fresh.
        self.task_name=config.get('task','nav-model-drive');self.switched=self.task_name=='dual'
        task=runtime.documents(f'src/tasks/{self.task_name}.yml')[0]
        self.goal_timeout=float(runtime.documents('src/nav/local_control.node.yml')[1]['model_goals_stale_sec']
                                if self.switched else task['nodes']['local_control']['goals_stale_sec'])
        if not np.isfinite(self.goal_timeout) or self.goal_timeout<=0:
            raise ValueError('nav-model-drive requires a positive model-goal timeout')
        meta,params=runtime.documents('src/nav_model/nav_model.node.yml')
        meta.pop('includes',None)
        params.update(task['nodes']['nav_model'])
        params.update(config.get('params',{}))  # node.yml overrides, e.g. obstacle_max_height_m
        if self.arbitrated:params['goals_topic']='/nav/goals'
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
        # GPU kernels differ from the CPU-traced reference by ~0.13 (sub-pixel), as on the Orin.
        limit=.25 if config['device']=='cuda' else 1e-3
        if error is not None and (not np.isfinite(error) or error>limit):
            raise ValueError(f'Navigation package selftest failed: {error}')
        self.fixture,self.node=runtime.create_node('nav_model',lambda:SimNavModel(frame_sources={},package_loader=lambda _:package),{})
        self.node.model=package;self.node._loaded_source=config['package'];self.node._debug_factory=None
        if self.grid_capable:
            package.reset()
            self.node._configure_model()
        self.node.mux_state.requested=core.ControlSelect.AUTONOMY
        self.last_capture=None;self.goals=core.WaypointsMsg();self.result={}
        root=Path(config['package'])
        self.info=dict(task=self.task_name,node='Rhizome NavModel.Tick → native LocalControl',
            package=str(root),device=config['device'],selftest_max_error=error,warmup_ms=warm,
            parameters=params,goal_watchdog_seconds=self.goal_timeout,files_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file()},
            input='RGB, camera calibration and truth odometry; no plant metadata, row truth, depth or masks',
            clocks='Simulation time for NavModel and 0.5 s model-goal watchdog; synchronous inference pauses simulation',
            limitations=[('dual: route has_rows and the RTK plan are supplied by the harness; ' if self.switched else 'Mapless mode; ')+
                         'navigation/geofence/drive_mux services are not running',
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
        if not self.r.estimator:
            node.odom=p.OdometryMsg(dict(header=dict(timestamp=dict(secs=capture//10**9,nsecs=capture%10**9)),
                pose=self.r.pose,twist=dict(v=self.r.forward_mps,w=self.r.yaw_rps)))
            node._on_odom(node.odom)
        # TransformationGraph.at(t) selects strictly BEFORE t. Publish the
        # synchronous capture pose 1 ns earlier so its snapshot uses this pose,
        # including the first frame, rather than the preceding camera tick.
        transform_ns=capture-1
        transform_time=dict(secs=transform_ns//10**9,nsecs=transform_ns%10**9)
        if not self.r.estimator:
            odom_frame=p.CoordinateFrame(json.loads(str(self.r.odom_frame)))
            odom_frame.header.timestamp=p.nanoseconds(transform_ns)
            node.transforms.update(odom_frame)
        if self.fields:
            if (field:=nearest_field(self.fields,self.r.pose['x'],self.r.pose['y']))!=self.field:self.load_field(field)
            # RTK: MAP == ODOM == the simulation world.
            map_frame=p.make_frame(p.Pose(frame=p.Frame.MAP),p.Pose(frame=p.Frame.ODOM))
            map_frame.header.topic='/frames/map';map_frame.header.timestamp=p.nanoseconds(transform_ns)
            node.transforms.update(map_frame)
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
        if self.shadow or self.arbitrated or self.switched:node.goals=self.r.goals
        if self.switched:
            rows=self.config.get('route_has_rows',False)
            if isinstance(rows,list):rows=([False]+[v for start,v in rows if start<=t])[-1]
            node.route.has_rows=rows;node.route.header.timestamp=stamp
            # navigation is not running: always healthy. older_than() reads
            # the wall clock, so this one stamp is wall time.
            node.nav_health.header.timestamp=p.now()
        start=time.perf_counter();node.Tick()
        prediction=self.messages.get('/nav_model/prediction',{})
        healthy=all(self.invariants.values())
        goals=self.messages.get(getattr(node,'goals_topic','/nav_model/goals'),{}) if healthy else {}
        self.goals=p.WaypointsMsg(goals);self.last_capture=t
        self.result=dict(simulation_time=t,prediction=prediction,goals=json.loads(str(self.goals)),
            nav_goal=self.messages.get('/nav/goal'),invariants=self.invariants.copy(),
            driving=healthy and bool(prediction.get('driving')) and bool(self.goals.waypoints),
            map_mode=bool(getattr(node,'_map_mode',False)),route_has_rows=bool(node.route.has_rows) if self.switched else None,
            inference_wall_seconds=time.perf_counter()-start)
        if self.grid_capable:
            self.obstacle=self.messages.get('/vision/obstacle',self.obstacle)
            self.result.update(obstacle=self.obstacle,obstacle_health=self.messages.get('nav_model/obstacle_health',self.messages.get('nav_obstacle/health')),
                               service_requests=self.services.copy(),seen=self.messages.pop('/nav_model/seen',None))
        if self.r.course_distance is not None and self.grid_capable and prediction.get('obstacle_enabled'):
            post = node.model.post
            zones = getattr(post, 'last_zone_id', None)
            if zones is not None:
                # Zone IDs [0, K) identify chassis clearances; K identifies a hole.
                hole_id = post.envelope.zones.shape[0]
                self.result['violation_cells'] = {
                    'hole': (zones == hole_id).sum(dim=(1, 2)).tolist(),
                    'obstacle': ((zones >= 0) & (zones < hole_id)).sum(dim=(1, 2)).tolist(),
                }
            if hasattr(post, 'last_ground_z'):
                nav_tick = round(t * 10)
                periodic = nav_tick % 100 == 0
                watch = 'shuttle' in self.r.config and (self.first_violation_tick is None or
                                                         nav_tick == self.first_violation_tick + 1)
                if periodic or watch:
                    folder = self.r.output/'grid'
                    folder.mkdir(exist_ok=True)
                    grid = dict(time=t, ground_z_m=post.last_ground_z.cpu().numpy().copy(),
                                valid=post.last_ground_valid.cpu().numpy().copy(),
                                origin_xy_m=np.array(post.accumulator.origin), cell_m=node.model.cell_m)
                    if hasattr(post, 'last_plane_z'):
                        grid['plane_z_m'] = post.last_plane_z.cpu().numpy().copy()
                    if zones is not None:
                        grid['zone_id'] = zones.cpu().numpy().copy()
                        grid['hole_id'] = hole_id
                    if periodic:
                        np.savez_compressed(folder/f'{t:.1f}.npz', **grid)
                    if watch:
                        if self.first_violation_tick is not None:
                            np.savez_compressed(folder/'after_first_violation.npz', **grid)
                        elif any(obstacle(prediction, d)['violations'] for d in ('fwd', 'rev')):
                            self.first_violation_tick = nav_tick
                            if self.previous_grid is not None:
                                np.savez_compressed(folder/'before_first_violation.npz', **self.previous_grid)
                            np.savez_compressed(folder/'first_violation.npz', **grid)
                        self.previous_grid = grid
        return self.result

    def set_source(self,name):
        self.source=name
        message=self.r.p.NavSourceMsg();message.requested=getattr(self.r.p.NavSource,name)
        self.node.nav_source=message  # what Subscribe stores before the callback
        self.node._on_nav_source(message)

    def load_field(self,field):
        """Write the field's route and point /settings/navigation at it."""
        p=self.r.p;self.field=field;corners=field_corners(field);x0,y0=corners[0]
        origin=p.Point(x=-88.,y=40.,z=0.,frame=p.Frame.GPS)
        def gps(u,v):
            g=p.geo.map_to_gps(p.Point(x=float(u),y=float(v),z=0.,frame=p.Frame.MAP),origin)
            return dict(x=g.x,y=g.y,z=0.)
        # One file per field: nav_model reloads the boundary when the name changes.
        path=self.r.output/f'nav_route_{x0:g}_{y0:g}.json'
        path.write_text(json.dumps(dict(origin=dict(x=origin.x,y=origin.y,z=0.),
            field_boundary=[gps(x,y) for x,y in corners])))
        self.node.nav_settings.route_name=str(path.resolve())

    @property
    def owns(self):return self.arbitrated and self.source=='NAV_MODEL'

    def command_goals(self,t):
        if self.shadow or (self.arbitrated and not self.owns):return self.r.goals
        if self.switched and not self.node.drive:
            return self.r.goals if all(self.invariants.values()) else self.r.p.WaypointsMsg()
        fresh=self.last_capture is not None and 0<=t-self.last_capture<=self.goal_timeout
        return self.goals if fresh and self.result.get('driving') else self.r.p.WaypointsMsg()


def field_corners(field):
    """A field's polygon: [[xmin, ymin], [xmax, ymax]] bounds, or its corners."""
    if len(field)!=2:return [tuple(c) for c in field]
    (x0,y0),(x1,y1)=field
    return [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]

def nearest_field(fields,x,y):
    """The field (bounds or corners) closest to (x, y); 0 inside."""
    from shapely.geometry import Point,Polygon
    return min(fields,key=lambda f:Polygon(field_corners(f)).distance(Point(x,y)))

def sim_description(core,runtime,dimensions):
    """RobotDescriptionMsg of robot_cad.yml's default robot, its chassis
    resized to `dimensions` if given (dimensions.yml layout: robot, drive, wheel)."""
    from robot_description.resolve import resolve
    resolved=resolve(runtime.documents('src/robot_description/robot_cad.yml')[0],
                     runtime.documents('src/base_control/tools.yml')[0],
                     runtime.config.get('chassis_configuration', ''),
                     runtime.config.get('payload_configuration', ''))
    message=resolved.message()
    if not dimensions:
        return core.RobotDescriptionMsg(json.dumps(message))
    description=json.loads(message['description'])
    chassis=description['chassis']
    chassis['body'].update(length=dimensions['robot']['length'],width=dimensions['robot']['width'])
    chassis['drive']['trackwidth']=dimensions['drive']['trackwidth']
    chassis['wheel']['radius']=dimensions['wheel']['radius']
    if runtime.config.get('localization', {}).get('mode') == 'odometry':
        chassis['wheel']['effective_radius'] = dimensions['wheel']['radius']
    description['wheelbase']=dimensions['drive']['wheelbase']
    message['description']=json.dumps(description)
    return core.RobotDescriptionMsg(json.dumps(message))

def obstacle(prediction,direction):
    """One direction ('fwd'|'rev') of a NavModelPrediction's obstacle verdict as
    {blocked, grid_stop, clearance_m, nearest, violations}: the NavObstacleDirection
    struct, or the flat fields of rhizome builds before it."""
    value=prediction.get('obstacle_'+direction)
    if isinstance(value,dict):return value
    return dict(blocked=bool(value),grid_stop=bool(prediction.get('grid_stop_'+direction)),
                clearance_m=prediction.get(f'clearance_{direction}_m',0.),
                nearest=prediction.get('nearest_'+direction),violations=prediction.get('violations_'+direction,0))
