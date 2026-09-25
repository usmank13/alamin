"""Isolated, synchronous adapter to a local Rhizome host build.

MuJoCo owns time and dynamics. The worker owns native node globals and its
in-process message bus; each rollout gets a fresh process and track-id sequence.
"""
import json
import math
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import numpy as np

from .contracts import PipelineError, write_json


def validate_config(config):
    if not isinstance(config,dict):
        raise PipelineError('RHIZOME_CONFIG','controller must be a JSON object')
    cfg = dict(config)
    if not cfg.get('root'):
        raise PipelineError('RHIZOME_CONFIG', 'controller.root must name a Rhizome checkout')
    cfg['root']=str(Path(cfg['root']).expanduser().resolve())
    cfg['host_build']=str(Path(cfg.get('host_build',os.environ.get('RHIZOME_HOST_BUILD',
                          Path(cfg['root'])/'build/host'))).expanduser().resolve())
    if 'targeting_diagnostics' in cfg and not isinstance(cfg['targeting_diagnostics'],bool):
        raise PipelineError('RHIZOME_CONFIG','targeting_diagnostics must be boolean')
    if 'physical_arm' in cfg and (not isinstance(cfg['physical_arm'],str) or not cfg['physical_arm']):
        raise PipelineError('RHIZOME_CONFIG','physical_arm must name an arm in the robot profile')
    if 'physical_arms' in cfg:
        ids=cfg['physical_arms']
        if (not isinstance(ids,list) or not ids or not all(isinstance(a,str) and a for a in ids)
                or len(set(ids))!=len(ids) or 'physical_arm' in cfg or 'shadow_arm' in cfg):
            raise PipelineError('RHIZOME_CONFIG','physical_arms requires unique arm IDs and excludes physical_arm/shadow_arm')
    if (cfg.get('physical_arm') or cfg.get('physical_arms')) and cfg.get('perception') not in ('oracle_visible','neural'):
        raise PipelineError('RHIZOME_CONFIG','physical_arm requires perception')
    waypoints = cfg.get('waypoints', [])
    if not isinstance(waypoints, list) or not waypoints:
        raise PipelineError('RHIZOME_CONFIG', 'Provide nonempty controller.waypoints in world/odom metres')
    for waypoint in waypoints:
        if (not isinstance(waypoint, dict) or set(waypoint)-{'xy','direction','type'}
                or not isinstance(waypoint.get('xy'),list)
                or len(waypoint.get('xy', [])) != 2
                or not all(isinstance(x, (int,float)) and math.isfinite(x) for x in waypoint['xy'])
                or waypoint.get('direction','FORWARD') not in ('FORWARD','REVERSE')
                or waypoint.get('type','NORMAL') not in ('NORMAL','TURN','ARMS_UP')):
            raise PipelineError('RHIZOME_CONFIG', 'Invalid waypoint: expected xy, FORWARD/REVERSE direction and NORMAL/TURN/ARMS_UP type')
    params = cfg.get('parameters', {})
    limits = {'auto_speed_max_ms':(0,1), 'turn_speed':(0,1), 'max_turn_rate':(.01,1),
              'lookahead_distance':(.05,10), 'arrival_distance':(.01,2), 'kp':(.01,10)}
    if not isinstance(params,dict) or set(params)-limits.keys():
        raise PipelineError('RHIZOME_CONFIG', f'Controller parameters must be among {list(limits)}')
    for key, value in params.items():
        low,high=limits[key]
        if not isinstance(value,(float,int)) or not math.isfinite(value) or not low<=value<=high:
            raise PipelineError('RHIZOME_CONFIG', f'{key} must be between {low} and {high}')
    if cfg.get('perception','none') not in ('none','oracle_visible','neural'):
        raise PipelineError('RHIZOME_CONFIG', 'perception must be none, oracle_visible or neural')
    if cfg.get('perception')=='neural':
        neural=cfg.get('neural')
        if not isinstance(neural,dict) or not neural.get('package') or not neural.get('crop_species'):
            raise PipelineError('RHIZOME_CONFIG','neural requires package and crop_species')
        neural={**dict(input_width=848,input_height=480,device='cpu',tube_length_m=.569629),**neural}
        neural['package']=str(Path(neural['package']).expanduser().resolve())
        if neural['device'] not in ('cpu','cuda'):
            raise PipelineError('RHIZOME_CONFIG','neural.device must be cpu or cuda')
        for key in ('input_width','input_height'):
            if not isinstance(neural[key],int) or not 32<=neural[key]<=2048:
                raise PipelineError('RHIZOME_CONFIG',f'Invalid neural.{key}')
        if not isinstance(neural['tube_length_m'],(float,int)) or not 0<neural['tube_length_m']<2:
            raise PipelineError('RHIZOME_CONFIG','Invalid neural.tube_length_m')
        targets=neural.get('target_positions_chassis_m',{} if (cfg.get('physical_arm') or cfg.get('physical_arms')) else None)
        if not isinstance(targets,dict) or (not targets and not (cfg.get('physical_arm') or cfg.get('physical_arms'))):
            raise PipelineError('RHIZOME_CONFIG','Specify neural.target_positions_chassis_m for each camera')
        for position in targets.values():
            if (not isinstance(position,list) or len(position)!=3
                    or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in position)):
                raise PipelineError('RHIZOME_CONFIG','Invalid neural target mount')
        quaternions=neural.get('target_quaternions_chassis_wxyz',{})
        if not isinstance(quaternions,dict) or set(quaternions)-targets.keys():
            raise PipelineError('RHIZOME_CONFIG','Target rotations must name configured cameras')
        for quaternion in quaternions.values():
            if (not isinstance(quaternion,list) or len(quaternion)!=4
                    or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in quaternion)
                    or not math.isclose(sum(v*v for v in quaternion),1,abs_tol=1e-6)):
                raise PipelineError('RHIZOME_CONFIG','Target rotations must be unit wxyz quaternions')
        cfg['neural']=neural
    shadow=cfg.get('shadow_arm')
    if shadow is not None:
        if cfg.get('perception') not in ('oracle_visible','neural'):
            raise PipelineError('RHIZOME_CONFIG', 'shadow_arm requires perception')
        if (not isinstance(shadow,dict) or not isinstance(shadow.get('position_chassis_m'),list)
                or len(shadow['position_chassis_m'])!=3):
            raise PipelineError('RHIZOME_CONFIG', 'shadow_arm needs an explicit provisional position_chassis_m')
        if not all(isinstance(x,(float,int)) and math.isfinite(x) for x in shadow['position_chassis_m']):
            raise PipelineError('RHIZOME_CONFIG', 'Nonfinite shadow arm mount')
        if not isinstance(shadow.get('camera'),str):
            raise PipelineError('RHIZOME_CONFIG', 'shadow_arm.camera must select a rig camera')
        q=shadow.get('quaternion_chassis_wxyz',[1,0,0,0])
        if (not isinstance(q,list) or len(q)!=4 or not all(isinstance(v,(float,int)) and math.isfinite(v) for v in q)
                or not math.isclose(sum(v*v for v in q),1,abs_tol=1e-6)):
            raise PipelineError('RHIZOME_CONFIG','Arm rotation must be a unit wxyz quaternion')
        if cfg.get('perception')=='neural' and cfg['neural']['target_positions_chassis_m'].get(shadow['camera'])!=shadow['position_chassis_m']:
            raise PipelineError('RHIZOME_CONFIG','The selected camera target must match shadow_arm.position_chassis_m')
        if (cfg.get('perception')=='neural' and
                cfg['neural'].get('target_quaternions_chassis_wxyz',{}).get(shadow['camera'],[1,0,0,0])!=q):
            raise PipelineError('RHIZOME_CONFIG','The selected camera target rotation must match the arm')
    return cfg


def bind_physical_arm(config, profile):
    """Resolve payload frames from the compiled assembly, before starting native code."""
    import copy
    config=copy.deepcopy(config)
    if not config.get('physical_arm'):return config,None
    arms={a['id']:a for a in profile.get('arms',[])}
    if config['physical_arm'] not in arms:
        raise PipelineError('RHIZOME_CONFIG','physical_arm is absent from the robot profile')
    arm=arms[config['physical_arm']]
    expected={key:arm[key] for key in ('camera','position_chassis_m','quaternion_chassis_wxyz','motor_limits_rad')}
    if config.get('shadow_arm') and any(config['shadow_arm'].get(key)!=value for key,value in expected.items()):
        raise PipelineError('RHIZOME_CONFIG','physical_arm mount must come from the compiled robot profile')
    config['shadow_arm']=expected
    if config.get('perception')=='neural':
        for key,source in (('target_positions_chassis_m','position_chassis_m'),
                           ('target_quaternions_chassis_wxyz','quaternion_chassis_wxyz')):
            values={a['camera']:a[source] for a in arms.values()}
            if config['neural'].get(key) and config['neural'][key]!=values:
                raise PipelineError('RHIZOME_CONFIG','Physical inference targets must match the compiled arm frames')
            config['neural'][key]=values
    return validate_config(config),arm


class RhizomeController:
    def __init__(self, config, output, *, timeout=30, worker_name=None):
        self.config=validate_config(config)
        self.output=Path(output)/'rhizome'
        if worker_name is not None:self.output=self.output/worker_name
        self.output.mkdir(parents=True)
        self.deadline=time.monotonic()+timeout
        self.log=(self.output/'worker.log').open('w')
        self.events=(self.output/'messages.jsonl').open('w')
        env=os.environ.copy()
        # The worker explicitly selects its build, node configuration and bus.
        for key in ('NODE_NAME','AIGEN_CONFIG_PATH','PYTHONPATH','PYTHONHOME'):
            env.pop(key,None)
        self.process=None
        try:
            self.process=subprocess.Popen(
                [str(Path(config.get('python',sys.executable)).expanduser().absolute()), '-u', '-X', 'faulthandler',
                 str(Path(__file__).with_name('rhizome_worker.py'))],
                stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,
                text=True,env=env,cwd=self.output)
            self.info=self.request(dict(op='init',config=self.config,output=str(self.output.resolve())))
            write_json(self.output/'provenance.json',self.info)
        except Exception:
            self.close()
            raise

    def request(self, message):
        remaining=self.deadline-time.monotonic()
        if remaining<=0:
            raise PipelineError('RHIZOME_TIMEOUT','Rhizome worker exceeded the rollout wall-clock budget')
        try:
            self.process.stdin.write(json.dumps(message,allow_nan=False)+'\n')
            self.process.stdin.flush()
            readable,_,_=select.select([self.process.stdout],[],[],min(remaining,30))
            if not readable:
                raise PipelineError('RHIZOME_TIMEOUT','No worker response; see rhizome/worker.log')
            line=self.process.stdout.readline()
            if not line:
                raise PipelineError('RHIZOME_WORKER','Worker exited; see rhizome/worker.log')
            result=json.loads(line)
        except PipelineError:
            raise
        except (OSError,ValueError) as exc:
            raise PipelineError('RHIZOME_WORKER',f'Worker communication failed: {exc}; see rhizome/worker.log') from exc
        if 'error' in result:
            raise PipelineError('RHIZOME_WORKER',result['error']+'; see rhizome/worker.log')
        if message['op']!='init':
            self.events.write(json.dumps(dict(simulation_time=message['time'],op=message['op'],
                                             input=message,output=result),allow_nan=False)+'\n')
        return result

    def step(self, t, position, quaternion, forward_mps, arm_measured=None):
        message=dict(op='step',time=t,position=list(position),quaternion=list(quaternion),forward_mps=float(forward_mps))
        if arm_measured is not None:message['arm_measured']=list(arm_measured)
        return self.request(message)

    def observe(self, t, camera, detections, row_profile, foliage=None):
        return self.request(dict(op='observe',time=t,camera=camera,detections=detections,
                                 row_profile=row_profile,foliage=foliage))

    def infer(self,t,camera,**arrays):
        # Only one synchronous request is in flight. HDF5 holds the permanent
        # frames; this bounded scratch file avoids JSON/base64 image transfers.
        path=self.output/'camera_input.npz'
        try:
            np.savez(path,**arrays)
            return self.request(dict(op='infer',time=t,camera=camera,frame_file=str(path.resolve())))
        finally:
            path.unlink(missing_ok=True)

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait()
            for stream in (self.process.stdin,self.process.stdout):
                try:stream.close()
                except OSError:pass  # A dead worker may leave buffered input.
        self.events.close();self.log.close()


def bind_physical_arms(config, profile):
    """Bind each independent payload to the assembly's camera and motor frames."""
    import copy
    config=validate_config(copy.deepcopy(config))
    arms={};children={}
    for name in config['physical_arms']:
        child=copy.deepcopy(config);child.pop('physical_arms');child['physical_arm']=name
        child,arm=bind_physical_arm(child,profile)
        arms[name]=arm;children[name]=child
    if len({a['camera'] for a in arms.values()})!=len(arms):
        raise PipelineError('RHIZOME_CONFIG','Each physical arm requires a distinct camera')
    if config.get('perception')=='neural':config['neural']=next(iter(children.values()))['neural']
    return config,arms


class RhizomeMultiController:
    """One isolated native runtime per payload; only the first drives navigation."""
    def __init__(self, config, output, arms, *, timeout=30):
        import copy
        self.workers={};self.camera_owners={a['camera']:name for name,a in arms.items()}
        self.navigation_arm=next(iter(arms));deadline=time.monotonic()+timeout
        try:
            for name,arm in arms.items():
                child=copy.deepcopy(config);child.pop('physical_arms')
                child.update(physical_arm=name,navigation_enabled=name==self.navigation_arm,
                    shadow_arm={key:arm[key] for key in ('camera','position_chassis_m','quaternion_chassis_wxyz','motor_limits_rad')})
                worker=RhizomeController(child,output,timeout=max(.001,deadline-time.monotonic()),worker_name=name)
                worker.deadline=deadline;self.workers[name]=worker
            primary=self.workers[self.navigation_arm].info
            self.info=dict(adapter='Isolated native runtime per arm; one LocalControl',
                navigation_arm=self.navigation_arm,camera_owners=self.camera_owners,
                workers={name:w.info for name,w in self.workers.items()},
                local_control_parameters=primary['local_control_parameters'],limitations=primary['limitations']+
                    ['Independent payload planners; native peer-arm semaphore/barrier exchange is not connected'])
            write_json(Path(output)/'rhizome/provenance.json',self.info)
        except Exception:
            self.close();raise

    def _call(self, name, method, *args, **kwargs):
        try:return getattr(self.workers[name],method)(*args,**kwargs)
        except Exception as exc:
            self.close()
            if isinstance(exc,PipelineError):
                raise PipelineError(exc.code,f'{name}: {exc}') from exc
            raise

    def step(self,t,position,quaternion,forward_mps,arm_measured):
        if set(arm_measured)!=set(self.workers):
            raise PipelineError('RHIZOME_CONFIG','Measured feedback must cover every active arm exactly')
        results={name:self._call(name,'step',t,position,quaternion,forward_mps,arm_measured[name]) for name in self.workers}
        return dict(twist=results[self.navigation_arm]['twist'],arms={name:r['arm'] for name,r in results.items()})

    def _owner(self,camera):
        if camera not in self.camera_owners:
            raise PipelineError('RHIZOME_CONFIG',f'No arm owns camera {camera}')
        return self.camera_owners[camera]

    def observe(self,t,camera,detections,row_profile,foliage=None):
        return self._call(self._owner(camera),'observe',t,camera,detections,row_profile,foliage)

    def infer(self,t,camera,**arrays):
        return self._call(self._owner(camera),'infer',t,camera,**arrays)

    def close(self):
        for worker in self.workers.values():worker.close()


def visible_row_geometry(instance_image, instance_ids, manifest):
    """Oracle row identities/lines for visible crops; no canopy polygons.

    Each generated row remains an independent boundary. A root-only oracle
    does not infer a connecting contour from adjacent detections in X.
    """
    visible=set(np.unique(instance_image).tolist())
    by_row={}
    for name,plant in manifest['instances'].items():
        if (instance_ids.get(name) in visible and plant.get('category')=='crop'
                and plant.get('row_id') is not None):
            by_row.setdefault(plant['row_id'],[]).append(plant['position_m'])
    rows=[]
    for number,row in enumerate(sorted(manifest.get('rows',[]),key=lambda row:row['id'])):
        if row['id'] not in by_row:continue
        support=np.asarray(by_row[row['id']],dtype=float)
        endpoints=np.asarray(row['endpoints_m'],dtype=float)
        direction=endpoints[1,:2]-endpoints[0,:2]
        length=np.linalg.norm(direction)
        if not np.isfinite(length) or length<1e-6:
            raise PipelineError('ORACLE_ROWS',f'Invalid row line: {row["id"]}')
        direction/=length
        z=float(np.median(support[:,2]))
        anchor=np.array([*endpoints.mean(0)[:2],z])
        along=(support[:,:2]-anchor[:2])@direction
        def point(value):return dict(zip(('x','y','z'),map(float,value)),frame='odom')
        rows.append(dict(id=number,anchor=point(anchor),direction=point([*direction,0]),
            begin=point(anchor+np.array([*direction,0])*along.min()),
            end=point(anchor+np.array([*direction,0])*along.max()),
            established=True,supported=True,height_valid=True,clipped=False,
            residual_m=0.,score=1.,age_m=0.,age_s=0.))
    if len(rows)>4:
        raise PipelineError('ORACLE_ROWS','Visible crop rows exceed the native four-row geometry capacity')
    return dict(present=True,row_epoch=1,rows=rows,polygons=[],low_overflow=False,
                high_overflow=False,row_overflow=False,canopy_top_valid=False)


def visible_detections(instance_image, instance_ids, manifest):
    """Oracle keypoints for rendered visible plants, explicitly in ODOM/world.

    Visibility comes from segmentation, but positions/classes come from the
    generator. This is a plumbing fixture, not an RGB perception algorithm.
    """
    visible=set(np.unique(instance_image).tolist())
    detections=[]
    for name,plant in manifest['instances'].items():
        if instance_ids.get(name) not in visible or plant.get('category') not in ('crop','weed'):
            continue
        x,y,z=plant['position_m']
        point=dict(x=x,y=y,z=z,frame='odom')
        detections.append(dict(class_label='CROP' if plant['category']=='crop' else 'WEED',score=1.,
                               keypoint=dict(kp=point,size=.02,vis=1.,z_g=0.),
                               rect=dict(center=point,lx=.02,ly=.02)))
    return detections


def neural_overlay(rgb,prediction,K,camera_rotation,camera_position,chassis_rotation,chassis_position,mount,
                   target_quaternion_chassis=(1,0,0,0)):
    """Project the model's target-frame keypoints onto the captured RGB frame."""
    from PIL import Image,ImageDraw
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    target=np.asarray(chassis_position)+np.asarray(chassis_rotation)@np.asarray(mount)
    from .rhizome_perception import quaternion_rotation
    target_rotation=np.asarray(chassis_rotation)@quaternion_rotation(target_quaternion_chassis)
    optical=np.asarray(camera_rotation)@np.diag([1,-1,-1])
    visible=0
    for detection in prediction['detections']:
        point=detection['keypoint']['kp']
        world=target_rotation@np.array([point[k] for k in ('x','y','z')])+target
        local=optical.T@(world-camera_position)
        if local[2]<=0:continue
        uv=np.asarray(K)@local;u,v=uv[:2]/uv[2]
        if not (0<=u<image.width and 0<=v<image.height):continue
        visible+=1
        color='#33ee66' if detection['class_label']=='CROP' else '#ff55cc'
        draw.ellipse((u-4,v-4,u+4,v+4),outline=color,width=2)
    draw.rectangle((0,0,image.width,18),fill='#151515')
    draw.text((3,2),f'Neural keypoints: {visible} in view / {len(prediction["detections"])}',fill='white')
    return image
