"""Oracle perception and public-service acceptance checks for native weeding."""
import gc
import json
import math
import os
from pathlib import Path
import threading
import time

import mujoco
import numpy as np

from sim_harness.scene import load_scene
from .agriculture_drive import camera_rig, shadow_row_profile
from .rhizome import visible_detections, visible_row_geometry
from .rhizome_sim_sensors import STATE


def oracle(config):
    from pyzome import core
    output=Path(config['output'])
    device=int(os.environ['AIGEN_HOST_DEVICE_INDEX'])
    profile=json.loads((output/'camera_profile.json').read_text())
    manifest=json.loads((Path(config['scene'])/'manifest.json').read_text())
    model,data=load_scene(output/'world.mjz')
    rig=camera_rig(profile,640,480)
    ids={name:i+1 for i,name in enumerate(manifest['instances'])}
    chassis=model.body('element/chassis').id

    class Oracle(core.BaseNode):
        def __init__(self,slot,arm):
            cfg=core.NodeConfig('crop_ie_'+str(slot))
            cfg.trigger.type=core.TriggerType.ON_TIMER
            cfg.trigger.period=core.milliseconds(100)
            super().__init__(cfg)
            self.slot,self.arm=slot,arm
            self.data=mujoco.MjData(model)
            self.count=0
            self.seen=0
        def Tick(self):
            data=self.data
            with np.load(STATE) as state:
                ns=int(state['timestamp_ns'])
                fresh=core.now().seconds()-ns/1e9<.25
                self.assert_invariant(fresh,'ORACLE_SNAPSHOT_STALE')
                if not fresh:
                    return
                data.qpos[:]=state['qpos']; data.qvel[:]=state['qvel']
            mujoco.mj_kinematics(model,data)
            mujoco.mj_camlight(model,data)
            name=self.arm['camera']; cid=model.camera('element/'+name).id
            rotation=data.cam_xmat[cid].reshape(3,3)@np.diag([1,-1,-1])
            k=np.array(rig[name]['K'])
            near,far=profile['cameras'][name]['optics']['depth_range_m']
            visible=[]
            for key,plant in manifest['instances'].items():
                point=rotation.T@(np.asarray(plant['position_m'])-data.cam_xpos[cid])
                if point[2]<near or point[2]>far:
                    continue
                u,v=(k@point)[:2]/point[2]
                if 0<=u<640 and 0<=v<480:
                    visible.append(ids[key])
            image=np.array(visible,dtype=int)
            detections=visible_detections(image,ids,manifest)
            foliage=visible_row_geometry(image,ids,manifest)
            msg=core.TrackerMsg(dict(detections=detections,foliage=foliage))
            msg.camera_timestamp=core.nanoseconds(ns)
            self.Publish(f'/vision/crop_infs/{self.slot}',msg)
            row=shadow_row_profile(model,data,chassis,self.arm['position_chassis_m'],self.arm['quaternion_chassis_wxyz'])
            for point in row:
                point['frame']='arm_'+str(self.slot)
            msg=core.RowProfileMsg(dict(profile=row));msg.camera_timestamp=core.nanoseconds(ns)
            self.Publish(f'/vision/row_profile/{self.slot}',msg)
            self.count+=1;self.seen+=len(detections)
            if self.count%10==0:
                path=Path('/dev/shm')/f'oracle_{self.slot}@{device}.json'
                path.write_text(json.dumps(dict(frames=self.count,detections=self.seen,
                    perception='Ground-truth roots in camera frustum; no occlusion or neural inference')))

    nodes=[Oracle(i,profile['arms'][2*(device-1)+i]) for i in range(2)]
    gc.freeze()
    for node in nodes:
        threading.Thread(target=node.Run,daemon=True).start()
    (output/f'oracle-ready@{device}').touch()
    threading.Event().wait()


def check(config):
    from pyzome import core
    from .rhizome_agent import save
    output=Path(config['output'])
    latest={}
    poses={}
    commands={}
    endpoints=[(slot,pid) for pid in (1,2) for slot in (0,1)]
    keys=[f'{slot}@{pid}' for slot,pid in endpoints]
    result=dict(passed=False,scope='Native arm startup homing, tracker, physics planner, mux and motor controls',
                perception='Oracle roots in crop camera frustum',factory_calibration='Ideal calibrated encoders tied to simulated mechanical stops; native startup homing still runs',
                model_response_verified=False)
    class Observer(core.BaseNode):
        def __init__(self):
            super().__init__(core.NodeConfig('alamin_weeding_check'))
            topics=['/odom','/drive_mux/state','/drive_mux/reason']
            for slot,pid in endpoints:
                topics += [f'arm_control_{slot}/config@{pid}']
                topics += [f'{name}_{slot}/health@{pid}' for name in ('arm_control','physics','tracker')]
                topics += [f'/arm_mux/state@{pid}',f'/arm_mux/reason@{pid}',f'/arm_planner/plan/{slot}@{pid}',f'/motors/arm/angle/{slot}@{pid}']
            self.log=Path('/dev/shm/weeding_topics.jsonl').open('w',buffering=1)
            for topic in set(topics):
                self.Subscribe(topic,None,lambda m,t=topic:self.receive(t,m))
        def receive(self,topic,msg):
            value=json.loads(str(msg));latest[topic]=value
            self.log.write(json.dumps(dict(wall=time.time(),topic=topic,message=value))+'\n')
            if topic.startswith('/motors/arm/angle/'):
                poses.setdefault(topic,[]).append(value)
            if topic.startswith('/arm_planner/plan/'):
                commands.setdefault(topic,[]).append(value)
        def Tick(self): pass
    node=Observer()
    threading.Thread(target=node.Run,daemon=True).start()
    def wait(predicate,seconds):
        until=time.monotonic()+seconds
        while not predicate():
            if time.monotonic()>until:
                raise AssertionError(f'Timed out; health and mux state: { {k:v for k,v in latest.items() if "health" in k or "mux" in k} }')
            time.sleep(.05)
    def service(name,msg,timeout=3):
        replies=core.sendRequest(name,msg,timeout)
        bodies=[json.loads(str(b)) for _,b in replies if b is not None]
        print(name,bodies,flush=True)
        return bodies
    def healthy(topic):
        return topic in latest and not latest[topic].get('invariant_failures',[])
    try:
        (output/'clock-start').touch()
        wait(lambda:'/odom' in latest and all(f'arm_control_{slot}/health@{pid}' in latest for slot,pid in endpoints),15)
        wait(lambda:all(healthy(f'arm_control_{slot}/health@{pid}') for slot,pid in endpoints),60)
        result['homing_healthy']=keys
        wait(lambda:all(healthy(f'physics_{slot}/health@{pid}') for slot,pid in endpoints),15)
        for pid in (1,2):
            replies=service(f'set-arm-mux@{pid}',core.MuxSelectMsg(dict(requested='AUTONOMY')))
            if not any(b.get('requested')=='AUTONOMY' for b in replies):
                raise AssertionError(f'Arm mux {pid} rejected autonomy')
        replies=service('set-drive-mux',core.MuxSelectMsg(dict(requested='AUTONOMY')))
        if not any(b.get('requested')=='AUTONOMY' for b in replies):
            raise AssertionError('Drive mux rejected autonomy')
        start=latest['/odom']['pose']; begin=(start['x'],start['y'])
        yaw=math.atan2(2*(start['qw']*start['qz']+start['qx']*start['qy']),1-2*(start['qy']**2+start['qz']**2))
        goal=core.WaypointsMsg(dict(waypoints=[dict(point=dict(x=start['x']+4*math.cos(yaw),y=start['y']+4*math.sin(yaw),frame='odom'))]))
        poses.clear(); commands.clear()
        until=time.monotonic()+config['seconds']
        while time.monotonic()<until:
            if any(latest.get(f'/arm_mux/state@{pid}',{}).get('requested')=='NONE' for pid in (1,2)) or latest.get('/drive_mux/state',{}).get('requested')=='NONE':
                raise AssertionError('Native mux dropped autonomy during the driving/weeding phase')
            node.Publish('/nav/goals',goal)
            time.sleep(.1)
        service('set-drive-mux',core.MuxSelectMsg(dict(requested='NONE')))
        end=latest['/odom']['pose']
        result['distance_m']=math.dist(begin,(end['x'],end['y']))
        result['motor_samples']={k:len(v) for k,v in poses.items()}
        result['planner_commands']={k:len(v) for k,v in commands.items()}
        result['arm_range_rad']={k:{axis:max(m[axis] for m in samples)-min(m[axis] for m in samples)
                                   for axis in ('yaw','pitch')} for k,samples in poses.items()}
        result['planner_range_rad']={k:{axis:max(m[axis]['position_ref'] for m in samples)-min(m[axis]['position_ref'] for m in samples)
                                       for axis in ('yaw','pitch')} for k,samples in commands.items()}
        result['final_health']={k:v for k,v in latest.items() if 'health' in k}
        if result['distance_m']<3.5 or len(poses)!=4 or len(commands)!=4:
            raise AssertionError('Expected arrival near the 4 m goal and all four planner/arm streams; allow at least 15 seconds')
        if any(max(r.values())<.03 for r in result['arm_range_rad'].values()):
            raise AssertionError('Every arm must physically articulate during driving')
        if any(max(r.values())<.03 for r in result['planner_range_rad'].values()):
            raise AssertionError('Every planner must change its target during driving')
        if not all(healthy(f'{name}_{slot}/health@{pid}') for name in ('arm_control','physics','tracker') for slot,pid in endpoints):
            raise AssertionError('Native arm, planner and tracker health must remain healthy')
        result['passed']=True
    finally:
        save(output/'control_check.json',result)
    print(json.dumps(result,indent=2),flush=True)
    os._exit(0)
