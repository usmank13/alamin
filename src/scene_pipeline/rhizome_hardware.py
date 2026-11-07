"""Bind the simulated robot's joints to the selected Rhizome hardware contract."""
import json
from pathlib import Path

import yaml

from .external_motors import MotorBinding, MotorSocketServer, MujocoMotors
from .rhizome_task import Task


def description(config):
    from robot_description.resolve import resolve
    root=Path(config['root'])
    return resolve(yaml.safe_load((root/'src/robot_description/robot_cad.yml').read_text()),
                   yaml.safe_load((root/'src/base_control/tools.yml').read_text()),
                   config['chassis_configuration'],config['payload_configuration'])


def connect(model,data,profile,config):
    from pyzome import core
    bindings={0:{}}
    for wheel in profile['wheels']:
        mid=int(getattr(core.MotorPosition,'DRIVE_'+wheel['id'].upper()))
        polarity=1 if wheel['id'].endswith('left') else -1
        bindings[0][0,mid]=MotorBinding('element/'+wheel['id'],int(wheel['drive_sign']*polarity))
    if config.get('weeding_test'):
        task=Task(config['root'],config['task'])
        resolved=json.loads(description(config).message()['description'])
        arms=[]
        for pid,items in sorted(resolved['payloads'].items(),key=lambda pair:int(pair[0])):
            bindings[int(pid)]={}
            for name,item in sorted(items.items()):
                if not name.startswith('arm_'):
                    continue
                arms.append((int(pid),name,item))
        if len(arms)!=len(profile['arms']):
            raise ValueError('Scene and selected Rhizome payload have different arm counts')
        for physical,(pid,name,item) in zip(profile['arms'],arms):
            cfg=item['config']
            node=task.node('arm_control_'+name.removeprefix('arm_'))['config']
            for axis,sign in zip(('yaw','pitch'),physical['motor_to_joint_sign']):
                jid=model.joint('element/'+physical[axis+'_joint']).id
                # Mechanical stops extend beyond software operating extents.
                extent=cfg['extents'][axis]
                if axis=='pitch':
                    low=cfg['mount']['pitch']['hardstop']-node['calibration']['hardstop_offset']
                    high=max(extent['max'],node['calibration']['yaw_zero_pitch']+.1)
                else:
                    low=min(extent['min'],cfg['mount']['yaw']['hardstop'])
                    high=max(extent['max'],-cfg['mount']['yaw']['hardstop'])
                model.jnt_range[jid]=sorted([sign*low,sign*high])
                model.jnt_solref[jid]=[.004,1]
                model.jnt_solimp[jid,:3]=[.99,.999,.001]
                load_offset=cfg['zero']['yaw']['load']+cfg['mount']['yaw']['hardstop'] if axis=='yaw' else 0
                mid=int(getattr(core.MotorPosition,node['motor_ids'][axis]))
                bindings[pid][0,mid]=MotorBinding('element/'+physical[axis+'_joint'],sign,load_offset_rad=load_offset,encoder_zeroed=True)
    adapters={pid:MujocoMotors(model,data,b) for pid,b in bindings.items()}
    servers=[MotorSocketServer(f'/tmp/motors-{pid}.sock',adapter) for pid,adapter in adapters.items()]
    return adapters,servers
