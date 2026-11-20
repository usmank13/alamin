"""Bind the simulated robot's joints to the selected Rhizome hardware contract."""
from contextlib import ExitStack
import json
import os
from pathlib import Path

from .external_motors import MotorBinding, MotorSocketServer, MujocoMotors


def description(config):
    import yaml
    from robot_description.resolve import resolve
    root=Path(config['root'])
    return resolve(yaml.safe_load((root/'src/robot_description/robot_cad.yml').read_text()),
                   yaml.safe_load((root/'src/base_control/tools.yml').read_text()),
                   config['chassis_configuration'],config['payload_configuration'])


def motor_bindings(model, profile, config):
    from pyzome import core
    bindings={0:{}}
    for wheel in profile['wheels']:
        mid=int(getattr(core.MotorPosition,'DRIVE_'+wheel['id'].upper()))
        polarity=1 if wheel['id'].endswith('left') else -1
        bindings[0][0,mid]=MotorBinding('element/'+wheel['id'],int(wheel['drive_sign']*polarity))
    if config.get('weeding_test'):
        from .rhizome_task import Task
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
    return bindings


class RhizomeMotorAdapter:
    """One simulated motor endpoint per Rhizome device, polled on the physics thread."""

    def __init__(self, model, data, bindings, socket_dir=Path('/tmp')):
        self.devices = {}
        self.servers = []
        with ExitStack() as opening:
            for device, joints in bindings.items():
                motors = MujocoMotors(model, data, joints)
                server = MotorSocketServer(socket_dir / f'motors-{device}.sock', motors)
                self.servers.append(opening.enter_context(server))
                self.devices[device] = motors
            self._resources = opening.pop_all()

    def before_step(self):
        for motors in self.devices.values():
            motors.apply()
        self.poll()

    def poll(self):
        for server in self.servers:
            server.poll(0)

    def close(self):
        try:
            for motors in self.devices.values():
                motors.stop(motors.motors)
        finally:
            self._resources.close()


def from_launch(model, data):
    """Open the motor endpoints specified by a native control run's launch file."""
    from .rhizome_install import inspect_install

    config = json.loads(Path(os.environ['ALAMIN_RHIZOME_LAUNCH']).read_text())
    if not config.get('control_test'):
        raise ValueError('The selected launch must enable native motor control')
    install = inspect_install(config['root'], config['host_build'])
    from pyzome import core
    if not Path(core.__file__).resolve().is_relative_to(Path(install['host_build']) / 'py'):
        raise ValueError('Set PYTHONPATH to the selected Rhizome host build and source')
    profile = json.loads((Path(config['scene']) / 'robot_profile.json').read_text())
    return RhizomeMotorAdapter(model, data, motor_bindings(model, profile, config))
