"""Public-topic control acceptance checks; all control policy runs in Rhizome."""
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import threading
import time

from .rhizome_task import Task


def supervise_controls(config):
    from .rhizome_agent import MODULE, save
    from .rhizome_sim_sensors import CONTROL_STATS, PUBLISHER_STATS
    output = Path(config['output'])
    children = []
    logs = Path('/dev/shm/control-logs')
    logs.mkdir()
    report = dict(passed=False, scope='Native LocalControl, drive mux, drive control and MuJoCo wheels',
                  motion_connected=True, model_response_verified=False)

    def start(name, command, node=None, device=0):
        log = (logs/(name+'.log')).open('w')
        env = os.environ.copy()
        if node:
            env['NODE_NAME'] = node
        cwd = Path('/tmp') if not device else Path('/tmp')/f'payload{device}'
        if device:
            (cwd/'build/host').mkdir(parents=True,exist_ok=True)
            env.update(AIGEN_DEVICE='weeder',AIGEN_HOST_DEVICE_INDEX=str(device),AIGEN_MOTOR_SOCKET=f'/tmp/motors-{device}.sock')
        process = subprocess.Popen(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append((name,process,log))
        return process

    def role(name,device=0):
        return start(name+(f'@{device}' if device else ''),[sys.executable,'-u','-m',MODULE,'--child',name,
                           '--runtime-config',str(output/'launch.json')],device=device)

    def wait(predicate, seconds=60):
        until = time.monotonic()+seconds
        while not predicate():
            for name,p,_ in children:
                if p.poll() is not None:
                    raise RuntimeError(f'{name} exited {p.returncode}; see {name}.log')
            if time.monotonic()>until:
                raise TimeoutError('Native control readiness timed out')
            time.sleep(.05)

    try:
        from .rhizome_build import verify
        build = verify(config['host_build'],config['root'])
        report['build'] = build
        services = Task(config['root'],config['task']).services()
        nodes = services.get('rover',[])
        expected = {'local_control','drive_mux','drive_control'}
        if not config.get('weeding_test') and (set(nodes) != expected or any(v for k,v in services.items() if k!='rover')):
            raise ValueError('Control check needs a task selecting local_control, drive_mux and drive_control only')
        if config.get('weeding_test'):
            from .rhizome_hardware import description
            import yaml
            ids=sorted(description(config).description.payloads,key=int)
            if ids != ['1','2']:
                raise ValueError('Weeding bench currently requires two payloads')
            provisioning=Path(config['root'])/'tools/provisioning/robot/config/payloads.yml'
            payloads=yaml.safe_load(provisioning.read_text())['payloads'][:len(ids)]
            Path('/tmp/conf/payloads.yml').write_text(yaml.safe_dump(dict(payloads=payloads)))
            Path('/opt/aigen/task/aigen-weeder.target').write_text('\n'.join(f'Wants=aigen@{n}.service' for n in services['weeder'])+'\n')
        role('relay')
        wait(lambda: Path('build/host/aigen-proxy-req').exists(),10)
        if config.get('weeding_test'):
            for device in (1,2):
                role('relay',device)
                wait(lambda d=device: Path(f'/tmp/payload{d}/build/host/aigen-proxy-req').exists(),10)
        role('observe')
        role('sensors')
        wait(CONTROL_STATS.exists)
        publisher = role('publish')
        wait(PUBLISHER_STATS.exists)
        def launch_node(name,device=0):
            binary=Path(config['host_build'])/'bin'/name
            source=list((Path(config['root'])/'src').glob(f'*/{name}.main.py'))
            if source:
                command=[sys.executable,'-u',str(source[0])]
            elif binary.is_file():
                if name not in build.get('binaries_sha256',{}):
                    raise ValueError(f'Native node {name} is not in the verified build; rebuild with --task')
                command=['stdbuf','-oL','-eL',str(binary)]
            else:
                raise FileNotFoundError(f'Build native node {name}')
            return start(name+(f'@{device}' if device else ''),command,name,device)
        if config.get('weeding_test'):
            for device in (1,2):
                role('oracle',device)
            wait(lambda: all((output/f'oracle-ready@{d}').exists() for d in (1,2)))
        for name in nodes:
            launch_node(name)
        if config.get('weeding_test'):
            for device in (1,2):
                for name in services['weeder']:
                    launch_node(name,device)
        config['publisher_pid'] = publisher.pid
        save(output/'launch.json',config)
        checker = role('control-check')
        until = time.monotonic()+(300 if config.get('weeding_test') else 90)
        while checker.poll() is None:
            for name,p,_ in children:
                if name!='control-check' and p.poll() is not None:
                    raise RuntimeError(f'{name} exited {p.returncode}')
            if time.monotonic()>until:
                raise TimeoutError('Control check timed out')
            time.sleep(.1)
        if (output/'control_check.json').exists():
            report.update(json.loads((output/'control_check.json').read_text()))
        if checker.returncode:
            raise RuntimeError('Control assertions failed; see control-check.log')
        report['sensors'] = json.loads(CONTROL_STATS.read_text())
        if any(report['sensors']['warnings']):
            raise RuntimeError('MuJoCo warnings occurred')
        report['passed'] = True
    except BaseException as error:
        report['passed'] = False
        report['error'] = str(error)
        raise
    finally:
        # A paused sensor publisher still must be reapable on failure.
        for _,p,log in reversed(children):
            if p.poll() is None:
                p.send_signal(signal.SIGCONT)
                p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill(); p.wait()
            log.close()
        if Path('/dev/shm/physics.jsonl').exists():
            shutil.copyfile('/dev/shm/physics.jsonl',output/'physics.jsonl')
        if CONTROL_STATS.exists():
            shutil.copyfile(CONTROL_STATS,output/'sensors.json')
            report['sensors'] = json.loads(CONTROL_STATS.read_text())
        for path in [*logs.glob('*.log'), PUBLISHER_STATS, Path('/dev/shm/topics.jsonl'),
                     Path('/dev/shm/weeding_topics.jsonl'), *Path('/dev/shm').glob('oracle_*.json')]:
            if path.is_file():
                shutil.copyfile(path,output/path.name)
        save(output/'report.json',report)


def run(config):
    if config.get('weeding_test'):
        from .rhizome_weeding_check import check
        return check(config)
    from pyzome import core
    from .rhizome_agent import save
    output = Path(config['output'])
    result = dict(passed=False)
    latest = {}

    class Observer(core.BaseNode):
        def __init__(self):
            super().__init__(core.NodeConfig('alamin_control_check'))
            for topic in ('/odom','/drive_mux/state','/drive_mux/reason','/nav/drive','drive_control/health','local_control/health'):
                self.Subscribe(topic,None,lambda msg,t=topic: latest.__setitem__(t,json.loads(str(msg))))
        def Tick(self):
            pass

    node = Observer()
    threading.Thread(target=node.Run,daemon=True).start()

    def wait(predicate, seconds):
        until=time.monotonic()+seconds
        while not predicate():
            if time.monotonic()>until:
                raise AssertionError(f'Condition timed out; topics={latest}')
            time.sleep(.05)

    def pose():
        p=latest['/odom']['pose']
        return p['x'],p['y']

    def speed():
        return abs(latest['/odom']['twist']['v'])

    def select(state):
        replies=core.sendRequest('set-drive-mux',core.MuxSelectMsg(dict(requested=state)),2)
        bodies=[json.loads(str(b)) for _,b in replies if b is not None]
        if not any(b.get('requested')==state for b in bodies):
            raise AssertionError(f'Mux rejected {state}: {bodies}; topics={latest}')

    def goal(distance):
        x,y=pose()
        p=latest['/odom']['pose']
        heading=math.atan2(2*(p['qw']*p['qz']+p['qx']*p['qy']),1-2*(p['qy']**2+p['qz']**2))
        return core.WaypointsMsg(dict(waypoints=[dict(point=dict(x=x+distance*math.cos(heading),y=y+distance*math.sin(heading),z=0,frame='odom'))]))

    target=None
    sending=True
    def goals():
        while True:
            if sending and target is not None:
                node.Publish('/nav/goals',target)
            time.sleep(.1)
    threading.Thread(target=goals,daemon=True).start()
    try:
        (output/'clock-start').touch()
        wait(lambda:'/odom' in latest,10)
        time.sleep(2)
        start=pose()
        target=goal(1.2)
        select('AUTONOMY')
        wait(lambda:speed()>.08,10)
        wait(lambda:math.dist(start,pose())>.85 and speed()<.04,20)
        result['arrival_distance_m']=math.dist(start,pose())
        result['arrival_speed_mps']=speed()

        target=goal(5)
        wait(lambda:speed()>.15,8)
        before=pose(); begin=time.monotonic()
        select('NONE')
        wait(lambda:speed()<.03,3)
        result['mux_stop_seconds']=time.monotonic()-begin
        result['mux_stop_distance_m']=math.dist(before,pose())

        select('AUTONOMY')
        wait(lambda:speed()>.15,8)
        sending=False
        begin=time.monotonic()
        wait(lambda:speed()<.03,3)
        result['stale_goal_stop_seconds']=time.monotonic()-begin

        sending=True
        wait(lambda:speed()>.15,8)
        os.kill(config['publisher_pid'],signal.SIGSTOP)
        # Odometry is deliberately absent: inspect physical state, not cached /odom.
        from .rhizome_sim_sensors import STATE
        import numpy as np
        begin=time.monotonic()
        def physical_speed():
            with np.load(STATE) as state:
                return float(np.linalg.norm(state['root_velocity']))
        wait(lambda:latest.get('/drive_mux/state',{}).get('requested')=='NONE' and physical_speed()<.04,4)
        result['stale_odom_stop_seconds']=time.monotonic()-begin
        result['stale_odom_speed_mps']=physical_speed()
        result['stale_odom_mux_reason']=latest.get('/drive_mux/reason')
        os.kill(config['publisher_pid'],signal.SIGCONT)
        result['passed']=True
    finally:
        save(output/'control_check.json',result)
    print(json.dumps(result,indent=2),flush=True)
    # BaseNode has no stop binding; this dedicated child owns its native event thread.
    os._exit(0)
