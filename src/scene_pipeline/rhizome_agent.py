"""Launch the actual Rhizome AiAgent with live MuJoCo observations.

This is an observation-only integration milestone, not the full task launcher.
Rhizome owns the agent loop and services. Motion/control nodes are not connected.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from .rhizome_install import inspect_install
from .rhizome_task import Task, merge_patch

MODULE = 'scene_pipeline.rhizome_agent'

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n')


def stage(root, output, task_name, agent_config=None):
    task = Task(root, task_name)
    if 'ai_agent' not in task.services().get('rover', []):
        raise ValueError(f'Task {task_name} does not select ai_agent on rover')
    runtime = output/'runtime'
    conf = runtime/'conf'
    (conf/'nodes').mkdir(parents=True)
    (runtime/'build/host').mkdir(parents=True)
    (runtime/'opt/image/bin').mkdir(parents=True)
    (runtime/'opt/task').mkdir(parents=True)
    (runtime/'etc').mkdir()
    (runtime/'etc/aigen.env').write_text('AIGEN_DEVICE=dev_machine\n')
    (runtime/'data').mkdir()
    (runtime/'home').mkdir()
    for name, path in task.nodes.items():
        (conf/'nodes'/f'{name}{path.suffix}').symlink_to(path)
    for directory in ('robot_description','base_control'):
        (conf/directory).symlink_to(root/'src'/directory)
    overrides = dict(agent_config or {})
    overrides.update(memory_dir='/data/agent/memory', captures_dir='/data/agent/captures')
    overrides['voice'] = {**overrides.get('voice', {}), 'dir': '/data/agent/voice'}
    # A simulation launch must not activate a provisioned Telegram connection.
    overrides['keys'] = {**overrides.get('keys', {}), 'telegram': '/tmp/no-telegram.json'}
    effective_task = merge_patch(task.config, {'nodes': {'ai_agent': overrides}})
    save(runtime/'opt/task/task.json', effective_task)
    return runtime


def child(role, config):
    from pyzome import core
    output = Path(config['output'])
    if role == 'relay':
        import threading
        relay = core.Relay('')
        relay.start()
        threading.Event().wait()
    elif role == 'sensors':
        from .rhizome_sim_sensors import run
        run(config)
    elif role == 'render':
        from .rhizome_sim_sensors import render
        render(config)
    elif role == 'publish':
        from .rhizome_sim_sensors import publish
        publish(config)
    elif role == 'observe':
        class Observer(core.BaseNode):
            def Tick(self):
                pass

            def __init__(self):
                super().__init__(core.NodeConfig('alamin_observer'))
                self.log = (output/'topics.jsonl').open('w', buffering=1)
                for topic in ('/agent/transcript','/agent/enable','/agent/busy','/nav/goals',
                              '/drive/limits','/odom','/camera/nav/front/intrinsics/color','/frames/nav_front'):
                    self.Subscribe(topic, None, lambda msg, t=topic: self.log.write(
                        json.dumps(dict(topic=t,wall=time.time(),message=json.loads(str(msg))))+'\n'))
        Observer().Run()
    elif role == 'camera-check':
        from camera.shm_stream_pipeline import ShmStreamParams, ShmStreamPipeline
        from PIL import Image
        result = {}
        for position in ('front','rear'):
            reader = ShmStreamPipeline(ShmStreamParams('alamin_check_'+position,
                f'/tmp/combo/nav/{position}/shm',core.ImageFormat.RGBA_GRAY16_LE,640,480,15))
            try:
                frames, metadata = reader.get_latest_frame(core.ImageFormat.RGBA_GRAY16_LE)
                rgb, depth = frames
                assert rgb.shape == (480,640,4) and depth.shape == (480,640)
                assert metadata.capture_timestamp_ns > 0
                Image.fromarray(rgb[:,:,:3]).save(output/f'received_{position}.png')
                result[position] = dict(color_shape=list(rgb.shape),depth_shape=list(depth.shape),
                    capture_timestamp_ns=metadata.capture_timestamp_ns,
                    capture_age_s=core.now().seconds()-metadata.capture_timestamp_ns/1e9)
            finally:
                reader.stop_pipeline()
                if reader.main_loop:
                    reader.main_loop.quit()
        save(output/'camera_check.json', result)


def supervise(config):
    from pyzome import core
    output = Path(config['output'])
    children = []
    report = dict(passed=False, actual_agent_entry_point=config['entry_point'],
                  scope='Stationary node, IPC and sensor wiring; excludes model behavior and motion',
                  motion_connected=False, model_response_verified=False,
                  mission_submitted=False, navigation_timing_validated=False)

    def start(name, command, node=None):
        log = (output/(name+'.log')).open('w')
        env = os.environ.copy()
        if node:
            env['NODE_NAME'] = node
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
        children.append((name,process,log))
        return process

    def check():
        for name,p,_ in children:
            if name != 'camera-check' and p.poll() is not None:
                raise RuntimeError(f'{name} exited {p.returncode}; see {name}.log')

    def wait_for(predicate, timeout=60):
        deadline = time.monotonic()+timeout
        while not predicate():
            check()
            if time.monotonic() >= deadline:
                raise TimeoutError('Runtime readiness timeout; see process logs')
            time.sleep(.1)

    def spawn_role(role):
        return start(role, [sys.executable,'-u','-m',MODULE,'--child',role,'--runtime-config',str(output/'launch.json')])

    try:
        spawn_role('relay')
        wait_for(lambda: Path('build/host/aigen-proxy-req').exists(), 10)
        spawn_role('observe')
        spawn_role('sensors')
        wait_for(lambda: (output/'sensors.json').exists())
        spawn_role('publish')
        spawn_role('render')
        start('ai_agent',[sys.executable,'-u',config['entry_point']], 'ai_agent')
        bodies = []
        deadline = time.monotonic()+30
        while time.monotonic() < deadline:
            check()
            replies = core.sendRequest('agent-send',core.AgentChatMsg(),1)
            bodies = [json.loads(str(body)) for _,body in replies if body is not None]
            if bodies:
                break
            time.sleep(.2)
        if not any(b.get('str') == 'empty message' for b in bodies):
            raise RuntimeError(f'Actual agent service validation failed: {bodies}')
        report['agent_send_validation'] = bodies
        wait_for(lambda: (output/'render.json').exists())
        reader = spawn_role('camera-check')
        wait_for(lambda: reader.poll() is not None, 90)
        if reader.returncode:
            raise RuntimeError('Rhizome SHM reader failed; see camera-check.log')
        report['cameras'] = json.loads((output/'camera_check.json').read_text())
        if Path('/opt/aigen/image/bin/aigen_topics').exists():
            cli = subprocess.run(['/opt/aigen/image/bin/aigen','topics'],capture_output=True,text=True,timeout=5)
            (output/'cli_topics.log').write_text(cli.stdout+cli.stderr)
            if cli.returncode or '/odom' not in cli.stdout:
                raise RuntimeError('Native aigen topics CLI failed')
            report['native_cli_topics_passed'] = True
        print('READY: actual ai_agent and live MuJoCo camera streams; motion disconnected',flush=True)
        if config['mission']:
            msg = core.AgentChatMsg(dict(entry=dict(role='operator',source='portal',text=config['mission'])))
            report['mission_reply'] = [json.loads(str(body)) for _,body in core.sendRequest('agent-send',msg,3) if body is not None]
            if not any(b.get('str') == 'ok' for b in report['mission_reply']):
                raise RuntimeError('agent-send did not accept mission')
            report['mission_submitted'] = True
        until = time.monotonic()+config['seconds']
        while time.monotonic() < until:
            check()
            time.sleep(.1)
        report['sensors'] = json.loads((output/'sensors.json').read_text())
        report['render'] = json.loads((output/'render.json').read_text())
        report['publisher'] = json.loads((output/'publisher.json').read_text())
        report['passed'] = not any(report['sensors']['warnings'])
        if not report['passed']:
            raise RuntimeError('MuJoCo reported simulation warnings')
    except BaseException as error:
        report['error'] = str(error)
        raise
    finally:
        for _,process,log in reversed(children):
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            log.close()
        save(output/'report.json',report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--rhizome',type=Path)
    p.add_argument('--host-build',type=Path)
    p.add_argument('--task',default='dual-agent')
    p.add_argument('--agent-config',type=Path,help='JSON of native ai_agent config overrides, including credential FILE paths')
    p.add_argument('--mission',help='Sent through agent-send after sensor readiness; requires credentials')
    p.add_argument('--seconds',type=float,default=30,help='Wall seconds after readiness')
    p.add_argument('--camera-hz',type=float,default=1,help='Maximum RGB-D frame pairs/s; actual throughput is measured')
    p.add_argument('--chassis-configuration',default='')
    p.add_argument('--payload-configuration',default='')
    p.add_argument('--runtime-config',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--child',choices=['supervise','relay','sensors','render','publish','observe','camera-check'],help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.child:
        config = json.loads(args.runtime_config.read_text())
        return supervise(config) if args.child == 'supervise' else child(args.child,config)
    if not args.scene or not args.output:
        p.error('--scene and --output are required')
    if not all(math.isfinite(v) and v > 0 for v in (args.seconds,args.camera_hz)):
        p.error('seconds and camera-hz must be finite and positive')
    if args.mission and not args.agent_config:
        p.error('--mission requires --agent-config with Rhizome credential-file paths')
    install = inspect_install(args.rhizome,args.host_build)
    root, host = Path(install['root']), Path(install['host_build'])
    scene, output = args.scene.resolve(), args.output.resolve()
    if not shutil.which('bwrap'):
        p.error('bubblewrap is required for private Rhizome runtime paths')
    overrides = json.loads(args.agent_config.read_text()) if args.agent_config else {}
    if not isinstance(overrides,dict):
        p.error('agent-config must be a JSON object')
    output.mkdir(parents=True, exist_ok=False)
    runtime = stage(root,output,args.task,overrides)
    for name in ('aigen','aigen_helpers.sh'):
        (runtime/'opt/image/bin'/name).symlink_to(root/'scripts'/name)
    programs = {}
    for name in ('aigen_echo','aigen_topics','aigen_services','set_nav_source','set_drive_mux'):
        program = host/'bin'/name
        if program.is_file():
            (runtime/'opt/image/bin'/name).symlink_to(program)
            programs[name] = hashlib.sha256(program.read_bytes()).hexdigest()
    save(output/'cli_programs.json',programs)
    config = dict(scene=str(scene),output=str(output),root=str(root),host_build=str(host),
                  entry_point=str(root/'src/ai_agent/ai_agent.main.py'),task=args.task,
                  seconds=args.seconds,camera_hz=args.camera_hz,mission=args.mission,
                  chassis_configuration=args.chassis_configuration,payload_configuration=args.payload_configuration)
    save(output/'launch.json',config)
    save(output/'install.json',install)
    entry = Path(config['entry_point'])
    save(output/'source.json',{'entry_point_sha256':hashlib.sha256(entry.read_bytes()).hexdigest()})
    env = os.environ.copy()
    env.pop('NODE_NAME',None)
    env.update(PYTHONPATH=os.pathsep.join([str(host/'py'),str(root/'src'),str(Path(__file__).resolve().parents[1])]),
               AIGEN_DEVICE='dev_machine', AIGEN_CONFIG_PATH='/tmp/conf', PYTHONDONTWRITEBYTECODE='1',
               HOME='/tmp/home',XDG_CACHE_HOME='/tmp/home/.cache',
               MUJOCO_GL=os.environ.get('MUJOCO_GL','egl'),LP_NUM_THREADS=os.environ.get('LP_NUM_THREADS','1'))
    etc_mounts = ['--tmpfs','/etc']
    for path in sorted(Path('/etc').iterdir()):
        if path.name != 'aigen':
            etc_mounts += ['--ro-bind',str(path),str(path)]
    command = ['bwrap','--die-with-parent','--unshare-pid','--unshare-ipc',
               '--ro-bind','/','/','--dev-bind','/dev','/dev','--tmpfs','/dev/shm','--proc','/proc',*etc_mounts,
               '--bind',str(output),str(output),'--bind',str(runtime),'/tmp',
               '--bind',str(runtime/'opt'),'/opt/aigen','--bind',str(runtime/'etc'),'/etc/aigen',
               '--bind',str(runtime/'data'),'/data','--chdir','/tmp',
               sys.executable,'-u','-m',MODULE,'--child','supervise','--runtime-config',str(output/'launch.json')]
    result = subprocess.run(command,env=env)
    if result.returncode:
        raise SystemExit(f'Rhizome simulation exited {result.returncode}; see {output}')


if __name__ == '__main__':
    main()
