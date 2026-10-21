#!/usr/bin/env python3
"""Drive a Rhizome rover in MuJoCo with the on-rover ai_agent in the loop.

The agent is rhizome's own Conversation + tools (Claude plans, Gemini ER picks
image points) behind a simulated AgentInterface. Native LocalControl (rhizome
worker) turns its waypoints into wheel commands. Agent time is simulation
time: each Claude/Gemini call lands at its start time plus its real wall
latency, and the sim never runs ahead of wall clock, so the rover keeps rolling
on its last goals while the agent thinks, at any sim speed.
Type operator messages on stdin while it runs; stop keywords cancel_nav first.
A simulated operator re-engages AUTONOMY 15 s after the agent's own cancel_nav.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import sys
import threading
import time

import numpy as np

REPO = Path(__file__).resolve().parents[1]
NO_GOALS = {'waypoints': []}
# Obstacle grid step height. Its 0.25 m default let the robot (0.21 m wheels)
# drive onto a 0.2 m cinder block and tip over; about half a wheel radius holds.
STEP_M = 0.12


def parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scene', type=Path, default=REPO / 'outputs/soybean30_obstacle_gen3')
    p.add_argument('--rhizome', type=Path, default=REPO.parent / 'rhizome-orin-agent')
    p.add_argument('--host-build', type=Path, default=REPO / 'outputs/rhizome_host_agent/host')
    p.add_argument('--worker-python', type=Path, default=REPO.parent / 'ml-aigen-tools/venv312/bin/python')
    p.add_argument('--mission', default='Drive forward along the row you are in for about 4 metres, then stop and report where you ended up.')
    p.add_argument('--seconds', type=float, default=180, help='Simulated duration')
    p.add_argument('--realtime', type=float, default=1.0, help='Sim speed vs wall clock; 1.0 keeps LLM latency faithful')
    p.add_argument('--llm', choices=['sdk', 'api'], default='sdk',
                   help='sdk: Claude Agent SDK on the Claude Code login; api: Anthropic API key (as on the rover)')
    p.add_argument('--claude-model', help='Default: claude-sonnet-5 (sdk) or models.claude in ai_agent.node.yml (api)')
    p.add_argument('--nav-package', type=Path, default=REPO / 'outputs/nav_obstacle_package/nav_grid_4_nav',
                   help='nav_model obstacle-grid package, run in shadow mode as on the rover')
    p.add_argument('--max-speed', type=float, default=0.3, help='LocalControl auto_speed_max_ms')
    p.add_argument('--spawn', type=float, nargs=3, metavar=('X', 'Y', 'YAW_DEG'))
    p.add_argument('--viewer', action='store_true', help='Live MuJoCo viewer (uses GLFW instead of EGL)')
    p.add_argument('--target', help='Goal-directed eval: manifest instance id the mission asks the agent to find')
    p.add_argument('--target-radius', type=float, default=2.5,
                   help='Reached = robot within this many metres of the target, surface to surface '
                        '(missions ask for a ~2 m standoff)')
    p.add_argument('--output', type=Path)
    return p.parse_args()


class SimState:
    """Everything the agent thread and the physics loop share. Guarded by lock."""

    def __init__(self):
        self.lock = threading.Lock()
        self.t = 0.0
        self.pose = None  # (x, y, yaw) in odom == MuJoCo world
        self.camera = None  # latest nav_front capture
        self.mux = 'AUTONOMY'  # the sim operator has engaged autonomy
        self.source = 'NONE'
        self.enabled = True
        self.goals = None  # WaypointsMsg JSON from the agent's /nav/goals
        self.reengage_at = None  # sim time the operator re-engages AUTONOMY
        self.agent_error = None
        self.capture = None  # splat capture in progress: {dir, keyframer, frames}

    def effective_goals(self):
        # Mirrors rover gating: only AGENT-sourced goals under AUTONOMY drive.
        if self.mux == 'AUTONOMY' and self.source == 'AGENT' and self.goals:
            return self.goals
        return NO_GOALS


class Log:
    def __init__(self, path, state):
        self.f = path.open('w', buffering=1)
        self.state = state

    def __call__(self, kind, **data):
        t = self.state.t
        self.f.write(json.dumps(dict(t=round(t, 3), wall=time.time(), kind=kind, **data), default=str) + '\n')
        text = data.get('text') or json.dumps(data, default=str)
        print(f'[{t:7.2f}s] {kind}: {text[:400]}', flush=True)


def sim_clock(state):
    """(wait_until, latency): asyncio helpers that measure agent time in sim seconds."""
    async def wait_until(t):
        while state.t < t:
            await asyncio.sleep(0.005)

    def latency(fn):
        async def wrapped(*a, **k):
            t0, w0 = state.t, time.perf_counter()
            result = await fn(*a, **k)
            await wait_until(t0 + time.perf_counter() - w0)
            return result
        return wrapped
    return wait_until, latency


def make_host(state, log, out, cfg, ai_main, tools, Image, ImageDraw, latency, cam_K):
    """Simulated AgentInterface with the same contract as AiAgent's."""
    frames = out / 'frames'
    frames.mkdir()
    latest = {'v': None}
    counter = {'n': 0}

    guard = state.coverage

    def publish_locked(topic, msg):
        if topic == '/nav/goals':
            goals = json.loads(str(msg))
            for w in goals.get('waypoints', []):  # sim: map == odom (no RTK origin)
                if w['point']['frame'] == 'map':
                    w['point']['frame'] = 'odom'
            wps = goals.get('waypoints') or [{}]
            last = wps[-1].get('point', {})
            guard.on_goals([(w['point']['x'], w['point']['y']) for w in goals.get('waypoints', [])],
                           any(w.get('dir') == 'REVERSE' for w in goals.get('waypoints', [])))
            with state.lock:
                state.goals = goals
                state.progress.set_goal((last['x'], last['y']) if last.get('frame') == 'odom' else None)
            log('goals', waypoints=[(round(w['point']['x'], 2), round(w['point']['y'], 2), w['point']['frame'])
                                    for w in goals.get('waypoints', [])])

    def call_service_locked(name, msg, timeout_s=2.0):
        if name == 'set-drive-mux':
            with state.lock:
                state.mux = json.loads(str(msg))['requested']
            log('service', text=f'{name} -> {state.mux}')
            return [(None, 'ok')]
        log('service', text=f'{name} (not simulated)')
        return []

    def cancel_nav():
        with state.lock:
            state.goals = None
            state.progress.set_goal(None)
            state.mux = 'NONE'
            state.reengage_at = state.t + 15.0  # the sim operator's response time
        log('cancel_nav', text='goals cleared, drive_mux NONE')
        return 'ok'

    def set_enabled(on):
        state.enabled = bool(on)

    def grab_frame(position='front'):
        with state.lock:
            cam = (state.camera or {}).get(position)
        if cam is None:
            raise RuntimeError(f'no nav_{position} frame rendered yet')
        buf = io.BytesIO()
        Image.fromarray(cam['rgb']).save(buf, 'JPEG', quality=85)
        jpeg = buf.getvalue()
        counter['n'] += 1
        fid = counter['n']
        (frames / f'{fid:04d}_{position}.jpg').write_bytes(jpeg)
        latest['v'] = dict(frame_id=fid, depth=cam['depth'], intrinsics=cam['intrinsics'],
                           extrinsics=cam['extrinsics'], jpeg_bytes=jpeg, n_bytes=len(jpeg))
        return dict(frame_id=fid, jpeg_bytes=jpeg, snapshot=dict(
            frame_id=fid, n_jpeg_bytes=len(jpeg), intrinsics=cam['intrinsics'],
            extrinsics=cam['extrinsics'], has_depth=bool(cam['depth'].any())))

    def debug_overlay_update(points=None, projected=None, **_):
        # Save Gemini's picks over the frame they were made on.
        v = latest['v']
        if not points or v is None:
            return
        img = Image.open(io.BytesIO(v['jpeg_bytes'])).convert('RGB')
        draw = ImageDraw.Draw(img)
        w, h = img.size
        for i, p in enumerate(points):
            y, x = p['point']
            u, vv = x / 1000 * w, y / 1000 * h
            draw.ellipse([u - 5, vv - 5, u + 5, vv + 5], outline='red', width=2)
            draw.text((u + 7, vv - 7), f"{i}:{p.get('label', '')}", fill='red')
        img.save(frames / f"{v['frame_id']:04d}_gemini.jpg")

    def odom_pose():
        with state.lock:
            return state.pose

    def start_capture(name):
        from ai_agent.capture import Keyframer
        root = out / 'captures' / ''.join(c if c.isalnum() or c in '-_' else '_' for c in name)
        (root / 'images').mkdir(parents=True, exist_ok=True)
        (root / 'depth').mkdir(exist_ok=True)
        with state.lock:
            state.capture = dict(dir=root, keyframer=Keyframer(), frames=[])
        log('capture', text=f'start {name} -> {root}')

    def stop_capture():
        with state.lock:
            cap, state.capture = state.capture, None
        if cap is None:
            return {'frames': 0, 'path': None}
        K = cam_K
        meta = dict(camera_model='OPENCV', fl_x=K[0][0], fl_y=K[1][1], cx=K[0][2], cy=K[1][2], w=640, h=480,
                    k1=0, k2=0, p1=0, p2=0, depth_unit_scale_factor=1e-3, frames=cap['frames'])
        (cap['dir'] / 'transforms.json').write_text(json.dumps(meta, indent=1))
        log('capture', text=f"stop: {len(cap['frames'])} frames in {cap['dir']}")
        return {'frames': len(cap['frames']), 'path': str(cap['dir'])}

    # AiAgent's Gemini methods only touch these two attributes on self.
    class GeminiSelf:
        cfg_models = cfg['models']
        _client = None

        def _ensure_gemini_client(self):
            if GeminiSelf._client is None:
                from google import genai
                GeminiSelf._client = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
            return GeminiSelf._client

    g = GeminiSelf()

    @latency
    async def gemini_trajectory(jpeg, goal, constraints):
        pts = await ai_main.AiAgent.gemini_trajectory(g, jpeg, goal, constraints)
        log('gemini', text=f'{goal!r} -> {pts}')
        return pts

    @latency
    async def gemini_describe(jpeg, prompt):
        return await ai_main.AiAgent.gemini_describe(g, jpeg, prompt)

    state.stop_capture = stop_capture
    return tools.AgentInterface(
        publish_locked=publish_locked, call_service_locked=call_service_locked, cancel_nav=cancel_nav,
        set_enabled=set_enabled, is_enabled=lambda: state.enabled, grab_frame=grab_frame,
        latest_frame=lambda: latest['v'], gemini_trajectory=gemini_trajectory,
        gemini_describe=gemini_describe, debug_overlay_update=debug_overlay_update, odom_pose=odom_pose,
        start_capture=start_capture, stop_capture=stop_capture,
        check_coverage=lambda points, reverse: guard.check(state.pose, points, reverse, state.t))


def sim_aigen_cli(state, log, tools):
    """Stand-in for the rover's `aigen` CLI: echo of the topics the prompt uses, plus mode switches."""

    async def aigen_cli(ctx, args, timeout_s=10.0):
        a = [str(x) for x in args]
        if refusal := tools.cli_refusal(a):
            log('aigen', text=f'{" ".join(a)} -> {refusal}')
            return refusal
        with state.lock:
            pose, mux, source, enabled, goals = state.pose, state.mux, state.source, state.enabled, state.goals
        out = None
        if a[:1] == ['echo'] and len(a) > 1:
            topic = a[1].split('@')[0]
            if topic in ('/pose/map', '/odom', '/pose/odom') and pose:
                out = json.dumps(dict(x=round(pose[0], 3), y=round(pose[1], 3),
                                      yaw_deg=round(math.degrees(pose[2]), 1),
                                      frame='map (sim: map == odom, truth pose)'))
            elif topic == '/drive_mux/state':
                out = json.dumps(dict(requested=mux, actual=mux))
            elif topic == '/nav/source':
                out = json.dumps(dict(requested=source))
            elif topic == '/agent/enable':
                out = 'true' if enabled else 'false'
            elif topic == '/nav/goals':
                out = json.dumps(goals or {'waypoints': []})
        elif a[:1] == ['set-nav-source'] and len(a) > 1 and a[1].upper() in ('NONE', 'GPS_ROUTE', 'AGENT'):
            with state.lock:
                state.source = a[1].upper()
                if state.source != 'AGENT':
                    state.progress.set_goal(None)
            out = f'nav source -> {state.source}'
        elif a[:1] == ['set-drive-mux'] and len(a) > 1:
            with state.lock:
                state.mux = a[1].upper()
            out = f'drive mux -> {state.mux}'
        elif a[:1] == ['status']:
            out = f'sim rover: drive_mux={mux} nav_source={source} agent_enable={enabled} pose={pose}'
        if out is None:
            out = (f'[sim: `aigen {" ".join(a)}` is not simulated. Supported: echo /pose/map|/odom|'
                   '/drive_mux/state|/nav/source|/agent/enable|/nav/goals, set-nav-source, set-drive-mux, status]')
        log('aigen', text=f'{" ".join(a)} -> {out}')
        return out

    return aigen_cli


def run_agent(state, log, out, args, loop_box, ready, stop):
    """Agent asyncio runtime: mirrors AiAgent._agent_loop with sim I/O."""
    import importlib.util
    import yaml
    from anthropic import AsyncAnthropic
    from PIL import Image, ImageDraw
    from ai_agent import tools
    from ai_agent.conversation import Conversation
    from ai_agent.memory import Memory
    from ai_agent.prompts import L1_SYSTEM_PROMPT
    spec = importlib.util.spec_from_file_location('ai_agent_main', args.rhizome / 'src/ai_agent/ai_agent.main.py')
    ai_main = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ai_main)

    cfg = list(yaml.safe_load_all((args.rhizome / 'src/ai_agent/ai_agent.node.yml').read_text()))[1]
    if args.claude_model:
        cfg['models']['claude'] = args.claude_model
    memory = Memory(str(out / 'memory'))
    memory.write('sim-consent', 'reference', 'Gemini consent for this simulated mission',
                 'Simulation run: no people are in the scene. Gemini ER calls are consented for this mission.')
    wait_until, latency = sim_clock(state)
    host = make_host(state, log, out, cfg, ai_main, tools, Image, ImageDraw, latency, args.cam_K)
    tools._IMPLS['aigen_cli'] = sim_aigen_cli(state, log, tools)

    async def wait_for_event(ctx, topics=None, timeout_s=300.0):
        # tools._wait_for_event with a simulation-time deadline.
        deadline = state.t + timeout_s
        while state.t < deadline:
            try:
                ev = await asyncio.wait_for(ctx.event_queue.get(), 0.05)
            except asyncio.TimeoutError:
                continue
            if not topics or ev.get('event') in topics:
                return json.dumps(ev)
        return json.dumps({'event': 'timeout', 'details': {}})
    tools._IMPLS['wait_for_event'] = wait_for_event

    class Operator:
        async def send_message(self, chat_id, text, photo=None):
            log('agent_says', text=text)

    async def main():
        loop_box['loop'] = asyncio.get_running_loop()
        queue = asyncio.Queue()
        loop_box['queue'] = queue
        ctx = tools.ToolContext(memory=memory, telegram=Operator(), event_queue=queue, primary_chat_id=1, agent=host)
        conv = Conversation(client=AsyncAnthropic() if args.llm == 'api' else None, model=cfg['models']['claude'],
                            system_prompt=L1_SYSTEM_PROMPT, memory=memory, tool_ctx=ctx)
        if args.llm == 'api':
            conv.client.messages.create = latency(conv.client.messages.create)
        # Log each tool call as Claude makes it.
        dispatch = tools.dispatch

        async def logged(ctx_, name, inp):
            log('tool', text=f'{name}({json.dumps(inp)[:300]})')
            res = await dispatch(ctx_, name, inp)
            log('result', text=f'{name} -> {str(res)[:4000]}')
            return res
        tools.dispatch = logged
        if args.llm == 'sdk':
            ready.set()
            await sdk_loop(conv, ctx, queue, tools, wait_until, state, log, out, stop, args)
            return
        ready.set()
        while not stop.is_set():
            ev = await queue.get()
            await conv.add_user_event(ev)
            started = time.perf_counter()
            try:
                final = await conv.run_until_quiescent()
            except Exception as e:
                final = f'[internal error: {e}]'
                state.agent_error = final  # the run is meaningless without the agent
            log('agent_final', text=final or '', wall_s=round(time.perf_counter() - started, 2))
        (out / 'conversation.json').write_text(json.dumps(conv.messages, indent=1, default=str))

    asyncio.run(main())


async def sdk_loop(conv, ctx, queue, tools, wait_until, state, log, out, stop, args):
    """The agent loop on the Claude Agent SDK instead of the Messages API.

    Same prompt, tools and dispatch as the rover; the SDK runs the tool loop.
    Model calls are not visible here, so their latency is charged at tool
    boundaries: wall time since the previous tool returned (or the event was
    sent) is the model's, and the next tool waits until sim time catches up.
    """
    from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, ClaudeSDKClient, ResultMessage,
                                  TextBlock, create_sdk_mcp_server, tool)
    mark = {'t': 0.0, 'wall': time.perf_counter()}

    def make(name, desc, schema):
        @tool(name, desc, schema)
        async def handler(inp):
            await wait_until(mark['t'] + time.perf_counter() - mark['wall'])
            result = await tools.dispatch(ctx, name, inp)
            mark['t'], mark['wall'] = state.t, time.perf_counter()
            return {'content': [{'type': 'text', 'text': str(result)}]}
        return handler

    server = create_sdk_mcp_server('rover', tools=[make(d['name'], d['description'], d['input_schema'])
                                                   for d in tools.TOOL_DEFS])
    names = [f"mcp__rover__{d['name']}" for d in tools.TOOL_DEFS]
    options = ClaudeAgentOptions(
        system_prompt=conv.system_prompt + '\n\n# Persistent memory index\n\n' + conv.memory.read_index(),
        mcp_servers={'rover': server}, tools=[], allowed_tools=names, setting_sources=[],
        permission_mode='bypassPermissions', model=args.claude_model or 'claude-sonnet-5',
        max_turns=200, cwd=str(out))
    transcript = []
    async with ClaudeSDKClient(options) as client:
        while not stop.is_set():
            ev = await queue.get()
            await conv.add_user_event(ev)  # rover's event -> user text formatting
            mark['t'], mark['wall'] = state.t, time.perf_counter()
            started = time.perf_counter()
            await client.query(conv.messages[-1]['content'])
            final = ''
            async for msg in client.receive_response():
                transcript.append(repr(msg))
                if isinstance(msg, AssistantMessage):
                    final = ' '.join(b.text for b in msg.content if isinstance(b, TextBlock)) or final
                elif isinstance(msg, ResultMessage) and msg.is_error:
                    state.agent_error = f'[sdk error: {msg.result}]'
            log('agent_final', text=final, wall_s=round(time.perf_counter() - started, 2))
            (out / 'conversation.json').write_text(json.dumps(transcript, indent=1))


def surface_distance(model, target, max_m=50.0):
    """fn(data) -> closest distance between the robot's and the target's collision geoms."""
    import mujoco
    names = [model.geom(i).name for i in range(model.ngeom)]
    robot = [i for i, n in enumerate(names) if n.startswith('element/') and n.endswith('_collision')]
    goal = [i for i, n in enumerate(names) if n.startswith(target + '/') and model.geom_contype[i]]
    fromto = np.zeros(6)
    return lambda data: min(mujoco.mj_geomDistance(model, data, a, b, max_m, fromto) for a in robot for b in goal)


def agent_event(name, details, loop_box):
    """System event into the agent's queue (what wait_for_event wakes on)."""
    asyncio.run_coroutine_threadsafe(loop_box['queue'].put({'event': name, 'details': details}), loop_box['loop'])


def operator_say(text, host_cancel, loop_box, log, stop_keywords):
    if any(text.strip().lower() == k or text.strip().lower().startswith(k + ' ') for k in stop_keywords):
        host_cancel()  # reflexive, before Claude sees it
    log('operator', text=text)
    asyncio.run_coroutine_threadsafe(
        loop_box['queue'].put({'event': 'telegram_msg', 'details': {'user': 'operator', 'chat_id': 1, 'text': text}}),
        loop_box['loop'])


def main():
    args = parse()
    for key in ('GEMINI_API_KEY',) + (('ANTHROPIC_API_KEY',) if args.llm == 'api' else ()):
        if not os.environ.get(key):
            sys.exit(f'{key} is not set')
    os.chdir(REPO)
    os.environ['MUJOCO_GL'] = 'glfw' if args.viewer else 'egl'
    args.rhizome = args.rhizome.expanduser().resolve()
    sys.path.insert(0, str(args.rhizome / 'src'))
    sys.path.insert(0, str(args.host_build.expanduser().resolve() / 'py'))  # pyzome for tools.py
    import mujoco
    import yaml
    from scipy.spatial.transform import Rotation
    from sim_harness.scene import _read_spec
    from scene_pipeline.agriculture import compose_robot, robot_footprint
    from scene_pipeline.agriculture_drive import camera_rig, trajectory_image, wheel_targets
    from scene_pipeline.contracts import read_json
    from scene_pipeline.navigation import add_cameras, raw_depth, robot_dimensions
    from scene_pipeline.rhizome import RhizomeController

    scene = args.scene.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = (args.output or REPO / 'outputs' / f'agent_nav_{stamp}').expanduser().resolve()
    out.mkdir(parents=True)
    generation = read_json(scene / 'generation.json')
    profile = read_json(scene / 'robot_profile.json')
    manifest = read_json(scene / 'manifest.json')
    bounds = np.asarray(manifest['terrain']['bounds_m'])
    spawn = generation['spawn']
    if args.spawn:
        spawn = {**spawn, 'position_m': list(args.spawn[:2]), 'yaw_rad': math.radians(args.spawn[2])}
    robot = _read_spec(scene / 'robot.mjz')
    profile = add_cameras(robot, profile, dict(cameras=['front', 'rear']))
    spec, model, data = compose_robot(_read_spec(scene / 'environment.mjz'), robot, profile, spawn, bounds)
    model.vis.quality.shadowsize = 1024
    # alamin's recording format, so `pipeline replay <out> --viewer` works on agent runs.
    from sim_harness.export import export_mujoco
    from scene_pipeline.dataset import Recorder
    export_mujoco(spec, data, out / 'rollout_scene.mjz')
    recorder = Recorder(out / 'data.h5', dict(flow='agent-nav', scene=str(scene), mission=args.mission,
                                              clock='simulation_seconds'))
    rig = camera_rig(profile, 640, 480)
    K = np.asarray(rig['nav_front']['K'])
    args.cam_K = K.tolist()  # both nav cameras share it
    intrinsics = dict(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]), width=640, height=480)
    chassis = model.body('element/chassis').id
    # Stow the weeding arms (the rover lifts them when not weeding). Hanging at
    # zero, their tools plough into the ground when reversing.
    for arm in profile.get('arms', []):
        sign = np.asarray(arm['motor_to_joint_sign'])
        jid = model.joint('element/' + arm['pitch_joint']).qposadr[0]
        site = model.site('element/' + arm['tool_site']).id
        best = None
        for pitch in [arm['home_motor_rad'][1], *arm['motor_limits_rad'][1]]:
            data.qpos[jid] = sign[1] * pitch
            mujoco.mj_forward(model, data)
            if best is None or data.site_xpos[site][2] > best[1]:
                best = (pitch, data.site_xpos[site][2])
        data.qpos[jid] = sign[1] * best[0]
        for key, motor in (('yaw_joint', 0.0), ('pitch_joint', best[0])):
            data.ctrl[model.actuator('element/' + arm[key]).id] = sign[0 if key == 'yaw_joint' else 1] * motor
    mujoco.mj_forward(model, data)
    wheel_names = ['element/' + w['id'] for w in profile['wheels']]
    controls = np.array([model.actuator(n).id for n in wheel_names])
    renderer = mujoco.Renderer(model, height=480, width=640)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0

    state = SimState()
    log = Log(out / 'events.jsonl', state)
    node_cfg = list(yaml.safe_load_all((args.rhizome / 'src/ai_agent/ai_agent.node.yml').read_text()))[1]
    stop_keywords = node_cfg['stop_keywords']
    from ai_agent.coverage import CoverageGuard
    from ai_agent.progress import ProgressMonitor, obstacle_event
    from nav_model.swept_coverage import SeenMap
    state.progress = ProgressMonitor(node_cfg['agent_arrival_radius_m'], node_cfg['agent_stuck_radius_m'],
                                     node_cfg['agent_stuck_time_s'], node_cfg['agent_approach_m'])
    dims = robot_dimensions(profile)['robot']
    half_track = robot_dimensions(profile)['drive']['trackwidth'] / 2
    state.coverage = CoverageGuard(
        dims['length'] / 2, dims['width'] / 2, max_unseen_m2=node_cfg['coverage_max_unseen_m2'] if node_cfg['coverage_slowdown'] else math.inf,
        recent_s=node_cfg['coverage_recent_s'], horizon_m=node_cfg['coverage_horizon_m'],
        lookahead_m=node_cfg['coverage_lookahead_m'], camera_x_m=node_cfg['camera_x_m'],
        hfov_deg=math.degrees(2 * math.atan(320 / intrinsics['fx'])), max_speed=node_cfg['coverage_max_speed'])
    limit_until = -1.0  # sim time the agent's /drive/limits lease lapses (drive_control's 1 s)
    controller = RhizomeController(dict(root=str(args.rhizome), host_build=str(args.host_build), python=str(args.worker_python),
                                        perception='none',  # goals arrive via set_goals
                                        parameters=dict(auto_speed_max_ms=args.max_speed),
                                        nav_model=dict(package=str(args.nav_package), device='cpu', cameras=['front', 'rear'],
                                                       depth=True, obstacle_detection=True, drive=False, threads=8,
                                                       params=dict(obstacle_max_height_m=STEP_M),
                                                       robot_dimensions=robot_dimensions(profile))),
                                   out, timeout=1e9)

    def cancel():
        with state.lock:
            state.goals = None
            state.mux = 'NONE'
        log('cancel_nav', text='reflexive stop keyword')

    def render(position):
        """nav_<position> RGB + raw GRAY16 depth + optical pose in the chassis frame."""
        name = 'nav_' + position
        cid = model.camera('element/' + name).id
        renderer.update_scene(data, 'element/' + name, scene_option=option)
        rgb = renderer.render().copy()
        renderer.enable_depth_rendering()
        try:
            depth = raw_depth(renderer.render().copy())
        finally:
            renderer.disable_depth_rendering()
        R = data.xmat[chassis].reshape(3, 3)
        rot = R.T @ data.cam_xmat[cid].reshape(3, 3) @ np.diag([1, -1, -1])  # optical: x right, y down, z fwd
        return dict(rgb=rgb, depth=depth, K=np.asarray(rig[name]['K']), rotation_robot=rot,
                    position_robot=R.T @ (data.cam_xpos[cid] - data.xpos[chassis]),
                    height_m=np.asarray(rig[name]['cam_height_m']))

    def capture():
        mujoco.mj_forward(model, data)
        views = {pos: render(pos) for pos in ('front', 'rear')}
        agent_views = {}
        for name, f in views.items():
            pos = f['position_robot'].copy()
            pos[2] = float(f['height_m'])  # rhizome body frame sits on the ground plane
            qx, qy, qz, qw = Rotation.from_matrix(f['rotation_robot']).as_quat()
            ext = dict(tx=pos[0], ty=pos[1], tz=pos[2], qw=qw, qx=qx, qy=qy, qz=qz)
            agent_views[name] = dict(rgb=f['rgb'], depth=f['depth'], intrinsics=intrinsics,
                                     extrinsics={k: float(v) for k, v in ext.items()})
        return agent_views, {f'{p}_{k}': v for p, view in views.items() for k, v in view.items()}

    def record_keyframe(t, arrays):
        """Splat capture: posed RGB-D keyframes from both nav cameras."""
        from PIL import Image
        with state.lock:
            cap, pose = state.capture, state.pose
        if cap is None or pose is None or not cap['keyframer'].accept(pose):
            return
        for pos in ('front', 'rear'):
            cid = model.camera('element/nav_' + pos).id
            c2w = np.eye(4)  # MuJoCo camera axes (x right, y up, looking -z) = nerfstudio/OpenGL
            c2w[:3, :3], c2w[:3, 3] = data.cam_xmat[cid].reshape(3, 3), data.cam_xpos[cid]
            i = len(cap['frames'])
            rgb_path, depth_path = f'images/{pos}_{i:05d}.png', f'depth/{pos}_{i:05d}.png'
            Image.fromarray(arrays[pos + '_rgb']).save(cap['dir'] / rgb_path, compress_level=1)
            mm = arrays[pos + '_depth'].astype(np.uint32) // 10  # 0.1 mm -> mm
            mm[mm > 1500] = 0  # the D405 is only useful to ~1.5 m outdoors
            Image.fromarray(mm.astype(np.uint16)).save(cap['dir'] / depth_path, compress_level=1)
            cap['frames'].append(dict(file_path=rgb_path, depth_file_path=depth_path, time=round(t, 3),
                                      transform_matrix=c2w.round(6).tolist()))

    loop_box, ready, stop = {}, threading.Event(), threading.Event()
    threading.Thread(target=run_agent, args=(state, log, out, args, loop_box, ready, stop), daemon=True).start()
    if not ready.wait(60):
        sys.exit('agent runtime did not start')

    def stdin_reader():
        for line in sys.stdin:
            if line.strip():
                operator_say(line.strip(), cancel, loop_box, log, stop_keywords)
    if sys.stdin.isatty():
        threading.Thread(target=stdin_reader, daemon=True).start()

    viewer = None
    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)

    with state.lock:
        state.camera = capture()[0]
    blocked = (False, False)
    # Exploration score: field cells and manifest objects that entered the front
    # camera's view wedge. ponytail: no occlusion, so an upper bound on what was seen.
    half_fov = math.atan(320 / intrinsics['fx'])
    cell = 0.5
    gx, gy = np.meshgrid(np.arange(bounds[0, 0] + cell / 2, bounds[1, 0], cell),
                         np.arange(bounds[0, 1] + cell / 2, bounds[1, 1], cell))
    seen_cells = np.zeros(gx.shape, bool)
    objects = {k: np.asarray(v['position_m'][:2]) for k, v in manifest.get('instances', {}).items()
               if v.get('category') != 'crop'}
    seen_objects = set()
    if args.target and args.target not in objects:
        sys.exit(f'--target {args.target!r} is not a manifest instance: {sorted(objects)}')
    target_min, target_reached_at = math.inf, None
    target_distance = surface_distance(model, args.target) if args.target else None

    def in_view(x, y, pose, max_range):
        dx, dy = x - pose[0], y - pose[1]
        rng = np.hypot(dx, dy)
        bearing = np.angle(np.exp(1j * (np.arctan2(dy, dx) - pose[2])))
        return (rng >= 0.5) & (rng <= max_range) & (np.abs(bearing) <= half_fov)
    operator_say(args.mission, cancel, loop_box, log, stop_keywords)
    steps = round(args.seconds / 0.002)
    sent = None
    yaw_i = 0.0
    # Obstacles lower than the obstacle grid's max step height count as bumps, not crashes.
    LOW_M = STEP_M
    mujoco.mj_forward(model, data)
    tops = {}  # object prefix -> highest point of its geoms
    for i in range(model.ngeom):
        name = model.geom(i).name
        if name.startswith('element/') or name == 'terrain' or '/' not in name:
            continue
        R = data.geom_xmat[i].reshape(3, 3)
        top = data.geom_xpos[i][2] + R[2] @ model.geom_aabb[i][:3] + np.abs(R[2]) @ model.geom_aabb[i][3:]
        tops[name.split('/')[0]] = max(tops.get(name.split('/')[0], -np.inf), top)
    low_geoms = {i for i in range(model.ngeom)
                 if '/' in model.geom(i).name and tops.get(model.geom(i).name.split('/')[0], np.inf) < LOW_M}
    bumped = set()
    positions = []
    lag_max = 0.0
    wall0 = time.perf_counter()
    failure = None
    try:
        for tick in range(steps + 1):
            t = float(data.time)
            if tick % 5 == 0:  # 100 Hz control, as in agriculture-drive
                mujoco.mj_forward(model, data)
                quat = data.xquat[chassis].copy()
                yaw = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_euler('xyz')[2]
                with state.lock:
                    state.t = t
                    state.pose = (float(data.xpos[chassis, 0]), float(data.xpos[chassis, 1]), float(yaw))
                    goals = state.effective_goals()
                if goals is not sent:
                    controller.request(dict(op='set_goals', time=t, goals=goals))
                    sent = goals
                vel = np.zeros(6)
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, chassis, vel, 0)  # world frame
                forward = vel[3] * math.cos(yaw) + vel[4] * math.sin(yaw)
                twist = controller.step(t, data.xpos[chassis], quat, forward)['twist']
                fastest = abs(twist[0]) + half_track * abs(twist[1])  # drive_control.apply_drive_limit
                if t < limit_until and fastest > state.coverage.max_speed:
                    twist = [v * state.coverage.max_speed / fastest for v in twist]
                # ponytail: yaw-rate PI stands in for a base that tracks its turn
                # commands; open-loop skid steer here gets 4-35% of small rates.
                # Replace with tuned wheel/soil contact if turning dynamics matter.
                yaw_err = twist[1] - vel[2]
                yaw_i = float(np.clip(yaw_i + 2.0 * yaw_err * 0.01, -1.0, 1.0)) if twist[1] else 0.0
                data.ctrl[controls], _ = wheel_targets(profile, twist[0], twist[1] + 1.5 * yaw_err * bool(twist[1]) + yaw_i)
                positions.append(data.xpos[chassis].copy())
                if args.target:
                    d = target_distance(data)
                    target_min = min(target_min, d)
                    if target_reached_at is None and d <= args.target_radius:
                        target_reached_at = t
                        log('target_reached', text=f'{args.target} within {d:.2f} m')
                # Pace to wall clock so the agent's real latency is simulated time.
                ahead = t / args.realtime - (time.perf_counter() - wall0)
                if ahead > 0:
                    time.sleep(ahead)
                lag_max = max(lag_max, -ahead)
            if tick % 500 == 0:  # 1 Hz, as the node's Tick
                with state.lock:
                    ev = state.progress.update(state.pose, t)
                if ev:
                    log(ev['event'], **ev['details'])
                    agent_event(ev['event'], ev['details'], loop_box)
                if state.reengage_at is not None and t >= state.reengage_at:
                    with state.lock:
                        state.mux, state.reengage_at = 'AUTONOMY', None
                    operator_say('I re-engaged drive_mux AUTONOMY after your cancel_nav; carry on.',
                                 cancel, loop_box, log, stop_keywords)
            if tick % 50 == 0:  # 10 Hz nav camera
                cam, nav_arrays = capture()
                record_keyframe(t, nav_arrays)
                recorder.append('state', t, qpos=data.qpos.copy(), qvel=data.qvel.copy())
                if state.pose:
                    seen_cells |= in_view(gx, gy, state.pose, 6.0)
                    seen_objects.update(k for k, xy in objects.items() if in_view(xy[0], xy[1], state.pose, 8.0))
                with state.lock:
                    state.camera = cam
                nav = controller.nav_infer(t, **nav_arrays)
                prediction = nav['prediction']
                if nav.get('seen'):
                    s = nav['seen']
                    blind = state.coverage.on_map(SeenMap.decode(s['t_s'], s['origin_x'], s['origin_y'], s['cell_m'],
                                                                 s['size'], bytes(s['age'])), state.pose, t)
                    if blind:  # AiAgent._on_seen: renew the /drive/limits lease
                        if t > limit_until:
                            log('blind', unseen_m2=blind['unseen_m2'], first_unseen_body=blind['nearest_xy'])
                        limit_until = t + 1.0
                now_blocked = (bool(prediction.get('obstacle_fwd')), bool(prediction.get('obstacle_rev')))
                if now_blocked != blocked:
                    blocked = now_blocked
                    log('obstacle', front=blocked[0], rear=blocked[1],
                        clearance_fwd_m=round(prediction.get('clearance_fwd_m', 0), 2))
                    xyz = lambda p: (p['x'], p['y'], p['z']) if p else None
                    ev = obstacle_event(blocked, xyz(prediction.get('nearest_fwd')), xyz(prediction.get('nearest_rev')))
                    agent_event(ev['event'], ev['details'], loop_box)
                if viewer:
                    viewer.sync()
            corners = robot_footprint(profile) @ data.xmat[chassis].reshape(3, 3)[:2, :2].T + data.xpos[chassis, :2]
            if (corners < bounds[0]).any() or (corners > bounds[1]).any():
                failure = 'Robot footprint left the field'
            if data.xmat[chassis].reshape(3, 3)[2, 2] < .4:
                failure = 'Robot tipped over'
            for c in data.contact:
                names = [model.geom(int(g)).name for g in c.geom]
                if any(n.startswith('element/') for n in names) and not all(n == 'terrain' or n.startswith('element/') for n in names):
                    other = next(int(g) for g, n in zip(c.geom, names) if not n.startswith('element/'))
                    if other in low_geoms and not any('chassis' in n for n in names):
                        if other not in bumped:  # wheels may drive over what the grid calls passable
                            bumped.add(other)
                            log('bump', text=f'{names} (object under {LOW_M} m)')
                    else:
                        failure = f'Robot contacted {names}'
            if state.agent_error:
                failure = 'Agent error: ' + state.agent_error[:300]
            if failure:
                break
            mujoco.mj_step(model, data)
    except KeyboardInterrupt:
        failure = 'interrupted'
    finally:
        if state.capture:
            state.stop_capture()  # keep what was recorded if the run ended mid-capture
        stop.set()
        controller.close()
        recorder.close()
        renderer.close()
        if viewer:
            viewer.close()
    wall = time.perf_counter() - wall0
    positions = np.asarray(positions)
    trajectory_image(out / 'trajectory.png', positions, bounds, manifest['rows'])
    np.save(out / 'positions.npy', positions)  # chassis xyz at 100 Hz
    (out / 'run.json').write_text(json.dumps(dict(scene=str(scene), spawn=spawn, target=args.target)))
    report = dict(failure=failure, sim_seconds=float(data.time), wall_seconds=round(wall, 1),
                  realtime_factor=round(float(data.time) / wall, 3), max_lag_s=round(lag_max, 3),
                  start_xy=positions[0, :2].round(3).tolist(), end_xy=positions[-1, :2].round(3).tolist(),
                  displacement_m=float(np.linalg.norm(positions[-1, :2] - positions[0, :2])),
                  path_length_m=float(np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1).sum()),
                  coverage_pct=round(100 * float(seen_cells.mean()), 1),
                  objects_seen=f'{len(seen_objects)}/{len(objects)}', objects_missed=sorted(set(objects) - seen_objects),
                  mission=args.mission)
    if args.target:
        final = target_distance(data)
        report.update(target=args.target, target_reached_at_s=target_reached_at,
                      target_min_distance_m=round(target_min, 2), target_final_distance_m=round(final, 2),
                      target_success=final <= args.target_radius)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    log('done', **report)
    print(f'output: {out}')


if __name__ == '__main__':
    main()
