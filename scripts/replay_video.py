#!/usr/bin/env python3
"""Render a scripts/run_agent_nav.py run to a 1080p replay video.

usage: .venv/bin/python scripts/replay_video.py RUN_DIR [--out RUN_DIR/replay.mp4]
       [--start S --end S]   sim-second window, for test clips
       [--fast 8 --slow 2]   playback speed between and around agent activity
       [--scene SCENE_DIR]   draw the run in another build of its scene (same layout, new looks)
       [--export-hero DIR]   write the hero view for scripts/blender_hero.py instead of a video
       [--hero-frames DIR]   use DIR/frames (hero view) and DIR/nav (robot view) rendered by blender_hero.py
       [--rhizome ROOT]      rhizome checkout (default $RHIZOME_ROOT): production robot look on older runs
       --out x.png writes the first frame only.

Left: chase camera, with who is driving (agent / nav_model), the sim clock,
a mission timeline and a minimap. Right: the robot's front nav camera,
re-rendered at the run's own settings with nav_model's row lines while it
drives, and the agent's trace (thinking, messages, tool calls, events).
"""
import argparse
import bisect
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

W, H, SIDE, FPS = 1920, 1080, 560, 30
HERO_W = W - SIDE
NAV_W, NAV_H = 528, 396
TRACE_Y = 488  # top of the trace feed in the sidebar
FONTS = '/usr/share/fonts/truetype/roboto/unhinted/RobotoTTF/Roboto-{}.ttf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'

BG, SURFACE, BORDER = (15, 18, 22), (24, 28, 35), (44, 51, 61)
TEXT, MUTED = (232, 234, 237), (139, 148, 158)
AMBER = (245, 158, 11)
DRIVER = {'AGENT': ((167, 139, 250), 'AGENT'), 'NAV_MODEL': ((45, 212, 191), 'NAV MODEL'), 'NONE': ((107, 114, 128), 'STOPPED')}
QUIET_TOOLS = {'memory_read', 'robot_state', 'wait_for_event', 'send_message', 'set_nav_source', 'aigen_cli'}
SLOW_KINDS = {'think', 'says', 'to_operator', 'operator', 'handoff'}


def mix(c, f, base=BG):
    return tuple(round(b + (a - b) * f) for a, b in zip(c, base))


def clock(t):
    return f'{int(t) // 60:02d}:{int(t) % 60:02d}'


def trace(events):
    """Sidebar entries (t, kind, text) from events.jsonl rows."""
    out, steer = [], None
    for e in events:
        t, kind, text = e['t'], e['kind'], ' '.join((e.get('text') or '').split())
        if kind == 'thinking':
            out.append((t, 'think', text))
        elif kind == 'says':
            out.append((t, 'says', text))
        elif kind in ('agent_says', 'agent_final'):
            if out and out[-1][1:] == ('says', text):  # the same words, streamed and then sent
                out.pop()
            out.append((t, 'to_operator', text))
        elif kind == 'operator':
            out.append((t, 'operator', text))
        elif kind == 'nav_source':
            out.append((t, 'handoff', DRIVER.get(text, DRIVER['NONE'])[1]))
        elif kind == 'tool':
            name, _, rest = text.partition('(')
            if name == 'grab_frame':
                out.append((t, 'event', 'looks at the camera'))
            elif name not in QUIET_TOOLS:
                try:
                    args = ', '.join(f'{k}={json.dumps(v)}' for k, v in json.loads(rest[:-1]).items())
                except ValueError:
                    args = rest[:-1]
                out.append((t, 'tool', f'{name}\t{args}'))
        elif kind == 'nav_model':
            if steer is not None and e.get('steer_source') != steer:
                out.append((t, 'event', 'row-end turn' if e.get('steer_source') == 'TURN' else 'back on the rows'))
            steer = e.get('steer_source')
        elif kind == 'arrived':
            out.append((t, 'event', 'arrived at goal'))
        elif kind in ('stuck', 'cancel_nav', 'edge_stop'):
            out.append((t, 'warn', kind.replace('_', ' ')))
        elif kind == 'obstacle':
            out.append((t, 'warn', 'obstacle stop' if e.get('front') or e.get('rear') else 'obstacle cleared'))
    return out


def drivers(events, end):
    """[(t0, t1, source)] covering 0..end; the agent holds the rover until the first hand-off."""
    cuts = [(0.0, 'AGENT')] + [(e['t'], e.get('text') or 'NONE') for e in events if e['kind'] == 'nav_source']
    return [(t0, t1, s) for (t0, s), (t1, _) in zip(cuts, cuts[1:] + [(end, None)]) if t1 > t0]


def schedule(start, end, keys, fast, slow):
    """Sim time of each output frame: `slow` from 1 s before to 4 s after each key time, else `fast`."""
    times, t = [], start
    while t < end:
        times.append(t)
        i = bisect.bisect_left(keys, t - 4)
        t += (slow if i < len(keys) and keys[i] <= t + 1 else fast) / FPS
    return times


def wrap(draw, text, font, width, max_lines):
    lines, line = [], ''
    for word in text.split():
        trial = f'{line} {word}'.strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and draw.textlength(lines[-1] + '…', font=font) > width:
            lines[-1] = lines[-1][:-1]
        lines[-1] += '…'
    return lines


class Hud:
    def __init__(self, entries, segments, end):
        from PIL import ImageFont
        self.entries, self.segments, self.end = entries, segments, end
        f = lambda style, size: ImageFont.truetype(FONTS.format(style), size)
        self.caps, self.small, self.body = f('Medium', 13), f('Regular', 14), f('Regular', 18)
        self.italic, self.medium, self.big = f('Italic', 17), f('Medium', 16), f('Medium', 20)
        self.mono, self.clock_font = ImageFont.truetype(MONO, 14), ImageFont.truetype(MONO, 22)
        self.cache = {}

    def cached(self, key, make):
        if key not in self.cache:
            self.cache = {k: v for k, v in self.cache.items() if k[0] != key[0]}
            self.cache[key] = make()
        return self.cache[key]

    def static(self):
        """Scrims over the hero and the sidebar chrome."""
        from PIL import Image, ImageDraw
        im = Image.new('RGBA', (W, H), (0, 0, 0, 0))
        ramp = np.zeros((H, HERO_W, 4), np.uint8)
        ramp[:120, :, 3] = np.linspace(150, 0, 120)[:, None]
        ramp[-140:, :, 3] = np.linspace(0, 170, 140)[:, None]
        im.paste(Image.fromarray(ramp), (0, 0))
        d = ImageDraw.Draw(im)
        d.rectangle([HERO_W, 0, W, H], fill=BG + (255,))
        d.line([HERO_W, 0, HERO_W, H], fill=BORDER + (255,))
        x = HERO_W + 16
        d.text((x, 20), 'ROBOT VIEW', font=self.caps, fill=TEXT)
        d.text((W - 16, 20), 'front nav camera', font=self.small, fill=MUTED, anchor='ra')
        d.text((x, TRACE_Y - 28), 'AGENT TRACE', font=self.caps, fill=TEXT)
        d.line([x, TRACE_Y - 6, W - 16, TRACE_Y - 6], fill=BORDER + (255,))
        return im

    def driver(self, source, sim_since):
        """Segmented control top-left of the hero, plus a hand-off toast for 3 sim s."""
        from PIL import Image, ImageDraw
        im = Image.new('RGBA', (HERO_W, 150), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((24, 20), 'IN CONTROL', font=self.caps, fill=TEXT)
        x, y, w, h = 24, 44, 172, 44
        d.rounded_rectangle([x, y, x + 2 * w, y + h], 22, fill=BG + (215,), outline=BORDER + (255,))
        for k, key in enumerate(('AGENT', 'NAV_MODEL')):
            colour, label = DRIVER[key]
            on, cx, cy = source == key, x + k * w + 30, y + h // 2
            if on:
                d.rounded_rectangle([x + k * w + 4, y + 4, x + (k + 1) * w - 4, y + h - 4], 18, fill=colour + (255,))
            ink = BG if on else MUTED
            if key == 'AGENT':  # spark
                d.polygon([(cx, cy - 9), (cx + 3, cy - 3), (cx + 9, cy), (cx + 3, cy + 3), (cx, cy + 9), (cx - 3, cy + 3), (cx - 9, cy), (cx - 3, cy - 3)], fill=ink)
            else:  # crop rows
                for dx in (-6, 0, 6):
                    d.line([cx + dx, cy - 8, cx + dx, cy + 8], fill=ink, width=3)
            d.text((cx + 18, cy), label, font=self.medium, fill=ink, anchor='lm')
        if source not in ('AGENT', 'NAV_MODEL'):
            d.rounded_rectangle([x + 2 * w + 12, y + 6, x + 2 * w + 124, y + h - 6], 16, fill=DRIVER['NONE'][0] + (255,))
            d.text((x + 2 * w + 68, y + h // 2), 'STOPPED', font=self.medium, fill=TEXT, anchor='mm')
        elif sim_since is not None and sim_since < 3:
            colour, label = DRIVER[source]
            a = round(255 * min(1, 3 - sim_since))
            msg = f'Hand-off to {label}'
            tw = d.textlength(msg, font=self.big)
            d.rounded_rectangle([HERO_W / 2 - tw / 2 - 24, 100, HERO_W / 2 + tw / 2 + 24, 146], 23, fill=BG + (round(a * .85),), outline=colour + (a,), width=2)
            d.text((HERO_W / 2, 123), msg, font=self.big, fill=colour + (a,), anchor='mm')
        return im

    def timeline(self, d, t, speed):
        t0, t1, x0, x1, y = 0, self.end, 24, HERO_W - 24, H - 36
        px = lambda s: x0 + (x1 - x0) * (min(max(s, t0), t1) - t0) / (t1 - t0)
        for a, b, source in self.segments:
            if b > t0 and a < t1:
                colour = DRIVER.get(source, DRIVER['NONE'])[0]
                d.rectangle([px(a), y, px(b), y + 6], fill=mix(colour, .4))
                if t > a:
                    d.rectangle([px(a), y, px(min(b, t)), y + 6], fill=colour)
        for et, kind, text in self.entries:
            if t0 <= et <= t1 and (kind == 'handoff' or text == 'arrived at goal'):
                d.rectangle([px(et) - 1, y - 5, px(et) + 1, y + 11], fill=TEXT)
        d.ellipse([px(t) - 8, y - 5, px(t) + 8, y + 11], fill=TEXT, outline=BG, width=2)
        d.text((x0, y + 16), clock(t0), font=self.small, fill=MUTED)
        d.text((x1, y + 16), clock(t1), font=self.small, fill=MUTED, anchor='ra')
        # clock + speed, top right of the hero
        d.rounded_rectangle([HERO_W - 204, 24, HERO_W - 24, 68], 22, fill=BG, outline=BORDER)
        d.text((HERO_W - 188, 46), clock(t), font=self.clock_font, fill=TEXT, anchor='lm')
        d.rounded_rectangle([HERO_W - 92, 32, HERO_W - 32, 60], 14, fill=SURFACE)
        d.text((HERO_W - 62, 46), f'{speed:g}×', font=self.medium, fill=MUTED, anchor='mm')

    def feed(self, n):
        """The trace with entries[:n] visible, newest at the bottom, older ones dimmed."""
        from PIL import Image, ImageDraw
        im = Image.new('RGB', (SIDE, H - TRACE_Y), BG)
        d = ImageDraw.Draw(im)
        x, right, y = 72, SIDE - 16, im.height - 16
        for age, (t, kind, text) in enumerate(reversed(self.entries[:n])):
            f = 1.0 if age == 0 else .8 if age < 3 else .55
            ink, dim = mix(TEXT, f), mix(MUTED, f)
            if kind == 'tool':
                name, _, args = text.partition('\t')
                lines = wrap(d, f'{name} {args}', self.mono, right - x - 20, 2)
                h = 10 + 20 * len(lines)
                y -= h
                d.rounded_rectangle([x, y, right, y + h], 6, fill=mix(SURFACE, f), outline=mix(BORDER, f))
                for k, line in enumerate(lines):
                    d.text((x + 10, y + 6 + 20 * k), line, font=self.mono, fill=dim)
                d.text((x + 10, y + 6), name, font=self.mono, fill=ink)
            elif kind in ('event', 'warn', 'handoff'):
                h = 30
                y -= h
                colour = AMBER if kind == 'warn' else DRIVER['AGENT'][0] if text.startswith('looks') else next((c for c, label in DRIVER.values() if label == text), DRIVER['NAV_MODEL'][0])
                label = text if kind != 'handoff' else 'rover stopped' if text == 'STOPPED' else f'hand-off to {text}'
                tw = d.textlength(label, font=self.small)
                d.rounded_rectangle([x, y + 2, x + tw + 34, y + h - 2], 13, fill=mix(colour, .16 * f), outline=mix(colour, .6 * f))
                d.ellipse([x + 10, y + 11, x + 18, y + 19], fill=mix(colour, f))
                d.text((x + 26, y + h // 2), label, font=self.small, fill=mix(colour, f), anchor='lm')
            else:
                box = kind in ('to_operator', 'operator')
                font, lh = (self.italic, 24) if kind == 'think' else (self.body, 26)
                lines = wrap(d, text, font, right - x - (24 if box else 0), 10 if kind == 'operator' else 6)
                h = lh * len(lines) + (40 if box else 0)
                y -= h
                if box:
                    colour = DRIVER['AGENT'][0] if kind == 'to_operator' else MUTED
                    d.rounded_rectangle([x, y, right, y + h], 10, fill=mix(colour, .14 * f), outline=mix(colour, .5 * f))
                    d.text((x + 12, y + 10), 'AGENT TO OPERATOR' if kind == 'to_operator' else 'OPERATOR', font=self.caps, fill=mix(colour, f))
                else:
                    d.rectangle([x - 12, y + 4, x - 10, y + h - 4], fill=mix(DRIVER['AGENT'][0], (.45 if kind == 'think' else 1) * f))
                for k, line in enumerate(lines):
                    d.text((x + (12 if box else 0), y + (30 if box else 0) + k * lh), line, font=font, fill=dim if kind == 'think' else ink)
            d.text((16, y + 4), clock(t), font=self.small, fill=dim)
            y -= 14
            if y < 0:
                break
        fade = np.asarray(im).astype(np.float32)
        ramp = np.linspace(0, 1, 48)[:, None, None]
        fade[:48] = np.asarray(BG) + (fade[:48] - BG) * ramp
        return Image.fromarray(fade.astype(np.uint8))


def export_scene(model, data, robot_root, out):
    """What scripts/blender_hero.py needs to redraw the hero view: every drawn geom (mesh, UVs,
    colour or texture, pose in its body), which bodies move, the sun, and the textures as PNGs."""
    import mujoco
    from PIL import Image
    out.mkdir(parents=True, exist_ok=True)
    mujoco.mj_forward(model, data)
    arrays, geoms, textures = {}, [], {}
    for g in range(model.ngeom):
        if model.geom_group[g] >= 3 or model.geom_rgba[g, 3] == 0:
            continue
        body, mat, kind = int(model.geom_bodyid[g]), int(model.geom_matid[g]), int(model.geom_type[g])
        entry = dict(name=model.geom(g).name, kind=kind, body=body, size=model.geom_size[g].tolist(),
                     pos=model.geom_pos[g].tolist(), quat=model.geom_quat[g].tolist(), rgba=model.geom_rgba[g].tolist())
        if mat >= 0:
            tex = int(model.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB])
            entry.update(material=model.material(mat).name, rgba=model.mat_rgba[mat].tolist(), shininess=float(model.mat_shininess[mat]),
                         repeat=model.mat_texrepeat[mat].tolist())
            if tex >= 0:
                if tex not in textures:
                    h, w, c = int(model.tex_height[tex]), int(model.tex_width[tex]), int(model.tex_nchannel[tex])
                    textures[tex] = f'texture_{tex}.png'
                    Image.fromarray(model.tex_data[model.tex_adr[tex]:model.tex_adr[tex] + h * w * c].reshape(h, w, c).squeeze()).save(out / textures[tex])
                entry['texture'] = textures[tex]
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            m = int(model.geom_dataid[g])
            entry['mesh'] = m
            if f'v{m}' not in arrays:
                v, f = slice(model.mesh_vertadr[m], model.mesh_vertadr[m] + model.mesh_vertnum[m]), slice(model.mesh_faceadr[m], model.mesh_faceadr[m] + model.mesh_facenum[m])
                arrays[f'v{m}'], arrays[f'f{m}'] = model.mesh_vert[v], model.mesh_face[f]
                if model.mesh_texcoordnum[m] > 0:
                    uv = model.mesh_texcoord[model.mesh_texcoordadr[m]:model.mesh_texcoordadr[m] + model.mesh_texcoordnum[m]].copy()
                    uv[:, 1] = 1 - uv[:, 1]  # MuJoCo's compiled V runs the other way
                    arrays[f'uv{m}'], arrays[f'fuv{m}'] = uv, model.mesh_facetexcoord[f]
        geoms.append(entry)
    bodies = [dict(name=model.body(b).name, moving=bool(model.body_rootid[b] == robot_root), pos=data.xpos[b].tolist(), quat=data.xquat[b].tolist())
              for b in range(model.nbody)]
    (out / 'scene.json').write_text(json.dumps(dict(geoms=geoms, bodies=bodies, sun=model.light_dir[0].tolist(), fovy=float(model.vis.global_.fovy),
                                                    size=[HERO_W, H], nav_fovy=float(model.cam_fovy[model.camera('element/nav_front').id]), nav_size=[640, 480])))
    np.savez(out / 'arrays.npz', **arrays)


def predictions(run):
    """Sim time -> nav_model's front-camera prediction, from the worker log."""
    times, preds = [], []
    path = run / 'rhizome/navigation/messages.jsonl'
    if path.exists():
        with open(path) as f:
            for line in f:
                if '"op": "nav_infer"' in line[:80]:
                    m = json.loads(line)
                    times.append(m['simulation_time'])
                    p = m['output']['prediction']
                    preds.append({**p.get('front', {}), 'steer': p.get('steer_source', 'ROWS') if p.get('gate', 'NONE') == 'NONE' else p['gate'].replace('_', ' ')})
    return times, preds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--out', type=Path)
    ap.add_argument('--start', type=float, default=0)
    ap.add_argument('--end', type=float)
    ap.add_argument('--fast', type=float, default=8)
    ap.add_argument('--slow', type=float, default=2)
    ap.add_argument('--scene', type=Path, help='render the run in this scene build instead of the one it ran in (same layout)')
    ap.add_argument('--export-hero', type=Path, help='write the hero view (scene, poses, camera path) here for scripts/blender_hero.py, and stop')
    ap.add_argument('--hero-frames', type=Path, help='a blender_hero.py output: its frames/ become the hero view and its nav/, if present, the robot view')
    ap.add_argument('--rhizome', default=os.environ.get('RHIZOME_ROOT'), help='rhizome checkout, for the production robot look on older runs')
    args = ap.parse_args()
    run = args.run.resolve()
    out = args.out or run / 'replay.mp4'
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import h5py
    import mujoco
    from PIL import Image, ImageDraw

    events = [json.loads(l) for l in open(run / 'events.jsonl')]
    scene = Path(json.loads((run / 'run.json').read_text())['scene'])
    bounds = np.asarray(json.loads((scene / 'manifest.json').read_text())['terrain']['bounds_m'], float)
    with h5py.File(run / 'data.h5', 'r') as f:
        time, qpos = f['state']['time'][:], f['state']['qpos'][:]
    end = float(time[-1])
    t0, t1 = args.start, min(args.end or end, end)
    entries, segments = trace(events), drivers(events, end)
    entry_t = [e[0] for e in entries]
    frames = schedule(t0, t1, [e[0] for e in entries if e[1] in SLOW_KINDS], args.fast, args.slow)
    gemini = [(e['t'], eval(e['text'].rpartition(' -> ')[2], {})) for e in events if e['kind'] == 'gemini']  # our own log's repr
    looks = [e['t'] for e in events if e['kind'] == 'tool' and e['text'].startswith('grab_frame')]
    pred_t, preds = predictions(run)

    # The robot's view matches the run (single-sampled, 1024 shadows); the hero gets the scene's full quality.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    if args.scene:  # the run's poses in another build of its scene, composed as run_agent_nav does
        from scene_pipeline.agriculture import compose_robot
        from scene_pipeline.navigation import add_cameras
        from sim_harness.scene import _read_spec
        scene, meta = args.scene.resolve(), json.loads((run / 'run.json').read_text())
        robot = _read_spec(scene / 'robot.mjz')
        profile = add_cameras(robot, json.loads((scene / 'robot_profile.json').read_text()), dict(cameras=['front', 'rear']))
        spec = compose_robot(_read_spec(scene / 'environment.mjz'), robot, profile, meta['spawn'], bounds)[0]
    else:
        spec = mujoco.MjSpec.from_zip(str(run / 'rollout_scene.mjz'))
    if args.rhizome and not any(m.name == 'element/robot_frame' for m in spec.materials):
        from scene_pipeline.robot_skin import reskin
        reskin(spec, args.rhizome, 'element/')  # runs recorded before the production skin
    hero_model = spec.compile()
    nav_model = copy.copy(hero_model)
    hero_model.vis.global_.offwidth, hero_model.vis.global_.offheight = HERO_W, H
    nav_model.vis.quality.offsamples, nav_model.vis.quality.shadowsize = 0, 1024
    hero_data, nav_data = mujoco.MjData(hero_model), mujoco.MjData(nav_model)
    hero, nav = mujoco.Renderer(hero_model, H, HERO_W), mujoco.Renderer(nav_model, 480, 640)
    chassis = hero_model.body('element/chassis').id
    robot_root = hero_model.body_rootid[chassis]
    nav_cam = hero_model.camera('element/nav_front').id
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(hero_model, cam)
    cam.distance, cam.elevation = 8.0, -30
    opt = mujoco.MjvOption()
    opt.geomgroup[3:] = 0

    hud = Hud(entries, segments, t1)  # the timeline ends where the video does
    static = hud.static()
    # Minimap: the scene's ground texture (north up), path drawn as it is driven.
    MW = 220
    MH = round(MW * np.ptp(bounds[:, 1]) / np.ptp(bounds[:, 0]))
    ground = scene / 'ground.png'
    minimap = (Image.open(ground).convert('RGB').resize((MW, MH)) if ground.exists() else Image.new('RGB', (MW, MH), SURFACE))
    minimap = Image.blend(minimap, Image.new('RGB', (MW, MH), BG), .2)
    to_map = lambda xy: ((xy[0] - bounds[0, 0]) / np.ptp(bounds[:, 0]) * MW, (bounds[1, 1] - xy[1]) / np.ptp(bounds[:, 1]) * MH)
    path_draw, last = ImageDraw.Draw(minimap), None
    for i in range(0, int(np.searchsorted(time, t0))):  # path before a clip's start
        p = to_map(qpos[i, :2])
        if last:
            path_draw.line([last, p], fill=DRIVER[next(s for a, b, s in segments if a <= time[i] <= b)][0], width=2)
        last = p

    track = []  # per frame: moving bodies' poses, camera eye/forward/up, bodies ghosted
    if args.export_hero:
        export_scene(hero_model, hero_data, robot_root, args.export_hero)
    ff = None
    if out.suffix != '.png' and not args.export_hero:
        ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS),
                               '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '20', str(out)], stdin=subprocess.PIPE)
    yaw_s = None
    for n, t in enumerate(frames):
        i = min(int(np.searchsorted(time, t, side='right')) - 1, len(time) - 2)
        a = (t - time[i]) / (time[i + 1] - time[i])
        q = qpos[i] + (qpos[i + 1] - qpos[i]) * min(max(a, 0), 1)
        for model, data in ((hero_model, hero_data), (nav_model, nav_data)):
            data.qpos[:] = q
            mujoco.mj_normalizeQuat(model, data.qpos)
            mujoco.mj_forward(model, data)
        fwd = hero_data.xmat[chassis].reshape(3, 3)[:, 0]
        yaw = math.degrees(math.atan2(fwd[1], fwd[0]))
        speed = round((frames[n + 1] - t if n + 1 < len(frames) else frames[n] - frames[n - 1]) * FPS, 1) if len(frames) > 1 else args.fast
        # Low-pass the chase heading per sim second, so pivots don't whip the camera and speed changes don't show.
        yaw_s = yaw if yaw_s is None else yaw_s + ((yaw - yaw_s + 180) % 360 - 180) * (1 - math.exp(-speed / FPS / 1.5))
        cam.lookat[:] = hero_data.xpos[chassis]
        cam.azimuth = yaw_s
        hero.update_scene(hero_data, cam, opt)
        # Ghost scenery (a tree, the tractor) that stands between the camera and the rover.
        eye = np.mean([c.pos for c in hero.scene.camera], axis=0)
        ray = cam.lookat - eye
        scenery = [(g, hero_model.geom_bodyid[g.objid]) for g in hero.scene.geoms[:hero.scene.ngeom] if g.objtype == mujoco.mjtObj.mjOBJ_GEOM]
        blocking = set()
        for g, body in scenery:
            bound = hero_model.geom_rbound[g.objid]
            nearest = eye + ray * np.clip((g.pos - eye) @ ray / (ray @ ray), 0, 1)
            if body and hero_model.body_rootid[body] != robot_root and 0 < bound < 8 and np.linalg.norm(g.pos - nearest) < bound:
                blocking.add(body)
        for g, body in scenery:
            if body in blocking:
                g.rgba[3] = .12
        if args.export_hero:
            view = hero.scene.camera[0]
            track.append((np.hstack([hero_data.xpos, hero_data.xquat]), np.concatenate([eye, view.forward, view.up]), sorted(int(b) for b in blocking),
                          np.concatenate([hero_data.cam_xpos[nav_cam], hero_data.cam_xmat[nav_cam]])))
            continue
        frame = Image.new('RGB', (W, H), BG)
        frame.paste(Image.open(args.hero_frames / 'frames' / f'{n:06d}.jpg') if args.hero_frames else Image.fromarray(hero.render()), (0, 0))
        frame.paste(static, (0, 0), static)
        d = ImageDraw.Draw(frame)

        source = next((s for a_, b_, s in segments if a_ <= t <= b_), 'NONE')
        since = t - max((a_ for a_, b_, s in segments if a_ <= t), default=0)
        toast = round(since, 1) if 0 < since < 3 and t > 3 else None
        badge = hud.cached(('driver', source, toast), lambda: hud.driver(source, toast))
        frame.paste(badge, (0, 0), badge)
        hud.timeline(d, t, speed)

        p = to_map(hero_data.xpos[chassis, :2])
        if last:
            path_draw.line([last, p], fill=DRIVER.get(source, DRIVER['NONE'])[0], width=2)
        last = p
        mx, my = HERO_W - MW - 24, H - MH - 76
        frame.paste(minimap, (mx, my))
        d.rectangle([mx - 1, my - 1, mx + MW, my + MH], outline=BORDER)
        r = math.radians(yaw)
        tri = [(mx + p[0] + 9 * math.cos(r + k), my + p[1] - 9 * math.sin(r + k)) for k in (0, 2.5, -2.5)]
        d.polygon(tri, fill=TEXT, outline=BG)

        # Robot view
        if args.hero_frames and (args.hero_frames / 'nav').is_dir():  # the storytelling cut: Blender's picture of the robot's view
            view = Image.open(args.hero_frames / 'nav' / f'{n:06d}.jpg').resize((NAV_W, NAV_H), Image.LANCZOS)
        else:
            nav.update_scene(nav_data, 'element/nav_front', opt)
            view = Image.fromarray(nav.render()).resize((NAV_W, NAV_H), Image.LANCZOS)
        vd, s = ImageDraw.Draw(view), NAV_W / 640
        k = bisect.bisect_right(pred_t, t) - 1
        steer = preds[k]['steer'] if k >= 0 else 'ROWS'
        if source == 'NAV_MODEL' and k >= 0 and steer == 'ROWS':
            for line, prob in zip(preds[k].get('rows_image', []), preds[k].get('row_vis_prob', [])):
                if prob >= .5:
                    vd.line([line['x1'] * s, line['y1'] * s, line['x2'] * s, line['y2'] * s], fill=DRIVER['NAV_MODEL'][0], width=3)
        for gt, points in gemini:  # ponytail: assumes Gemini looked through the front camera; read the tool's camera arg if rear goals appear
            if 0 <= t - gt < 4:
                for pt in points:
                    u, v = pt['point'][1] / 1000 * NAV_W, pt['point'][0] / 1000 * NAV_H
                    vd.ellipse([u - 6, v - 6, u + 6, v + 6], fill=DRIVER['AGENT'][0], outline=BG, width=2)
        nx, ny = HERO_W + 16, 48
        frame.paste(view, (nx, ny))
        looking = any(0 <= t - lt < 2 for lt in looks) or any(0 <= t - gt < 4 for gt, _ in gemini)
        d.rectangle([nx - 1, ny - 1, nx + NAV_W, ny + NAV_H], outline=DRIVER['AGENT'][0] if looking else BORDER, width=3 if looking else 1)
        if source == 'NAV_MODEL' or looking:
            colour, label = (DRIVER['AGENT'][0], 'AGENT LOOKING') if looking else (DRIVER['NAV_MODEL'][0], f'NAV MODEL · {steer}')
            tw = d.textlength(label, font=hud.caps)
            d.rounded_rectangle([nx + 10, ny + 10, nx + 30 + tw, ny + 36], 13, fill=BG)
            d.text((nx + 20, ny + 23), label, font=hud.caps, fill=colour, anchor='lm')

        shown = bisect.bisect_right(entry_t, t)
        frame.paste(hud.cached(('feed', shown), lambda: hud.feed(shown)), (HERO_W + 1, TRACE_Y))
        if ff is None:
            frame.save(out)
            break
        ff.stdin.write(frame.tobytes())
    if args.export_hero:
        np.savez(args.export_hero / 'track.npz', poses=np.array([p for p, _, _, _ in track]), camera=np.array([c for _, c, _, _ in track]),
                 ghosted=np.array([json.dumps(g) for _, _, g, _ in track]), nav=np.array([v for _, _, _, v in track]))
        print(args.export_hero, len(track), 'frames')
        return
    if ff:
        ff.stdin.write(frame.tobytes() * (2 * FPS))  # hold the last frame
        ff.stdin.close()
        ff.wait()
    print(out, f'{len(frames) / FPS:.0f} s')


if __name__ == '__main__':
    main()
