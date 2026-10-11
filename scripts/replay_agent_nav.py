#!/usr/bin/env python3
"""Render a scripts/run_agent_nav.py run to MP4.

usage: .venv/bin/python scripts/replay_agent_nav.py RUN_DIR [--speed 2] [--out RUN_DIR/replay.mp4]

Left: chase camera behind the rover. Right: the last image the agent looked at
(with Gemini's picked points when it asked for them). Bottom: what the agent was
doing (tool calls, events, its messages to the operator), in simulation time,
and a top-down map (path so far, heading, scene objects, target in red).
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np

REPO = Path(__file__).resolve().parents[1]
W, H, CAPTION = 640, 480, 150


def captions(run):
    """(t, line) for what the operator would want to read, in sim time."""
    out = []
    for line in (run / 'events.jsonl').read_text().splitlines():
        d = json.loads(line)
        kind, text = d['kind'], d.get('text') or ''
        if kind == 'tool':
            name, _, rest = text.partition('(')
            if name in ('aigen_cli', 'memory_read', 'grab_frame'):
                continue
            out.append((d['t'], f'> {name}({rest[:-1][:90]})'))
        elif kind in ('arrived', 'approaching', 'stuck', 'obstacle', 'target_reached', 'cancel_nav'):
            if kind == 'obstacle':
                text = 'obstacle stop ' + ('ON' if d.get('front') or d.get('rear') else 'cleared')
            out.append((d['t'], f'[{kind}] {text}'[:110]))
        elif kind in ('agent_says', 'operator'):
            who = 'agent' if kind == 'agent_says' else 'operator'
            out.append((d['t'], f'{who}: ' + ' '.join(text.split())[:200]))
        elif kind == 'gemini':
            out.append((d['t'], 'gemini: ' + text[:110]))
    return out


def agent_images(run):
    """(t, path) of each image the agent grabbed, Gemini overlays replacing their frame."""
    walls = [(json.loads(l)['wall'], json.loads(l)['t']) for l in (run / 'events.jsonl').read_text().splitlines()]
    walls = np.asarray(walls)
    out = []
    for p in sorted((run / 'frames').glob('*.jpg')):
        out.append((float(np.interp(p.stat().st_mtime, walls[:, 0], walls[:, 1])), p))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', type=Path)
    ap.add_argument('--speed', type=float, default=2.0, help='video seconds per sim second = 1/speed')
    ap.add_argument('--out', type=Path)
    args = ap.parse_args()
    run = args.run.resolve()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / 'src'))
    import mujoco
    from PIL import Image, ImageDraw, ImageFont
    from scipy.spatial.transform import Rotation
    from sim_harness.scene import _read_spec
    from scene_pipeline.agriculture import compose_robot
    from scene_pipeline.contracts import read_json
    from scene_pipeline.navigation import add_cameras

    meta = json.loads((run / 'run.json').read_text())
    scene = Path(meta['scene'])
    profile = read_json(scene / 'robot_profile.json')
    manifest = read_json(scene / 'manifest.json')
    robot = _read_spec(scene / 'robot.mjz')
    profile = add_cameras(robot, profile, dict(cameras=['front', 'rear']))  # same model as the run
    _, model, data = compose_robot(_read_spec(scene / 'environment.mjz'), robot, profile, meta['spawn'],
                                   np.asarray(manifest['terrain']['bounds_m']))
    rec = np.load(run / 'replay.npz')
    chassis = model.body('element/chassis').id
    renderer = mujoco.Renderer(model, height=H, width=W)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation = 7.0, -28.0
    lines, images = captions(run), agent_images(run)
    font = ImageFont.load_default()
    out = args.out or run / 'replay.mp4'
    fps = 10 * args.speed
    ff = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                           '-s', f'{2 * W}x{H + CAPTION}', '-r', str(fps), '-i', '-', '-c:v', 'libx264',
                           '-pix_fmt', 'yuv420p', '-crf', '23', str(out)], stdin=subprocess.PIPE)
    yaw_smooth = None
    bounds = np.asarray(manifest['terrain']['bounds_m'], float)
    objects = {k: v['position_m'][:2] for k, v in manifest.get('instances', {}).items()}
    M = CAPTION - 8  # map side, pixels
    to_map = lambda xy: (2 * W - M - 4 + (xy[0] - bounds[0, 0]) / np.ptp(bounds[:, 0]) * M,
                         H + 4 + (bounds[1, 1] - xy[1]) / np.ptp(bounds[:, 1]) * M)
    path = []
    for t, qpos in zip(rec['t'], rec['qpos']):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        q = data.xquat[chassis]
        yaw = Rotation.from_quat(q[[1, 2, 3, 0]]).as_euler('xyz')[2]
        # Low-pass the chase heading so pivots don't whip the camera around.
        yaw_smooth = yaw if yaw_smooth is None else yaw_smooth + 0.15 * math.atan2(
            math.sin(yaw - yaw_smooth), math.cos(yaw - yaw_smooth))
        cam.lookat[:] = data.xpos[chassis]
        cam.azimuth = math.degrees(yaw_smooth)
        renderer.update_scene(data, cam, scene_option=option)
        frame = Image.new('RGB', (2 * W, H + CAPTION), (20, 20, 24))
        frame.paste(Image.fromarray(renderer.render()), (0, 0))
        seen = [p for ti, p in images if ti <= t]
        if seen:
            frame.paste(Image.open(seen[-1]).convert('RGB').resize((W, H)), (W, 0))
        draw = ImageDraw.Draw(frame)
        draw.rectangle([0, 0, 220, 18], fill=(0, 0, 0))
        draw.text((6, 4), f't = {t:6.1f} s   x{args.speed:g} speed', fill=(255, 255, 255), font=font)
        draw.rectangle([W, 0, W + 230, 18], fill=(0, 0, 0))
        draw.text((W + 6, 4), 'agent view: ' + (seen[-1].stem if seen else 'none yet'), fill=(255, 255, 255), font=font)
        path.append(to_map(data.xpos[chassis, :2]))
        draw.rectangle([2 * W - M - 4, H + 4, 2 * W - 4, H + 4 + M], fill=(45, 45, 50), outline=(90, 90, 90))
        for k, xy in objects.items():
            u, v = to_map(xy)
            red = k == meta.get('target')
            draw.ellipse([u - (4 if red else 2), v - (4 if red else 2), u + (4 if red else 2), v + (4 if red else 2)],
                         fill=(230, 60, 50) if red else (160, 160, 160))
        if len(path) > 1:
            draw.line(path, fill=(80, 160, 255), width=2)
        u, v = path[-1]
        draw.line([u, v, u + 9 * math.cos(yaw), v - 9 * math.sin(yaw)], fill=(255, 220, 60), width=3)
        recent = [(ti, s) for ti, s in lines if ti <= t][-6:]
        y = H + 6
        for ti, s in recent:
            for chunk in textwrap.wrap(f'{ti:6.1f}  {s}', 165)[:2]:
                draw.text((8, y), chunk, fill=(235, 235, 235) if ti > t - 5 else (150, 150, 150), font=font)
                y += 12
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    renderer.close()
    print(out)


if __name__ == '__main__':
    main()
