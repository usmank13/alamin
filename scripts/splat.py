#!/usr/bin/env python3
"""Gaussian splats from agent captures (scripts/run_agent_nav.py captures/<name>/).

Runs in .venv-splat (nerfstudio 1.1.5 + gsplat + mujoco):
  splat.py prep  CAPTURE                     # seed points.ply from the depth frames
  splat.py train CAPTURE [--iters 15000]     # ns-train splatfacto, poses kept in the sim world frame
  splat.py eval  CAPTURE --scene SCENE --center X Y [--radius R]
      # held-out ground-truth views from MuJoCo on rings around (X, Y) at the
      # capture camera height and higher; PSNR/SSIM + a contact sheet

Poses are not recentred or rescaled (`--orientation-method none` etc.), so a
pose in the scene is the same pose in the splat and eval needs no transform.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def prep(cap: Path, max_points=200_000):
    """Back-project the (range-clipped) depth frames into a coloured cloud."""
    from PIL import Image
    meta = json.loads((cap / 'transforms.json').read_text())
    fx, fy, cx, cy = meta['fl_x'], meta['fl_y'], meta['cx'], meta['cy']
    v, u = np.mgrid[0:meta['h'], 0:meta['w']]
    pts, cols = [], []
    for f in meta['frames']:
        d = np.asarray(Image.open(cap / f['depth_file_path']), np.float32) * meta['depth_unit_scale_factor']
        ok = d > 0
        # OpenGL camera: x right, y up, looking -z.
        xyz = np.stack([(u[ok] - cx) / fx * d[ok], -(v[ok] - cy) / fy * d[ok], -d[ok]], 1)
        c2w = np.asarray(f['transform_matrix'])
        pts.append(xyz @ c2w[:3, :3].T + c2w[:3, 3])
        cols.append(np.asarray(Image.open(cap / f['file_path']).convert('RGB'))[ok])
    pts, cols = np.concatenate(pts), np.concatenate(cols)
    keep = np.random.default_rng(0).choice(len(pts), min(max_points, len(pts)), replace=False)
    pts, cols = pts[keep], cols[keep]
    with open(cap / 'points.ply', 'w') as fh:
        fh.write(f'ply\nformat ascii 1.0\nelement vertex {len(pts)}\nproperty float x\nproperty float y\n'
                 'property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
        for p, c in zip(pts, cols):
            fh.write(f'{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n')
    meta['ply_file_path'] = 'points.ply'
    (cap / 'transforms.json').write_text(json.dumps(meta, indent=1))
    print(f'{len(pts)} seed points from {len(meta["frames"])} frames')


def train(cap: Path, iters: int):
    if not (cap / 'points.ply').exists():
        prep(cap)
    cmd = [str(Path(sys.executable).with_name('ns-train')), 'splatfacto', '--data', str(cap),
           '--output-dir', str(cap / 'train'), '--max-num-iterations', str(iters),
           '--viewer.quit-on-train-completion', 'True', '--vis', 'tensorboard',
           '--pipeline.model.camera-optimizer.mode', 'off',  # sim poses are exact
           'nerfstudio-data', '--orientation-method', 'none', '--center-method', 'none',
           '--auto-scale-poses', 'False', '--eval-mode', 'interval', '--eval-interval', '8']
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)
    config = sorted((cap / 'train').rglob('config.yml'))[-1]
    print('config:', config)


def look_at(eye, target):
    """OpenGL camera-to-world: x right, y up, looking -z."""
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    up = np.cross(r, f)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([r, up, -f], 1)
    c2w[:3, 3] = eye
    return c2w


def eval_(cap: Path, scene: Path, center, radius: float, n=12):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import mujoco
    import torch
    from PIL import Image
    from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    from nerfstudio.cameras.cameras import Cameras, CameraType
    from nerfstudio.utils.eval_utils import eval_setup

    meta = json.loads((cap / 'transforms.json').read_text())
    W, H, fx, fy = meta['w'], meta['h'], meta['fl_x'], meta['fl_y']
    cam_h = float(np.median([f['transform_matrix'][2][3] for f in meta['frames']]))
    spec = mujoco.MjSpec.from_zip(str(scene / 'environment.mjz'))
    spec.worldbody.add_camera(name='eval', fovy=math.degrees(2 * math.atan(H / 2 / fy)))
    model = spec.compile()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    cid = model.camera('eval').id

    def gt(c2w):
        # MuJoCo camera frame matches OpenGL; cam_quat is world-from-camera (w, x, y, z).
        model.cam_pos[cid] = c2w[:3, 3]
        q = np.empty(4)
        mujoco.mju_mat2Quat(q, c2w[:3, :3].flatten())
        model.cam_quat[cid] = q
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, 'eval', scene_option=option)
        return renderer.render()

    config = sorted((cap / 'train').rglob('config.yml'))[-1]
    _, pipeline, _, _ = eval_setup(config, test_mode='inference')
    psnr, ssim = PeakSignalNoiseRatio(data_range=1.0).cuda(), StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    target = [center[0], center[1], 1.0]
    rows, report = [], {}
    for label, height in (('capture height', cam_h), ('raised 1.6 m', 1.6)):
        scores = []
        row = []
        for k in range(n):
            a = 2 * math.pi * k / n
            c2w = look_at(np.array([center[0] + radius * math.cos(a), center[1] + radius * math.sin(a), height]), target)
            truth = torch.tensor(gt(c2w) / 255.0, dtype=torch.float32, device='cuda')
            camera = Cameras(camera_to_worlds=torch.tensor(c2w[None, :3], dtype=torch.float32), fx=fx, fy=fy,
                             cx=W / 2, cy=H / 2, width=W, height=H, camera_type=CameraType.PERSPECTIVE).to('cuda')
            with torch.no_grad():
                pred = pipeline.model.get_outputs_for_camera(camera)['rgb'].clamp(0, 1)
            a_, b_ = pred.permute(2, 0, 1)[None], truth.permute(2, 0, 1)[None]
            scores.append((float(psnr(a_, b_)), float(ssim(a_, b_))))
            if k % 3 == 0:
                row.append(np.concatenate([(truth.cpu().numpy() * 255).astype(np.uint8),
                                           (pred.cpu().numpy() * 255).astype(np.uint8)], 0))
        report[label] = dict(height_m=round(height, 2), psnr=round(float(np.mean([s[0] for s in scores])), 2),
                             ssim=round(float(np.mean([s[1] for s in scores])), 3), views=n)
        rows.append(np.concatenate(row, 1))
    sheet = Image.fromarray(np.concatenate(rows, 0)).resize((len(row) * W // 2, 2 * 2 * H // 2))
    sheet.save(cap / 'eval_sheet.jpg', quality=90)
    report.update(radius_m=radius, center=list(center), frames=len(meta['frames']),
                  sheet='eval_sheet.jpg (per height: top row ground truth, bottom row splat)')
    (cap / 'eval.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('prep').add_argument('capture', type=Path)
    t = sub.add_parser('train')
    t.add_argument('capture', type=Path)
    t.add_argument('--iters', type=int, default=15000)
    e = sub.add_parser('eval')
    e.add_argument('capture', type=Path)
    e.add_argument('--scene', type=Path, required=True)
    e.add_argument('--center', type=float, nargs=2, required=True)
    e.add_argument('--radius', type=float, default=5.0, help='eval ring radius around the centre')
    a = ap.parse_args()
    cap = a.capture.resolve()
    if a.cmd == 'prep':
        prep(cap)
    elif a.cmd == 'train':
        train(cap, a.iters)
    else:
        eval_(cap, a.scene.resolve(), a.center, a.radius)


if __name__ == '__main__':
    main()
