"""Simulator renders, not generative illustrations. Shared by retrieved/grounded pilots."""
import html
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
import numpy as np
from PIL import Image, ImageDraw


def bounds(model, data, visual_only=True):
    points = []
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        if visual_only and (model.geom_contype[i] or model.geom_conaffinity[i]):
            continue
        if visual_only and model.geom_rgba[i, 3] == 0:
            continue  # Transparent semantic region boxes are not visible solids.
        rotation = data.geom_xmat[i].reshape(3, 3)
        center = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
        radius = abs(rotation) @ model.geom_aabb[i, 3:]
        points.extend([center-radius, center+radius])
    if not points and visual_only:
        return bounds(model, data, False)
    return np.min(points, axis=0), np.max(points, axis=0)


def render_case(model, output, title, distance=None):
    """Scalar-joint fixtures only; poses are kinematic, not successful manipulation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    low, high = bounds(model, data)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, cam)
    cam.lookat[:] = (low+high)/2
    cam.distance = distance or float(np.linalg.norm(high-low)*1.65)
    if distance is None:
        # Fit both endpoints, so a tall opened lid is not cropped.
        for j in range(model.njnt):
            data.qpos[model.jnt_qposadr[j]] = model.jnt_range[j, np.argmax(abs(model.jnt_range[j]))] if model.jnt_limited[j] else np.pi
        mujoco.mj_forward(model, data)
        open_low, open_high = bounds(model, data)
        view_low, view_high = np.minimum(low, open_low), np.maximum(high, open_high)
        cam.lookat[:] = (view_low+view_high)/2
        cam.distance = float(np.linalg.norm(view_high-view_low)*1.65)
    cam.azimuth, cam.elevation = 65, -22
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    # Use explicit contact roles, not dataset-dependent visual/collision groups.
    original_rgba = model.geom_rgba.copy()
    original_mat = model.geom_matid.copy()
    collision = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    # Some fixture sources omit lighting because the environment supplies it.
    model.vis.headlight.active = 1
    model.vis.headlight.ambient[:] = .45
    model.vis.headlight.diffuse[:] = .7
    model.vis.quality.shadowsize = 1024
    model.vis.global_.offwidth = max(640, model.vis.global_.offwidth)
    model.vis.global_.offheight = max(480, model.vis.global_.offheight)
    images = []
    try:
        with mujoco.Renderer(model, height=480, width=640) as renderer:
            for fraction, overlay, caption in [(0, False, 'Closed / source zero'), (1, False, 'Open / kinematic'), (1, True, 'Collision geometry')]:
                mujoco.mj_resetData(model, data)
                for j in range(model.njnt):
                    end = model.jnt_range[j, np.argmax(abs(model.jnt_range[j]))] if model.jnt_limited[j] else np.pi
                    data.qpos[model.jnt_qposadr[j]] = fraction*end
                mujoco.mj_forward(model, data)
                model.geom_rgba[:] = original_rgba
                model.geom_matid[:] = original_mat
                if overlay:
                    model.geom_rgba[~collision, 3] = 0
                    model.geom_matid[collision] = -1
                    model.geom_rgba[collision] = [.9, .35, .15, 1]
                else:
                    model.geom_rgba[collision, 3] = 0
                renderer.update_scene(data, cam, scene_option=opt)
                im = Image.fromarray(renderer.render())
                ImageDraw.Draw(im).text((12, 12), caption, fill='white', stroke_width=1, stroke_fill='black')
                images.append(im)
            model.geom_rgba[:] = original_rgba
            model.geom_rgba[collision, 3] = 0
            model.geom_matid[:] = original_mat
            frames = []
            for fraction in np.r_[np.linspace(0, 1, 12), np.linspace(1, 0, 12)]:
                mujoco.mj_resetData(model, data)
                for j in range(model.njnt):
                    end = model.jnt_range[j, np.argmax(abs(model.jnt_range[j]))] if model.jnt_limited[j] else np.pi
                    data.qpos[model.jnt_qposadr[j]] = fraction*end
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, cam, scene_option=opt)
                frames.append(Image.fromarray(renderer.render()).resize((480, 360)))
            frames[0].save(output/'sweep.gif', save_all=True, append_images=frames[1:], duration=90, loop=0)
    finally:
        model.geom_rgba[:] = original_rgba
        model.geom_matid[:] = original_mat
    panel = Image.new('RGB', (1920, 520), '#20252b')
    for index, im in enumerate(images):
        panel.paste(im, (640*index, 40))
    ImageDraw.Draw(panel).text((12, 12), title, fill='white')
    panel.save(output/'comparison.png')
    return dict(closed_bounds_m=[low.tolist(), high.tolist()], camera_distance_m=cam.distance)


def gallery(root, rows):
    root = Path(root)
    chunks = ['<!doctype html><meta charset="utf-8"><title>Grounded / retrieved pilots</title>',
              '<style>body{background:#20252b;color:#eee;font:16px sans-serif;margin:30px}img{max-width:100%}section{margin:40px 0}a{color:#9cf}pre{white-space:pre-wrap}</style>',
              '<h1>Grounded / retrieved pilots</h1><p>Actual MuJoCo renders. Open poses and GIFs are kinematic sweeps, not robot manipulation. Camera distance is shared across scale variants of each source.</p>',
              '<p>Retrieved assets: RoboCasa Team / Lightwheel, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Modifications: reported uniform scale only. Grounded P4 objects are procedural proxies, not manufacturer CAD.</p>']
    for row in rows:
        case = row['case']
        chunks.append(f'<section><h2>{html.escape(case)}</h2><a href="{case}/sweep.gif">Joint animation</a> | <a href="{case}/scene.mjz">Portable MuJoCo asset</a> | <a href="{case}/result.json">Evidence / metrics</a><br><img src="{case}/comparison.png"><pre>{html.escape(str(row.get("summary", "")))}</pre></section>')
    (root/'index.html').write_text('\n'.join(chunks))
