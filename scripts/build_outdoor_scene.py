#!/usr/bin/env python3
"""Outdoor (non-agriculture) scene dirs for scripts/run_agent_nav.py.

usage: .venv/bin/python scripts/build_outdoor_scene.py OUT construction|urban

Assets live in outputs/outdoor_assets (not in git):
  kenney/   CC0 kits, unzipped from kenney.nl: car-kit, city-kit-commercial_2.1,
            city-kit-industrial_2.0, city-kit-roads
  store/    `pipeline --asset-store outputs/outdoor_assets/store asset-fetch
            gazebo:<name>` for jersey_barrier, construction_cone,
            construction_barrel, euro_pallet, cinder_block, bus_stop,
            stop_sign, lamp_post; gz_packages.txt maps name -> asset.mjz.
Gazebo cones and barrels are ~1.5x real size; they are scaled down on placement.
"""
import io, json, math, shutil, sys, xml.etree.ElementTree as ET
from pathlib import Path
import mujoco, numpy as np, trimesh
from PIL import Image
from trimesh.exchange.obj import export_obj

A = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A / 'src'))
from scene_pipeline.agriculture import compile_environment, compose_robot
from scene_pipeline.navigation import add_cameras
from sim_harness.scene import _read_spec

S = A / 'outputs/outdoor_assets'
K = S / 'kenney'
GZ = {l.split()[0]: str(S / l.split()[1]) for l in (S / 'gz_packages.txt').read_text().splitlines()}  # name -> asset.mjz


def glb_prop(glb, length_m, out):
    """GLB (Y-up, one atlas texture) -> static asset.mjz: textured visual + one convex-hull collider."""
    mesh = trimesh.util.concatenate(trimesh.load(glb, force='scene').dump())
    mesh.apply_transform([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    mesh.apply_scale(length_m / max(mesh.extents[:2]))
    lo, hi = mesh.bounds; mesh.apply_translation([-(lo[0] + hi[0]) / 2, -(lo[1] + hi[1]) / 2, -lo[2]])
    obj, files = export_obj(mesh, include_texture=True, return_texture=True)
    img = next(v for k, v in files.items() if k.endswith('.png'))
    buf = io.BytesIO(); Image.open(io.BytesIO(img)).convert('RGB').save(buf, 'PNG')
    assets = {'m.obj': obj.encode(), 'c.obj': export_obj(mesh.convex_hull).encode(), 'a.png': buf.getvalue()}
    x = ET.Element('mujoco'); a = ET.SubElement(x, 'asset')
    ET.SubElement(a, 'mesh', name='m', file='m.obj'); ET.SubElement(a, 'mesh', name='c', file='c.obj')
    ET.SubElement(a, 'texture', name='t', type='2d', file='a.png'); ET.SubElement(a, 'material', name='s', texture='t')
    b = ET.SubElement(ET.SubElement(x, 'worldbody'), 'body', name='root')
    ET.SubElement(b, 'geom', type='mesh', mesh='m', material='s', group='2', contype='0', conaffinity='0')
    ET.SubElement(b, 'geom', type='mesh', mesh='c', group='3', rgba='0 0 0 0')
    spec = mujoco.MjSpec.from_string(ET.tostring(x, encoding='unicode'), assets=assets)
    out.parent.mkdir(parents=True, exist_ok=True); spec.to_zip(str(out)); return str(out)


def scaled(mjz, s, out):
    """Copy of a static asset.mjz uniformly scaled by s (meshes and offsets)."""
    spec = mujoco.MjSpec.from_zip(mjz)
    for m in spec.meshes:
        m.scale = [v * s for v in m.scale]
    for e in list(spec.bodies) + list(spec.geoms):
        e.pos = [v * s for v in e.pos]
    out.parent.mkdir(parents=True, exist_ok=True); spec.to_zip(str(out)); return str(out)


def yawq(deg): return [math.cos(math.radians(deg) / 2), 0, 0, math.sin(math.radians(deg) / 2)]


def build(out, kind):
    out = Path(out); out.mkdir(parents=True)
    n = 301; xs = np.linspace(-15, 15, n); X, Y = np.meshgrid(xs, xs)
    z = np.zeros((n, n))
    props = []
    def P(name, model, x, y, yaw=0): props.append(dict(name=name, model=model, position_m=[x, y, 0], quaternion_wxyz=yawq(yaw)))
    roads = K / 'kenney_city-kit-roads/Models/GLB format'; cars = K / 'kenney_car-kit/Models/GLB format'
    ind = K / 'kenney_city-kit-industrial_2.0/Models/GLB format'; com = K / 'kenney_city-kit-commercial_2.1/Models/GLB format'
    if kind == 'construction':
        for cx, cy, r, h in [(8, 8, 2.5, 1.5), (-9, 9, 2, 1.0)]:  # dirt piles: heightfield mounds
            z = np.maximum(z, h * np.clip(1 - ((X - cx) ** 2 + (Y - cy) ** 2) / r ** 2, 0, 1) ** 1.5)
        P('container1', glb_prop(ind / 'shipping-container-a.glb', 6.1, out / 'p/c1.mjz'), 10, -10, 0)
        P('container2', glb_prop(ind / 'shipping-container-b.glb', 6.1, out / 'p/c2.mjz'), 10, -6.5, 0)
        P('loader', glb_prop(cars / 'tractor-shovel.glb', 5.0, out / 'p/loader.mjz'), -8, -8, 30)
        P('truck', glb_prop(cars / 'truck.glb', 6.5, out / 'p/truck.mjz'), 2, 11, 90)
        for i, y in enumerate(np.arange(-12, 13, 4.5)):
            P(f'jersey{i}', GZ['jersey_barrier'], -12.5, y, 90)
        cone = scaled(GZ['construction_cone'], 0.65, out / 'p/cone.mjz')  # ~0.7 m, a real road cone
        barrel = scaled(GZ['construction_barrel'], 0.65, out / 'p/barrel.mjz')  # ~1.0 m
        for i, (x, y) in enumerate([(-3, -4), (-1, -4), (1, -4), (3, -4), (4, 3), (6, 3)]):
            P(f'cone{i}', cone, x, y)
        for i, (x, y) in enumerate([(3, -10), (5, -10), (7, -2)]):
            P(f'barrel{i}', barrel, x, y)
        for i, (x, y) in enumerate([(-4, 6), (-4, 7.4), (-2.6, 6)]):
            P(f'pallet{i}', GZ['euro_pallet'], x, y)
        for i, (x, y) in enumerate([(0, 6), (0.5, 6.3), (0.2, 6.6)]):
            P(f'block{i}', GZ['cinder_block'], x, y, 20 * i)
        spawn = dict(position_m=[-6.0, 0.0], yaw_rad=0.0)
    else:
        for i, (f, x, y, yaw) in enumerate([('building-a', -11, 10, 0), ('building-c', -3, 11, 0), ('building-e', 5, 11, 0), ('building-g', 12, 10, 0),
                                            ('building-b', -11, -10, 180), ('building-d', -3, -11, 180), ('building-f', 5, -11, 180), ('building-h', 12, -10, 180)]):
            P(f'bldg{i}', glb_prop(com / f'{f}.glb', 7, out / f'p/b{i}.mjz'), x, y, yaw)
        for i, (f, x, y, yaw) in enumerate([('sedan', -6, 3.5, 90), ('suv', 1, 3.5, 90), ('van', 8, -3.5, -90), ('taxi', -1, -3.5, -90)]):
            P(f'car{i}', glb_prop(cars / f'{f}.glb', 4.5, out / f'p/car{i}.mjz'), x, y, yaw)
        P('busstop', GZ['bus_stop'], -8, -6.5, 180)
        P('stop', GZ['stop_sign'], 3, 6, 0)
        for i, x in enumerate([-12, 0, 12]):
            P(f'lamp{i}', GZ['lamp_post'], x, -6.2, 0)
        P('dumpster', glb_prop(roads / 'dumpster.glb', 2.0, out / 'p/dump.mjz'), 9, 6.5, 0)
        spawn = dict(position_m=[-12.0, 0.0], yaw_rad=0.0)
    np.savez(out / 'terrain.npz', elevation_m=z)
    bounds = [[-15, -15], [15, 15]]
    bundle = dict(terrain=dict(file='terrain.npz', bounds_m=bounds), instances=[], class_names=['terrain'], rows=[])
    env, manifest = compile_environment(bundle, out, props)
    env.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                    rgb1=[.55, .7, .9], rgb2=[.95, .95, .95], width=512, height=3072)
    src = A / 'outputs/soybean30_obstacle_gen3'
    shutil.copy(src / 'robot.mjz', out); shutil.copy(src / 'robot_profile.json', out)
    profile = json.loads((out / 'robot_profile.json').read_text())
    robot = _read_spec(out / 'robot.mjz'); profile = add_cameras(robot, profile, dict(cameras=['front']))
    spec, model, data = compose_robot(env, robot, profile, spawn, np.asarray(bounds, float))
    env.to_zip(str(out / 'environment.mjz'))
    (out / 'manifest.json').write_text(json.dumps({**manifest, 'domain': kind}))
    (out / 'generation.json').write_text(json.dumps(dict(domain=kind, spawn=spawn)))
    r = mujoco.Renderer(model, 480, 640); opt = mujoco.MjvOption(); opt.geomgroup[3:] = 0
    mujoco.mj_forward(model, data); r.update_scene(data, 'element/nav_front', scene_option=opt)
    Image.fromarray(r.render()).save(out / 'nav_front.png')
    cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0]; cam.distance = 40; cam.elevation = -55; cam.azimuth = 120
    r.update_scene(data, cam, scene_option=opt); Image.fromarray(r.render()).save(out / 'overview.png')
    print(out, model.ngeom, 'geoms')


if __name__ == '__main__':
    build(sys.argv[1], sys.argv[2])
