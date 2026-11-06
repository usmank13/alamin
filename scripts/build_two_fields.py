#!/usr/bin/env python3
"""Build a portable two-plot soybean navigation environment from local Inicio assets.

MUJOCO_GL=egl .venv/bin/python scripts/build_two_fields.py --output outputs/two_fields_corner
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from scene_pipeline.agriculture import compile_environment, compose_robot, load_bundle, sha256
from scene_pipeline.navigation import add_cameras
from scene_pipeline.inspection import scene_page
from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec, load_scene
from build_outdoor_scene import glb_prop, yawq

BOUNDS = np.array([[-22., -10.], [14., 31.]])
PLOTS = [dict(id='west', centre_m=[-14., 0.], yaw_deg=0,
              bounds_m=[[-18., -2.7], [-10., 2.7]]),
         dict(id='east', centre_m=[5., 23.], yaw_deg=90,
              bounds_m=[[2.3, 19.], [7.7, 27.]])]
ROAD_WIDTH = 3.6
PLOT_GAP = float(np.linalg.norm(np.array(PLOTS[1]['bounds_m'][0]) - PLOTS[0]['bounds_m'][1]))


def rotation(yaw):
    angle = math.radians(yaw)
    return np.array([[math.cos(angle), -math.sin(angle)],
                     [math.sin(angle), math.cos(angle)]])


def plot_coordinates(x, y, plot):
    r = rotation(plot['yaw_deg'])
    dx, dy = x-plot['centre_m'][0], y-plot['centre_m'][1]
    return r[0,0]*dx+r[1,0]*dy, r[0,1]*dx+r[1,1]*dy


def reference_route():
    # Tangent straight segments and a 4 m radius left turn, sampled at <= 0.1 m.
    horizontal = np.column_stack((np.linspace(-16.5, 1, 176), np.zeros(176)))
    angle = np.linspace(-np.pi/2, 0, 64)
    corner = np.column_stack((1+4*np.cos(angle), 4+4*np.sin(angle)))
    vertical = np.column_stack((np.full(216, 5.), np.linspace(4, 25.5, 216)))
    return np.concatenate((horizontal, corner[1:], vertical[1:]))


def road_distance(x, y):
    horizontal = np.hypot(x-np.clip(x, -10.4, 1), y)
    angle = np.clip(np.arctan2(y-4, x-1), -np.pi/2, 0)
    corner = np.hypot(x-(1+4*np.cos(angle)), y-(4+4*np.sin(angle)))
    vertical = np.hypot(x-5, y-np.clip(y, 4, 19.4))
    return np.minimum(np.minimum(horizontal, corner), vertical)


def road_fade(x, y):
    return np.clip((x+10.4)/.8, 0, 1) * np.clip((19.4-y)/.8, 0, 1)


def surface(x, y):
    """Continuous soil surface; rotated furrows and ruts follow the curved road."""
    x, y = np.broadcast_arrays(np.asarray(x), np.asarray(y))
    z = .012 * np.sin(.7*x) * np.sin(.8*y) + .005 * np.sin(3*x+2*y)
    for plot in PLOTS:
        along, across = plot_coordinates(x, y, plot)
        weight = np.clip((4.15-abs(along))/.3, 0, 1) * np.clip((2.85-abs(across))/.3, 0, 1)
        z += weight * .025 * (np.cos(2*np.pi*(across-.381)/.762)-1)
    distance = road_distance(x, y)
    road = np.clip((ROAD_WIDTH/2+.2-distance)/.4, 0, 1) * road_fade(x, y)
    return z - road * .018 * np.exp(-((distance-.85)/.18)**2)


def landscape_texture(path):
    rng = np.random.default_rng(19)
    x, y = np.meshgrid(np.linspace(*BOUNDS[:, 0], 1801), np.linspace(*BOUNDS[::-1, 1], 2051))
    noise = rng.normal(0, 1, x.shape)
    broad = gaussian_filter(noise, 25)
    broad /= broad.std()
    rgb = np.broadcast_to([93., 103., 51.], (*x.shape, 3)).copy()
    rgb += (noise * 7 + broad * 8)[..., None]
    for plot in PLOTS:
        along, across = plot_coordinates(x, y, plot)
        mask = (abs(along) <= 4) & (abs(across) <= 2.7)
        furrow = 8 * np.cos(2*np.pi*(across-.381)/.762)
        soil = np.array([105., 77., 49.]) + (noise*8 + furrow)[..., None]
        rgb[mask] = soil[mask]
    distance = road_distance(x, y)
    edge_noise = .055 * np.sin(9*x+7*y) + .035 * np.sin(23*x+19*y)
    road = np.clip((ROAD_WIDTH/2 + edge_noise-distance)/.16, 0, 1) * road_fade(x, y)
    # Pale compacted wheel tracks with a darker, partly grassy centre strip.
    tracks = np.exp(-((distance-.85)/.28)**2)
    dirt = np.array([144., 118., 81.]) + (noise*5 + tracks*12)[..., None]
    rgb = rgb*(1-road[..., None]) + dirt*road[..., None]
    # Gravelly parking pull-off joins the south shoulder.
    pull_off = ((x+3)/3.5)**2 + ((y+4)/3.8)**2 < 1
    rgb[pull_off] = (np.array([127., 108., 76.]) + noise[..., None]*8)[pull_off]
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(path)


def add_ground_visual(spec, xs, ys, z, texture):
    x, y = np.meshgrid(xs, ys)
    vertices = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    a = np.arange(z.size).reshape(z.shape)[:-1, :-1].ravel()
    n = len(xs)
    faces = np.concatenate((np.column_stack((a, a+1, a+n)), np.column_stack((a+1, a+n+1, a+n))))
    uv = np.column_stack(((x.ravel()-xs[0])/(xs[-1]-xs[0]), (y.ravel()-ys[0])/(ys[-1]-ys[0])))
    spec.add_mesh(name='ground_visual', uservert=vertices.ravel(), userface=faces.ravel(),
                  usertexcoord=uv.ravel(), userfacetexcoord=faces.ravel(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, maxhullvert=64)
    spec.assets['ground.png'] = texture.read_bytes()
    spec.add_texture(name='ground_color', type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file='ground.png', vflip=True)
    material = spec.add_material(name='ground_material', specular=0, shininess=0)
    material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'ground_color'
    spec.worldbody.add_geom(name='ground_visual', type=mujoco.mjtGeom.mjGEOM_MESH,
                            meshname='ground_visual', material='ground_material',
                            contype=0, conaffinity=0, group=2, mass=0)
    next(g for g in spec.geoms if g.name == 'terrain').group = 3


def add_plants(spec, manifest, bundle, root):
    """Share source mesh assets across four tiles, retaining individual plant IDs."""
    crops, source_manifest = compile_environment(bundle, root)
    crops.delete(next(g for g in crops.geoms if g.name == 'terrain'))
    for light in list(crops.lights):
        crops.delete(light)
    spec.attach(crops, prefix='crops/', frame=spec.worldbody.add_frame())
    source_geoms = {g.name.removeprefix('crops/'): g for g in spec.geoms if g.name.startswith('crops/')}
    for tile in range(4):
        layout = PLOTS[tile//2]
        plot = layout['id']
        r = rotation(layout['yaw_deg'])
        centre = np.asarray(layout['centre_m']) + r @ [-2 if tile%2 == 0 else 2, 0]
        for item in bundle['instances']:
            identity = f'{plot}_{tile%2}_{item["id"]}'
            old = np.asarray(item['position_m'])
            xy = r @ old[:2] + centre
            new = np.array([*xy, surface(*xy)])
            shift = new - np.array([*(r @ old[:2]), old[2]])
            record = copy.deepcopy(source_manifest['instances'][item['id']])
            record.update(id=identity, position_m=new.tolist(), plot_id=plot)
            if record.get('row_id'):
                record['row_id'] = f'{plot}_{record["row_id"]}'
            manifest['instances'][identity] = record
            for i, _ in enumerate(item['meshes']):
                original_name = f'{item["id"]}_{i}'
                geom = source_geoms[original_name]
                name = f'{identity}_{i}'
                if tile == 0:
                    geom.name = name
                    geom.pos = shift
                else:
                    geom = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=geom.meshname, rgba=geom.rgba, pos=shift, quat=yawq(layout['yaw_deg']),
                        contype=0, conaffinity=0, mass=0, group=2)
                manifest['geom_labels'][name] = dict(instance=identity, class_id=item['meshes'][i].get('class_id', 0))
    for layout in PLOTS:
        for row in bundle['rows']:
            ends = np.array(row['endpoints_m'], dtype=float)
            ends[0, 0] -= 2
            ends[1, 0] += 2
            ends = ends @ rotation(layout['yaw_deg']).T + layout['centre_m']
            manifest['rows'].append(dict(id=f"{layout['id']}_{row['id']}",
                                        plot_id=layout['id'], endpoints_m=ends.tolist()))


def vegetation(tree, seed):
    """Irregular leafy crowns and visible woody branches; rigid navigation proxies."""
    rng = np.random.default_rng(seed)
    spec = mujoco.MjSpec()
    body = spec.worldbody.add_body(name='root')
    height = 5.8 if tree else 1.25
    if tree:
        body.add_geom(name='trunk', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                      fromto=[0, 0, 0, .12, 0, 3.8], size=[.19, 0, 0], rgba=[.25, .18, .11, 1])
        for i in range(9):
            angle = i * 2.4
            end = [1.2*math.cos(angle), 1.2*math.sin(angle), 3.5 + rng.uniform(0, 1)]
            body.add_geom(name=f'branch_{i}', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                          fromto=[.08, 0, 2.3 + .12*i, *end], size=[.055, 0, 0],
                          rgba=[.28, .21, .13, 1], contype=0, conaffinity=0, group=2)
    else:
        body.add_geom(name='bush_collision', type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                      pos=[0, 0, .5], size=[.85, .7, .55], group=3, rgba=[0, 0, 0, 0])
    # Each crown combines many small leaf sprays instead of smooth green spheres.
    for shade in range(4):
        verts, faces = [], []
        for _ in range(900 if tree else 250):
            p = rng.normal(size=3)
            p /= max(1, np.linalg.norm(p))
            p *= [2, 1.7, 1.55] if tree else [.95, .8, .65]
            p += [0, 0, height-1.55 if tree else .65]
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            v = np.cross(u, rng.normal(size=3)); v /= np.linalg.norm(v)
            length = rng.uniform(.10, .24) if tree else rng.uniform(.06, .13)
            u *= length; v *= length * .6
            start = len(verts)
            verts.extend([p-u, p+v, p+u, p-v, p+[0, 0, .025]])
            faces.extend([[start+j, start+(j+1)%4, start+4] for j in range(4)])
        spec.add_mesh(name=f'leaves_{shade}', uservert=np.asarray(verts).ravel(),
                      userface=np.asarray(faces).ravel(), maxhullvert=32,
                      inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        body.add_geom(name=f'leaves_{shade}', type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f'leaves_{shade}',
                      rgba=[.16+.025*shade, .25+.035*shade, .065+.015*shade, 1],
                      contype=0, conaffinity=0, group=2)
    return spec


def add_prop(spec, manifest, child, name, xy, yaw=0):
    z = float(surface(*xy))
    for i, geom in enumerate(child.geoms):
        geom.name = f'geom_{i}'
        manifest['geom_labels'][f'{name}/geom_{i}'] = dict(instance=name, class_id=9)
    spec.attach(child, prefix=name+'/', frame=spec.worldbody.add_frame(pos=[*xy, z], quat=yawq(yaw)))
    manifest['instances'][name] = dict(id=name, category='obstruction', position_m=[*xy, z])


def check_scene(model, data, profile, route):
    """Check full robot footprint along the route and passive stability at spawn."""
    initial = data.qpos.copy()
    root = model.joint('element/root').qposadr[0]
    hits = set()
    low, high = np.asarray(profile['footprint_xy_m'])
    radius = float(np.max(np.linalg.norm([[x,y] for x in (low[0],high[0]) for y in (low[1],high[1])], axis=1)))
    # A swept enclosing disc gives a conservative XY clearance for all headings.
    for geom in range(model.ngeom):
        name = model.geom(geom).name or ''
        if name.startswith('element/') or name == 'terrain' or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        centre = data.geom_xpos[geom, :2]
        # Bounding sphere is intentionally conservative for the parked tractor.
        clearance = np.linalg.norm(np.asarray(route) - centre, axis=1) - radius - model.geom_rbound[geom]
        if np.any(clearance < .15):
            hits.add(name)
    if hits:
        raise ValueError(f'Route clearance failed: {sorted(hits)}')
    mujoco.mj_step(model, data, nstep=1500)
    stable = bool(np.isfinite(data.qpos).all() and not any(w.number for w in data.warning)
                  and data.body('element/chassis').xmat.reshape(3,3)[2,2] > .98)
    drift = float(np.linalg.norm(data.qpos[root:root+2]-initial[root:root+2]))
    if not stable or drift > .15:
        raise ValueError(f'Robot settling failed: stable={stable}, drift={drift}')
    data.qpos[:] = initial
    data.qvel[:] = 0
    data.time = 0
    mujoco.mj_forward(model, data)
    return dict(passed=True, physics_seconds=3, spawn_drift_m=drift,
                route_samples=len(route), route_obstacle_clearance_passed=True,
                physics_warnings=0, autonomous_navigation_tested=False)


def previews(out, model, data):
    option = mujoco.MjvOption(); option.geomgroup[3:] = 0
    with mujoco.Renderer(model, height=720, width=1280) as renderer:
        for name, lookat, distance, azimuth, elevation in (
                ('overview', [-3, 10, 0], 49, 115, -62),
                ('road', [1, 5, 1], 27, 145, -40),
                ('west_plot', [-14, 0, 0], 10, 135, -42)):
            camera = mujoco.MjvCamera(); camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
            renderer.update_scene(data, camera, scene_option=option)
            Image.fromarray(renderer.render()).save(out/f'{name}.png')
        renderer.update_scene(data, 'element/nav_front', scene_option=option)
        Image.fromarray(renderer.render()).save(out/'nav_front.png')
    shutil.copy(out/'overview.png', out/'preview.png')


def build(source, tractor, out):
    source, tractor, out = Path(source), Path(tractor), Path(out)
    bundle = load_bundle(source/'bundle')
    profile = json.loads((source/'robot_profile.json').read_text())
    if not tractor.is_file():
        raise FileNotFoundError(tractor)
    out.mkdir(parents=True, exist_ok=False)
    xs, ys = np.linspace(*BOUNDS[:,0], 361), np.linspace(*BOUNDS[:,1], 411)
    x, y = np.meshgrid(xs, ys); z = surface(x, y)
    np.savez_compressed(out/'terrain.npz', elevation_m=z)
    empty = dict(terrain=dict(file='terrain.npz', bounds_m=BOUNDS.tolist()), instances=[],
                 rows=[], class_names={**bundle['class_names'], '9': 'obstruction'})
    spec, manifest = compile_environment(empty, out)
    landscape_texture(out/'ground.png')
    add_ground_visual(spec, xs, ys, z, out/'ground.png')
    add_plants(spec, manifest, bundle, source/'bundle')
    props = [('oak_inside_corner', True, [-1, 5]), ('oak_north', True, [10, 13]),
             ('bush_1', False, [-7, 4]), ('bush_2', False, [-7, -6.6]),
             ('bush_3', False, [0, 10]), ('bush_4', False, [9, 18]),
             ('bush_5', False, [-19, 5]), ('bush_6', False, [0, 27])]
    for i, (name, tree, xy) in enumerate(props):
        add_prop(spec, manifest, vegetation(tree, 40+i), name, xy)
    tractor_mjz = Path(glb_prop(tractor, 4.2, out/'props/tractor.mjz'))
    add_prop(spec, manifest, _read_spec(tractor_mjz), 'parked_tractor', [-3, -6], -15)
    license_file = tractor.parents[2]/'License.txt'
    shutil.copy(license_file, out/'props/tractor_LICENSE.txt')
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.42,.62,.82],
                     rgb2=[.86,.89,.84], width=512, height=3072)
    spec.visual.quality.shadowsize = 4096
    spawn = dict(position_m=[-16.5, 0.], yaw_rad=0.)
    # Keep the robot-only bundle free of navigation cameras so run_agent_nav can add its own rig.
    shutil.copy(source/'robot.mjz', out/'robot.mjz')
    (out/'robot_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    robot = _read_spec(out/'robot.mjz')
    add_cameras(robot, profile, dict(cameras=['front', 'rear']))
    composed, model, data = compose_robot(spec, robot, profile, spawn, BOUNDS)
    route = reference_route()
    report = check_scene(model, data, profile, route)
    export_mujoco(composed, data, out/'scene.mjz')
    spec.to_zip(str(out/'environment.mjz'))
    # Exercise the portable archive loader, including its saved initial state.
    portable, restored = load_scene(out/'scene.mjz')
    # MJCF serializes state floats to six significant digits.
    np.testing.assert_allclose(restored.qpos, data.qpos, atol=1e-6)
    assert portable.ngeom == model.ngeom
    del portable, restored
    manifest.update(plots=PLOTS, road=dict(width_m=ROAD_WIDTH, centreline_m=route.tolist()))
    mission = dict(start=spawn, goal=dict(position_m=[5, 25.5], yaw_rad=math.pi/2, tolerance_m=.6),
                   plot_gap_m=PLOT_GAP, road_width_m=ROAD_WIDTH,
                   reference_waypoints_m=np.concatenate((route[::10], route[-1:])).tolist(),
                   instruction='Leave the west soybean plot, follow the dirt road past the parked tractor and turn left around the tree, enter the northeast soybean plot with north-south rows, and stop near its far end. Stay on the road between plots and avoid vegetation and equipment.')
    reference_drive = dict(seconds=240, controller=dict(
        root=str(REPO.parent/'rhizome'), host_build=str(REPO/'outputs/rhizome_host_current/host'),
        perception='none', waypoints=[dict(xy=xy) for xy in mission['reference_waypoints_m']],
        parameters=dict(auto_speed_max_ms=.3, arrival_distance=.15)))
    provenance = dict(source=str(source.resolve()), bundle_sha256=sha256(source/'bundle/bundle.json'),
                      tractor_sha256=sha256(tractor), tractor_license='Kenney Car Kit, CC0', seed=19)
    generation = dict(domain='agriculture', spawn=spawn, provenance=provenance,
                      limitations=['Rigid soil and tires; uncalibrated robot dynamics.',
                                   'Crops and foliage are visual; tree trunks, bush envelopes and tractor hull collide.',
                                   'Procedural scenery and stylized tractor, not photorealistic assets.',
                                   'Repeated source plant tiles; terrain outside the supplied bounds is undefined.'])
    report.update(plots=2, rows=len(manifest['rows']), crop_count=sum(v.get('category')=='crop' for v in manifest['instances'].values()),
                  plot_edge_gap_m=PLOT_GAP, road_width_m=ROAD_WIDTH, archive_reload_passed=True, geoms=model.ngeom)
    for filename, value in [('manifest.json', manifest), ('generation.json', generation),
                            ('mission.json', mission), ('reference_drive.json', reference_drive),
                            ('validation.json', report)]:
        (out/filename).write_text(json.dumps(value, indent=2)+'\n')
    previews(out, model, data)
    scene_page(out)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'outputs/soybean30_early_gen3')
    parser.add_argument('--tractor', type=Path, default=REPO/'outputs/outdoor_assets/kenney/kenney_car-kit/Models/GLB format/tractor.glb')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.tractor, args.output)
