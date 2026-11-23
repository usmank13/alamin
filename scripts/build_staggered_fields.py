#!/usr/bin/env python3
"""Build two adjoining six-row soybean blocks with stepped row ends.

MUJOCO_GL=egl .venv/bin/python scripts/build_staggered_fields.py --output outputs/staggered_soybean
"""
import argparse
import copy
import json
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
from scene_pipeline.inspection import scene_page
from scene_pipeline.navigation import add_cameras
from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec, load_scene
from build_two_fields import add_ground_visual, check_scene

SPACING = .762
LENGTH = 10.
OFFSET = 3.
HALF_WIDTH = 6 * SPACING
BOUNDS = np.array([[-20., -12.], [22., 12.]])  # K-turns past a coarse boundary stay on it
SPAWN = dict(position_m=[-8., -HALF_WIDTH / 2], yaw_rad=0.)
PLOTS = [dict(id='south', bounds_m=[[-5., -HALF_WIDTH], [5., 0.]]),
         dict(id='north', bounds_m=[[-5. + OFFSET, 0.], [5. + OFFSET, HALF_WIDTH]])]
BOUNDARY = [[-5., -HALF_WIDTH], [5., -HALF_WIDTH], [5., 0.],
            [5. + OFFSET, 0.], [5. + OFFSET, HALF_WIDTH],
            [-5. + OFFSET, HALF_WIDTH], [-5. + OFFSET, 0.], [-5., 0.]]


def field_mask(x, y):
    return np.logical_or.reduce([(x >= p['bounds_m'][0][0]) & (x <= p['bounds_m'][1][0]) &
                                 (y >= p['bounds_m'][0][1]) & (y <= p['bounds_m'][1][1])
                                 for p in PLOTS])


def surface(x, y):
    x, y = np.broadcast_arrays(x, y)
    # Low relief soil; a level headland gives the robot a stable initial pose.
    return .006 * np.sin(x * .8) * np.sin(y * .6) * field_mask(x, y)


def landscape_texture(path):
    x, y = np.meshgrid(np.linspace(*BOUNDS[:, 0], 2520), np.linspace(*BOUNDS[::-1, 1], 1440))
    noise = np.random.default_rng(52).normal(size=x.shape)
    broad = gaussian_filter(noise, 20)
    broad /= broad.std()
    grain = noise * 4 + broad * 4
    grass = np.array([96., 112., 62.]) + grain[..., None]
    # Parallel cultivation marks share one phase across the join.
    furrows = 7 * np.cos(2 * np.pi * (y - SPACING / 2) / SPACING)
    soil = np.array([113., 86., 57.]) + (grain + furrows)[..., None]
    mask = field_mask(x, y)
    grass[mask] = soil[mask]
    Image.fromarray(np.uint8(np.clip(grass, 0, 255))).save(path)


def add_plants(spec, manifest, bundle, root):
    """Reuse full soybean meshes without scaling; trim repeated 4 m tiles to 10 m."""
    source_rows = sorted(bundle['rows'], key=lambda r: r['endpoints_m'][0][1])
    if len(source_rows) != 6:
        raise ValueError('The source must contain six soybean rows')
    row_lookup = {r['id']: (i, np.mean(np.asarray(r['endpoints_m'])[:, 1]))
                  for i, r in enumerate(source_rows)}
    crops, source_manifest = compile_environment(bundle, root)
    crops.delete(next(g for g in crops.geoms if g.name == 'terrain'))
    for light in list(crops.lights):
        crops.delete(light)
    spec.attach(crops, prefix='source/', frame=spec.worldbody.add_frame())
    originals = {g.name.removeprefix('source/'): g for g in spec.geoms if g.name.startswith('source/')}
    for plot in PLOTS:
        (xmin, ymin), (xmax, ymax) = plot['bounds_m']
        for i, source_row in enumerate(source_rows):
            y = ymin + (i + .5) * SPACING
            manifest['rows'].append(dict(id=f"{plot['id']}_{source_row['id']}", plot_id=plot['id'],
                                         endpoints_m=[[xmin, y], [xmax, y]]))
        for tile, centre in enumerate(np.arange(xmin + 2, xmax + 2, 4.)):
            for item in bundle['instances']:
                old = np.asarray(item['position_m'])
                x = old[0] + centre
                if not xmin <= x < xmax:
                    continue
                if item.get('row_id'):
                    i, source_y = row_lookup[item['row_id']]
                    y = ymin + (i + .5) * SPACING + old[1] - source_y
                else:
                    y = old[1] + (ymin + ymax) / 2
                if not ymin <= y < ymax:
                    continue
                new = np.array([x, y, float(surface(x, y))])
                identity = f"{plot['id']}_{tile}_{item['id']}"
                record = copy.deepcopy(source_manifest['instances'][item['id']])
                record.update(id=identity, position_m=new.tolist(), plot_id=plot['id'])
                if record.get('row_id'):
                    record['row_id'] = f"{plot['id']}_{record['row_id']}"
                manifest['instances'][identity] = record
                for j, mesh in enumerate(item['meshes']):
                    original = originals[f"{item['id']}_{j}"]
                    name = f'{identity}_{j}'
                    # Source vertices are in world coordinates, so subtract the original root.
                    spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=original.meshname, rgba=original.rgba, pos=new-old,
                        contype=0, conaffinity=0, mass=0, group=2)
                    manifest['geom_labels'][name] = dict(instance=identity, class_id=mesh.get('class_id', 0))
    for geom in originals.values():
        spec.delete(geom)


def validate_layout(manifest):
    rows = sorted(manifest['rows'], key=lambda r: r['endpoints_m'][0][1])
    ends = np.array([r['endpoints_m'] for r in rows])
    assert ends.shape == (12, 2, 2)
    np.testing.assert_allclose(ends[:, 1, 0] - ends[:, 0, 0], LENGTH)
    np.testing.assert_allclose(ends[:, 0, 1], ends[:, 1, 1])
    np.testing.assert_allclose(np.diff(ends[:, 0, 1]), SPACING)
    np.testing.assert_allclose(ends[6:, :, 0] - ends[:6, :, 0], OFFSET)
    for plot in PLOTS:
        assert sum(r['plot_id'] == plot['id'] for r in rows) == 6
    boundary = np.asarray(BOUNDARY)
    area = abs(np.dot(boundary[:, 0], np.roll(boundary[:, 1], -1)) -
               np.dot(boundary[:, 1], np.roll(boundary[:, 0], -1))) / 2
    np.testing.assert_allclose(area, 12 * SPACING * LENGTH)
    assert area < np.prod(np.ptp(boundary, axis=0))
    return dict(rows=12, rows_per_block=6, row_length_m=LENGTH, row_spacing_m=SPACING,
                lengthwise_offset_m=OFFSET, cultivated_area_m2=area, stepped_boundary_passed=True)


def previews(out, model, data):
    option = mujoco.MjvOption()
    option.geomgroup[3:] = 0
    with mujoco.Renderer(model, height=720, width=1280, max_geom=12000) as renderer:
        for name, lookat, distance, azimuth, elevation, ortho in [
                ('topdown', [0., 0., 0.], 17., 90., -90., 1),
                ('overview', [0., 0., 0.], 23., 115., -53., 0),
                ('robot', [-3.5, -2.3, .1], 7., 145., -28., 0)]:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = lookat
            camera.distance, camera.azimuth, camera.elevation, camera.orthographic = distance, azimuth, elevation, ortho
            renderer.update_scene(data, camera, scene_option=option)
            Image.fromarray(renderer.render()).save(out/f'{name}.png')
    shutil.copy(out/'overview.png', out/'preview.png')


def build(source, out):
    bundle = load_bundle(source/'bundle')
    profile = json.loads((source/'robot_profile.json').read_text())
    out.mkdir(parents=True, exist_ok=False)
    xs, ys = np.linspace(*BOUNDS[:, 0], 421), np.linspace(*BOUNDS[:, 1], 241)
    x, y = np.meshgrid(xs, ys)
    z = surface(x, y)
    np.savez_compressed(out/'terrain.npz', elevation_m=z)
    empty = dict(terrain=dict(file='terrain.npz', bounds_m=BOUNDS.tolist()),
                 instances=[], rows=[], class_names=bundle['class_names'])
    spec, manifest = compile_environment(empty, out)
    landscape_texture(out/'ground.png')
    add_ground_visual(spec, xs, ys, z, out/'ground.png')
    add_plants(spec, manifest, bundle, source/'bundle')
    layout_report = validate_layout(manifest)
    spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[.45, .64, .82],
                     rgb2=[.88, .9, .84], width=256, height=1536)
    spec.visual.quality.shadowsize = 2048
    shutil.copy(source/'robot.mjz', out/'robot.mjz')
    (out/'robot_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    robot = _read_spec(out/'robot.mjz')
    add_cameras(robot, profile, dict(cameras=['front', 'rear']))
    composed, model, data = compose_robot(spec, robot, profile, SPAWN, BOUNDS)
    report = check_scene(model, data, profile, [SPAWN['position_m']])
    export_mujoco(composed, data, out/'scene.mjz')
    spec.to_zip(str(out/'environment.mjz'))
    portable, restored = load_scene(out/'scene.mjz')
    np.testing.assert_allclose(restored.qpos, data.qpos, atol=1e-5)
    assert portable.ngeom == model.ngeom
    del portable, restored
    manifest.update(plots=PLOTS, field_boundary_m=BOUNDARY)
    generation = dict(domain='agriculture', spawn=SPAWN,
        provenance=dict(source=str(source.resolve()), source_bundle_sha256=sha256(source/'bundle/bundle.json'), seed=52),
        limitations=['Source soybean and weed meshes repeat every 4 metres, without scaling.',
                     'Rigid ground and uncalibrated robot dynamics; crops are visual-only.',
                     'Layout, passive physics and archive loading checked; autonomous driving not tested.'])
    report.update(layout_report, archive_reload_passed=True, geoms=model.ngeom,
                  crop_count=sum(i['category']=='crop' for i in manifest['instances'].values()),
                  weed_count=sum(i['category']=='weed' for i in manifest['instances'].values()))
    for name, value in [('manifest', manifest), ('generation', generation), ('validation', report)]:
        (out/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)
    previews(out, model, data)
    scene_page(out)
    print(f'Built {out}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPO/'outputs/soybean30_early_gen3')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.source, args.output)
