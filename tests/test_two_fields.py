"""Metric layout and plant/semantic alignment for the two-field builder."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('build_two_fields', SCRIPTS / 'build_two_fields.py')
scene = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(SCRIPTS))
try:
    spec.loader.exec_module(scene)
finally:
    sys.path.pop(0)


def test_tiles_share_meshes_and_keep_distinct_grounded_labels(tmp_path):
    np.savez(tmp_path/'terrain.npz', elevation_m=np.zeros((5, 5)))
    vertices = np.array([[0, 0, 0], [.1, 0, 0], [0, .1, 0], [0, 0, .2]])
    np.savez(tmp_path/'plant.npz', vertices=vertices,
             faces=[[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
    mesh = dict(file='plant.npz', rgba=[.1, .4, .1, 1], class_id=2)
    bundle = dict(terrain=dict(file='terrain.npz', bounds_m=[[-2,-2],[2,2]]),
                  instances=[dict(id='plant', position_m=[0,0,0], category='crop',
                                  row_id='row', meshes=[mesh])],
                  rows=[dict(id='row', endpoints_m=[[-1.9,0],[1.9,0]])], class_names={'2':'crop'})
    target, _ = scene.compile_environment({**bundle, 'instances': [], 'rows': []}, tmp_path)
    manifest = dict(instances={}, geom_labels={}, rows=[])
    scene.add_plants(target, manifest, bundle, tmp_path)
    model = target.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model, data)
    assert model.nmesh == 1
    names = ['west_0_plant', 'west_1_plant', 'east_0_plant', 'east_1_plant']
    assert set(manifest['instances']) == set(names)
    geom_centres = []
    for name, (x, y) in zip(names, [(-16,0), (-12,0), (5,21), (5,25)]):
        item = manifest['instances'][name]
        np.testing.assert_allclose(item['position_m'], [x, y, scene.surface(x,y)])
        assert item['row_id'] == name.split('_')[0]+'_row'
        assert manifest['geom_labels'][name+'_0'] == dict(instance=name, class_id=2)
        geom_centres.append(data.geom(name+'_0').xpos.copy())
    np.testing.assert_allclose(np.array(geom_centres)[1]-geom_centres[0], [4,0,scene.surface(-12,0)-scene.surface(-16,0)], atol=1e-7)
    np.testing.assert_allclose(np.array(geom_centres)[3]-geom_centres[2], [0,4,scene.surface(5,25)-scene.surface(5,21)], atol=1e-7)
    r = np.eye(3); r[:2,:2] = scene.rotation(90)
    np.testing.assert_allclose(data.geom('east_0_plant_0').xmat.reshape(3,3),
                               r @ data.geom('west_0_plant_0').xmat.reshape(3,3), atol=1e-7)
    assert len(manifest['rows']) == 2
    np.testing.assert_allclose(manifest['rows'][0]['endpoints_m'], [[-17.9,0],[-10.1,0]])
    np.testing.assert_allclose(manifest['rows'][1]['endpoints_m'], [[5,19.1],[5,26.9]])


def test_road_joins_plots_without_height_steps():
    assert 20 < scene.PLOT_GAP < 21
    route = scene.reference_route()
    np.testing.assert_allclose(route[[0,-1]], [[-16.5,0],[5,25.5]])
    steps = np.diff(route, axis=0)
    assert np.max(np.linalg.norm(steps, axis=1)) <= .101
    np.testing.assert_allclose(steps[0]/np.linalg.norm(steps[0]), [1,0])
    np.testing.assert_allclose(steps[-1]/np.linalg.norm(steps[-1]), [0,1])
    road = route[(route[:,0] >= -10) & (route[:,1] <= 19)]
    np.testing.assert_allclose(scene.road_distance(road[:,0],road[:,1]), 0, atol=1e-12)
    # The two rut tracks follow the tangent through the corner and transitions.
    tangent = np.gradient(route, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:,None]
    normal = np.column_stack((-tangent[:,1], tangent[:,0]))
    for offset in (-.85, .85):
        track = route + offset*normal
        assert np.max(np.abs(np.diff(scene.surface(track[:,0],track[:,1])))) < .025
