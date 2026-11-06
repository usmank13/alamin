"""Reference image layout, accessible field entries and semantic work patch."""
import importlib.util
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'osmesa')
import mujoco
import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
spec = importlib.util.spec_from_file_location('build_reference_farm', SCRIPTS/'build_reference_farm.py')
scene = importlib.util.module_from_spec(spec)
sys.path.insert(0,str(SCRIPTS))
try:
    spec.loader.exec_module(scene)
finally:
    sys.path.pop(0)


def test_reference_layout_and_clear_entry_routes():
    np.testing.assert_allclose(scene.metric([[0,1049],[731,0]]),scene.BOUNDS)
    assert scene.SPAWN['position_m'][0] > 80
    assert scene.SPAWN['position_m'][1] > 110
    road=scene.route_points(scene.ROAD_POINTS)
    np.testing.assert_allclose(scene.road_distance(road[:,0],road[:,1]),0,atol=1e-12)
    assert scene.ROAD_WIDTH/2 > np.hypot(1.02,1.08)+.15
    for destination,plot in zip(scene.DESTINATIONS,scene.PLOTS):
        xy=destination['position_m']
        assert scene.polygon_mask(*xy,plot['polygon_m'])
        route=scene.routes()[destination['id']]
        np.testing.assert_allclose(route[0],scene.SPAWN['position_m'])
        np.testing.assert_allclose(route[-1],xy)
        assert np.max(np.linalg.norm(np.diff(route,axis=0),axis=1)) <= .201
        assert np.all((route>=scene.BOUNDS[0])&(route<=scene.BOUNDS[1]))
        assert np.max(abs(np.diff(scene.surface(route[:,0],route[:,1])))) < .015
    assert scene.surface(*scene.SPAWN['position_m']) == 0


def test_semantic_patch_retains_crop_weed_ids_and_rotates_rows():
    bundle=dict(instances=[dict(id='crop',category='crop',position_m=[-1,1,0],row_id='row'),
                          dict(id='weed',category='weed',position_m=[1,-1,0],row_id=None)],
                rows=[dict(id='row',endpoints_m=[[-1,1],[1,1]])])
    spec=mujoco.MjSpec()
    manifest=dict(instances={},rows=[],geom_labels={})
    scene.add_crops(spec,manifest,bundle)
    model=spec.compile()
    assert model.nmesh==3
    assert model.ngeom==3
    assert manifest['instances']['west_crop']['row_id']=='west_row'
    assert manifest['instances']['west_weed']['category']=='weed'
    assert manifest['geom_labels']['west_weed']==dict(instance='west_weed',class_id=1)
    np.testing.assert_allclose(manifest['rows'][0]['endpoints_m'],[[19,51],[19,53]])
    np.testing.assert_allclose(manifest['instances']['west_crop']['position_m'],[19,51,scene.surface(19,51)])
