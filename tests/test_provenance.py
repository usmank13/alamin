import json
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('jsonschema')
import mujoco
from scene_pipeline.assets import instantiate,bounds
from scene_pipeline.provenance import classify
from scene_pipeline.contracts import PipelineError,write_json
from scene_pipeline.dsl import load
from scene_pipeline.layout import solve
from scene_pipeline.compiler import compile_scene


def test_procedural_origin_is_not_a_measurement_or_certification(tmp_path):
    _,asset=instantiate('jar',tmp_path)
    p=classify(asset)
    assert p['origin']=='procedural' and p['dimension_basis']=='engineering_design_parameters'
    assert not p['contact_rich_certified']
    assert p==asset['source_classification']


def test_generated_origin_defaults_to_visual_only():
    p=classify(dict(route='G6',category='fruit',provenance=dict(kind='generated_source',provider='fal',source='future-request')))
    assert p['origin']=='generated' and p['provider']=='fal'
    assert p['allowed_use']=='visual_only' and not p['contact_rich_certified']
    assert p['physical_basis']=='not_assigned_visual_only'
    with pytest.raises(PipelineError):classify(dict(route='G1',provenance=dict(kind='retrieved_source')))


def test_compiler_rejects_contact_enabled_generated_decoration(tmp_path):
    p=load('examples/cafe_program.py');ir=solve(p,1,tmp_path)
    path=tmp_path/ir['objects'][0]['asset'];asset=json.loads(path.read_text())
    asset['route']='G6';asset['provenance']={'kind':'generated_source','provider':'fal'}
    write_json(path,asset)
    with pytest.raises(PipelineError,match='contact geometry'):compile_scene(ir,tmp_path)


def test_generated_collision_is_still_unverified_generated_geometry():
    p=classify(dict(route='G6',category='fruit',provenance=dict(kind='generated_source',provider='fal',
        physical_use='static_collision',physical_policy='generated_static_convex_hull_v1')))
    assert p['origin']=='generated' and p['allowed_use']=='static_collision'
    assert p['dimension_basis']=='unverified_generated_geometry'
    assert p['physical_basis']=='generated_static_convex_hull_v1'
    assert not p['calibrated'] and not p['contact_rich_certified']


def test_generated_collision_cannot_claim_articulation(tmp_path):
    ir=solve(load('examples/cafe_program.py'),1,tmp_path)
    item=next(o for o in ir['objects'] if o['category']=='base_cabinet')
    path=tmp_path/item['asset'];asset=json.loads(path.read_text())
    asset['route']='G6';asset['provenance']=dict(kind='generated_source',provider='fal',physical_use='static_collision')
    write_json(path,asset)
    with pytest.raises(PipelineError,match='dynamics or articulation'):compile_scene(ir,tmp_path)


def test_retrieved_sources_and_countertop_support(tmp_path):
    if not Path('vendor/robocasa_native/fixtures/microwaves/Microwave075/model.xml').exists():pytest.skip('Optional native sources absent')
    ir=solve(load('examples/rich_kitchen_program.py'),17,tmp_path)
    write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    inventory=json.loads((tmp_path/'provenance.json').read_text())
    assert inventory['counts']=={'procedural':42,'retrieved':2}
    microwave=next(o for o in ir['objects'] if o['category']=='microwave')
    assert microwave['support_parent'] is not None
    assert .9<microwave['position'][2]<1.2
    assert microwave['source_classification']['origin']=='retrieved'
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['instances'][microwave['id']]['source_classification']==microwave['source_classification']
    model=mujoco.MjSpec.from_zip(str(tmp_path/'scene.mjz')).compile()
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('microwave_0/') and model.geom_contype[i]:assert model.geom_rgba[i,3]==0


def test_mesh_bounds_use_vertices_not_rotated_box():
    spec=mujoco.MjSpec.from_string('''<mujoco><asset><mesh name="tetra" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/></asset>
      <worldbody><geom mesh="tetra" type="mesh" euler="0 0 45" contype="0" conaffinity="0"/></worldbody></mujoco>''')
    m=spec.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    lo,hi=bounds(m,d)
    np.testing.assert_allclose(lo,[-2**-.5,0,0],atol=1e-6)
    np.testing.assert_allclose(hi,[2**-.5,2**-.5,1],atol=1e-6)


def test_required_wall_relation_guides_placement(tmp_path):
    p=load('examples/cafe_program.py')
    p['relations']=[dict(kind='against_wall',objects=['storage'],required=True)]
    ir=solve(p,1,tmp_path)
    assert all(r['satisfied'] for r in ir['provenance']['relations'])


def test_compiler_does_not_trust_program_origin_label(tmp_path):
    ir=solve(load('examples/cafe_program.py'),1,tmp_path)
    ir['objects'][0]['source_classification']={'origin':'generated','provider':'invented'}
    compile_scene(ir,tmp_path)
    manifest=json.loads((tmp_path/'manifest.json').read_text())
    assert manifest['instances'][ir['objects'][0]['id']]['source_classification']['origin']=='procedural'


def test_shelf_support_clear_of_corner_uprights(tmp_path):
    _,asset=instantiate('shelf',tmp_path)
    width,depth,_=asset['nominal_dimensions']
    for surface in asset['supports']:
        assert surface['size'][0]/2<=width/2-.06+1e-9
        assert surface['size'][1]/2<=depth/2-.06+1e-9
