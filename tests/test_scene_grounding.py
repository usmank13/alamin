"""Synthetic fixtures test mechanisms, not empirical realism. Real-data runs are separate."""
from copy import deepcopy
import hashlib
import json

import mujoco
import numpy as np
import pytest
pytest.importorskip('shapely')

from scene_pipeline.contracts import PipelineError,digest,validate_program
from scene_pipeline.reference_layout import parse_svg,transform,metric_geometry
from scene_pipeline.reconstruction import reference_ir,compare_compiled
from scene_pipeline.compiler import compile_scene
from scene_pipeline.layout_priors import verify_bundle,condition,evaluate
from scene_pipeline.generators import generate_layout
from scene_pipeline.scene_intent import freeze,check_revision
from scene_pipeline.scene_checks import check_scene
from scene_pipeline.dsl import load


SVG='''<svg xmlns="http://www.w3.org/2000/svg"><g transform="translate(2,3)">
<g class="Space Storage"><polygon points="0,0 8,0 8,5 0,5"/></g>
<g class="Wall"><polygon points="0,0 8,0 8,.2 0,.2"/>
<g class="Door"><polygon points="3,0 4,0 4,.2 3,.2"/></g></g>
<g class="Wall"><polygon points="0,4.8 8,4.8 8,5 0,5"/></g>
<g class="FixedFurniture BaseCabinet" transform="translate(1,1) scale(2)">
<g class="BoundaryPolygon"><polygon points="0,0 1,0 1,.5 0,.5"/></g></g>
</g></svg>'''


@pytest.fixture
def reference(tmp_path):
    path=tmp_path/'source.svg';path.write_text(SVG)
    source=dict(dataset='synthetic_unit_test_NOT_real',url='https://example.invalid/synthetic',
                license='test',member='synthetic.svg',sha256=hashlib.sha256(SVG.encode()).hexdigest())
    return parse_svg(path,source)


@pytest.fixture
def bundle():
    sources=[dict(sha256=f'synthetic_{i}',dataset='synthetic_unit_test_NOT_real') for i in range(9)]
    rows=[dict(source_group=s['sha256'],room_id='r',role='kitchen',split='train' if i<5 else 'calibration' if i<7 else 'test',
        values=dict(aspect=1.1+i*.07,rectangularity=1.,fixed_fixture_coverage=.01+i*.005,
                    fixed_fixture_wall_gap=.005*i,fixed_fixture_count=1+i%3)) for i,s in enumerate(sources)]
    b=dict(kind='empirical_prior_bundle',schema_version=1,sources=sources,rows=rows)
    b['sha256']=digest(b);return b


def rehash(b):
    b['sha256']=digest({k:v for k,v in b.items() if k!='sha256'});return b


def test_affine_and_source_identity(reference,tmp_path):
    np.testing.assert_allclose(reference['fixtures'][0]['polygon'],[[3,4],[5,4],[5,5],[3,5]])
    assert reference['metric_scale'] is None
    path=tmp_path/'changed.svg';path.write_text(SVG+' ')
    with pytest.raises(PipelineError,match='manifest'):parse_svg(path,reference['source'])
    with pytest.raises(PipelineError):transform('skewX(30)')


def test_scale_never_guessed(reference):
    with pytest.raises(PipelineError):metric_geometry(reference)
    with pytest.raises(PipelineError):metric_geometry(reference,area_m2=60,metres_per_unit=.01)
    with pytest.raises(PipelineError):metric_geometry(reference,area_m2=float('nan'))
    _,evidence=metric_geometry(reference,area_m2=60)
    assert not evidence['source_metric_verified'] and evidence['kind']=='target_area_normalization'


def test_reference_roundtrip_and_tamper(reference,tmp_path):
    ir=reference_ir(reference,area_m2=60)
    _,model,data=compile_scene(ir,tmp_path)
    report,_,_=compare_compiled(reference,model,data,area_m2=60)
    assert report['passed'],report
    assert report['fixture_annotations_not_reconstructed']==1
    assert not report['full_scene_validated']
    damaged=deepcopy(ir)
    damaged['architecture']['walls'][0]['polygon']=[[x,y+.2] for x,y in damaged['architecture']['walls'][0]['polygon']]
    # Leave provenance and claimed source hash unchanged: validator must detect geometry.
    _,model,data=compile_scene(damaged,tmp_path)
    report,_,_=compare_compiled(reference,model,data,area_m2=60)
    assert not report['passed']


def test_oblique_concave_architecture(reference,tmp_path):
    reference['rooms'][0]['polygon']=[[0,0],[8,0],[8,5],[3,5],[3,2],[0,2]]
    reference['walls'][0]['polygon']=[[0,0],[4,1],[4.1,1.3],[.1,.3]]
    reference['reference_sha256']=digest({k:v for k,v in reference.items() if k!='reference_sha256'})
    ir=reference_ir(reference,area_m2=60)
    _,m,d=compile_scene(ir,tmp_path)
    report,_,_=compare_compiled(reference,m,d,area_m2=60)
    assert report['passed'],report


def test_normalized_reference_tampering(reference):
    reference['rooms'][0]['polygon'][0][0]+=10
    with pytest.raises(PipelineError,match='normalized identity'):metric_geometry(reference,area_m2=60)


def test_source_split_guards(bundle):
    verify_bundle(bundle)
    corrupt=deepcopy(bundle);corrupt['rows'][0]['values']['aspect']=999
    with pytest.raises(PipelineError,match='identity'):verify_bundle(corrupt)
    corrupt=deepcopy(bundle);corrupt['rows'][0]['source_group']='invented'
    with pytest.raises(PipelineError,match='Unknown source'):verify_bundle(rehash(corrupt))
    corrupt=deepcopy(bundle);corrupt['rows'][5]['source_group']=corrupt['rows'][0]['source_group']
    with pytest.raises(PipelineError,match='partitions'):verify_bundle(rehash(corrupt))


def test_explicit_ood_backoff(bundle):
    with pytest.raises(PipelineError,match='training source'):condition(bundle,'workshop')
    p=condition(bundle,'workshop',allow_backoff=True)
    assert p['backoff']=='generic_architecture_only'
    assert p['features']==('aspect','rectangularity')


def test_test_partition_never_conditions_generation(bundle):
    first=condition(bundle,'kitchen')
    for r in bundle['rows']:
        if r['split']=='test':r['values']['aspect']=10000
    second=condition(rehash(bundle),'kitchen')
    np.testing.assert_array_equal(first['matrix'],second['matrix'])


def test_generic_space_and_frozen_intent():
    program=load('examples/cafe_program.py');program['space']['kind']='electronics repair workspace'
    validate_program(program)
    intent=freeze(program,authority='supplied_program');check_revision(intent,program)
    program['objects'][0]['count']+=1
    with pytest.raises(PipelineError,match='frozen'):check_revision(intent,program)


def test_backends_share_contract_and_repeat(bundle,tmp_path):
    p=load('examples/cafe_program.py');p['space']['kind']='kitchen'
    baseline=generate_layout(p,4,tmp_path/'base')
    a=generate_layout(p,4,tmp_path/'a',backend='empirical',priors=bundle,candidates=3)
    b=generate_layout(p,4,tmp_path/'b',backend='empirical',priors=bundle,candidates=3)
    assert set(a)==set(baseline) and digest(a)==digest(b)
    intent=freeze(p,authority='supplied_program')
    assert check_scene(intent,a,tmp_path/'a')['passed']
    _,m,d=compile_scene(a,tmp_path/'a');assert m.ngeom>0
    report=evaluate(a,tmp_path/'a',bundle)
    assert not report['full_scene_distribution_verified']


def test_scene_checks_ignore_fabricated_cached_bounds(tmp_path):
    p=load('examples/cafe_program.py');ir=generate_layout(p,2,tmp_path)
    intent=freeze(p,authority='supplied_program')
    assert check_scene(intent,ir,tmp_path)['passed']
    ir['objects'][0]['position'][0]=1000
    ir['objects'][0]['bounds']=[[0,0,0],[1,1,1]]
    assert any(e['code']=='OUTSIDE_ROOMS' for e in check_scene(intent,ir,tmp_path)['errors'])
    ir['objects'][1]['dimensions'][0]=1000
    assert any(e['code']=='INVENTED_DIMENSIONS' for e in check_scene(intent,ir,tmp_path)['errors'])


def test_missing_inventory_not_self_approved(tmp_path):
    p=load('examples/cafe_program.py');ir=generate_layout(p,2,tmp_path)
    intent=freeze(p,authority='supplied_program');ir['objects'].pop()
    assert not check_scene(intent,ir,tmp_path)['passed']


def test_required_relation_not_self_approved(tmp_path):
    p=load('examples/cafe_program.py');ir=generate_layout(p,2,tmp_path)
    p['relations']=[dict(kind='under',objects=['cabinet','prep'],required=True)]
    intent=freeze(p,authority='supplied_program')
    ir['provenance']['relations']=[dict(satisfied=True)]
    assert any(e['code']=='REQUIRED_RELATION' for e in check_scene(intent,ir,tmp_path)['errors'])
