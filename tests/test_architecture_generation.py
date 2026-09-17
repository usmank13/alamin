"""Synthetic fixtures test generic mechanisms; never counted as empirical evidence."""
from copy import deepcopy
from pathlib import Path
import hashlib
import math

import numpy as np
import pytest
from shapely.geometry import Polygon

from scene_pipeline.contracts import PipelineError,digest,validate_program,read_json,write_json
from scene_pipeline.architecture_priors import signature,family,extract_room,canonical_boundary,verify_bundle
from scene_pipeline.architecture_sampler import sample_architecture,eligible_rows
from scene_pipeline.architecture_checks import check_architecture
from scene_pipeline.reference_layout import parse_svg
from scene_pipeline.compiler import compile_scene
from scene_pipeline.scene_intent import freeze,check_revision


@pytest.fixture
def program():
    p=read_json('examples/architecture_hall.json');p['space']['kind']='workroom'
    return p


def make_bundle(concave=False):
    rows=[];sources=[]
    for i in range(5):
        width=1.2+i*.12
        points=np.array([[0,0],[2,0],[2,1],[1,1],[1,2],[0,2]],dtype=float) if concave else np.array([[0,0],[1,0],[1,1],[0,1]],dtype=float)
        points[:,0]*=width;points[:,1]/=width;points/=np.sqrt(Polygon(points).area);points=points.tolist()
        openings=[dict(kind='door',edge=0,offset=.3+i*.02,width=.15),dict(kind='window',edge=2,offset=.5,width=.2)]
        split='train' if i<3 else 'calibration' if i==3 else 'test';group=f'synthetic_{i}'
        sources.append(dict(group=group,split=split,dataset='SYNTHETIC_TEST_NOT_REAL',sha256=digest(group)))
        rows.append(dict(points=points,openings=openings,role='workroom',family=family(points),
                         signature=signature(points,openings),room_id='r',source_group=group,split=split))
    bundle=dict(schema_version=2,kind='architecture_prior_bundle',sources=sources,rows=rows)
    bundle['sha256']=digest(bundle);return bundle


def rehash(bundle):
    bundle['sha256']=digest({k:v for k,v in bundle.items() if k!='sha256'});return bundle


def test_schema_and_explicit_freeze(program):
    validate_program(program);intent=freeze(program,authority='supplied_program')
    program['architecture']['openings'][0]['count']=2
    with pytest.raises(PipelineError,match='explicit'):check_revision(intent,program)
    program['architecture']['openings'][0]['required']=False
    with pytest.raises(PipelineError):validate_program(program)


def test_inferred_preference_can_change(program):
    program['architecture']['openings'][0].update(origin='inferred',required=False)
    intent=freeze(program,authority='supplied_program')
    program['architecture']['openings'][0]['count']=2
    check_revision(intent,program)


def test_schema_rejects_llm_coordinates(program):
    program['architecture']['openings'][0]['position']=[1,2,0]
    with pytest.raises(PipelineError):validate_program(program)


def test_unsupported_requirement_not_silently_dropped(program,tmp_path):
    program['architecture']['unsupported_requirements']=['A doorway exactly 3 metres wide']
    intent=freeze(program,authority='supplied_program')
    with pytest.raises(PipelineError,match='cannot be realized'):sample_architecture(program,0,tmp_path,bundle=make_bundle())
    program['architecture']['unsupported_requirements']=[]
    with pytest.raises(PipelineError,match='dropped'):check_revision(intent,program)


@pytest.mark.parametrize('concave',[False,True])
def test_joint_sampling_reproducible_and_compiled(program,tmp_path,concave):
    bundle=make_bundle(concave)
    a=sample_architecture(program,5,tmp_path/'a',bundle=bundle,robot_radius=.3)
    b=sample_architecture(program,5,tmp_path/'b',bundle=bundle,robot_radius=.3)
    assert digest(a)==digest(b)
    _,m,d=compile_scene(a,tmp_path/'a')
    report=check_architecture(program,a,bundle=bundle,robot_radius=.3,model=m,data=d)
    assert report['passed'],report
    assert len(report['compiled_geometry'])==4
    assert Polygon(a['rooms'][0]['polygon']).area==pytest.approx(90)


def test_no_semantic_label_branches(program,tmp_path):
    bundle=make_bundle();outputs=[]
    for i,label in enumerate(['electronics repair','manufacturing hall','office']):
        p=deepcopy(program);p['space']['kind']=label
        with pytest.raises(PipelineError):eligible_rows(p,bundle)
        outputs.append(sample_architecture(p,7,tmp_path/str(i),bundle=bundle,allow_backoff=True))
    for ir in outputs[1:]:
        assert ir['rooms'][0]['polygon']==outputs[0]['rooms'][0]['polygon']
        assert ir['openings']==outputs[0]['openings']
    assert outputs[0]['provenance']['architecture_sampling']['conditioning']=='generic_architecture_backoff'


def test_new_seed_changes_layout_not_best_of_n(program,tmp_path):
    bundle=make_bundle();layouts=[]
    for seed in range(12):
        ir=sample_architecture(program,seed,tmp_path/str(seed),bundle=bundle)
        layouts.append(digest((ir['rooms'][0]['polygon'],ir['openings'])))
    assert len(set(layouts))==12


def test_holdout_not_used(program,tmp_path):
    bundle=make_bundle();a=sample_architecture(program,4,tmp_path/'a',bundle=bundle)
    bundle['rows'][-1]['points']=[[0,0],[10,0],[10,.1],[0,.1]]
    bundle['rows'][-1]['signature']=signature(bundle['rows'][-1]['points'],bundle['rows'][-1]['openings'])
    b=sample_architecture(program,4,tmp_path/'b',bundle=rehash(bundle))
    assert a['rooms']==b['rooms'] and a['openings']==b['openings']
    assert all(s['source_group']!='synthetic_4' for s in a['provenance']['architecture_sampling']['sources'])


def test_source_leakage_and_sparse_support(program):
    bundle=make_bundle();bundle['rows'][0]['split']='test'
    with pytest.raises(PipelineError,match='partition'):verify_bundle(rehash(bundle))
    bundle=make_bundle();bundle['rows']=bundle['rows'][1:]
    with pytest.raises(PipelineError,match='three independent'):eligible_rows(program,rehash(bundle))


def test_required_relationships(program,tmp_path):
    program['architecture']['openings'].append(dict(id='daylight',kind='window',count=1,role='daylight',required=True,origin='explicit'))
    program['architecture']['relationships']=[dict(kind='opposite_wall',openings=['entry','daylight'],required=True,origin='explicit')]
    ir=sample_architecture(program,1,tmp_path/'good',bundle=make_bundle())
    assert check_architecture(program,ir,bundle=make_bundle())['passed']
    program['architecture']['relationships'][0]['kind']='same_wall'
    with pytest.raises(PipelineError):sample_architecture(program,1,tmp_path/'bad',bundle=make_bundle())


def test_tamper_checks_geometry_not_claimed_flags(program,tmp_path):
    bundle=make_bundle();ir=sample_architecture(program,0,tmp_path,bundle=bundle)
    ir['openings'][0]['host_edge']=2
    assert any(e['code']=='OPENING_METADATA' for e in check_architecture(program,ir,bundle=bundle)['errors'])
    ir['rooms'][0]['polygon'][0][0]+=.05
    assert not check_architecture(program,ir,bundle=bundle)['passed']


def test_compiled_tamper(program,tmp_path):
    bundle=make_bundle();ir=sample_architecture(program,0,tmp_path,bundle=bundle)
    _,m,d=compile_scene(ir,tmp_path)
    import mujoco
    gid=next(i for i in range(m.ngeom) if m.geom(i).name.startswith('wall_reference') and m.geom(i).name.endswith('_collision'))
    m.geom_pos[gid,0]+=.1;mujoco.mj_forward(m,d)
    assert any(e['code']=='COMPILED_GEOMETRY' for e in check_architecture(program,ir,bundle=bundle,model=m,data=d)['errors'])


def test_narrow_entry_exhausts_budget(program,tmp_path):
    with pytest.raises(PipelineError,match='budget'):sample_architecture(program,0,tmp_path,bundle=make_bundle(),robot_radius=3.)
    assert len(read_json(tmp_path/'architecture_proposals.json'))==64


def test_two_entries_separated_by_narrow_bottleneck(program,tmp_path):
    bundle=make_bundle();program['architecture']['openings'][0]['count']=2
    outline=np.array([[0,0],[3,0],[3,1.4],[5,1.4],[5,0],[8,0],
                      [8,3],[5,3],[5,1.6],[3,1.6],[3,3],[0,3]],dtype=float)
    for i,row in enumerate(bundle['rows']):
        points=outline.copy();points[:,0]*=1+i*.05;points[:,1]/=1+i*.05
        points/=math.sqrt(Polygon(points).area)
        openings=[dict(kind='door',edge=e,offset=.5,width=.4) for e in (5,11)]
        row.update(points=points.tolist(),openings=openings,family=family(points),signature=signature(points,openings))
    with pytest.raises(PipelineError,match='budget'):sample_architecture(program,2,tmp_path,bundle=rehash(bundle),robot_radius=.3)
    trials=read_json(tmp_path/'architecture_proposals.json')
    assert any(e['code']=='ENTRY_CONNECTIVITY' for t in trials for e in t.get('errors',[]))


def test_unverified_robot_without_config(program,tmp_path):
    ir=sample_architecture(program,0,tmp_path,bundle=make_bundle())
    assert check_architecture(program,ir)['access']['status']=='unverified_no_robot_footprint'


def test_canonical_rotation_and_translation():
    points=np.array([[0,0],[4,0],[4,2],[0,2]])
    a,_,_=canonical_boundary(points)
    b,_,_=canonical_boundary(points@np.array([[0,-1],[1,0]])+[10,20])
    np.testing.assert_allclose(a,b,atol=1e-9)


def test_opening_requires_wall_intersection(tmp_path):
    svg='''<svg xmlns="http://www.w3.org/2000/svg"><g class="Space Office"><polygon points="0,0 8,0 8,5 0,5"/></g>
    <g class="Wall"><polygon points="0,-.2 8,-.2 8,0 0,0"/></g>
    <g class="Door"><polygon points="3,-.2 4,-.2 4,0 3,0"/></g></svg>'''
    path=tmp_path/'a.svg';path.write_text(svg)
    ref=parse_svg(path,dict(sha256=hashlib.sha256(svg.encode()).hexdigest()))
    assert len(extract_room(ref,ref['rooms'][0])['openings'])==1
    ref['openings'][0]['polygon']=[[3,0],[4,0],[4,.2],[3,.2]]
    with pytest.raises(PipelineError):extract_room(ref,ref['rooms'][0])


def test_furnishing_adapter_and_public_pipeline(program,tmp_path):
    from scene_pipeline.orchestrator import generate
    program['objects']=[dict(id='shelf',category='shelf',count=1,zone='main',required=True)]
    report=generate(program['prompt'],3,tmp_path/'generated',program=program,layout_backend='architecture',priors=make_bundle(),preview=False,robot_radius=.3)
    assert report['status']=='validated' and not report['distribution_verified']
    result=read_json(tmp_path/'generated/ir.json')
    assert len(result['objects'])==1
    assert result['provenance']['furnishing_grounding'].startswith('heuristic')


def test_static_architecture_export_is_not_articulation_proof(program,tmp_path):
    pytest.importorskip('pybullet')
    from scene_pipeline.portability import export_urdf,verify_urdf
    ir=sample_architecture(program,2,tmp_path,bundle=make_bundle())
    write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    report=verify_urdf(export_urdf(tmp_path))
    assert report['passed'] and not report['articulation_verified']
    assert report['verification_scope']=='static_package_load_only'


def test_no_forged_source_metadata(program,tmp_path):
    bundle=make_bundle();ir=sample_architecture(program,1,tmp_path,bundle=bundle)
    ir['provenance']['architecture_sampling']['source_records'][0]['dataset']='invented real source'
    assert any(e['code']=='SOURCE_METADATA' for e in check_architecture(program,ir,bundle=bundle)['errors'])


def test_checker_reports_invalid_inputs(program,tmp_path):
    ir=sample_architecture(program,1,tmp_path,bundle=make_bundle())
    with pytest.raises(PipelineError,match='radius'):
        check_architecture(program,ir,robot_radius=-.1)
    legacy=deepcopy(program);legacy['schema_version']=1
    with pytest.raises(PipelineError,match='v2'):
        check_architecture(legacy,ir)


def test_mapping_approach_tracks_room_frame(program,tmp_path):
    from scene_pipeline.flows import mapping_approaches
    ir=sample_architecture(program,1,tmp_path,bundle=make_bundle())
    pose,yaw=mapping_approaches(ir,.7)[0]
    rotation=np.array([[0,-1],[1,0]]);shift=np.array([12.,-8.])
    for item in ir['rooms']+ir['openings']:
        item['polygon']=(np.array(item['polygon'])@rotation.T+shift).tolist()
    transformed,angle=mapping_approaches(ir,.7)[0]
    np.testing.assert_allclose(transformed[:2],rotation@pose[:2]+shift)
    assert math.cos(angle-yaw)==pytest.approx(0.,abs=1e-9)
    assert math.sin(angle-yaw)==pytest.approx(1.)
    assert mapping_approaches(dict(schema_version=1),.7)==[([.7,0,0],0.)]


def test_stock_base_in_sampled_architecture(program,tmp_path):
    if not Path('vendor/mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml').exists():
        pytest.skip('Stock robot checkout not installed')
    from scene_pipeline.flows import robot_spec,Mapper
    ir=sample_architecture(program,1,tmp_path,bundle=make_bundle())
    write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    _,model,data,_,rig,_=robot_spec(tmp_path,'mapping')
    placement=rig['placement']
    assert placement['yaw']==pytest.approx(math.pi/2)
    mapper=Mapper(ir,model,data,placement['position'][:2],placement['yaw'])
    np.testing.assert_allclose(mapper.pose,[*placement['position'][:2],placement['yaw']])
    np.testing.assert_allclose(data.xpos[model.body('base/base').id][:2],mapper.pose[:2])
