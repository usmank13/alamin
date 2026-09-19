"""Small, offline recovery regressions. Recorded agents are not model benchmarks."""
from copy import deepcopy

import pytest

from scene_pipeline import orchestrator
from scene_pipeline.contracts import PipelineError,read_json
from scene_pipeline.dsl import load,parse
from scene_pipeline.runtime import Runtime
from scene_pipeline.scene_intent import freeze,check_revision


def proposal():
    program=load('examples/cafe_program.py')
    program['space']['inferred_fields']=['area_m2','shape','annexes']
    return program


def test_agent_can_change_inferred_space_and_asset_but_not_inventory():
    program=proposal();program['objects'][0]['asset_request']=dict(query='work table')
    intent=freeze(program,authority='agent_program_not_independently_verified')
    repair=deepcopy(program);repair['space']['area_m2']=60
    repair['objects'][0]['asset_request']['query']='preparation bench'
    check_revision(intent,repair)
    for field,value in [('count',2),('category','shelf'),('required',False),('zone','elsewhere')]:
        bad=deepcopy(repair);bad['objects'][0][field]=value
        with pytest.raises(PipelineError,match='frozen'):check_revision(intent,bad)
    repair['space']['kind']='warehouse'
    repair['space']['inferred_fields'].append('kind')  # Cannot retroactively unfreeze a requirement.
    with pytest.raises(PipelineError,match='space requirement'):check_revision(intent,repair)


def test_supplied_program_and_unannotated_space_remain_fixed():
    program=proposal();intent=freeze(program,authority='supplied_program')
    repair=deepcopy(program);repair['space']['area_m2']=60
    with pytest.raises(PipelineError,match='space requirement'):check_revision(intent,repair)
    replacement=deepcopy(program);replacement['objects'][0]['asset_ref']='a'*64
    with pytest.raises(PipelineError,match='frozen'):check_revision(intent,replacement)
    program['space'].pop('inferred_fields');intent=freeze(program,authority='agent_program_not_independently_verified')
    with pytest.raises(PipelineError,match='space requirement'):check_revision(intent,repair)


def test_quoted_dimensions_and_required_relations_cannot_be_repaired_away():
    p=parse('''scene("A 1.2 m wide table",space(),[
        place("table","prep_table",dimensions_m=[1.2,.8,.9],dimension_evidence={"prompt_quote":"1.2 m wide"})],
        [relation("against_wall",["table"],required=True)])''')
    intent=freeze(p,authority='agent_program_not_independently_verified')
    bad=deepcopy(p);bad['objects'][0].pop('dimension_evidence');bad['objects'][0].pop('dimensions_m')
    with pytest.raises(PipelineError,match='frozen'):check_revision(intent,bad)
    bad=deepcopy(p);bad['relations']=[]
    with pytest.raises(PipelineError,match='relation'):check_revision(intent,bad)


def test_patch_preserves_untouched_objects_and_sections():
    p=proposal();before=deepcopy(p)
    changed=dict(p['objects'][0],zone='work')
    added=dict(id='extra',category='shelf',count=1,zone='main',required=False)
    result=orchestrator.apply_repair(p,dict(objects=[changed,added]))
    assert p==before and result['objects'][1:-1]==p['objects'][1:]
    assert result['objects'][0]==changed and result['objects'][-1]==added
    assert result['space']==p['space'] and result['relations']==p['relations']
    with pytest.raises(PipelineError,match='remove'):orchestrator.apply_repair(p,dict(remove_objects=['unknown']))


def test_partial_resolution_reuses_success_and_remembers_rejected_queries(tmp_path,monkeypatch):
    from scene_pipeline import asset_library
    from test_open_vocabulary import DOC,FIELDS
    calls=[];fetches=[]
    def rejected(item,*args,**kwargs):
        calls.append(item['asset_request']['query'])
        raise PipelineError('ASSET_NO_MATCH','No suitable candidate',dict(query=calls[-1]))
    monkeypatch.setattr(asset_library,'resolve_request',rejected)
    p=proposal();p['objects']=[dict(id='table',category='sterilizer',count=1,zone='main',required=True,family='table',
        dimension_evidence=dict(url='https://example.test/product',identity='Steri-500',fields=FIELDS)),
        dict(id='machine',category='machine',count=1,zone='main',required=True,asset_request=dict(query='first query'))]
    cache={}
    def fetcher(url):fetches.append(url);return DOC
    for attempt in range(3):
        if attempt==2:p['objects'][1]['asset_request']['query']='different query'
        with pytest.raises(PipelineError,match='suitable') as error:
            orchestrator.resolve_program(p,tmp_path/str(attempt),resolution_cache=cache,cache_path=tmp_path/'evidence.json',fetcher=fetcher)
        assert error.value.details['object_id']=='machine'
        assert read_json(tmp_path/str(attempt)/'dimension_sources.json')['table']
    assert len(fetches)==1 and calls==['first query','different query']


def test_heuristic_retries_locally_without_changing_program(tmp_path,monkeypatch):
    from scene_pipeline import layout
    from scene_pipeline.generators import generate_layout
    real=layout.solve;seeds=[];p=proposal();before=deepcopy(p)
    def flaky(program,seed,root,**kwargs):
        seeds.append(seed)
        if len(seeds)==1:raise PipelineError('LAYOUT_UNSAT','Unlucky first placement')
        return real(program,seed,root,**kwargs)
    monkeypatch.setattr(layout,'solve',flaky)
    result=generate_layout(p,7,tmp_path)
    assert seeds==[7,8] and p==before
    assert result['provenance']['layout_generator']['selected_seed']==8
    assert len(read_json(tmp_path/'layout_candidates.json'))==2


def test_layout_retry_does_not_hide_missing_asset(tmp_path,monkeypatch):
    from scene_pipeline import layout
    from scene_pipeline.generators import generate_layout
    calls=[]
    def missing(*args,**kwargs):
        calls.append(1);raise PipelineError('MISSING_SOURCE','Unavailable source')
    monkeypatch.setattr(layout,'solve',missing)
    with pytest.raises(PipelineError,match='Unavailable'):generate_layout(proposal(),1,tmp_path)
    assert calls==[1]


def test_impossible_fixture_blocks_paid_generation(tmp_path,monkeypatch):
    p=parse('''scene("A table 10 m wide, 10 m deep and 1 m high",space(area_m2=30),[
        place("table","large_table",family="table",dimensions_m=[10,10,1],
              dimension_evidence={"prompt_quote":"10 m wide, 10 m deep and 1 m high"})])''')
    monkeypatch.setattr(orchestrator.fal,'prefetch',lambda *a,**k:pytest.fail('Must check fixture fit before spending'))
    with pytest.raises(PipelineError) as error:
        orchestrator.generate(p['prompt'],0,tmp_path/'scene',program=p,materials='fal',preview=False)
    assert error.value.details['attempts'][0]['error']['code']=='LAYOUT_UNSAT'
    assert read_json(tmp_path/'scene/cost.json')['fal_spend_usd']==0


def test_generated_support_is_checked_without_fake_geometry(tmp_path):
    p=proposal();p['objects']=[dict(id='decor',category='novel_decor',count=1,zone='main',required=True,
        generated_request=dict(prompt='a decorative prop',size_m=.2,placement='support',physical_use='visual_only'))]
    with pytest.raises(PipelineError,match='physical support'):orchestrator.preflight_layout(p,1,tmp_path)
    p['objects'].append(dict(id='table',category='prep_table',count=1,zone='main',required=True))
    report=orchestrator.preflight_layout(p,1,tmp_path)
    assert report['status']=='passed' and not report['generated_geometry_checked']
    assert report['deferred_objects']==['decor'] and report['placed_objects']==1


@pytest.mark.parametrize('domain',['kitchen','warehouse','workshop'])
@pytest.mark.parametrize('seed',[1,7])
def test_recorded_repair_completes_across_seeds_and_domain_labels(tmp_path,monkeypatch,domain,seed):
    # Fresh per-test stores, a real layout/compiler/physics path, but scripted
    # semantics: this measures recovery mechanics, not domain realism or LLM skill.
    import trimesh
    from scene_pipeline import fal
    from scene_pipeline.asset_library import register_generated
    from test_materials import FakeFal
    store=tmp_path/'library'
    monkeypatch.setenv('SCENE_PIPELINE_ASSET_STORE',str(store))
    monkeypatch.setenv('FAL_KEY','offline-test-key')
    monkeypatch.setattr(fal,'ROOT',tmp_path/'fal')
    request=dict(prompt='a long visual prop',size_m=.8,placement='support',physical_use='visual_only')
    fake=FakeFal(glb=trimesh.creation.box(extents=[.8,.1,.2]).export(file_type='glb'))
    fal.prefetch(fal.decor_jobs({'objects':[dict(category='long_prop',generated_request=request)]}),
                 http=fake.http,download=fake.download,interval=0)
    _,asset=register_generated('long_prop',request,store=store)
    monkeypatch.setattr(fal,'_http',lambda *a,**k:pytest.fail('No live provider calls'))
    p=proposal();p['prompt']=f'A {domain} with storage and a work surface.';p['space']['kind']=domain
    p['objects'].append(dict(id='prop',category='long_prop',count=1,zone='main',required=True,asset_ref=asset['key']))
    p['objects']=[o for o in p['objects'] if o['id'] not in ('prep','storage','goods')]
    # The .8 m prop cannot fit any .52 m cabinet surface, even rotated. Adding
    # a table repairs the real failure; the layout and validators are not mocked.
    runtime=Runtime(backend='recorded',responses=[dict(kind='program',output=p),dict(kind='repair',output=dict(objects=[
        dict(id='table',category='prep_table',count=1,zone='main',required=False)]))])
    root=tmp_path/'scene'
    result=orchestrator.generate(p['prompt'],seed,root,runtime=runtime,preview=False,max_iterations=3,cache_path=tmp_path/'evidence.json')
    assert result['passed'] and len(result['attempts'])==2
    assert [call['kind'] for call in runtime.calls]==['program','repair']
    assert read_json(root/'program.json')['objects'][:len(p['objects'])]==p['objects']
    assert read_json(root/'attempts/1/repair.json')['objects'][0]['id']=='table'
    assert result['fal_spend_usd']==0


def test_physics_validation_failure_repairs_until_success(tmp_path,capsys):
    p=parse('''scene("storage room",space(area_m2=40),[
        place("cab","base_cabinet",count=2),place("prep","prep_table")])''')
    p['space']['inferred_fields']=[]
    runtime=Runtime(backend='recorded',responses=[dict(kind='program',output=p),
        dict(kind='repair',output={}),dict(kind='repair',output=dict(objects=[
            dict(id='drawers',category='drawer_unit',count=3,zone='main',required=False)]))])
    root=tmp_path/'scene'
    result=orchestrator.generate(p['prompt'],1,root,runtime=runtime,preview=False,max_iterations=5)
    assert result['passed'] and result['attempts_used']==3
    assert result['stop_reason']=='validated'
    assert [a['seed'] for a in result['attempts']]==[1,4,7]
    assert all(a['error']['code']=='VALIDATION_FAILED' for a in result['attempts'][:2])
    assert read_json(root/'validation.json')['passed']
    progress=read_json(root/'progress.json')
    assert progress['attempt']==3 and progress['max_attempts']==5 and progress['status']=='validated'
    assert progress['last_failure'] is None
    captured=capsys.readouterr()
    assert not captured.out  # Keep CLI stdout available for the final JSON result.
    assert 'Attempt 3/5' in captured.err


def test_transient_codex_failures_retry_up_to_limit(tmp_path,monkeypatch):
    contexts=[]
    def failed(prompt,work,feedback,*args,**kwargs):
        contexts.append(feedback)
        raise PipelineError('AGENT_FAILED','Temporary worker failure')
    monkeypatch.setattr(orchestrator,'agent_program',failed)
    with pytest.raises(PipelineError) as error:
        orchestrator.generate('room',0,tmp_path/'scene',max_iterations=3,preview=False)
    assert len(contexts)==3 and len(contexts[2]['failure_history'])==2
    assert error.value.details['attempts_used']==3
    assert read_json(tmp_path/'scene/progress.json')['stop_reason']=='attempt_limit'


@pytest.mark.parametrize('code',['AGENT_CREDENTIALS','AGENT_REQUEST_INVALID','COST_UNKNOWN','BUDGET_EXHAUSTED'])
def test_unrecoverable_errors_stop_and_explain_why(tmp_path,monkeypatch,code):
    def failed(*args,**kwargs):raise PipelineError(code,'Cannot continue')
    monkeypatch.setattr(orchestrator,'agent_program',failed)
    with pytest.raises(PipelineError) as error:
        orchestrator.generate('room',0,tmp_path/'scene',max_iterations=5,preview=False)
    assert error.value.details['attempts_used']==1
    assert error.value.details['stop_reason']==code
    assert read_json(tmp_path/'scene/progress.json')['stop_reason']==code
