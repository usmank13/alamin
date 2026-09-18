"""Trust-boundary and portable-runtime regressions; never call a paid provider."""
import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from scene_pipeline.contracts import PipelineError,read_json,write_json
from scene_pipeline.evidence import resolve,measurement,public_address,cache_entry
from scene_pipeline.runtime import Runtime,using,request,redact


def test_prompt_axes_not_bag_of_numbers(tmp_path):
    quote='1.2 m wide, 0.7 m deep and 0.9 m high'
    item=dict(id='bench',category='bench',dimensions_m=[.7,.9,1.2],dimension_evidence=dict(prompt_quote=quote))
    with pytest.raises(PipelineError,match='quoted axis'):resolve(item,quote,cache_path=tmp_path/'cache.json')
    item['dimensions_m']=[1.2,.7,.9]
    assert resolve(item,quote,cache_path=tmp_path/'cache.json')[0]==[1.2,.7,.9]


@pytest.mark.parametrize('label,axis',[('Width','width'),('Shipping Width','width'),('Width','height')])
def test_source_fields_cannot_bind_shipping_or_another_axis(label,axis):
    with pytest.raises(PipelineError):
        measurement('<h1>Product</h1><p>Shipping Width 52 in</p>',identity='Product',label=label,number='52',unit='in',url='u',axis=axis)


def test_cache_rechecks_source_not_just_claimed_hash(tmp_path):
    from scene_pipeline.evidence import update_cache
    document='<h1>Bench</h1><p>Width 1 m</p><p>Depth 2 m</p><p>Height 3 m</p>'
    fields={a:[a.title(),str(i),'m'] for i,a in enumerate(('width','depth','height'),1)}
    path=tmp_path/'cache.json'
    resolve(dict(id='b',category='bench',family='table',dimension_evidence=dict(url='https://example.test',identity='Bench',fields=fields)),
            'bench',cache_path=path,fetcher=lambda _:document)
    entry=cache_entry('bench',path);assert entry['family']=='table'
    entry['dimensions_m'][0]=4;update_cache(path,'bench',entry)
    with pytest.raises(PipelineError,match='archived'):cache_entry('bench',path)


@pytest.mark.parametrize('url',['file:///etc/passwd','http://user:pass@example.org','http://127.0.0.1','https://[::1]','http://169.254.169.254'])
def test_private_evidence_urls_rejected(url):
    with pytest.raises(PipelineError):public_address(url)


def test_mixed_public_private_dns_rejected(monkeypatch):
    import socket
    monkeypatch.setattr(socket,'getaddrinfo',lambda *a,**k:[(2,1,6,'',('8.8.8.8',80)),(2,1,6,'',('127.0.0.1',80))])
    with pytest.raises(PipelineError,match='Private'):public_address('http://example.test')


SCHEMA=dict(type='object',properties={'count':dict(type='integer',minimum=1)},required=['count'],additionalProperties=False)


def test_recorded_runtime_validates_and_accounts(tmp_path):
    runtime=Runtime(backend='recorded',responses=[dict(kind='program',output={'count':2})])
    with using(runtime):assert request('intent',SCHEMA,tmp_path)=={'count':2}
    assert runtime.summary()['cost_usd']==0 and runtime.summary()['calls']==1
    assert read_json(tmp_path/'program.usage.json')['passed']


def test_invalid_recorded_response_cannot_bypass_schema(tmp_path):
    runtime=Runtime(backend='recorded',responses=[dict(kind='program',output={'count':0})])
    with pytest.raises(PipelineError,match='schema'):runtime.request('x',SCHEMA,tmp_path)
    assert not runtime.calls[0]['passed']


def test_deadline_and_cost_guard_before_provider(tmp_path):
    runtime=Runtime(deadline=time.monotonic()-1)
    with pytest.raises(PipelineError,match='deadline'):runtime.request('x',SCHEMA,tmp_path)
    runtime=Runtime(max_cost_usd=1,calls=[{'cost_usd':1}])
    with pytest.raises(PipelineError,match='spend'):runtime.request('x',SCHEMA,tmp_path)
    runtime.calls=[{'cost_usd':None}]
    with pytest.raises(PipelineError,match='reported costs'):runtime.request('x',SCHEMA,tmp_path)


def test_secret_redaction(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY','example-super-secret-token')
    assert redact('error example-super-secret-token')=='error [REDACTED]'


def test_codex_stdin_closed_and_transport_timeout(tmp_path,monkeypatch):
    import scene_pipeline.runtime as module
    def fail(command,**kwargs):
        assert kwargs['stdin']==subprocess.DEVNULL
        raise subprocess.TimeoutExpired(command,kwargs['timeout'])
    monkeypatch.setattr(module.subprocess,'run',fail)
    runtime=Runtime()
    with pytest.raises(PipelineError,match='wall-time'):runtime.request('x',SCHEMA,tmp_path)
    assert runtime.calls[0]['passed'] is False


def test_pydantic_adapter_offline():
    pytest.importorskip('pydantic_ai')
    from pydantic_ai.models.test import TestModel
    from scene_pipeline.openrouter_worker import execute
    reply=asyncio.run(execute(dict(instruction='Return a count',schema=SCHEMA,model='test',search=False,image=None),
                              model=TestModel(custom_output_args={'count':3})))
    assert reply['output']=={'count':3} and reply['usage']['provider_requests']==1


def test_openrouter_transport_offline():
    pytest.importorskip('pydantic_ai')
    import httpx2
    from openai import AsyncOpenAI
    from pydantic_ai.models.openrouter import OpenRouterModel
    from pydantic_ai.providers.openrouter import OpenRouterProvider
    from scene_pipeline.openrouter_worker import execute
    async def check():
        def handler(req):
            body=json.loads(req.content)
            assert body['provider']['allow_fallbacks'] is False
            tool=body['tools'][0]['function']['name']
            return httpx2.Response(200,json=dict(id='test',object='chat.completion',created=0,model='test/model',provider='offline',
                choices=[dict(index=0,finish_reason='tool_calls',message=dict(role='assistant',content=None,
                    tool_calls=[dict(id='call_test',type='function',function=dict(name=tool,arguments='{"count": 4}'))]))],
                usage=dict(prompt_tokens=10,completion_tokens=4,total_tokens=14,cost=.002)))
        async with httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) as client:
            api=AsyncOpenAI(api_key='offline-test',base_url='https://openrouter.ai/api/v1',http_client=client,max_retries=0)
            # Prime SDK metadata synchronously: this test covers HTTP, not the
            # platform probe's background-thread wakeup in restricted sandboxes.
            from openai._base_client import get_platform
            api._platform=get_platform()
            model=OpenRouterModel('test/model',provider=OpenRouterProvider(openai_client=api))
            return await asyncio.wait_for(execute(dict(instruction='Return a count',schema=SCHEMA,model='test/model',search=False,image=None),model=model),timeout=5)
    reply=asyncio.run(check())
    assert reply['output']=={'count':4}
    assert reply['usage']['cost_usd']==.002 and reply['usage']['input_tokens']==10


def test_untrusted_supplied_basis_rejected(tmp_path):
    from scene_pipeline.dsl import load
    from scene_pipeline.orchestrator import resolve_program
    program=load('examples/cafe_program.py')
    program['objects'][0].update(dimensions_m=[8,8,8],dimension_basis='sourced')
    with pytest.raises(PipelineError,match='claimed resolved basis'):resolve_program(program,tmp_path)


def test_supplied_program_offline_outside_repo(tmp_path):
    root=Path(__file__).resolve().parents[1]
    env={**os.environ,'PATH':'/nonexistent','SCENE_PIPELINE_CACHE':str(tmp_path/'cache')}
    env.pop('OPENROUTER_API_KEY',None)
    command=[sys.executable,'-m','scene_pipeline.cli','generate','--program',str(root/'examples/cafe_program.py'),
             '--prompt','Portable test','--output',str(tmp_path/'scene'),'--no-preview']
    result=subprocess.run(command,cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert read_json(tmp_path/'scene/cost.json')['usage']['calls']==0


def test_failed_scene_gallery_and_html_escaping(tmp_path):
    from scene_pipeline.inspection import scene_page,batch_page
    scene_page(tmp_path/'scene',dict(prompt='<script>bad()</script>',status='failed'))
    text=(tmp_path/'scene/index.html').read_text()
    assert '&lt;script&gt;' in text and '<script>' not in text and 'render/cycles.png' not in text
    scene_page(tmp_path/'scene')
    assert '&lt;script&gt;' in (tmp_path/'scene/index.html').read_text()
    batch_page(tmp_path,dict(runs=[dict(path='scene',domain='clinic',backend='test',model=None,status='failed',seed=1,prompt='failure')]))
    assert 'scene/index.html' in (tmp_path/'index.html').read_text()


def test_dataset_inspection_links_existing_capture_and_retains_failure(tmp_path):
    from scene_pipeline.inspection import inspect_path
    write_json(tmp_path/'dataset.json',dict(variants=[dict(variant=0,error=dict(code='LAYOUT_UNSAT')),dict(variant=1,report=dict(passed=True))]))
    write_json(tmp_path/'variant_001/mapping/report.json',dict(flow='mapping',passed=True))
    (tmp_path/'variant_001/mapping/data.h5').touch()
    result=inspect_path(tmp_path,open_browser=False)
    page=Path(result['page']).read_text()
    assert 'LAYOUT_UNSAT' in page and 'variant_001/mapping/data.h5' in page
    assert 'variant_000/mapping/data.h5' not in page


def test_dataset_retains_architecture_and_evidence(tmp_path,monkeypatch):
    from test_architecture_generation import make_bundle
    from scene_pipeline.orchestrator import generate
    from scene_pipeline import flows
    from scene_pipeline.dataset import collect_variants,Recorder
    program=read_json('examples/architecture_hall.json');program['space']['kind']='workroom'
    generate(program['prompt'],1,tmp_path/'source',program=program,preview=False,layout_backend='architecture',priors=make_bundle())
    def fake_run(scene,flow,output,**kwargs):
        Path(output).mkdir();rec=Recorder(Path(output)/'data.h5',{});rec.append('state',0.,value=1);rec.close()
        return dict(passed=True)
    monkeypatch.setattr(flows,'run',fake_run)
    report=collect_variants(tmp_path/'source',tmp_path/'variants',variants=1,seconds=.1,start_seed=101)
    assert report['passed'],report
    assert report['variants'][0]['factors']['layout_seed']==101
    assert read_json(tmp_path/'variants/variant_000/generation.json')['layout_backend']=='architecture'
    assert read_json(tmp_path/'variants/variant_000/validation.json')['acceptance_profile']=='scene_physics_no_articulated_quota'


def test_evaluation_resume_and_independent_intent(tmp_path,monkeypatch):
    from test_architecture_generation import make_bundle
    from scene_pipeline.evaluation import evaluate
    manifest=read_json('examples/evaluation.json');manifest['cases']=manifest['cases'][:1]
    # Deliberately incorrect externally authored expectation must not pass merely because physics does.
    manifest['cases'][0]['expected_counts']['jar']=999
    report=evaluate(manifest,tmp_path/'eval',priors=make_bundle(),seeds=[7],cycles=False)
    assert all(r['status']!='validated' for r in report['runs'])
    assert any(r.get('scores',{}).get('intent_passed') is False for r in report['runs'])
    again=evaluate(manifest,tmp_path/'eval',priors=make_bundle(),seeds=[7],cycles=False,resume=True)
    assert report==again
    manifest['cases'][0]['expected_counts']['jar']=6
    with pytest.raises(PipelineError,match='changed'):evaluate(manifest,tmp_path/'eval',priors=make_bundle(),seeds=[7],resume=True)


def test_live_evaluation_needs_explicit_models_and_budget(tmp_path):
    from scene_pipeline.evaluation import evaluate
    manifest=read_json('examples/evaluation.json')
    with pytest.raises(PipelineError,match='two distinct'):evaluate(manifest,tmp_path,live=True)
    with pytest.raises(PipelineError,match='spend cap'):evaluate(manifest,tmp_path,live=True,models=['a','b'])


def test_representative_domains_not_mechanical_fixtures(tmp_path):
    from scene_pipeline.evaluation import evaluate,validate_manifest,score
    manifest=read_json('examples/evaluation_domains.json');validate_manifest(manifest)
    with pytest.raises(PipelineError,match='live-only'):evaluate(manifest,tmp_path)
    case=next(c for c in manifest['cases'] if c['id']=='warehouse')
    objects=[dict(id=f'{category}_{i}',category=category,dimensions=[1,1,1])
             for category,count in case['expected_counts'].items() for i in range(count)]
    write_json(tmp_path/'ir.json',dict(objects=objects,meta=dict(area_m2=100),provenance={}))
    write_json(tmp_path/'manifest.json',dict(instances={o['id']:dict(category=o['category'],affordances=[],provenance=dict(kind='sourced_template')) for o in objects}))
    write_json(tmp_path/'validation.json',dict(passed=True))
    scored=score(tmp_path,case)
    assert scored['intent_passed'] and scored['physics_passed']
    assert {c['capability'] for c in scored['capability_coverage']}=={'pallet_support','pallet_engagement','loaded_lift','mobile_cart'}
    assert all(c['status']=='unsupported_or_unverified' and not c['required'] for c in scored['capability_coverage'])
    assert not scored['missing_capabilities']
    case=deepcopy(case);case['required_capabilities']=dict(pallet_jack=['loaded_lift'])
    assert score(tmp_path,case)['missing_capabilities'][0]['capability']=='loaded_lift'
    assert len(scored['proxy_instances'])==len(objects)


def test_flow_existing_output_is_not_modified(tmp_path):
    from scene_pipeline.flows import run
    write_json(tmp_path/'agent_calls.json',[{'original':True}])
    with pytest.raises(PipelineError,match=str(tmp_path)):run('missing','mapping',tmp_path)
    assert read_json(tmp_path/'agent_calls.json')==[{'original':True}]


def test_semantic_collision_detected(tmp_path):
    from scene_pipeline.dsl import load
    from scene_pipeline.layout import solve
    from scene_pipeline.compiler import compile_scene
    program=load('examples/cafe_program.py');ir=solve(program,1,tmp_path)
    a,b=next((a,b) for a in ir['objects'] for b in ir['objects'] if a['category']!=b['category'])
    b['class_id']=a['class_id']
    with pytest.raises(PipelineError,match='class ID'):compile_scene(ir,tmp_path)
