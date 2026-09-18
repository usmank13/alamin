"""Frozen cross-domain experiments. Never tune or repair code against live results."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import math
import os
import re
import time

from .contracts import PipelineError,digest,read_json,write_json,validate_program
from .inspection import scene_page,batch_page
from .runtime import Runtime,redact


def validate_manifest(manifest):
    if manifest.get('schema_version')!=1 or not manifest.get('cases'):
        raise PipelineError('EVALUATION_MANIFEST','Expected schema_version 1 and cases')
    ids=[]
    for case in manifest['cases']:
        if not re.fullmatch('[a-z][a-z0-9_]*',case['id']):raise PipelineError('EVALUATION_ID','Unsafe case identifier')
        ids.append(case['id'])
        if 'program' in case:
            validate_program(case['program'])
            if case['program']['prompt']!=case['prompt']:raise PipelineError('EVALUATION_PROMPT','Program/prompt mismatch')
        elif not 30<=case.get('expected_area_m2',0)<=120:
            raise PipelineError('EVALUATION_EXPECTATIONS','Live-only cases require expected_area_m2')
        if not isinstance(case.get('expected_counts'),dict):raise PipelineError('EVALUATION_EXPECTATIONS','Independent expected counts required')
    if len(ids)!=len(set(ids)):raise PipelineError('EVALUATION_ID','Duplicate cases')


def program_for(case,backend):
    program=deepcopy(case['program'])
    if backend=='heuristic':
        program['schema_version']=1;program.pop('architecture',None)
    return program


def score(scene,case):
    ir=read_json(scene/'ir.json');counts=Counter(o['category'] for o in ir['objects'])
    mismatches={k:dict(expected=v,actual=counts[k]) for k,v in case['expected_counts'].items() if counts[k]!=v}
    checks=read_json(scene/'validation.json') if (scene/'validation.json').exists() else {}
    manifest=read_json(scene/'manifest.json')
    proxies=[k for k,v in manifest['instances'].items() if v.get('provenance',{}).get('kind')=='sourced_template']
    dimensions=[]
    for category,wanted in case.get('expected_dimensions_m',{}).items():
        for obj in ir['objects']:
            if obj['category']==category and any(abs(a-b)>1e-6 for a,b in zip(obj['dimensions'],wanted)):
                dimensions.append(obj['id'])
    area_ok=abs(ir['meta']['area_m2']-case.get('expected_area_m2',case.get('program',{}).get('space',{}).get('area_m2',0)))<1e-6
    # Only known implemented capabilities can be credited. A requested label on a
    # grey box cannot establish machining, conveying, washing, seating, etc.
    missing_capabilities=[];capability_coverage=[]
    probes={k:list(v) for k,v in case.get('capability_probes',{}).items()}
    for category,capabilities in case.get('required_capabilities',{}).items():
        probes.setdefault(category,[]).extend(c for c in capabilities if c not in probes.get(category,[]))
    for category,capabilities in probes.items():
        instances=[o for o in manifest['instances'].values() if o['category']==category]
        for capability in capabilities:
            roles={'hinged_access':'door','sliding_access':'drawer'}
            covered=capability in roles and bool(instances) and all(any(a.get('role')==roles[capability] for a in o.get('affordances',[])) for o in instances)
            required=capability in case.get('required_capabilities',{}).get(category,[])
            entry=dict(category=category,capability=capability,required=required,
                       status='affordance_present_task_unverified' if covered else 'unsupported_or_unverified')
            capability_coverage.append(entry)
            if required and not covered:missing_capabilities.append(entry)
    return dict(intent_passed=not mismatches and not dimensions and area_ok,count_mismatches=mismatches,
                dimension_mismatches=dimensions,area_passed=area_ok,physics_passed=checks.get('passed',False),
                checks=checks.get('checks',{}),validation_profile=checks.get('acceptance_profile'),
                proxy_instances=proxies,articulated_objects=checks.get('articulated_objects'),
                functional_suitability='unverified',missing_capabilities=missing_capabilities,
                capability_coverage=capability_coverage,density=checks.get('density'),
                conditioning=ir['provenance'].get('architecture_sampling',{}).get('conditioning'),
                empirical_furnishings=False)


def evaluate(manifest,output,*,priors=None,models=None,agent_backend='openrouter',seeds=(7,11),
             live=False,max_cost_usd=None,timeout=900,cache_snapshot=None,resume=False,cycles=None,flows=False,
             interactions=False):
    from .orchestrator import generate
    validate_manifest(manifest);output=Path(output)
    if not live and any('program' not in c for c in manifest['cases']):
        raise PipelineError('EVALUATION_MODE','Representative domain prompts are live-only; use evaluation.json for mechanical fixtures')
    if live and (not models or len(models)!=2 or len(set(models))!=2):
        raise PipelineError('EVALUATION_MODELS','Live smoke requires two distinct explicit model IDs')
    if live and (max_cost_usd is None or not math.isfinite(max_cost_usd) or max_cost_usd<=0):
        raise PipelineError('COST_BUDGET','Live evaluation requires a positive spend cap')
    if live and agent_backend=='openrouter' and not os.getenv('OPENROUTER_API_KEY'):
        raise PipelineError('AGENT_CREDENTIALS','Set OPENROUTER_API_KEY before live evaluation')
    if priors is None:raise PipelineError('PRIOR_REQUIRED','Supply the frozen architecture prior bundle; no implicit download or synthetic empirical claim')
    seeds=list(seeds[:1] if live else seeds);cycles=live if cycles is None else cycles
    configuration=dict(manifest=manifest,prior_sha256=priors['sha256'],models=models if live else None,
        seeds=seeds,live=live,agent_backend=agent_backend,timeout=timeout,max_cost_usd=max_cost_usd,
        cache_sha256=digest(cache_snapshot or {}),cycles=cycles,flows=flows,interactions=interactions,
        implementation_sha256=digest({p.name:digest(p.read_text()) for p in sorted(Path(__file__).parent.glob('*.py'))}))
    key=digest(configuration)
    if output.exists():
        if not resume:raise PipelineError('OUTPUT_EXISTS','Use --resume for an unchanged frozen evaluation')
        if not (output/'evaluation_manifest.json').exists() or read_json(output/'evaluation_manifest.json')['sha256']!=key:
            raise PipelineError('EVALUATION_DRIFT','Manifest, evidence, code or settings changed; use a new output directory')
    else:
        output.mkdir(parents=True);write_json(output/'evaluation_manifest.json',dict(sha256=key,configuration=configuration))
        write_json(output/'priors.json',priors);write_json(output/'initial_cache.json',cache_snapshot or {})
    combinations=[(model,'architecture') for model in models] if live else [(None,'heuristic'),(None,'architecture')]
    report=dict(schema_version=1,mode='live_prompt_smoke' if live else 'fixed_program_regression',runs=[],passed=True,
                limitations=['No domain realism certification','Generic backoff is not domain-specific empirical evidence',
                             'No distribution pass threshold','Client budget stops between calls; not a hard billing cap'])
    spent=0.;cost_unknown=False;interaction_domains=set()
    for case in manifest['cases']:
        for model,backend in combinations:
            for seed in seeds:
                identity=f'{case["id"]}_{backend}_{digest(model)[:8]}_{seed}';root=output/'runs'/identity
                result_path=root/'run.json'
                if result_path.exists():
                    result=read_json(result_path)
                    if 'interaction' in result.get('stages',{}):interaction_domains.add(case['domain'])
                else:
                    root.mkdir(parents=True,exist_ok=True);attempt=0
                    while (root/f'scene_{attempt}').exists():attempt+=1
                    scene=root/f'scene_{attempt}';cache=root/f'cache_{attempt}.json';write_json(cache,cache_snapshot or {})
                    result=dict(id=identity,path=str(scene.relative_to(output)),domain=case['domain'],prompt=case['prompt'],
                                seed=seed,model=model,backend=backend,status='failed',stages={})
                    started=time.monotonic()
                    try:
                        if live and (cost_unknown or spent>=max_cost_usd):raise PipelineError('BUDGET_EXHAUSTED','Batch cost limit reached or previous usage unknown')
                        runtime=Runtime(backend=agent_backend,model=model,max_cost_usd=max_cost_usd-spent if live else None)
                        generated=generate(case['prompt'],seed,scene,program=None if live else program_for(case,backend),
                            runtime=runtime,model=model,max_iterations=3,timeout=timeout,preview=True,layout_backend=backend,
                            priors=priors if backend=='architecture' else None,allow_prior_backoff=True,robot_radius=.25,cache_path=cache)
                        result['status']=generated['status'];result['scores']=score(scene,case)
                        if not result['scores']['intent_passed']:result['status']='intent_failed'
                        elif result['scores']['missing_capabilities'] and result['status']=='validated':result['status']='capability_gap'
                        if result['status']=='validated':
                            from .portability import export_urdf,verify_urdf
                            result['stages']['export']=verify_urdf(export_urdf(scene))
                            if cycles:
                                from .render import cycles as render_cycles
                                result['stages']['cycles']=dict(image=str(render_cycles(scene)),passed=True)
                            if flows:
                                from .flows import run
                                result['stages']['mapping']=run(scene,'mapping',scene/'mapping',seconds=1,tier='state')
                            if interactions and case['domain'] not in interaction_domains:
                                interaction_domains.add(case['domain'])
                                if any(o['category']=='drawer_unit' for o in read_json(scene/'ir.json')['objects']):
                                    from .flows import run
                                    result['stages']['interaction']=run(scene,'interaction',scene/'interaction',seconds=60,tier='full')
                                else:result['stages']['interaction']=dict(status='unsupported',reason='No supported drawer target',passed=False)
                        if any(s.get('passed') is False for s in result['stages'].values()):result['status']='downstream_failed'
                    except PipelineError as exc:
                        result['error']=exc.as_dict()
                        if result['status']=='validated':result['status']='downstream_failed'
                    except Exception as exc:
                        result.update(status='tool_error',error=dict(code='TOOL_ERROR',message=redact(str(exc))))
                    result['seconds']=time.monotonic()-started
                    cost=read_json(scene/'cost.json') if (scene/'cost.json').exists() else {}
                    result['usage']=cost.get('usage',dict(cost_usd=0.,calls=0))
                    # If compilation preceded a failure, keep that diagnostic scene inspectable too.
                    if not (scene/'scene.mjz').exists():
                        compiled=sorted(scene.glob('attempts/*/scene.mjz'))
                        if compiled:
                            result['diagnostic_scene']=str(compiled[-1].parent.relative_to(output))
                            from .render import preview
                            try:preview(compiled[-1].parent)
                            except Exception as exc:result['preview_error']=redact(str(exc))
                    scene_page(scene,result);write_json(result_path,result)
                report['runs'].append(result)
                usage=result.get('usage',{});cost=usage.get('cost_usd')
                if cost is None and usage.get('calls'):cost_unknown=True
                if cost is not None:spent+=cost
                report.update(passed=all(r['status']=='validated' for r in report['runs']),known_cost_usd=spent,cost_complete=not cost_unknown)
                write_json(output/'evaluation.json',report);batch_page(output,report)
    return report
