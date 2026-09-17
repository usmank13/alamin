"""Bounded agent loop around deterministic scene tools; no scene-file editing."""
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from copy import deepcopy

from . import VERSION
from .contracts import PipelineError,PROGRAM_SCHEMA,PROGRAM_SCHEMA_V2,FAMILY,FIELD,obj,validate_program,write_json,read_json,digest
from .registry import search,fingerprint,CATALOG
from .evidence import resolve,fetch,CACHE
from . import fal
from .generators import generate_layout
from .scene_intent import freeze,check_revision
from .scene_checks import check_scene
from .compiler import compile_scene
from .validation import validate_scene


OPEN_VOCABULARY=('Prefer categories in this registry. For content the registry lacks, keep the user\'s own category name as lowercase_snake, '
    'choose the closest template family (box, table, shelf, cabinet, drawer) and, if the user stated its size, give dimensions_m with '
    'dimension_evidence.prompt_quote copied verbatim from the prompt; otherwise omit dimensions and the harness sources and verifies them. '
    'Do the same for a registry category whose size the user stated. Never supply dimensions without evidence. ')


def strict(schema):
    """Agent-facing copy for OpenAI strict structured outputs: anyOf only, every property required, optional -> nullable."""
    if isinstance(schema,list):return [strict(s) for s in schema]
    if not isinstance(schema,dict):return schema
    out={('anyOf' if k=='oneOf' else k):strict(v) for k,v in schema.items()}
    if out.get('type')=='object' and 'properties' in out:
        required=set(out.get('required',[]))
        for key,prop in out['properties'].items():
            if key not in required:
                options=prop['anyOf'] if list(prop)==['anyOf'] else [prop]
                out['properties'][key]={'anyOf':options+[{'type':'null'}]}
        out['required']=list(out['properties'])
    return out


def _strip_null(value):
    if isinstance(value,dict):return {k:_strip_null(v) for k,v in value.items() if v is not None}
    if isinstance(value,list):return [_strip_null(v) for v in value]
    return value


def _codex(instruction,schema,work,model,timeout,*,name='program',search=False):
    """One bounded structured-output call. Read-only sandbox; web search is a server-side tool."""
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    write_json(work/f'{name}.schema.json',strict(schema))
    command=['codex']+(['--search'] if search else [])+['exec']+(['--model',model] if model else [])+[
        '--sandbox','read-only','--skip-git-repo-check','--ignore-user-config','--ephemeral','--json',
        '--output-schema',str((work/f'{name}.schema.json').resolve()),'--output-last-message',str((work/f'{name}.json').resolve()),
        '--cd',str(work.resolve()),instruction]
    try:
        # Closed stdin: Codex otherwise waits on an inherited pipe ("Reading additional input
        # from stdin") and an unattended run hangs to the timeout with no trajectory.
        result=subprocess.run(command,capture_output=True,text=True,timeout=timeout,stdin=subprocess.DEVNULL)
    except (FileNotFoundError,subprocess.TimeoutExpired) as exc:
        raise PipelineError('AGENT_UNAVAILABLE',str(exc)) from exc
    (work/'trajectory.jsonl').write_text(result.stdout)
    (work/'agent.stderr.log').write_text(result.stderr)
    if result.returncode or not (work/f'{name}.json').exists():
        raise PipelineError('AGENT_FAILED','Agent failed; see attempt trajectory and stderr',{'returncode':result.returncode})
    return _strip_null(read_json(work/f'{name}.json'))


def source_dimensions(category,prompt,work,model=None,timeout=180,feedback=None):
    """Web-search call whose output is evidence only; the harness fetches the page and binds every number."""
    schema=obj({'url':{'type':'string','pattern':'^https?://'},'identity':{'type':'string','minLength':1},
                'fields':obj({'width':FIELD,'depth':FIELD,'height':FIELD}),'family':FAMILY})
    instruction=(f'Find one public manufacturer specification or standards page stating the overall width, depth and height of a typical '
        f'"{category}" as it would appear in this scene: {prompt}\nReturn only JSON matching the schema: the page URL; an identity string that '
        'appears verbatim on the page; and for width, depth and height the label, number and unit as three strings exactly as printed on the '
        'page, e.g. ["Overall Width","30","in"]. The harness fetches the page and rejects any field whose label, number and unit do not occur '
        'together verbatim, so do not compute, convert, round or estimate. Units may be written as printed (in, mm, cm, m, ft, ″, "). '
        'Prefer a plain HTML specification page over a PDF or a script-rendered page. Also pick the closest template family: box (static solid), '
        'table (top on legs), shelf (open levels), cabinet (hinged door), drawer (sliding drawer).'
        +(f'\nThe previous evidence was rejected by the verifier: {feedback}. Choose another page or copy the fields exactly.' if feedback else ''))
    return _codex(instruction,schema,work,model,timeout,name='dimension_evidence',search=True)


def resolve_program(program,work,*,model=None,timeout=180,sourcing=source_dimensions,cache_path=CACHE,fetcher=fetch):
    """Bind every open-vocabulary or user-stated dimension to evidence before intent is frozen.

    Resolution order: user quote in the prompt, dimension cache, harness-verified web page,
    then honest UNKNOWN_CATEGORY at layout. Resolved programs carry dimensions_m plus a
    basis and no raw evidence, so repairs compare equal when the evidence is unchanged.
    """
    resolved=deepcopy(program);sources={}
    for item in resolved['objects']:
        known=item['category'] in CATALOG
        if known:item.pop('family',None)  # registry categories own their family; agent noise here must not count as drift
        if item.get('dimension_basis') or (known and item.get('dimension_evidence') is None):continue
        default=CATALOG[item['category']].get('dimensions') if known else None
        def source_and_verify(tries=3):
            """Verified sourcing with the verifier's rejection fed back; populates the cache for the category."""
            feedback=None
            for attempt in range(tries):
                evidence=sourcing(item['category'],program['prompt'],Path(work)/'sourcing'/f"{item['category']}_{attempt}",model=model,timeout=timeout,feedback=feedback)
                family=evidence.pop('family',None)
                try:
                    found=resolve(dict(item,dimension_evidence=evidence),program['prompt'],cache_path=cache_path,fetcher=fetcher)
                except PipelineError as exc:
                    if exc.code not in ('UNIT','UNBOUND_MEASUREMENT','SOURCE_IDENTITY','EVIDENCE_FETCH','DIMENSION'):raise
                    feedback=exc.as_dict();continue
                if family:item.setdefault('family',family)
                return found
            raise PipelineError('DIMENSION_UNSOURCED',f"No verifiable dimensions for {item['category']} after {tries} sourcing attempts",feedback)
        try:found=resolve(item,program['prompt'],default=default,cache_path=cache_path,fetcher=fetcher)
        except PipelineError as exc:
            # User quoted some axes; source the category once so the rest come from a verified page.
            if known or sourcing is None or exc.code!='UNBOUND_MEASUREMENT' or 'unbound_axes' not in exc.details:raise
            source_and_verify()
            found=resolve(item,program['prompt'],cache_path=cache_path,fetcher=fetcher)
        if found is None and not known and sourcing is not None:
            found=source_and_verify()
        if found is None:continue
        item['dimensions_m'],record=found;item['dimension_basis']=record['basis'];item.pop('dimension_evidence',None)
        if not known:item.setdefault('family',record.get('family','box'))
        sources[item['id']]=record
    write_json(Path(work)/'dimension_sources.json',sources)
    return validate_program(resolved)


def agent_program(prompt,work,feedback=None,model=None,timeout=180,*,layout_backend='heuristic',architecture_only=False,clutter='off'):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    registry=search(routes=None if clutter=='fal' else ('G1','G3'))  # decor is offered only when its fal route is enabled
    schema=deepcopy(PROGRAM_SCHEMA_V2 if layout_backend=='architecture' else PROGRAM_SCHEMA)
    if layout_backend=='architecture':schema['properties']['architecture']['required'].append('unsupported_requirements')
    schema['properties']['objects']['items']['properties'].pop('dimension_basis')  # harness-owned; the agent supplies evidence only
    instruction=(
        'Produce only a SceneProgram JSON matching the supplied schema. You select categories, counts, zones and relations; '
        'deterministic tools compute geometry and placement. Do not execute shell commands, modify files or invent product measurements. '
        'Required means user-required; optional omissions must not change the requested purpose. Include populated storage/clutter where requested. '
        'Preserve every explicitly requested category and count across repairs. Do not reduce the inventory to bypass layout failures. '
        'Express the arrangement the user describes with relations (against_wall, near, in_row, under); mark one required only when the user requires it. '
        'Support placement for microwave and small objects is automatic; countertop microwave does not require a near relation to the counter category. '
        'For a robotics acceptance scene, request at least five articulated object instances when consistent with the prompt. '
        +OPEN_VOCABULARY+
        'Use space.area_m2 from the user or the explicit engineering area default of 60. '
        'Do not claim a walk-in or washing station is present by relabeling a cabinet. '
        f'Registry: {registry}\nPrompt: {prompt}\nLast deterministic failure: {feedback}\n'
    )
    if layout_backend=='architecture':
        instruction=(
            'Produce SceneProgram v2 JSON matching the schema. You supply semantics and relationships; '
            'tools supply geometry from a trusted empirical corpus. Do not execute commands or supply coordinates, '
            'source IDs, measurements, or fitted parameters. Area is user-specified or the labeled engineering default 60 m2. '
            'Use an open-ended space.kind label, shape sampled unless specified, and annexes 0. '
            'Describe requested doors/windows with IDs, counts and semantic roles. Same_wall and opposite_wall '
            'are available geometric relationships. Explicit user requirements have origin explicit and required true; '
            'your inferred preferences have origin inferred and required false. Never invent explicit requirements. '
            'Do not weaken explicit requirements during repair. Required unsupported semantics must be reported as '
            'unsupported rather than replaced with a different meaning: put them verbatim in architecture.unsupported_requirements; '
            'use an empty list when all requirements are expressible. Door/window absolute dimensions and multi-room '
            'requirements are not supported in this version. No room-specific placement rules are supplied. '
            +('This command builds architecture only: objects and relations must be empty. ' if architecture_only else
              f'For requested furnishings, select asset categories from {registry}. '+OPEN_VOCABULARY)
            +f'Prompt: {prompt}\nLast deterministic failure: {feedback}\n')
    program=_codex(instruction,schema,work,model,timeout)
    for item in program.get('objects',[]):item.pop('dimension_basis',None)
    return validate_program(program)


def generate(prompt,seed,output,*,program=None,model=None,max_iterations=20,timeout=900,preview=True,
             layout_backend='heuristic',priors=None,allow_prior_backoff=False,robot_radius=None,access_margin=.05,architecture_only=False,
             materials='flat',clutter='off',max_fal_usd=None):
    output=Path(output)
    if materials not in ('flat','fal') or clutter not in ('off','fal'):raise PipelineError('OPTION','materials must be flat|fal and clutter off|fal')
    if output.exists(): raise PipelineError('OUTPUT_EXISTS',f'Refusing to overwrite {output}')
    if not 1<=max_iterations<=20: raise PipelineError('ITERATION_LIMIT','Repair limit must be 1..20')
    if timeout<=0: raise PipelineError('TIME_BUDGET','Generation timeout must be positive')
    output.mkdir(parents=True)
    write_json(output/'generation.json',dict(schema_version=1,materials=materials,clutter=clutter,max_fal_usd=max_fal_usd))
    fal_records=[]
    started=time.perf_counter();attempts=[];feedback=None;intent=None;revisions=[]
    status='failed';chosen=None;best=None;previous=None
    for attempt in range(max_iterations if program is None else 1):
        directory=output/'attempts'/str(attempt);directory.mkdir(parents=True)
        start=time.perf_counter()
        try:
            remaining=timeout-(time.perf_counter()-started)
            if remaining<=0: raise PipelineError('BUDGET_EXHAUSTED','Generation wall-time budget exhausted')
            selected=program if program is not None else agent_program(prompt,directory,feedback,model,min(180,remaining),
                layout_backend=layout_backend,architecture_only=architecture_only,clutter=clutter)
            validate_program(selected)
            if architecture_only and (selected['objects'] or selected['relations']):
                raise PipelineError('ARCHITECTURE_ONLY','Architecture-only command does not accept furnishing requests')
            if selected['prompt']!=prompt:
                # Preserve original user intent, not the agent's rewritten provenance.
                selected={**selected,'prompt':prompt}
            selected=resolve_program(selected,directory,model=model,timeout=min(180,remaining),sourcing=source_dimensions if program is None else None,fetcher=fetch)
            write_json(directory/'program.json',selected)
            if intent is None:
                intent=freeze(selected,authority='supplied_program' if program is not None else 'agent_program_not_independently_verified')
                write_json(output/'intent.json',intent)
            check_revision(intent,selected)
            if intent.get('architecture')!=selected.get('architecture'):
                revisions.append(dict(attempt=attempt,reason=feedback,architecture=selected.get('architecture')))
                write_json(output/'architecture_revisions.json',revisions)
            # Paid fal jobs happen here, once, before any deterministic stage; repairs and variants hit the cache.
            jobs=(fal.material_jobs(0) if materials=='fal' else [])+(fal.decor_jobs(selected) if clutter=='fal' else [])
            if jobs:
                fal_records+=fal.prefetch(jobs,budget_usd=max_fal_usd,spent_usd=sum(r['cost_usd'] for r in fal_records),deadline_s=max(30.,remaining-60))
                write_json(directory/'fal_prefetch.json',fal_records)
            t=time.perf_counter();ir=generate_layout(selected,seed+attempt,directory,backend=layout_backend,priors=priors,
                                                    allow_prior_backoff=allow_prior_backoff,robot_radius=robot_radius,
                                                    access_margin=access_margin);layout_seconds=time.perf_counter()-t
            ir['meta']['appearance']={**ir['meta'].get('appearance',{}),'materials':dict(source=materials,seed=0)}
            scene_report=check_scene(intent,ir,directory)
            write_json(directory/'scene_checks.json',scene_report)
            if not scene_report['passed']:raise PipelineError('SCENE_CHECKS','Independent scene checks failed',scene_report)
            write_json(directory/'ir.json',ir)
            t=time.perf_counter();_,compiled_model,compiled_data=compile_scene(ir,directory);compile_seconds=time.perf_counter()-t
            if layout_backend=='architecture':
                from .architecture_checks import check_architecture
                architecture_report=check_architecture(selected,ir,bundle=priors,robot_radius=robot_radius,access_margin=access_margin,
                                                      model=compiled_model,data=compiled_data,allow_backoff=allow_prior_backoff,asset_root=directory)
                write_json(directory/'architecture_validation.json',architecture_report)
                if not architecture_report['passed']:raise PipelineError('ARCHITECTURE_CHECKS','Independent architecture validation failed',architecture_report)
            t=time.perf_counter();report=validate_scene(directory,require_articulated=layout_backend!='architecture',robot_radius=robot_radius);validation_seconds=time.perf_counter()-t
            if preview:
                from .render import preview as render_preview
                render_preview(directory,cutaway=layout_backend!='architecture')
            attempts.append(dict(attempt=attempt,passed=report['passed'],seconds=time.perf_counter()-start,
                                 layout_seconds=layout_seconds,compile_seconds=compile_seconds,validation_seconds=validation_seconds))
            if report['passed']:
                chosen=directory;status='validated';break
            feedback=dict(code='VALIDATION_FAILED',report=report)
            # A compiled scene that failed some checks is still loadable evidence; keep the best.
            score=sum(bool(v) for v in report['checks'].values())
            if best is None or score>best[0]:best=(score,directory,report)
        except PipelineError as exc:
            feedback=exc.as_dict();write_json(directory/'failure.json',feedback)
            attempts.append(dict(attempt=attempt,passed=False,error=feedback,seconds=time.perf_counter()-start))
            if exc.code in ('BUDGET_EXHAUSTED','AGENT_UNAVAILABLE','AGENT_FAILED','FAL_CREDENTIALS'): break
            if feedback==previous: break  # The agent cannot move this failure; stop paying for retries.
            previous=feedback
        except Exception as exc:
            # Tool failures retain diagnostics and costs instead of leaving an
            # apparently unfinished output with no machine-readable outcome.
            feedback=dict(code='TOOL_ERROR',message=str(exc),exception_type=type(exc).__name__)
            write_json(directory/'failure.json',feedback)
            (directory/'traceback.log').write_text(traceback.format_exc())
            attempts.append(dict(attempt=attempt,passed=False,error=feedback,seconds=time.perf_counter()-start))
            break
    if not chosen and best is not None:
        # Degrade explicitly rather than emit nothing: the scene loads, and unmet.json says what failed.
        chosen=best[1];status='partial';report=best[2]
        ir=read_json(chosen/'ir.json')
        write_json(chosen/'unmet.json',dict(failed_checks=[k for k,v in report['checks'].items() if not v],
            failed_sweeps=[s['joint'] for s in report['sweeps'] if not s['passed']],
            dropped=ir['provenance'].get('dropped',[]),unsatisfied_relations=[r['constraint'] for r in ir['provenance'].get('relations',[]) if not r['satisfied']],
            last_failure=feedback,note='Best failed attempt by passed-check count; not a validated scene.'))
    if chosen:
        for path in chosen.iterdir():
            if path.is_dir(): shutil.copytree(path,output/path.name)
            else: shutil.copy2(path,output/path.name)
    result=dict(schema_version=1,status=status,passed=status=='validated',prompt=prompt,seed=seed,attempts=attempts,
                replay_key=digest(dict(prompt=prompt,seed=seed,registry=fingerprint(),version=VERSION,program=program,
                                      layout_backend=layout_backend,prior_sha256=priors.get('sha256') if priors else None,
                                      allow_prior_backoff=allow_prior_backoff,robot_radius=robot_radius,access_margin=access_margin,
                                      materials=materials,clutter=clutter)),
                seconds=time.perf_counter()-started,api_spend_usd=None,api_spend_note='Not available from CLI; never inferred as zero',
                materials=materials,clutter=clutter,fal_calls=len(fal_records),fal_spend_usd=sum(r['cost_usd'] for r in fal_records),fal_spend_basis=fal.PRICE_BASIS,
                program_source='supplied declarative program' if program else 'Codex non-interactive',last_failure=feedback if status!='validated' else None,
                validation_scope='architecture_geometry_and_scene_physics' if layout_backend=='architecture' else 'implemented_scene_and_physics_checks',
                distribution_verified=False)
    if layout_backend=='architecture':
        result['architecture_metric_basis']='requested area and source proportions; absolute aperture sizes not calibrated'
        result['source_metric_scale_verified']=False
        result['conditioning']=read_json(chosen/'ir.json')['provenance']['architecture_sampling']['conditioning'] if chosen else None
    write_json(output/'cost.json',result)
    write_json(output/'fal_calls.json',fal_records)
    if not chosen: raise PipelineError('GENERATION_FAILED','No validated scene produced',result)
    return result
