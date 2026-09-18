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
from .runtime import Runtime, CURRENT, using, request, strict, strip_null as _strip_null
from .generators import generate_layout
from .scene_intent import freeze,check_revision
from .scene_checks import check_scene
from .compiler import compile_scene
from .validation import validate_scene


OPEN_VOCABULARY=('Prefer categories in this registry. For content the registry lacks, keep the user\'s own category name as lowercase_snake, '
    'and request real geometry using asset_request with a short search query, mode static and placement freestanding/support/wall. '
    'The resolver searches pinned asset catalogs and asks you to choose a candidate; use alternative search vocabulary if none matches. '
    'Never invent candidate IDs. Retrieved SDF props have native dimensions and no implied moving mechanisms. '
    'Do not substitute a sized proxy after retrieval failure without explicit prompt permission. '
    'When a parametric template is appropriate or explicitly requested, choose a family (box, table, shelf, cabinet, drawer) and, if the user stated its size, give dimensions_m with '
    'dimension_evidence.prompt_quote copied verbatim from the prompt; otherwise omit dimensions and the harness sources and verifies them. '
    'Do the same for a registry category whose size the user stated. Never supply dimensions without evidence. ')


def _codex(instruction,schema,work,model,timeout,*,name='program',search=False):
    """Compatibility alias; transport is selected by the enclosing runtime."""
    return request(instruction,schema,work,model,timeout,name=name,search=search)


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


def resolve_program(program,work,*,model=None,timeout=180,sourcing=source_dimensions,cache_path=None,fetcher=fetch,asset_store=None,resolution_cache=None):
    """Bind every open-vocabulary or user-stated dimension to evidence before intent is frozen.

    Resolution order: user quote in the prompt, dimension cache, harness-verified web page,
    then honest UNKNOWN_CATEGORY at layout. Resolved programs carry dimensions_m plus a
    basis and no raw evidence, so repairs compare equal when the evidence is unchanged.
    """
    if resolution_cache is not None:
        # Cache each verified binding independently: a later failed lookup must not
        # throw away earlier imports/evidence. This cache lives only for this run.
        resolved=deepcopy(program);records={name:{} for name in ('dimension_sources','asset_resolutions','evidence_cache')}
        for index,item in enumerate(program['objects']):
            key=digest(dict(prompt=program['prompt'],request=item))
            entry=resolution_cache.get(key)
            if entry and 'error' in entry:
                error=entry['error'];raise PipelineError(error['code'],error['message'],error['details'])
            if entry is None:
                single={**program,'objects':[item],'relations':[]};folder=Path(work)/'resolution'/item['id']
                try:
                    bound=resolve_program(single,folder,model=model,timeout=timeout,sourcing=sourcing,cache_path=cache_path,
                                          fetcher=fetcher,asset_store=asset_store)
                except PipelineError as exc:
                    error=exc.as_dict();error['details']=dict(object_id=item['id'],request=item,cause=exc.details)
                    if exc.code in ('ASSET_NOT_FOUND','ASSET_NO_MATCH'):
                        resolution_cache[key]=dict(error=error)
                    raise PipelineError(error['code'],error['message'],error['details']) from exc
                entry=dict(object=bound['objects'][0],records={name:read_json(folder/f'{name}.json') for name in records})
                resolution_cache[key]=deepcopy(entry)
            resolved['objects'][index]=deepcopy(entry['object'])
            if entry['object'].get('asset_ref'):
                from .asset_library import materialize
                materialize(entry['object']['asset_ref'],Path(work)/'assets',asset_store)
            for name in records:records[name].update(entry['records'][name])
            for name,value in records.items():write_json(Path(work)/f'{name}.json',value)
        for name,value in records.items():write_json(Path(work)/f'{name}.json',value)
        return validate_program(resolved)
    resolved=deepcopy(program);sources={};assets={}
    for index,item in enumerate(resolved['objects']):
        if item.get('generated_request'):continue  # resolved after the bounded fal prefetch
        if item.get('asset_request') or item.get('asset_ref'):
            from .asset_library import resolve_request
            resolved['objects'][index],assets[item['id']]=resolve_request(item,work,store=asset_store,allow_agent=sourcing is not None)
            sources[item['id']]=assets[item['id']]
            continue
        known=item['category'] in CATALOG
        if known:item.pop('family',None)  # registry categories own their family; agent noise here must not count as drift
        if item.get('dimension_basis'):
            raise PipelineError('UNTRUSTED_BASIS','Supply original dimension evidence, not a claimed resolved basis',item['id'])
        if known and item.get('dimension_evidence') is None:continue
        default=CATALOG[item['category']].get('dimensions') if known else None
        def source_and_verify(tries=3):
            """Verified sourcing with the verifier's rejection fed back; populates the cache for the category."""
            feedback=None
            for attempt in range(tries):
                remaining=CURRENT.get().remaining(timeout) if CURRENT.get() else timeout
                evidence=sourcing(item['category'],program['prompt'],Path(work)/'sourcing'/f"{item['category']}_{attempt}",model=model,timeout=remaining,feedback=feedback)
                family=evidence.pop('family',None)
                try:
                    found=resolve(dict(item,dimension_evidence=evidence,**({'family':family} if family else {})),program['prompt'],cache_path=cache_path,fetcher=fetcher)
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
    write_json(Path(work)/'asset_resolutions.json',assets)
    snapshot={}
    from .evidence import measurement
    for item in resolved['objects']:
        record=sources.get(item['id'],{})
        if 'document' not in record:continue
        evidence=record['evidence']
        dims=[measurement(record['document'],identity=evidence['identity'],label=evidence['fields'][a][0],
                          number=evidence['fields'][a][1],unit=evidence['fields'][a][2],url=evidence['url'],axis=a)['value_m']
              for a in ('width','depth','height')]
        snapshot[item['category']]={**record,'dimensions_m':dims,'basis':'sourced','family':item.get('family','box')}
    write_json(Path(work)/'evidence_cache.json',snapshot)
    return validate_program(resolved)


def agent_program(prompt,work,feedback=None,model=None,timeout=180,*,layout_backend='heuristic',architecture_only=False,clutter='off',fal_budget_usd=None,asset_store=None):
    work=Path(work);work.mkdir(parents=True,exist_ok=True)
    registry=search(routes=None if clutter=='fal' else ('G1','G3'))  # decor is offered only when its fal route is enabled
    if clutter=='fal':
        for category,entry in registry.items():
            if entry['route']=='G6':entry['generated_cache_available']=fal.cached(*fal.decor_request(category)) is not None
    schema=deepcopy(PROGRAM_SCHEMA_V2 if layout_backend=='architecture' else PROGRAM_SCHEMA)
    if layout_backend=='architecture':schema['properties']['architecture']['required'].append('unsupported_requirements')
    schema['properties']['objects']['items']['properties'].pop('dimension_basis')  # harness-owned; the agent supplies evidence only
    schema['properties']['space']['required'].append('inferred_fields')
    instruction=(
        'Produce only a SceneProgram JSON matching the supplied schema. You select categories, counts, zones and relations; '
        'deterministic tools compute geometry and placement. Do not execute shell commands, modify files or invent product measurements. '
        'Required means user-required; optional omissions must not change the requested purpose. Include populated storage/clutter where requested. '
        'Preserve every explicitly requested category and count across repairs. Do not reduce the inventory to bypass layout failures. '
        'Express the arrangement the user describes with relations (against_wall, near, in_row, under); mark one required only when the user requires it. '
        'Support placement for microwave and small objects is automatic; countertop microwave does not require a near relation to the counter category. '
        'For a robotics acceptance scene, request at least five articulated object instances when consistent with the prompt. '
        +OPEN_VOCABULARY+
        'Use space.area_m2 from the user; otherwise infer a plausible design area for the requested use within the schema bounds. '
        'An inferred area is an engineering estimate, not a measured or empirical floorplan. '
        'Do not claim a walk-in or washing station is present by relabeling a cabinet. '
        f'Registry: {registry}\nPrompt: {prompt}\nLast deterministic failure: {feedback}\n'
    )
    if layout_backend=='architecture':
        instruction=(
            'Produce SceneProgram v2 JSON matching the schema. You supply semantics and relationships; '
            'tools supply geometry from a trusted empirical corpus. Do not execute commands or supply coordinates, '
            'source IDs, measurements, or fitted parameters. Area is user-specified or an explicitly unverified design estimate within the schema bounds. '
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
    if clutter=='fal':
        instruction+=('\nVisual-only generation is enabled. The catalog is not a closed vocabulary: '
            'for any decorative category, generated_request accepts prompt, size_m (estimated largest extent, metres), '
            'placement (support or freestanding), and physical_use="visual_only". Use generated_request '
            'when stock decor does not fit; the tool caches it and registers a reusable asset_ref automatically. '
            'These are static visuals, never contact, support surfaces or articulation. Do not substitute them for requested functional equipment. '
            'Include some required generated clutter when the user explicitly requests it. Never invent measurements: size_m here '
            'is labeled an unverified visual estimate, not a sourced dimension. Retrieval can return no suitable match; '
            'revise the query or report an unavailable object rather than relabel a weak candidate.')
        instruction+=f'\nRemaining fal estimate budget: {fal_budget_usd}; each new distinct decor prompt costs {fal.PRICE_USD[fal.HUNYUAN]} USD; exact cache hits are free.'
    from .asset_library import registered_assets
    reusable=registered_assets(prompt,store=asset_store)
    if reusable:instruction+=f'\nReusable library entries (use asset_ref unchanged): {reusable}'
    instruction+=('\nList only your own design choices in space.inferred_fields (kind, area_m2, shape, annexes). '
        'Never mark a user-specified value inferred. Unlisted fields stay fixed during repair. '
        'Required objects preserve category, count and zone; their retrieved asset or inferred implementation can change. '
        'Keep explicit dimensions and required relationships unchanged.')
    previous=(feedback or {}).get('previous_program')
    if previous:
        # A repair is an upsert of complete object requests, not a new inventory.
        # Omitted sections/objects remain byte-for-byte unchanged.
        properties={k:deepcopy(schema['properties'][k]) for k in ('space','objects','relations')}
        properties['objects']['minItems']=0
        properties['remove_objects']={'type':'array','items':{'type':'string'},'uniqueItems':True}
        if 'architecture' in schema['properties']:properties['architecture']=schema['properties']['architecture']
        schema=obj(properties,required=[])
        instruction+=('\nREPAIR MODE overrides the full-program output instruction: return only a repair patch. '
            'objects contains complete replacements for changed IDs, or new objects such as a missing support; '
            'omit unchanged objects. remove_objects lists only dispensable optional IDs. '
            'space, relations and architecture, if supplied, replace those sections; otherwise leave them out. '
            'Fix the reported failure with the smallest change. Keep successful asset/evidence choices. '
            'For a rejected retrieval, use a genuinely different query/candidate or a verified library reference. '
            'Do not repeat rejected queries or swap functional equipment for visual-only clutter. '
            'Failure history and resolved asset dimensions in the context are tool feedback, not new user requirements.')
    result=_codex(instruction,schema,work,model,timeout,name='repair' if previous else 'program')
    if previous:
        write_json(work/'repair.json',result)
        program=apply_repair(previous,result)
    else:program=result
    for item in program.get('objects',[]):item.pop('dimension_basis',None)
    return validate_program(program)


def apply_repair(previous,patch):
    """Merge a small declarative repair; intent checks still guard the result."""
    program=deepcopy(previous)
    updates={item['id']:item for item in patch.get('objects',[])}
    if len(updates)!=len(patch.get('objects',[])):
        raise PipelineError('DUPLICATE_ID','Duplicate object in repair')
    removed=set(patch.get('remove_objects',[]))
    existing={item['id'] for item in program['objects']}
    if removed & updates.keys() or not removed<=existing:
        raise PipelineError('REPAIR_CONFLICT','Cannot remove an unknown object or remove and replace the same ID')
    program['objects']=[updates.pop(item['id'],item) for item in program['objects'] if item['id'] not in removed]
    program['objects'].extend(updates.values())
    for section in ('space','relations','architecture'):
        if section in patch:program[section]=deepcopy(patch[section])
    return validate_program(program)


def preflight_layout(program,seed,root,*,asset_store=None,**layout_options):
    """Check known geometry before fal spending; never fabricate clutter geometry."""
    core=deepcopy(program);deferred=[]
    for item in program['objects']:
        if item.get('generated_request') or (not item.get('asset_ref') and CATALOG.get(item['category'],{}).get('route')=='G6'):
            deferred.append(item)
    ids={item['id'] for item in deferred}
    core['objects']=[item for item in core['objects'] if item['id'] not in ids]
    core['relations']=[r for r in core['relations'] if not ids.intersection(r['objects'])]
    root=Path(root)
    from .asset_library import materialize
    for item in core['objects']:
        if item.get('asset_ref'):materialize(item['asset_ref'],root/'assets',asset_store)
    report=dict(scope='known_asset_layout_only',deferred_objects=sorted(ids),generated_geometry_checked=False)
    if not core['objects'] and core['schema_version']==1:
        if any(item['required'] and item.get('generated_request',CATALOG.get(item['category'],{})).get('placement')=='support' for item in deferred):
            raise PipelineError('LAYOUT_UNSAT','Required generated clutter needs a physical support',dict(objects=sorted(ids)))
        report['status']='skipped_no_known_geometry'
    else:
        ir=generate_layout(core,seed,root,**layout_options)
        checked=check_scene(freeze(core,authority='supplied_program'),ir,root)
        if not checked['passed']:raise PipelineError('SCENE_CHECKS','Preflight scene checks failed',checked)
        supports=[s for item in ir['objects'] for s in read_json(root/item['asset'])['supports']]
        for item in deferred:
            config=item.get('generated_request',CATALOG.get(item['category'],{}))
            if item['required'] and config.get('placement')=='support' and not supports:
                raise PipelineError('LAYOUT_UNSAT','Required generated clutter needs a physical support',dict(object_id=item['id']))
        report.update(status='passed',placed_objects=len(ir['objects']))
    write_json(root/'report.json',report)
    return report


def generate(prompt,seed,output,*,agent_backend='codex',model=None,timeout=900,max_cost_usd=None,runtime=None,cache_path=None,**kwargs):
    runtime=runtime or Runtime(backend=agent_backend,model=model,max_cost_usd=max_cost_usd)
    if not __import__('math').isfinite(timeout) or timeout<=0:raise PipelineError('TIME_BUDGET','Timeout must be finite and positive')
    deadline=time.monotonic()+timeout
    runtime.deadline=min(runtime.deadline,deadline) if runtime.deadline is not None else deadline
    with using(runtime):
        return _generate(prompt,seed,output,model=model,timeout=timeout,cache_path=cache_path,**kwargs)


def _generate(prompt,seed,output,*,program=None,model=None,max_iterations=20,timeout=900,preview=True,cache_path=None,
              layout_backend='heuristic',priors=None,allow_prior_backoff=False,robot_radius=None,access_margin=.05,architecture_only=False,asset_store=None,
              materials='flat',clutter='off',max_fal_usd=None):
    output=Path(output)
    if materials not in ('flat','fal') or clutter not in ('off','fal'):raise PipelineError('OPTION','materials must be flat|fal and clutter off|fal')
    if output.exists(): raise PipelineError('OUTPUT_EXISTS',f'Refusing to overwrite {output}')
    if not 1<=max_iterations<=20: raise PipelineError('ITERATION_LIMIT','Repair limit must be 1..20')
    if timeout<=0: raise PipelineError('TIME_BUDGET','Generation timeout must be positive')
    output.mkdir(parents=True)
    fal_records=[]
    configuration=dict(schema_version=1,layout_backend=layout_backend,allow_prior_backoff=allow_prior_backoff,
                       robot_radius=robot_radius,access_margin=access_margin,architecture_only=architecture_only,
                       agent_backend=CURRENT.get().backend,model=CURRENT.get().model,
                       materials=materials,clutter=clutter,max_fal_usd=max_fal_usd,
                       prior_sha256=priors.get('sha256') if priors else None)
    write_json(output/'generation.json',configuration)
    if priors is not None:write_json(output/'priors.json',priors)
    started=time.perf_counter();attempts=[];feedback=None;intent=None;revisions=[]
    status='failed';chosen=None;best=None;previous=None;last_program=None;resolution_cache={}
    layout_options=dict(backend=layout_backend,priors=priors,allow_prior_backoff=allow_prior_backoff,
                        robot_radius=robot_radius,access_margin=access_margin)
    for attempt in range(max_iterations if program is None else 1):
        directory=output/'attempts'/str(attempt);directory.mkdir(parents=True)
        start=time.perf_counter()
        try:
            remaining=timeout-(time.perf_counter()-started)
            if remaining<=0: raise PipelineError('BUDGET_EXHAUSTED','Generation wall-time budget exhausted')
            context=None if feedback is None else dict(failure=feedback,previous_program=last_program,
                initial_space=intent['space'] if intent else None,required_objects=[r for r in intent['requirements'] if r['required']] if intent else [],
                required_relations=[r for r in intent['relations'] if r['required']] if intent else [],
                rejected_retrievals=[v['error'] for v in resolution_cache.values() if 'error' in v],
                resolved_assets=[v['records']['asset_resolutions'] for v in resolution_cache.values() if 'records' in v and v['records']['asset_resolutions']])
            selected=program if program is not None else agent_program(prompt,directory,context,model,min(180,remaining),
                layout_backend=layout_backend,architecture_only=architecture_only,clutter=clutter,
                asset_store=asset_store,
                fal_budget_usd=None if max_fal_usd is None else max_fal_usd-sum(r['cost_usd'] for r in fal_records))
            validate_program(selected)
            if clutter!='fal' and any(o.get('generated_request') for o in selected['objects']):
                raise PipelineError('GENERATED_DISABLED','generated_request requires --clutter fal')
            if architecture_only and (selected['objects'] or selected['relations']):
                raise PipelineError('ARCHITECTURE_ONLY','Architecture-only command does not accept furnishing requests')
            if selected['prompt']!=prompt:
                # Preserve original user intent, not the agent's rewritten provenance.
                selected={**selected,'prompt':prompt}
            write_json(directory/'input_program.json',selected)
            if program is None:
                if intent is None:
                    intent=freeze(selected,authority='agent_program_not_independently_verified')
                    write_json(output/'intent.json',intent)
                check_revision(intent,selected)
            # Retain the last intent-safe proposal, never an invalid repair, as the
            # base for the next patch. Evidence requests stay replayable as inputs.
            last_program=deepcopy(selected)
            selected=resolve_program(selected,directory,model=model,timeout=CURRENT.get().remaining(),sourcing=source_dimensions if program is None else None,
                                     fetcher=fetch,cache_path=cache_path,asset_store=asset_store,resolution_cache=resolution_cache)
            CURRENT.get().remaining()
            write_json(directory/'program.json',selected)
            # Check grounded fixture fit before paying for decor/materials. Final
            # layout and physics still validate the actual generated geometry.
            jobs=(fal.material_jobs(0) if materials=='fal' else [])+(fal.decor_jobs(selected) if clutter=='fal' else [])
            if jobs:
                preflight_layout(selected,seed,directory/'preflight',asset_store=asset_store,**layout_options)
                remaining=CURRENT.get().remaining()
                fal_records+=fal.prefetch(jobs,budget_usd=max_fal_usd,spent_usd=sum(r['cost_usd'] for r in fal_records),deadline_s=remaining)
                write_json(directory/'fal_prefetch.json',fal_records)
            for item in selected['objects']:
                if not item.get('generated_request'):continue
                from .asset_library import register_generated,materialize
                _,asset=register_generated(item['category'],item['generated_request'],store=asset_store)
                materialize(asset['key'],directory/'assets',asset_store)
                item.pop('generated_request');item['asset_ref']=asset['key']
            write_json(directory/'program.json',selected)
            if intent is None:
                intent=freeze(selected,authority='supplied_program' if program is not None else 'agent_program_not_independently_verified')
                write_json(output/'intent.json',intent)
            if intent.get('architecture')!=selected.get('architecture'):
                revisions.append(dict(attempt=attempt,reason=feedback,architecture=selected.get('architecture')))
                write_json(output/'architecture_revisions.json',revisions)
            t=time.perf_counter();ir=generate_layout(selected,seed,directory,**layout_options);layout_seconds=time.perf_counter()-t
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
            CURRENT.get().remaining()
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
            if exc.code in ('BUDGET_EXHAUSTED','AGENT_UNAVAILABLE','AGENT_FAILED','AGENT_CREDENTIALS','AGENT_TIMEOUT','COST_UNKNOWN','FAL_CREDENTIALS'): break
            signature=(feedback,digest(last_program))
            if signature==previous: break  # Identical failure AND unchanged proposal; no progress.
            previous=signature
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
                seconds=time.perf_counter()-started,api_spend_usd=CURRENT.get().summary()['cost_usd'],usage=CURRENT.get().summary(),
                api_spend_note=CURRENT.get().summary()['cost_note'],
                materials=materials,clutter=clutter,fal_calls=len(fal_records),fal_spend_usd=sum(r['cost_usd'] for r in fal_records),fal_spend_basis=fal.PRICE_BASIS,
                program_source='supplied declarative program' if program else CURRENT.get().backend,last_failure=feedback if status!='validated' else None,
                validation_scope='architecture_geometry_and_scene_physics' if layout_backend=='architecture' else 'implemented_scene_and_physics_checks',
                distribution_verified=False)
    if layout_backend=='architecture':
        result['architecture_metric_basis']='requested area and source proportions; absolute aperture sizes not calibrated'
        result['source_metric_scale_verified']=False
        result['conditioning']=read_json(chosen/'ir.json')['provenance']['architecture_sampling']['conditioning'] if chosen else None
    write_json(output/'cost.json',result)
    write_json(output/'fal_calls.json',fal_records)
    write_json(output/'agent_calls.json',CURRENT.get().calls)
    if not chosen: raise PipelineError('GENERATION_FAILED','No validated scene produced',result)
    return result
