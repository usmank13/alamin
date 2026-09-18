"""Add seeded fal dressing to existing variants, then capture new matching datasets.

The source tree is read-only. Each prepared scene preserves the source objects and
adds static visual-only props on checked support surfaces. Run the subcommands in
order: plan, fetch, prepare, capture. Output directories support completed-stage
resume; failed captures are retained and require a fresh directory for a retry.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import time

os.environ.setdefault('MUJOCO_GL', 'osmesa')
os.environ.setdefault('LP_NUM_THREADS', '1')

from scene_pipeline import fal
from scene_pipeline.contracts import read_json, write_json, digest
from scene_pipeline.registry import CATALOG, MATERIALS, class_id

THEMES = {
    'kitchen': [
        ('terracotta_cream', 'matte terracotta quarry tiles with light grey grout', 'warm cream washable plaster', 'sage green satin cabinet paint', 'honey oak veneer'),
        ('slate_white', 'dark slate grey square porcelain floor tiles with narrow grout', 'clean white eggshell plaster', 'navy blue satin cabinet paint', 'pale ash wood veneer'),
        ('ivory_teal', 'ivory speckled terrazzo floor tiles', 'soft pale blue washable plaster', 'muted teal satin cabinet paint', 'natural birch wood veneer'),
        ('sand_olive', 'sand beige ceramic square floor tiles with cream grout', 'warm off-white plaster', 'olive green satin cabinet paint', 'warm walnut veneer'),
        ('charcoal_oak', 'medium grey matte stone square floor tiles', 'light warm grey washable plaster', 'warm white satin cabinet paint', 'light natural oak veneer'),
    ],
    'warehouse': [
        ('concrete_blue', 'light grey sealed concrete warehouse floor, fine aggregate and subtle wear, no markings', 'off-white painted industrial plaster', 'industrial blue powder-coated steel cabinet finish', 'unfinished pine plywood'),
        ('epoxy_sage', 'muted sage grey epoxy warehouse floor, fine matte aggregate, no markings', 'warm white painted industrial plaster', 'charcoal powder-coated steel cabinet finish', 'medium brown plywood'),
        ('terrazzo_red', 'grey industrial terrazzo floor, small dark aggregate, no markings', 'pale grey painted industrial plaster', 'oxide red powder-coated steel cabinet finish', 'pale plywood'),
        ('concrete_yellow', 'medium grey polished concrete warehouse floor, subtle mottling, no markings', 'warm cream painted industrial plaster', 'muted safety yellow powder-coated steel cabinet finish', 'rough warm pine plywood'),
        ('epoxy_navy', 'light beige grey epoxy industrial floor, subtle stipple, no markings', 'pale blue grey painted industrial plaster', 'navy powder-coated steel cabinet finish', 'natural birch plywood'),
    ],
}
WAREHOUSE_PROPS = {
    'shipping_parcel': ('a small sealed brown corrugated cardboard shipping parcel with tan packing tape, no readable text', .32),
    'packing_tape_dispenser': ('a red handheld packing tape dispenser with a roll of brown packing tape', .23),
    'barcode_scanner': ('a black handheld warehouse barcode scanner resting flat on its side', .20),
    'safety_hard_hat': ('a yellow industrial safety hard hat resting upright', .28),
    'work_gloves': ('a pair of orange and grey fabric work gloves lying together flat', .24),
}
KITCHEN_PROPS = ['mug', 'cutting_board', 'fruit_bowl', 'paper_towel_roll', 'kettle', 'knife_block', 'stack_of_plates']


def request(category):
    prompt, size = WAREHOUSE_PROPS[category]
    return dict(prompt=prompt, size_m=size, placement='support', physical_use='visual_only')


def plan(source, output):
    import numpy as np
    if (output/'plan.json').exists(): return read_json(output/'plan.json')
    variants = []; jobs = {}
    for domain, themes in THEMES.items():
        for i, theme in enumerate(themes):
            original = source/domain/'variants'/f'variant_{i:03d}'
            if not (original/'scene.mjz').exists(): raise ValueError(f'Missing source: {original}')
            seed = 7100 + (100 if domain == 'warehouse' else 0) + i
            rng = np.random.default_rng(seed)
            overrides = {}
            for finish, description in zip(('floor_tile', 'wall_paint', 'paint', 'wood'), theme[1:]):
                overrides[finish] = dict(prompt='Seamless '+description+', uniform material surface, no objects or perspective', tile_m=MATERIALS[finish]['tile_m'])
            options = dict(source='fal', seed=seed, overrides=overrides)
            material_keys=[]
            for finish, spec in MATERIALS.items():
                if 'prompt' not in spec: continue
                model, payload = fal.patina_request(finish, seed, prompt=overrides.get(finish, {}).get('prompt', spec['prompt']))
                key=fal.key(model,payload);material_keys.append(key)
                jobs[key]=dict(kind='material',finish=finish,seed=seed,model=model,payload=payload)
            categories = KITCHEN_PROPS if domain=='kitchen' else list(WAREHOUSE_PROPS)
            selected=list(rng.permutation(categories)[:4+i%2])
            selected += list(rng.choice(categories,size=2+i%3,replace=True))
            for category in set(selected):
                config=CATALOG[category] if domain=='kitchen' else {'prompt':WAREHOUSE_PROPS[category][0]}
                model,payload=fal.decor_request(category,config)
                jobs[fal.key(model,payload)]=dict(kind='decor',category=category,model=model,payload=payload)
            variants.append(dict(domain=domain,variant=i,source=str(original.resolve()),seed=seed,theme=theme[0],materials=options,
                                 material_keys=material_keys,clutter=selected,tier='full' if i<3 else 'state',
                                 source_ir_sha256=digest(read_json(original/'ir.json'))))
    result=dict(schema_version=1,source=str(source.resolve()),variants=variants,jobs=jobs,
                estimated_new_fal_usd=sum(fal.PRICE_USD[j['model']] for j in jobs.values() if fal.cached(j['model'],j['payload']) is None),
                physical_use='Added clutter is visual-only, not a manipulation target.',
                price_basis=fal.PRICE_BASIS)
    output.mkdir(parents=True,exist_ok=True);write_json(output/'plan.json',result)
    return result


def fetch(output, budget, env_file):
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            name,sep,value=line.removeprefix('export ').partition('=')
            if sep and name.strip() in ('FAL_KEY','FAL_API_KEY'):
                parts=shlex.split(value,comments=True)
                if len(parts)==1:os.environ.setdefault(name.strip(),parts[0])
    proposal=read_json(output/'plan.json')
    ledger=read_json(output/'fal_calls.json') if (output/'fal_calls.json').exists() else []
    jobs=list(proposal['jobs'].values())
    pending_estimate=sum(fal.PRICE_USD[j['model']] for j in jobs if not fal.cached(j['model'],j['payload']))
    spent=sum(x['cost_usd'] for x in ledger)
    if spent+pending_estimate>budget:raise ValueError('Plan exceeds the total fal list-price budget')
    # Put slow mesh requests first; bounded batches avoid flooding the provider.
    jobs.sort(key=lambda j:j['kind']!='decor')
    for offset in range(0,len(jobs),8):
        calls=fal.prefetch(jobs[offset:offset+8],budget_usd=budget,spent_usd=spent,deadline_s=1800)
        ledger+=calls;spent=sum(x['cost_usd'] for x in ledger)
        write_json(output/'fal_calls.json',ledger)
        print(json.dumps(dict(batch=offset//8,completed=len(ledger),estimated_usd=spent,errors=[x.get('error') for x in calls if x.get('error')])),flush=True)
        if any(x.get('error') for x in calls):raise RuntimeError('fal batch failed; inspect ledger before resuming')


def add_clutter(ir, root, variant, library=None):
    import numpy as np
    from scene_pipeline.assets import instantiate
    from scene_pipeline.layout import footprint,intersects
    from scene_pipeline.provenance import classify
    rng=np.random.default_rng(variant['seed']);added=[]
    packages={o['id']:read_json(root/o['asset']) for o in ir['objects']}
    for index,category in enumerate(variant['clutter']):
        cached_package=library/category/'package.json' if library else None
        if cached_package is not None and cached_package.exists():
            package=read_json(cached_package)
            folder=root/'assets'/package['key']
            shutil.copytree(library/category/package['key'],folder,dirs_exist_ok=True)
        else:
            folder,package=instantiate(category,root/'assets',generated_request=request(category) if category in WAREHOUSE_PROPS else None)
            if library is not None:
                (library/category).mkdir(parents=True,exist_ok=True)
                shutil.copytree(folder,library/category/package['key'],dirs_exist_ok=True)
                write_json(cached_package,package)
        lo,hi=np.array(package['bounds']);candidates=[]
        for support in ir['objects']:
            if support['id'] not in packages:continue
            for surface in packages[support['id']]['supports']:
                sx,sy,sz=surface['center'];z=support['position'][2]+sz
                if not .3<=z<=1.6:continue
                for turn in (0.,math.pi/2,math.pi,3*math.pi/2):
                    lower,upper=np.array(footprint(package,[0,0,0],turn,sweep=False))
                    xmin=sx-surface['size'][0]/2-lower[0]+.012;xmax=sx+surface['size'][0]/2-upper[0]-.012
                    ymin=sy-surface['size'][1]/2-lower[1]+.012;ymax=sy+surface['size'][1]/2-upper[1]-.012
                    if xmin>xmax or ymin>ymax:continue
                    for _ in range(32):
                        dx,dy=rng.uniform(xmin,xmax),rng.uniform(ymin,ymax)
                        c,s=math.cos(support['yaw']),math.sin(support['yaw'])
                        pos=[support['position'][0]+c*dx-s*dy,support['position'][1]+s*dx+c*dy,z-lo[2]+.001]
                        yaw=support['yaw']+turn;volume=footprint(package,pos,yaw,sweep=False)
                        if any(intersects(volume,o['bounds'],.006) for o in ir['objects'] if o['id']!=support['id']):continue
                        candidates.append((pos,yaw,support['id'],volume,support['zone']))
        if not candidates:raise ValueError(f'No safe support for required decoration {category} in {variant["theme"]}')
        pos,yaw,parent,volume,zone=candidates[int(rng.integers(len(candidates)))]
        obj=dict(id=f'fal_decor_{index}_0',category=category,class_id=class_id(category),asset=str(folder.relative_to(root)/'asset.json'),
                 asset_key=package['key'],dimensions=package['dimensions'],position=pos,yaw=float(yaw),zone=zone,
                 support_parent=parent,dynamic=False,initial_state={},bounds=volume,source_classification=classify(package))
        ir['objects'].append(obj);added.append(obj)
    return added


def compare_physics(source,root):
    import numpy as np
    from scene_pipeline.validation import load
    a,b=load(source),load(root);checks=0
    def body_id(i):
        if i==0:return 0
        name=a.body(i).name
        if name:return b.body(name).id
        parent=int(a.body_parentid[i]);new_parent=body_id(parent)
        before=[k for k in range(1,a.nbody) if a.body_parentid[k]==parent and not a.body(k).name]
        after=[k for k in range(1,b.nbody) if b.body_parentid[k]==new_parent and not b.body(k).name]
        if len(before)!=len(after):raise ValueError('Unnamed body membership changed')
        return after[before.index(i)]
    for kind,fields in [('geom',['geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity','geom_friction']),
                        ('body',['body_pos','body_quat','body_mass','body_inertia','body_ipos','body_iquat']),
                        ('joint',['jnt_type','jnt_axis','jnt_pos','jnt_range','jnt_stiffness'])]:
        count={'geom':a.ngeom,'body':a.nbody,'joint':a.njnt}[kind]
        for i in range(count):
            name=getattr(a,kind)(i).name
            if kind=='body':j=body_id(i)
            elif not name and kind in ('geom','joint'):
                owner_field='geom_bodyid' if kind=='geom' else 'jnt_bodyid'
                owner=int(getattr(a,owner_field)[i]);owner_name=a.body(owner).name
                new_owner=body_id(owner)
                before=[k for k in range(count) if getattr(a,owner_field)[k]==owner and not getattr(a,kind)(k).name]
                after=[k for k in range(b.ngeom if kind=='geom' else b.njnt) if getattr(b,owner_field)[k]==new_owner and not getattr(b,kind)(k).name]
                if len(before)!=len(after):raise ValueError(f'Unnamed {kind} membership changed for {owner_name}')
                j=after[before.index(i)]
            else:j=getattr(b,kind)(name).id
            for field in fields:
                if not np.allclose(getattr(a,field)[i],getattr(b,field)[j],atol=1e-10,rtol=0):raise ValueError(f'Physics changed: {name} {field}')
                checks+=1
    if a.nq!=b.nq or a.nv!=b.nv:raise ValueError('Dressing introduced dynamics')
    for field in ('dof_damping','dof_armature','dof_frictionloss'):
        if not np.array_equal(getattr(a,field),getattr(b,field)):raise ValueError(f'Physics changed: {field}')
    for field in ('timestep','gravity','integrator','solver'):
        if not np.array_equal(getattr(a.opt,field),getattr(b.opt,field)):raise ValueError(f'Physics changed: option {field}')
    return dict(passed=True,name_matched_array_comparisons=checks,original_nq=a.nq,decorated_nq=b.nq)


def prepare(output, domain=None, wait_for_assets=False):
    from scene_pipeline.compiler import compile_scene
    from scene_pipeline.validation import validate_scene
    from scene_pipeline.scene_checks import check_scene
    from scene_pipeline.render import preview,cycles
    from scene_pipeline.inspection import scene_page
    proposal=read_json(output/'plan.json')
    for variant in proposal['variants']:
        if domain and variant['domain']!=domain:continue
        target=output/variant['domain']/f'variant_{variant["variant"]:03d}'
        if (target/'decoration.json').exists() and read_json(target/'decoration.json').get('passed'):continue
        if wait_for_assets:
            deadline=time.monotonic()+1800
            while any(not (fal.ROOT/key/'meta.json').exists() for key in variant['material_keys']):
                if time.monotonic()>deadline:raise TimeoutError(f'Materials still unavailable: {target}')
                time.sleep(5)
        started=time.monotonic();source=Path(variant['source'])
        target.mkdir(parents=True,exist_ok=True)
        write_json(target/'decoration_request.json',variant)
        shutil.copytree(source/'assets',target/'assets',dirs_exist_ok=True)
        for name in ('program.json','input_program.json','intent.json','generation.json','dimension_sources.json','asset_resolutions.json','evidence_cache.json'):
            if (source/name).exists():shutil.copy2(source/name,target/name)
        ir=deepcopy(read_json(source/'ir.json'))
        assert digest(ir)==variant['source_ir_sha256']
        ir['meta']['appearance']['materials']=variant['materials']
        ir['meta']['appearance']['tint']=[1.,1.,1.]
        added=add_clutter(ir,target,variant,output/'decoration_library')
        ir['provenance']['decoration']=dict(seed=variant['seed'],theme=variant['theme'],source_ir_sha256=variant['source_ir_sha256'],added_instances=[o['id'] for o in added])
        write_json(target/'ir.json',ir)
        compile_scene(ir,target)
        realized=read_json(target/'manifest.json')['materials']
        if realized['fallback']:raise ValueError(f'Missing PBR sets: {realized["fallback"]}')
        unchanged=compare_physics(source,target)
        checks=check_scene(read_json(target/'intent.json'),ir,target);write_json(target/'scene_checks.json',checks)
        validation=validate_scene(target)
        if not checks['passed'] or not validation['passed']:raise ValueError(f'Decorated scene failed validation: {target}')
        config=read_json(target/'generation.json');config.update(materials='fal',clutter='fal',decoration_pass=True)
        write_json(target/'generation.json',config)
        preview(target);cycles(target,samples=32,resolution=1024,view='overview')
        for name in ('scene.blend','scene.blend1','blender.log'):(target/'render'/name).unlink(missing_ok=True)
        report=dict(passed=True,theme=variant['theme'],seed=variant['seed'],source=str(source),source_ir_sha256=variant['source_ir_sha256'],
                    physical_objects_unchanged=unchanged,added_instances=[dict(id=o['id'],category=o['category'],support_parent=o['support_parent']) for o in added],
                    material_sets=realized['realized'],seconds=time.monotonic()-started,added_clutter_use='static visual-only; zero contact geometry')
        write_json(target/'decoration.json',report);scene_page(target)
        print(json.dumps(dict(prepared=str(target),props=len(added),seconds=report['seconds'])),flush=True)


def capture(output,domain,wait_for_assets=False):
    from scene_pipeline.flows import run
    from scene_pipeline.dataset import inspect
    from scene_pipeline.inspection import scene_page
    import mujoco
    from OpenGL import GL
    context=mujoco.GLContext(16,16);context.make_current()
    device=dict(backend=os.environ.get('MUJOCO_GL','osmesa'),vendor=GL.glGetString(GL.GL_VENDOR).decode(),
                renderer=GL.glGetString(GL.GL_RENDERER).decode(),physics='CPU',video_encoding='CPU ffmpeg')
    context.free()
    proposal=read_json(output/'plan.json');rows=[]
    original=read_json(Path(proposal['source'])/domain/'variants/dataset.json')
    for variant in proposal['variants']:
        if variant['domain']!=domain:continue
        scene=output/domain/f'variant_{variant["variant"]:03d}'
        if wait_for_assets:
            deadline=time.monotonic()+1800
            while not (scene/'decoration.json').exists():
                if time.monotonic()>deadline:raise TimeoutError(f'Decorated scene still unavailable: {scene}')
                time.sleep(5)
        assert read_json(scene/'decoration.json')['passed']
        target=scene/'mapping'
        if (target/'report.json').exists():report=read_json(target/'report.json')
        else:report=run(scene,'mapping',target,seconds=60,tier=variant['tier'],seed=variant['variant'],timeout=7200)
        if not report['passed']:raise ValueError(f'Failed capture: {target}')
        report['rendering']=device
        write_json(target/'report.json',report)
        factors={**original['variants'][variant['variant']]['factors'],'materials':variant['materials'],
                 'decoration_seed':variant['seed'],'decoration_theme':variant['theme'],'fal_clutter':variant['clutter'],'tint':[1.,1.,1.]}
        rows.append(dict(variant=variant['variant'],tier=variant['tier'],factors=factors,report=report,streams=inspect(target/'data.h5')))
        scene_page(target);scene_page(scene)
        (target/'replay.stderr.log').unlink(missing_ok=True)
        write_json(output/domain/'capture_progress.json',dict(completed=len(rows),variants=rows))
        print(json.dumps(dict(captured=str(scene),passed=report['passed'],seconds=report['wall_seconds'])),flush=True)
    summary={**original,'variants':rows,'passed':all(r['report']['passed'] for r in rows),
             'randomization':'Original layout/clutter seeds retained; ten domain-specific fal PBR themes plus seeded generated prop selection, count and support placement. All captures re-recorded after decoration.'}
    write_json(output/domain/'dataset.json',summary)
    (output/domain/'DATA_CARD.md').write_text('# Decorated mapping variants\n\nFive 60-second runs: three full RGB-D and two state/range/IMU. All observations, segmentation, replay and scene archives were captured after fal decoration.\n\n'+summary['randomization']+'\n\nAdded props are static visual-only geometry, not contact/manipulation assets. Sensor noise is synthetic. See each decoration.json, rig.json, report.json and dataset.json for exact provenance, factors and results.\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','fetch','prepare','capture'])
    p.add_argument('--source',type=Path,default=Path('deliverables/outputs'))
    p.add_argument('--output',type=Path,default=Path('outputs/fal_variant_refresh'))
    p.add_argument('--cache',type=Path,default=Path('vendor/fal_cache/fal'))
    p.add_argument('--max-fal-usd',type=float,default=8.)
    p.add_argument('--env-file',type=Path,default=Path('.env'))
    p.add_argument('--domain',choices=list(THEMES))
    p.add_argument('--wait-for-assets',action='store_true',help='Wait up to 30 minutes per variant for a concurrent fetch stage')
    args=p.parse_args();fal.ROOT=args.cache.resolve()
    if args.command=='plan':
        result=plan(args.source,args.output);print(json.dumps(dict(variants=len(result['variants']),jobs=len(result['jobs']),estimated_new_fal_usd=result['estimated_new_fal_usd'])))
    elif args.command=='fetch':fetch(args.output,args.max_fal_usd,args.env_file)
    elif args.command=='prepare':prepare(args.output,args.domain,args.wait_for_assets)
    else:
        if not args.domain:p.error('capture requires --domain')
        capture(args.output,args.domain,args.wait_for_assets)


if __name__=='__main__':main()
