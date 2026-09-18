"""Reproduce the primary home-kitchen dataset from a saved scene and cached fal assets.

Run prepare, then capture (optionally partition --indices among workers), then
finalize. A new output tree is required; completed stages can resume. No model or
paid fal calls are allowed. The source scene remains read-only.
"""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import time
import numpy as np
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LP_NUM_THREADS','1')
from scene_pipeline import fal
from scene_pipeline.contracts import read_json,write_json,digest
from scene_pipeline.dataset import randomize_program,inspect
from scene_pipeline.registry import MATERIALS
from decorate_variants import THEMES


def prepare(args):
    from scene_pipeline.orchestrator import generate
    from scene_pipeline.compiler import compile_scene
    from scene_pipeline.validation import validate_scene
    from scene_pipeline.scene_checks import check_scene
    from scene_pipeline.render import preview,cycles
    from scene_pipeline.inspection import scene_page
    source=args.source;output=args.output;output.mkdir(parents=True,exist_ok=True)
    store=output/'asset_library';program=read_json(source/'input_program.json')
    resolved={o['id']:o for o in read_json(source/'program.json')['objects']}
    for item in program['objects']:
        ref=resolved[item['id']].get('asset_ref')
        if ref:
            target=store/'packages'/ref
            if not target.exists():shutil.copytree(source/'assets'/ref,target)
            item.pop('generated_request',None);item.pop('asset_request',None);item['asset_ref']=ref
    evidence=output/'evidence_cache.json'
    if not evidence.exists():write_json(evidence,read_json(source/'evidence_cache.json'))
    for i in args.indices:
        scene=output/f'variant_{i:03d}'
        if (scene/'variant.json').exists():continue
        started=time.monotonic();seed=args.start_seed+i;rng=np.random.default_rng(seed)
        variant,counts=randomize_program(program,rng)
        theme_index=[0,1,2,3,4,1,3,0,4,2][i];theme=THEMES['kitchen'][theme_index]
        overrides={finish:dict(prompt='Seamless '+description+', uniform material surface, no objects or perspective',tile_m=MATERIALS[finish]['tile_m']) for finish,description in zip(('floor_tile','wall_paint','paint','wood'),theme[1:])}
        materials=dict(source='fal',seed=7100+theme_index,overrides=overrides)
        for finish,spec in MATERIALS.items():
            if 'prompt' in spec:
                model,payload=fal.patina_request(finish,materials['seed'],prompt=overrides.get(finish,{}).get('prompt',spec['prompt']))
                if fal.cached(model,payload) is None:raise RuntimeError(f'Missing cached texture: {finish}, seed {materials["seed"]}')
        rejected=[]
        for retry in range(3):
            seed=args.start_seed+i+1000*retry;rng=np.random.default_rng(seed)
            variant,counts=randomize_program(program,rng)
            generation=generate(program['prompt'],seed,scene,program=variant,preview=False,cache_path=evidence,asset_store=store,materials='fal',clutter='fal',max_fal_usd=0.,layout_backend='heuristic')
            if generation['passed']:break
            failure=output/'rejected_layouts'/f'variant_{i:03d}_seed_{seed}';failure.mkdir(parents=True,exist_ok=True)
            for name in ('cost.json','validation.json','scene_checks.json','ir.json','program.json','input_program.json','unmet.json','generation.json'):
                if (scene/name).exists():shutil.copy2(scene/name,failure/name)
            rejected.append(dict(seed=seed,evidence=str(failure),reason=generation['last_failure']))
            shutil.rmtree(scene)  # reports/inputs above preserve rejection evidence without duplicate generated assets
        else:raise RuntimeError(f'No accepted layout for variant {i} after three seeds; see rejected_layouts')
        ir=read_json(scene/'ir.json')
        appearance=dict(light_multiplier=float(rng.uniform(.8,1.2)),tint=[1.,1.,1.],texture_repeat=float(rng.uniform(5,12)),materials=materials)
        ir['meta']['appearance']=appearance
        ir['provenance']['dataset_variant']=dict(source_scene=str(source.resolve()),source_ir_sha256=digest(read_json(source/'ir.json')),variant=i,layout_seed=seed)
        write_json(scene/'ir.json',ir);compile_scene(ir,scene)
        checks=check_scene(read_json(scene/'intent.json'),ir,scene);write_json(scene/'scene_checks.json',checks)
        validation=validate_scene(scene)
        assert checks['passed'] and validation['passed'],str(scene)
        manifest=read_json(scene/'manifest.json');assert not manifest['materials']['fallback']
        # Canonical inputs, assets and reports are retained; successful build copies are redundant.
        for name in ('attempts','resolution','preflight'):
            if (scene/name).exists():shutil.rmtree(scene/name)
        preview(scene);cycles(scene,samples=32,resolution=1024,view='overview')
        for p in [scene/'render'/n for n in ('scene.blend','scene.blend1','blender.log','recipe.json')]+list((scene/'render').glob('texture_*.png')):p.unlink(missing_ok=True)
        record=dict(passed=True,variant=i,tier='full' if i<3 else 'state',theme=theme[0],factors=dict(layout_seed=seed,clutter_counts=counts,**appearance),seconds=time.monotonic()-started,source_scene=str(source.resolve()),rejected_layouts=rejected,new_fal_usd=0.,new_model_calls=0)
        write_json(scene/'variant.json',record);scene_page(scene)
        print(__import__('json').dumps(dict(prepared=str(scene),seconds=record['seconds'])),flush=True)


def capture(args):
    from scene_pipeline.flows import run
    from scene_pipeline.inspection import scene_page
    import mujoco
    from OpenGL import GL
    context=mujoco.GLContext(16,16);context.make_current()
    device=dict(backend=os.environ['MUJOCO_GL'],vendor=GL.glGetString(GL.GL_VENDOR).decode(),renderer=GL.glGetString(GL.GL_RENDERER).decode(),physics='CPU',video_encoding='CPU ffmpeg');context.free()
    for i in args.indices:
        scene=args.output/f'variant_{i:03d}';deadline=time.monotonic()+3600
        while not (scene/'variant.json').exists():
            if time.monotonic()>deadline:raise TimeoutError(str(scene))
            time.sleep(5)
        variant=read_json(scene/'variant.json');target=scene/'mapping'
        if (target/'report.json').exists():continue
        report=run(scene,'mapping',target,seconds=60,tier=variant['tier'],seed=i,timeout=7200)
        assert report['passed'],str(target)
        report['rendering']=device;write_json(target/'report.json',report)
        (target/'replay.stderr.log').unlink(missing_ok=True)
        scene_page(target);scene_page(scene)
        print(__import__('json').dumps(dict(captured=str(scene),seconds=report['wall_seconds'])),flush=True)


def finalize(args):
    from scene_pipeline.inspection import dataset_page
    rows=[]
    for i in range(10):
        scene=args.output/f'variant_{i:03d}';variant=read_json(scene/'variant.json');report=read_json(scene/'mapping/report.json')
        assert variant['passed'] and report['passed'] and abs(report['simulated_seconds']-60)<1e-6
        rows.append(dict(variant=i,tier=variant['tier'],factors=variant['factors'],theme=variant['theme'],report=report,streams=inspect(scene/'mapping/data.h5')))
    result=dict(schema_version=1,start_seed=args.start_seed,variants=rows,passed=True,primary=True,source_scene=str(args.source.resolve()),tier_rationale='Three full RGB-D runs provide image/depth/semantic/instance observations and truth; seven state/range/IMU runs retain dynamics and noisy sensors while avoiding per-frame RGB-D rendering/storage. Every run lasts 60 simulated seconds.',randomization='One home-kitchen program; ten accepted layout seeds and clutter counts/placements (failed robot-access/layout checks are retained separately; bounded replacement seeds add 1000), independent lighting multipliers, and five cached fal PBR themes reused across distinct layouts. Generated clutter is visual-only. Exact factors are recorded per variant.',new_fal_usd=0.,new_model_calls=0)
    write_json(args.output/'dataset.json',result)
    (args.output/'DATA_CARD.md').write_text('# Primary home-kitchen dataset\n\n'+result['tier_rationale']+'\n\n'+result['randomization']+'\n\n10 runs, 600 simulated seconds; 1,803 camera frames at 320x240, 60,010 state samples. The schema and loader are documented in the combined collection data card. Clocks use simulation seconds on the same 0.002 s physics grid. Sensor noise is synthetic and uncalibrated; camera viewpoint is low, layouts use a constrained program, generated props have no contacts, and results do not establish broad manipulation or downstream learning performance.\n')
    dataset_page(args.output)
    print('Verified primary home-kitchen batch: 10 variants, 3 full + 7 state',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','capture','finalize'])
    parser.add_argument('--source',type=Path,default=Path('deliverables/outputs/home_kitchen/scene'))
    parser.add_argument('--output',type=Path,default=Path('outputs/home_kitchen_primary'))
    parser.add_argument('--cache',type=Path,default=Path('vendor/fal_cache/fal'))
    parser.add_argument('--start-seed',type=int,default=300)
    parser.add_argument('--indices',type=int,nargs='+',default=list(range(10)))
    args=parser.parse_args()
    if any(i not in range(10) for i in args.indices):parser.error('indices must be 0..9')
    fal.ROOT=args.cache.resolve()
    globals()[args.command](args)

if __name__=='__main__':main()
