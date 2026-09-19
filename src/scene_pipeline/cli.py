"""Public commands; expensive integrations are imported only when requested."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LP_NUM_THREADS','1')

from .contracts import PipelineError,read_json


def main(argv=None):
    parser=argparse.ArgumentParser(prog='pipeline')
    parser.add_argument('--resource-root',type=Path,help='Root containing vendor/ (also SCENE_PIPELINE_RESOURCE_ROOT)')
    parser.add_argument('--cache-dir',type=Path,help='Writable evidence cache (also SCENE_PIPELINE_CACHE)')
    parser.add_argument('--asset-store',type=Path,help='Retrieved asset library (also SCENE_PIPELINE_ASSET_STORE)')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('generate');p.add_argument('--prompt',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',type=Path);p.add_argument('--model');p.add_argument('--max-iterations',type=int,default=20);p.add_argument('--timeout',type=float,default=900);p.add_argument('--no-preview',action='store_true')
    p.add_argument('--use-skill',action='store_true',help='Let one Codex agent follow the repository pipeline skill and run the scene tools')
    p.add_argument('--layout-backend',choices=['heuristic','empirical','architecture'],default='heuristic');p.add_argument('--priors',type=Path);p.add_argument('--allow-prior-backoff',action='store_true')
    p.add_argument('--robot-radius',type=float);p.add_argument('--access-margin',type=float,default=.05)
    p=sub.add_parser('architecture');p.add_argument('--prompt',required=True);p.add_argument('--program',type=Path);p.add_argument('--priors',type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-prior-backoff',action='store_true')
    p.add_argument('--robot-radius',type=float);p.add_argument('--access-margin',type=float,default=.05);p.add_argument('--no-preview',action='store_true')
    p.add_argument('--model');p.add_argument('--max-iterations',type=int,default=20);p.add_argument('--timeout',type=float,default=900)
    p=sub.add_parser('architecture-fit');p.add_argument('--sample',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('architecture-check');p.add_argument('scene',type=Path);p.add_argument('--priors',type=Path,required=True)
    p.add_argument('--allow-prior-backoff',action='store_true');p.add_argument('--robot-radius',type=float);p.add_argument('--access-margin',type=float,default=.05)
    p=sub.add_parser('architecture-evaluate');p.add_argument('--program',type=Path,required=True);p.add_argument('--priors',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--seeds',type=int,default=32);p.add_argument('--allow-prior-backoff',action='store_true')
    p.add_argument('--robot-radius',type=float);p.add_argument('--access-margin',type=float,default=.05)
    p=sub.add_parser('layout-audit');p.add_argument('sample',type=Path);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('reconstruct');p.add_argument('reference',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--no-preview',action='store_true')
    scale=p.add_mutually_exclusive_group(required=True);scale.add_argument('--area-m2',type=float);scale.add_argument('--metres-per-unit',type=float)
    sub.add_parser('registry')
    p=sub.add_parser('asset-index');p.add_argument('--sources',type=Path,help='JSON list of revision-pinned GitHub SDF sources')
    p=sub.add_parser('asset-search');p.add_argument('query');p.add_argument('--limit',type=int,default=12)
    p=sub.add_parser('asset-generate',help='Generate and register any static prop with approximate collision using fal')
    p.add_argument('--category',required=True);p.add_argument('--prompt',required=True)
    p.add_argument('--size-m',type=float,required=True,help='Estimated largest extent; not a measured dimension')
    p.add_argument('--placement',choices=['support','freestanding'],default='support')
    p.add_argument('--max-fal-usd',type=float,default=.4)
    p=sub.add_parser('asset-fetch');p.add_argument('candidate_id');p.add_argument('--category',required=True)
    p.add_argument('--mode',choices=['static','articulated','dynamic'],default='static')
    p.add_argument('--placement',choices=['freestanding','support','wall'],default='freestanding')
    p.add_argument('--timeout',type=float,default=180);p.add_argument('--no-preview',action='store_true')
    p=sub.add_parser('validate');p.add_argument('scene',type=Path)
    p=sub.add_parser('render');p.add_argument('scene',type=Path);p.add_argument('--blender');p.add_argument('--samples',type=int,default=32);p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--view',choices=['auto','interior','overview'],default='auto')
    p=sub.add_parser('preview');p.add_argument('scene',type=Path)
    p=sub.add_parser('replay');p.add_argument('rollout',type=Path);p.add_argument('--fps',type=int,default=5)
    p=sub.add_parser('export');p.add_argument('scene',type=Path);p.add_argument('--format',choices=['urdf'],default='urdf');p.add_argument('--verify',action='store_true')
    p=sub.add_parser('assets');p.add_argument('action',choices=['build','inspect','validate','promote']);p.add_argument('target');p.add_argument('--output',type=Path,default=Path('outputs/library'));p.add_argument('--scale',type=float,default=1)
    p=sub.add_parser('run');p.add_argument('scene',type=Path);p.add_argument('--flow',choices=['mapping','interaction','navigate'],required=True);p.add_argument('--goal');p.add_argument('--policy',choices=['planner','vlm'],default='planner');p.add_argument('--no-video',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--tier',choices=['full','state'],default='full');p.add_argument('--seed',type=int,default=0)
    p=sub.add_parser('dataset');p.add_argument('scene',type=Path);p.add_argument('--variants',type=int,default=10);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60)
    p.add_argument('--start-seed',type=int,default=100,help='First layout/randomization seed; each following variant increments it')
    p=sub.add_parser('costs');p.add_argument('artifacts',type=Path,nargs='+');p.add_argument('--output',type=Path,required=True);p.add_argument('--usd-per-mtok-in',type=float);p.add_argument('--usd-per-mtok-out',type=float)
    for command in ('generate','architecture'):
        p=sub.choices[command];p.add_argument('--materials',choices=['flat','fal'],default='flat',help='PBR material sets from fal PATINA (needs FAL_KEY)')
        p.add_argument('--clutter',choices=['off','fal'],default='off',help='Offer fal-generated decor categories with approximate collision to the agent')
        p.add_argument('--max-fal-usd',type=float,help='List-price cap on new fal jobs; cache hits are free')
    p=sub.add_parser('doctor');p.add_argument('--smoke',action='store_true')
    p=sub.add_parser('inspect');p.add_argument('path',type=Path);p.add_argument('--viewer',action='store_true');p.add_argument('--no-open',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('manifest',type=Path);p.add_argument('--priors',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--live',action='store_true');p.add_argument('--models',nargs=2);p.add_argument('--max-cost-usd',type=float)
    p.add_argument('--agent-backend',choices=['codex','openrouter'],default='openrouter');p.add_argument('--seeds',nargs='+',type=int,default=[7,11])
    p.add_argument('--timeout',type=float,default=900);p.add_argument('--cache-snapshot',type=Path);p.add_argument('--resume',action='store_true')
    p.add_argument('--no-cycles',action='store_true');p.add_argument('--flows',action='store_true');p.add_argument('--interactions',action='store_true')
    for command in ('generate','architecture','run'):
        p=sub.choices[command]
        p.add_argument('--agent-backend',choices=['codex','openrouter','recorded'],default='codex')
        p.add_argument('--responses',type=Path,help='Recorded response list; offline adapter only')
        p.add_argument('--max-cost-usd',type=float)
        if command=='run':p.add_argument('--model');p.add_argument('--timeout',type=float,default=900)
    args=parser.parse_args(argv)
    if args.resource_root:os.environ['SCENE_PIPELINE_RESOURCE_ROOT']=str(args.resource_root.resolve())
    if args.cache_dir:os.environ['SCENE_PIPELINE_CACHE']=str(args.cache_dir.resolve())
    if args.asset_store:os.environ['SCENE_PIPELINE_ASSET_STORE']=str(args.asset_store.resolve())
    try:
        if os.environ.get('SCENE_PIPELINE_SKILL_JOB'):
            from .skill_runner import prepare_build
            prepare_build(args)
        agent_options={}
        if args.command in ('generate','architecture','run'):
            from .runtime import Runtime
            runtime=Runtime(backend=args.agent_backend,model=args.model,max_cost_usd=args.max_cost_usd,
                            responses=read_json(args.responses) if args.responses else [])
            agent_options=dict(runtime=runtime)
        if args.command=='asset-index':
            from .asset_catalog import index_sources
            result=index_sources(sources=read_json(args.sources) if args.sources else None)
        elif args.command=='asset-search':
            from .asset_catalog import search
            from .asset_library import registered_assets
            result=dict(candidates=search(args.query,limit=args.limit),registered_assets=registered_assets(args.query,limit=args.limit),
                        note='Lexical candidates, not verified semantic matches')
        elif args.command=='asset-generate':
            from . import fal
            from .asset_library import register_generated,asset_preview
            from .contracts import ID
            import jsonschema
            jsonschema.validate(args.category,ID)
            request=dict(prompt=args.prompt,size_m=args.size_m,placement=args.placement,physical_use='static_collision')
            calls=fal.prefetch(fal.decor_jobs({'objects':[dict(category=args.category,generated_request=request)]}),budget_usd=args.max_fal_usd)
            if any(c.get('error') for c in calls):raise PipelineError('FAL_JOB','Generated asset unavailable',calls)
            folder,asset=register_generated(args.category,request)
            page=asset_preview(folder)
            result=dict(passed=True,asset_ref=asset['key'],package=str(folder),gallery=str(page),fal_calls=calls,
                        dimensions_m=asset['dimensions'],capabilities=asset['capabilities'])
        elif args.command=='asset-fetch':
            from .asset_library import acquire
            folder,asset=acquire(args.candidate_id,args.category,mode=args.mode,placement=args.placement,timeout=args.timeout,preview=not args.no_preview)
            result=dict(passed=True,asset_ref=asset['key'],package=str(folder),dimensions_m=asset['dimensions'],capabilities=asset['capabilities'])
        elif args.command=='doctor':
            from .doctor import diagnose
            result=diagnose(args.smoke)
        elif args.command=='inspect':
            from .inspection import inspect_path
            result=inspect_path(args.path,viewer=args.viewer,open_browser=not args.no_open)
        elif args.command=='evaluate':
            from .evaluation import evaluate
            report=evaluate(read_json(args.manifest),args.output,priors=read_json(args.priors),models=args.models,
                            live=args.live,agent_backend=args.agent_backend,seeds=args.seeds,max_cost_usd=args.max_cost_usd,
                            timeout=args.timeout,cache_snapshot=read_json(args.cache_snapshot) if args.cache_snapshot else None,
                            resume=args.resume,cycles=False if args.no_cycles else None,flows=args.flows or args.live,
                            interactions=args.interactions or args.live)
            result={k:v for k,v in report.items() if k!='runs'}
            result.update(runs=len(report['runs']),report=str(args.output/'evaluation.json'),gallery=str(args.output/'index.html'))
        elif args.command=='registry':
            from .registry import search
            result=search()
        elif args.command=='generate':
            if args.use_skill:
                if args.agent_backend!='codex' or args.program or args.layout_backend!='heuristic' or args.priors or args.robot_radius is not None or args.access_margin!=.05 or args.max_cost_usd is not None:
                    raise PipelineError('SKILL_OPTIONS','--use-skill supports Codex prompt generation with default layout settings and token/time limits')
                from .skill_runner import generate
                result=generate(args.prompt,args.seed,args.output,model=args.model,max_iterations=args.max_iterations,
                                timeout=args.timeout,materials=args.materials,clutter=args.clutter,max_fal_usd=args.max_fal_usd)
            else:
                from .orchestrator import generate
                from .dsl import load
                result=generate(args.prompt,args.seed,args.output,program=load(args.program) if args.program else None,model=args.model,
                            max_iterations=args.max_iterations,timeout=args.timeout,preview=not args.no_preview,
                            layout_backend=args.layout_backend,priors=read_json(args.priors) if args.priors else None,allow_prior_backoff=args.allow_prior_backoff,
                            robot_radius=args.robot_radius,access_margin=args.access_margin,
                            materials=args.materials,clutter=args.clutter,max_fal_usd=args.max_fal_usd,**agent_options)
        elif args.command=='architecture-fit':
            from .architecture_priors import build_bundle
            result=build_bundle(args.sample,args.output)
        elif args.command=='architecture-check':
            import mujoco
            from .architecture_checks import check_architecture
            from .validation import load
            from .scene_intent import check_revision
            from .contracts import validate_program
            program=validate_program(read_json(args.scene/'program.json'))
            if (args.scene/'intent.json').exists():check_revision(read_json(args.scene/'intent.json'),program)
            model=load(args.scene);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
            result=check_architecture(program,read_json(args.scene/'ir.json'),bundle=read_json(args.priors),
                allow_backoff=args.allow_prior_backoff,robot_radius=args.robot_radius,access_margin=args.access_margin,
                model=model,data=data,asset_root=args.scene)
        elif args.command=='architecture':
            from .orchestrator import generate
            from .dsl import load
            result=generate(args.prompt,args.seed,args.output,program=load(args.program) if args.program else None,
                model=args.model,max_iterations=args.max_iterations,timeout=args.timeout,preview=not args.no_preview,
                layout_backend='architecture',priors=read_json(args.priors),allow_prior_backoff=args.allow_prior_backoff,
                robot_radius=args.robot_radius,access_margin=args.access_margin,architecture_only=True,
                materials=args.materials,clutter=args.clutter,max_fal_usd=args.max_fal_usd,**agent_options)
        elif args.command=='architecture-evaluate':
            from .architecture_evaluation import evaluate_batch
            from .dsl import load
            report=evaluate_batch(load(args.program),read_json(args.priors),args.output,seeds=args.seeds,
                allow_backoff=args.allow_prior_backoff,robot_radius=args.robot_radius,access_margin=args.access_margin)
            result={k:v for k,v in report.items() if k!='runs'}
        elif args.command=='layout-audit':
            from .reference_layout import audit_sample
            result=audit_sample(args.sample,args.output)
        elif args.command=='reconstruct':
            from .reconstruction import reconstruct
            result=reconstruct(read_json(args.reference),args.output,area_m2=args.area_m2,
                               metres_per_unit=args.metres_per_unit,render=not args.no_preview)
        elif args.command=='validate':
            from .validation import validate_scene
            config=read_json(args.scene/'generation.json') if (args.scene/'generation.json').exists() else {}
            result=validate_scene(args.scene,require_articulated=config.get('layout_backend')!='architecture',robot_radius=config.get('robot_radius'))
        elif args.command=='render':
            from .render import cycles
            result={'image':str(cycles(args.scene,blender=args.blender,samples=args.samples,resolution=args.resolution,view=args.view))}
        elif args.command=='preview':
            from .render import preview
            preview(args.scene);result={'gallery':str(args.scene/'index.html')}
        elif args.command=='replay':
            from .replay import video
            result=video(args.rollout,args.fps)
        elif args.command=='export':
            from .portability import export_urdf,verify_urdf
            path=export_urdf(args.scene);result=verify_urdf(path) if args.verify else {'package':str(path)}
        elif args.command=='assets':
            from .assets import instantiate
            from .validation import validate_asset
            if args.action=='build':
                path,asset=instantiate(args.target,args.output,scale=args.scale);result={'path':str(path),'asset':asset}
            elif args.action=='inspect': result=read_json(Path(args.target)/'asset.json')
            else: result=validate_asset(args.target,promote=args.action=='promote')
        elif args.command=='run':
            from .flows import run
            result=run(args.scene,args.flow,args.output,seconds=args.seconds,tier=args.tier,seed=args.seed,goal=args.goal,policy=args.policy,video=not args.no_video,timeout=args.timeout,**agent_options)
        elif args.command=='dataset':
            from .dataset import collect_variants
            result=collect_variants(args.scene,args.output,args.variants,args.seconds,start_seed=args.start_seed)
        elif args.command=='costs':
            from .costs import table
            result=table(args.artifacts,args.output,args.usd_per_mtok_in,args.usd_per_mtok_out)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result.get('passed',True) else 1
    except PipelineError as exc:
        print(json.dumps({'error':exc.as_dict()},indent=2),file=sys.stderr)
        return 2
    except (OSError,ValueError) as exc:
        from .runtime import redact
        print(json.dumps({'error':dict(code='INPUT_ERROR',message=redact(str(exc)))}),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
