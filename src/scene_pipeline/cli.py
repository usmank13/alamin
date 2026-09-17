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
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('generate');p.add_argument('--prompt',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--program',type=Path);p.add_argument('--model');p.add_argument('--max-iterations',type=int,default=20);p.add_argument('--timeout',type=float,default=900);p.add_argument('--no-preview',action='store_true')
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
    p=sub.add_parser('validate');p.add_argument('scene',type=Path)
    p=sub.add_parser('render');p.add_argument('scene',type=Path);p.add_argument('--blender');p.add_argument('--samples',type=int,default=32);p.add_argument('--resolution',type=int,default=1024)
    p.add_argument('--view',choices=['auto','interior','overview'],default='auto')
    p=sub.add_parser('preview');p.add_argument('scene',type=Path)
    p=sub.add_parser('replay');p.add_argument('rollout',type=Path);p.add_argument('--fps',type=int,default=5)
    p=sub.add_parser('export');p.add_argument('scene',type=Path);p.add_argument('--format',choices=['urdf'],default='urdf');p.add_argument('--verify',action='store_true')
    p=sub.add_parser('assets');p.add_argument('action',choices=['build','inspect','validate','promote']);p.add_argument('target');p.add_argument('--output',type=Path,default=Path('outputs/library'));p.add_argument('--scale',type=float,default=1)
    p=sub.add_parser('run');p.add_argument('scene',type=Path);p.add_argument('--flow',choices=['mapping','interaction'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--tier',choices=['full','state'],default='full');p.add_argument('--seed',type=int,default=0)
    p=sub.add_parser('dataset');p.add_argument('scene',type=Path);p.add_argument('--variants',type=int,default=10);p.add_argument('--output',type=Path,required=True);p.add_argument('--seconds',type=float,default=60)
    args=parser.parse_args(argv)
    try:
        if args.command=='registry':
            from .registry import search
            result=search()
        elif args.command=='generate':
            from .orchestrator import generate
            from .dsl import load
            result=generate(args.prompt,args.seed,args.output,program=load(args.program) if args.program else None,model=args.model,
                            max_iterations=args.max_iterations,timeout=args.timeout,preview=not args.no_preview,
                            layout_backend=args.layout_backend,priors=read_json(args.priors) if args.priors else None,allow_prior_backoff=args.allow_prior_backoff,
                            robot_radius=args.robot_radius,access_margin=args.access_margin)
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
                robot_radius=args.robot_radius,access_margin=args.access_margin,architecture_only=True)
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
            result=validate_scene(args.scene)
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
            result=run(args.scene,args.flow,args.output,seconds=args.seconds,tier=args.tier,seed=args.seed)
        elif args.command=='dataset':
            from .dataset import collect_variants
            result=collect_variants(args.scene,args.output,args.variants,args.seconds)
        print(json.dumps(result,indent=2,allow_nan=False))
        return 0 if result.get('passed',True) else 1
    except PipelineError as exc:
        print(json.dumps({'error':exc.as_dict()},indent=2),file=sys.stderr)
        return 2


if __name__=='__main__': raise SystemExit(main())
