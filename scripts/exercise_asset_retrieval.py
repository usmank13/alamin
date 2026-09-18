"""Small real-asset smoke: direct tool calls, no model/API key and no scene-specific rules."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import time

from scene_pipeline.asset_catalog import search
from scene_pipeline.asset_library import acquire,materialize
from scene_pipeline.contracts import PipelineError,read_json,write_json
from scene_pipeline.inspection import STYLE


def gallery(output,report):
    e=html.escape;cards=[]
    for run in report['assets']:
        images=''
        if run.get('path'):
            for name in ('preview','collision'):
                images+=f'<figure><figcaption>{name}</figcaption><img src="{e(run["path"])}/{name}.png"></figure>'
            images+=f'<a href="{e(run["path"])}/index.html">Inspect asset, provenance and checks</a>'
        cards.append(f'<article><h2>{e(run["domain"])} / {e(run["category"])}</h2><p>{e(run["status"])}</p>{images}<pre>{e(json.dumps(run,indent=2))}</pre></article>')
    scenes=''.join(f'<li><a href="scenes/{e(r["domain"])}/index.html">{e(r["domain"])}</a>: {e(r["status"])}</li>' for r in report['scenes'])
    (output/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Real asset retrieval smoke</title><style>{STYLE}</style>'
        '<h1>Real asset retrieval smoke</h1><p>Direct search → selected import → validation → scene references. No model calls. '
        'Static props only; geometry/physics checks do not certify semantic suitability or task performance.</p>'
        f'<a href="report.json">Machine-readable report</a><ul>{scenes}</ul><div class="grid">'+''.join(cards)+'</div>')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,default=Path('examples/retrieval_assets.json'))
    parser.add_argument('--store',type=Path,default=Path('vendor/asset_library'));parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--priors',type=Path);args=parser.parse_args()
    if args.output.exists():raise PipelineError('OUTPUT_EXISTS','Use a fresh diagnostic output directory')
    args.output.mkdir(parents=True);report=dict(kind='asset_retrieval_smoke',assets=[],scenes=[],model_calls=0);groups=defaultdict(list)
    for case in read_json(args.manifest)['cases']:
        run={**case,'status':'failed'};start=time.monotonic()
        try:
            matches=search(case['query'],args.store,limit=50)
            run['search_candidates']=[m['candidate_id'] for m in matches]
            if case['candidate_id'] not in run['search_candidates']:raise PipelineError('SEARCH_MISS','Selected candidate was not returned')
            _,asset=acquire(case['candidate_id'],case['category'],store=args.store)
            folder,_=materialize(asset['key'],args.output/'assets',args.store)
            run.update(status='validated_static',asset_ref=asset['key'],path=str(folder.relative_to(args.output)),dimensions_m=asset['dimensions'],
                       capabilities=asset['capabilities'],import_report=read_json(folder/'import.json'))
            groups[case['domain']].append(run)
        except Exception as exc:run['error']=exc.as_dict() if isinstance(exc,PipelineError) else dict(code='TOOL_ERROR',message=str(exc))
        run['seconds']=time.monotonic()-start;report['assets'].append(run)
        write_json(args.output/'report.json',report);gallery(args.output,report)
        print(json.dumps({k:run[k] for k in ('category','status','seconds','error') if k in run}),flush=True)
    if args.priors:
        from scene_pipeline.orchestrator import generate
        from scene_pipeline.portability import export_urdf,verify_urdf
        from scene_pipeline.inspection import scene_page
        for domain,runs in groups.items():
            output=args.output/'scenes'/domain;result=dict(domain=domain,status='failed')
            prompt=f'A 100 square metre {domain} asset-integration test. Place the selected static props with a clear entrance.'
            program=dict(schema_version=2,prompt=prompt,space=dict(kind=domain,area_m2=100,shape='rectangle',annexes=0),
                objects=[dict(id=r['category'],category=r['category'],count=1,zone='main',required=True,asset_ref=r['asset_ref']) for r in runs],relations=[],
                architecture=dict(openings=[dict(id='entry',kind='door',count=1,role='entrance',required=True,origin='explicit')],relationships=[]))
            try:
                generated=generate(prompt,7,output,program=program,asset_store=args.store,layout_backend='architecture',
                    priors=read_json(args.priors),allow_prior_backoff=True,robot_radius=.25)
                result.update(status=generated['status'],generation=generated)
                if generated['passed']:
                    result['export']=verify_urdf(export_urdf(output))
                    if not result['export']['passed']:result['status']='export_failed'
            except Exception as exc:result['error']=exc.as_dict() if isinstance(exc,PipelineError) else dict(code='TOOL_ERROR',message=str(exc))
            result['coverage']='successfully imported assets only; see asset failures for missing requested content'
            scene_page(output,result);report['scenes'].append(result)
            write_json(args.output/'report.json',report);gallery(args.output,report)
            print(json.dumps(dict(domain=domain,status=result['status'])),flush=True)
    return 0 if all(r['status']=='validated_static' for r in report['assets']) and all(r['status']=='validated' for r in report['scenes']) else 1


if __name__=='__main__':raise SystemExit(main())
