"""Offline, repeatable feasibility experiment using a previously audited real corpus."""
import argparse
from collections import Counter
import html
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.distance import cdist

from scene_pipeline.contracts import PipelineError,read_json,write_json,digest
from scene_pipeline.dsl import load
from scene_pipeline.generators import generate_layout
from scene_pipeline.layout_priors import condition,evaluate,FEATURES
from scene_pipeline.reconstruction import reconstruct
from scene_pipeline.scene_intent import freeze
from scene_pipeline.scene_checks import check_scene
from scene_pipeline.compiler import compile_scene
from scene_pipeline.validation import validate_scene


def comparison_plot(records,path):
    image=Image.new('RGB',(1200,660),'white');draw=ImageDraw.Draw(image)
    draw.text((20,10),'Seeded layout comparison: same assets and area / sampled real proportions; no floorplan copying',fill='black')
    for column,backend in enumerate(('heuristic','empirical')):
        record=next(r for r in records if r['backend']==backend and r['role']=='kitchen' and 'error' not in r)
        ir=read_json(Path(record['path'])/'ir.json');room=ir['rooms'][0]
        points=np.array(room['polygon']);low=points.min(axis=0);high=points.max(axis=0)
        scale=min(520/(high-low).max(),520/(high-low).max());offset=column*600
        def xy(p):return (offset+35+(p[0]-low[0])*scale,590-(p[1]-low[1])*scale)
        draw.text((offset+30,40),backend+' / '+record['distribution']['conditioning'],fill='black')
        draw.polygon([xy(p) for p in room['polygon']],fill='#e9edf1',outline='black')
        for o in ir['objects']:
            # Inspection outline is a labeled conservative sweep bbox, not ground truth.
            a,b=o['bounds'];p,q=xy(a),xy(b)
            draw.rectangle((min(p[0],q[0]),min(p[1],q[1]),max(p[0],q[0]),max(p[1],q[1])),outline='#48759a')
            draw.text(xy(o['position']),o['id'],fill='black')
    draw.text((20,635),'Blue: conservative placement/sweep bounds. Sparse fixture priors do not certify scene realism.',fill='black')
    image.save(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,default=4);parser.add_argument('--render',action='store_true')
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite output')
    if not 1<=args.seeds<=20:parser.error('seeds must be 1..20')
    args.output.mkdir(parents=True)
    priors=read_json(args.corpus/'priors.json');reconstructions=[]
    for reference_path in sorted((args.corpus/'references').glob('*.json')):
        name=reference_path.stem
        try:
            report=reconstruct(read_json(reference_path),args.output/'reconstruction'/name,area_m2=90,
                               render=args.render and name in ('000','003','018'))
            reconstructions.append(dict(reference=name,**report))
        except Exception as exc:
            reconstructions.append(dict(reference=name,passed=False,error=str(exc)))
        print('reconstruction',name,reconstructions[-1]['passed'],flush=True)
    records=[]
    for role in ('kitchen','storage','workshop'):
        for backend in ('heuristic','empirical'):
            for seed in range(args.seeds):
                program=load('examples/cafe_program.py');program['space']['kind']=role
                program['prompt']=f'A {role} workspace with work surfaces, storage cabinets and supported containers.'
                # Same inventory deliberately isolates layout from asset/LLM changes.
                path=args.output/f'{role}_{backend}_{seed}';record=dict(role=role,backend=backend,seed=seed,path=str(path))
                try:
                    ir=generate_layout(program,seed,path,backend=backend,priors=priors,allow_prior_backoff=(role=='workshop'))
                    write_json(path/'ir.json',ir);write_json(path/'program.json',program)
                    record['checks']=check_scene(freeze(program,authority='supplied_program'),ir,path)
                    compile_scene(ir,path)
                    record['physics']=validate_scene(path)['passed']
                    record['distribution']=evaluate(ir,path,priors,allow_backoff=(role=='workshop'))
                    record['pose_digest']=digest([(o['category'],o['position'],o['yaw']) for o in ir['objects']])
                    if args.render and seed==0:
                        from scene_pipeline.render import preview
                        preview(path)
                except Exception as exc:record['error']=str(exc)
                records.append(record)
                print(role,backend,seed,record.get('error',record.get('physics')),flush=True)
    # Test partition first used here, after generation. Training-scaled joint distance
    # is reported, not tuned, and compared to a calibration-to-test real-data baseline.
    batch=[]
    for role in ('kitchen','storage','workshop'):
        prior=condition(priors,role,allow_backoff=role=='workshop')
        columns=list(prior['features'])
        test=[r for r in priors['rows'] if r['split']=='test' and (role=='workshop' or r['role'].casefold()==role)]
        if not test:
            batch.append(dict(role=role,status='no_held_out_test_rooms'));continue
        observed=np.array([[r['values'][f] for f in columns] for r in test])/prior['scale']
        calibration=np.array([[r['values'][f] for f in columns] for r in prior['calibration']])/prior['scale']
        def energy(a,b):return float(2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean())
        for backend in ('heuristic','empirical'):
            selected=[r for r in records if r['role']==role and r['backend']==backend and 'distribution' in r]
            if not selected:continue
            generated=np.array([[r['distribution']['features'][f] for f in columns] for r in selected])/prior['scale']
            batch.append(dict(role=role,backend=backend,samples=len(selected),test_rooms=len(test),features=columns,
                joint_energy_to_test=energy(generated,observed),
                real_calibration_to_test=energy(calibration,observed) if len(calibration) else None,
                unique_layouts=len({r['pose_digest'] for r in selected}),
                interpretation='small-sample diagnostic; no generalization or realism acceptance claim'))
    report=dict(schema_version=1,reconstructions=reconstructions,runs=records,batch_metrics=batch,
                summary=dict(reconstructions_passed=sum(r['passed'] for r in reconstructions),reconstructions_total=len(reconstructions),
                             generated=len(records),generation_errors=sum('error' in r for r in records),
                             physics_passed=sum(r.get('physics',False) for r in records)),
                untested=['LLM intent extraction','functional zones/groups','furnished reconstruction','commercial domain generalization'])
    write_json(args.output/'report.json',report)
    comparison_plot(records,args.output/'comparison.png')
    links=''.join(f'<li><a href="reconstruction/{r["reference"]}/index.html">Reference {r["reference"]}: {r["passed"]}</a></li>' for r in reconstructions)
    (args.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Scene grounding feasibility</h1>'
        '<p>Real CubiCasa annotations; CC BY-NC 4.0. Area-normalized reconstruction, not verified metric recovery. '
        'Layout experiment uses identical supplied inventories to isolate the generator; no LLM run.</p>'
        '<img width="100%" src="comparison.png"><h2>Source-to-simulator checks</h2><ul>'+links+'</ul><pre>'
        +html.escape(str(report['summary']))+'</pre><a href="report.json">Full metrics and failures</a>')
    print(report['summary'])


if __name__=='__main__':main()
