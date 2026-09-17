"""Held-out batch evaluation, separate from proposal selection and hard validity."""
from collections import Counter
from pathlib import Path
import html
import math

import numpy as np
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon
from PIL import Image,ImageDraw

from .contracts import PipelineError,read_json,write_json,digest,validate_program
from .architecture_sampler import sample_architecture,assign_requests,eligible_rows
from .architecture_priors import family,signature
from .architecture_checks import check_architecture
from .compiler import compile_scene

FEATURES=['aspect','rectangularity','vertices','doors','windows','opening_offset_mean','opening_offset_std',
          'opening_width_mean','opening_width_std','clear_wall_fraction']


def feature_vector(row):
    points=np.asarray(row['points']);p=Polygon(points);x0,y0,x1,y1=p.bounds
    offsets=[o['offset'] for o in row['openings']];widths=[o['width'] for o in row['openings']]
    lengths=np.linalg.norm(np.roll(points,-1,axis=0)-points,axis=1)
    occupied=sum(lengths[o['edge']]*o['width'] for o in row['openings'])
    return [max(x1-x0,y1-y0)/min(x1-x0,y1-y0),p.area/((x1-x0)*(y1-y0)),len(points),
            sum(o['kind']=='door' for o in row['openings']),sum(o['kind']=='window' for o in row['openings']),
            float(np.mean(offsets)) if offsets else 0.,float(np.std(offsets)) if offsets else 0.,
            float(np.mean(widths)) if widths else 0.,float(np.std(widths)) if widths else 0.,1-occupied/sum(lengths)]


def row_from_ir(ir):
    points=np.array(ir['rooms'][0]['polygon']);points/=math.sqrt(Polygon(points).area);openings=[]
    for o in ir['openings']:
        edge=o['host_edge'];a,b=points[edge],points[(edge+1)%len(points)];length=np.linalg.norm(b-a)
        # Recover parameters from actual opening polygon, not sampled provenance.
        op=np.array(o['polygon'])/math.sqrt(ir['meta']['area_m2']);u=(b-a)/length
        along=(op-a)@u
        openings.append(dict(kind=o['kind'],edge=edge,offset=float((along.min()+along.max())/(2*length)),
                             width=float((along.max()-along.min())/length)))
    return dict(points=points.tolist(),openings=openings,family=family(points),signature=signature(points,openings))


def visual_grid(irs,path):
    columns=4;rows=max(1,math.ceil(len(irs)/columns));image=Image.new('RGB',(1200,rows*280),'white');draw=ImageDraw.Draw(image)
    for index,(seed,ir) in enumerate(irs):
        x=(index%columns)*300;y=(index//columns)*280;points=np.array(ir['rooms'][0]['polygon'])
        low=points.min(axis=0);high=points.max(axis=0);scale=220/max(high-low)
        def xy(p):return (x+35+(p[0]-low[0])*scale,y+245-(p[1]-low[1])*scale)
        draw.text((x+15,y+12),f'Seed {seed} / {family(points)}',fill='black')
        draw.polygon([xy(p) for p in points],fill='#e4e9ed',outline='#384858',width=3)
        for o in ir['openings']:
            draw.polygon([xy(p) for p in o['polygon']],fill='#2386c8' if o['kind']=='door' else '#13a586')
    image.save(path)


def evaluate_batch(program,bundle,output,*,seeds=32,allow_backoff=False,robot_radius=None,access_margin=.05):
    validate_program(program);output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS','Refusing to overwrite evaluation')
    if not 1<=seeds<=256:raise PipelineError('EVALUATION_BUDGET','Seeds must be 1..256')
    if program['objects']:raise PipelineError('EVALUATION_SCOPE','This evaluation isolates architecture; omit objects')
    eligible,mode=eligible_rows(program,bundle,allow_backoff);output.mkdir(parents=True)
    # Freeze inputs/selection before any held-out feature computation.
    write_json(output/'run_manifest.json',dict(prior_sha256=bundle['sha256'],program=program,seeds=list(range(seeds)),
              algorithm='architecture-joint-v1',conditioning=mode,robot_radius=robot_radius,access_margin=access_margin))
    records=[];accepted=[];views=[];proposed=[]
    lookup={(r['source_group'],r['room_id']):r for r in bundle['rows'] if r['split']=='train'}
    for seed in range(seeds):
        folder=output/f'seed_{seed:03d}'
        try:
            ir=sample_architecture(program,seed,folder,bundle=bundle,allow_backoff=allow_backoff,robot_radius=robot_radius,access_margin=access_margin)
            write_json(folder/'ir.json',ir);write_json(folder/'program.json',program)
            _,model,data=compile_scene(ir,folder)
            report=check_architecture(program,ir,bundle=bundle,allow_backoff=allow_backoff,robot_radius=robot_radius,
                                      access_margin=access_margin,model=model,data=data)
            write_json(folder/'architecture_validation.json',report)
            record=dict(seed=seed,passed=report['passed'],errors=report['errors'])
            if report['passed']:
                row=row_from_ir(ir);accepted.append(row);record['features']=feature_vector(row)
                if len(views)<16:views.append((seed,ir))
                if seed<3:
                    from .render import preview
                    preview(folder,cutaway=False)
        except PipelineError as exc:record=dict(seed=seed,passed=False,error=exc.as_dict())
        trials=read_json(folder/'architecture_proposals.json') if (folder/'architecture_proposals.json').exists() else []
        record['proposal_count']=len(trials)
        record['rejections']=[e for t in trials if not t['passed'] for e in t.get('errors',[t.get('error',{})])]
        for t in trials:
            if not t.get('sources'):continue
            a,b=[lookup[(s['source_group'],s['room_id'])] for s in t['sources']];w=t['weight']
            pts=(1-w)*np.array(a['points'])+w*np.array(b['points'])
            opens=[dict(kind=x['kind'],edge=x['edge'],offset=(1-w)*x['offset']+w*y['offset'],width=(1-w)*x['width']+w*y['width']) for x,y in zip(a['openings'],b['openings'])]
            proposed.append(dict(points=pts.tolist(),openings=opens))
        records.append(record)
    # Only now inspect calibration/test spatial feature values. Do not restrict them
    # to supported training topology: coverage of unseen configurations must be visible.
    def context_matches(r):
        if mode=='exact_label' and r['role'].casefold()!=program['space']['kind'].casefold():return False
        shape=program['space']['shape']
        if shape!='sampled' and not (shape==r['family'] or shape=='concave' and r['family']=='l_shape'):return False
        return assign_requests(r['points'],[dict(kind=o['kind'],host_edge=o['edge']) for o in r['openings']],program['architecture']) is not None
    partitions={split:[r for r in bundle['rows'] if r['split']==split and context_matches(r)] for split in ('train','calibration','test')}
    train=np.array([feature_vector(r) for r in partitions['train']]);scale=np.maximum(train.std(axis=0),.05)
    def matrix(rows):return np.array([feature_vector(r) for r in rows])/scale
    def energy(a,b):return float(2*cdist(a,b).mean()-cdist(a,a).mean()-cdist(b,b).mean())
    metrics=dict(feature_names=FEATURES,feature_scale=scale.tolist(),partitions={k:len(v) for k,v in partitions.items()})
    test=partitions['test'];calibration=partitions['calibration'];supported={r['signature'] for r in eligible}
    if test:
        target=matrix(test)
        metrics.update(generated_to_test_energy=energy(matrix(accepted),target) if accepted else None,
                       proposal_to_test_energy=energy(matrix(proposed),target) if proposed else None,
                       real_calibration_to_test_energy=energy(matrix(calibration),target) if calibration else None,
                       supported_test_configuration_fraction=sum(r['signature'] in supported for r in test)/len(test))
        from .registry import POLICY
        width=math.sqrt(POLICY['aspect']);height=1/width
        baseline=dict(points=[[0,0],[width,0],[width,height],[0,height]],openings=[dict(kind='door',edge=0,offset=.5,
            width=POLICY['door_clear_width_m']/math.sqrt(program['space']['area_m2'])/width)])
        if assign_requests(baseline['points'],[dict(kind='door',host_edge=0)],program['architecture']) is not None:
            metrics['heuristic_architecture_to_test_energy']=energy(matrix([baseline]*seeds),target)
        else:metrics['heuristic_baseline']='incompatible_with_requested_opening_constraints'
    else:metrics['held_out_status']='no_matching_test_evidence'
    def identity(r):return digest(dict(points=np.round(r['points'],6).tolist(),openings=[(o['kind'],o['edge'],round(o['offset'],6),round(o['width'],6)) for o in r['openings']]))
    metrics['unique_accepted_layouts']=len({identity(r) for r in accepted})
    train_ids={identity(r) for r in partitions['train']}
    metrics['exact_training_layout_matches']=sum(identity(r) in train_ids for r in accepted)
    def descriptor(r):
        pts=np.asarray(r['points']);pts=pts/math.sqrt(Polygon(pts).area)
        return np.r_[pts.ravel(),[v for o in r['openings'] for v in (o['offset'],o['width'])]]
    def nearest(query,candidates):
        values=[float(np.sqrt(np.mean((descriptor(query)-descriptor(r))**2))) for r in candidates if r['signature']==query['signature']]
        return min(values) if values else None
    near=[nearest(r,partitions['train']) for r in accepted]
    within=[nearest(r,accepted[:i]+accepted[i+1:]) for i,r in enumerate(accepted)]
    metrics['nearest_training_parameter_rms']=near
    metrics['near_training_layouts']=sum(x is not None and x<.001 for x in near)
    metrics['near_duplicate_generated_layouts']=sum(x is not None and x<.001 for x in within)
    metrics['near_duplicate_parameter_rms_tolerance']=.001
    metrics['accepted_configuration_counts']=dict(Counter(r['signature'] for r in accepted))
    metrics['test_configuration_counts']=dict(Counter(r['signature'] for r in test))
    metrics['source_copy_precision']=1e-6
    report=dict(schema_version=1,geometry_passed=sum(r['passed'] for r in records),requested=seeds,runs=records,metrics=metrics,
                proposal_count=sum(r['proposal_count'] for r in records),
                rejection_reasons=dict(Counter(e.get('code','unknown') for r in records for e in r['rejections'])),
                distribution_acceptance='diagnostics_only_no_tuned_pass_threshold',conditioning=mode,
                limitations=['No domain-specific realism certification','No manufacturing evidence','Source scale unverified',
                             'Area normalization also scales apertures; absolute doorway widths are not calibrated',
                             'Held-out target is intent-filtered, not filtered for the requested robot footprint',
                             'Held-out data scored after generation, not a generator objective','Geometry-feature agreement does not certify semantics'])
    write_json(output/'report.json',report);visual_grid(views,output/'layouts.png')
    links=''.join(f'<li><a href="seed_{s:03d}/index.html">Seed {s} native preview</a></li>' for s,_ in views[:3])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Distribution-conditioned architecture</h1>'
        '<p>Blue: doors; green: windows. Explicit area; unverified source scale. No empirical furnishing claim.</p>'
        '<img width="100%" src="layouts.png"><ul>'+links+'</ul><a href="report.json">All metrics and failures</a><pre>'
        +html.escape(str(metrics))+'</pre>')
    return report
