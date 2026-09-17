"""Conditional joint interpolation with bounded rejection, not best-of-N ranking."""
from collections import defaultdict,Counter
from copy import deepcopy
import itertools
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely import constrained_delaunay_triangles

from .contracts import PipelineError,validate_program,validate_ir,write_json,digest
from .architecture_priors import verify_bundle,family
from .architecture_checks import check_architecture,relationship
from .registry import POLICY


def assign_requests(points,openings,intent):
    """Bounded symbolic assignment; semantic role strings never affect geometry."""
    requests=sorted(intent['openings'],key=lambda r:not r['required']);nodes=0
    for kind in ('door','window'):
        hard=[r for r in requests if r['required'] and r['kind']==kind]
        if hard and sum(o['kind']==kind for o in openings)!=sum(r['count'] for r in hard):return None
    def walk(i,assigned):
        nonlocal nodes
        nodes+=1
        if nodes>512:return None
        if i==len(requests):
            return assigned if all(not r['required'] or relationship(points,assigned,r) for r in intent['relationships']) else None
        r=requests[i];available=[j for j,o in enumerate(assigned) if o['kind']==r['kind'] and 'request_id' not in o]
        for chosen in itertools.combinations(available,r['count']):
            next_assigned=deepcopy(assigned)
            for j in chosen:next_assigned[j].update(request_id=r['id'],role=r['role'])
            result=walk(i+1,next_assigned)
            if result is not None:return result
        return walk(i+1,assigned) if not r['required'] else None
    return walk(0,deepcopy(openings))


def eligible_rows(program,bundle,allow_backoff=False):
    verify_bundle(bundle)
    training=[r for r in bundle['rows'] if r['split']=='train']
    def supported(rows):
        usable=[]
        for r in rows:
            shape=program['space']['shape']
            if shape!='sampled' and not (r['family']==shape or shape=='concave' and r['family']=='l_shape'):continue
            openings=[dict(kind=o['kind'],host_edge=o['edge']) for o in r['openings']]
            if assign_requests(r['points'],openings,program['architecture']) is not None:usable.append(r)
        groups=defaultdict(set)
        for r in usable:groups[r['signature']].add(r['source_group'])
        return [r for r in usable if len(groups[r['signature']])>=3]
    exact=[r for r in training if r['role'].casefold()==program['space']['kind'].casefold()]
    rows=supported(exact);mode='exact_label'
    if not rows and allow_backoff:rows=supported(training);mode='generic_architecture_backoff'
    if not rows:raise PipelineError('PRIOR_UNSUPPORTED','No compatible configuration with three independent training sources',
                                   dict(role=program['space']['kind'],exact_label_rows=len(exact),backoff_allowed=allow_backoff))
    return rows,mode


def realize(program,a,b,weight,seed,bundle_hash,conditioning):
    points=(1-weight)*np.array(a['points'])+weight*np.array(b['points']);p=Polygon(points)
    if not p.is_valid or p.area<=0 or not p.exterior.is_ccw:raise PipelineError('PROPOSAL_POLYGON','Interpolated polygon invalid')
    points*=math.sqrt(program['space']['area_m2']/p.area);points=points.tolist();room=Polygon(points)
    wall=room.buffer(POLICY['wall_thickness_m'],join_style=2).difference(room)
    walls=[dict(id=f'wall_{i}',polygon=list(map(list,t.exterior.coords[:-1]))) for i,t in enumerate(constrained_delaunay_triangles(wall).geoms)]
    openings=[]
    for i,(x,y) in enumerate(zip(a['openings'],b['openings'])):
        offset=(1-weight)*x['offset']+weight*y['offset'];fraction=(1-weight)*x['width']+weight*y['width']
        edge=x['edge'];start,end=np.array(points[edge]),np.array(points[(edge+1)%len(points)])
        length=np.linalg.norm(end-start);u=(end-start)/length;out=np.array([u[1],-u[0]])*POLICY['wall_thickness_m']
        left=start+(end-start)*(offset-fraction/2);right=start+(end-start)*(offset+fraction/2)
        openings.append(dict(id=f'opening_{i}',kind=x['kind'],host_edge=edge,
            polygon=np.array([left,right,right+out,left+out]).tolist(),position=[*list((left+right)/2),0.],
            width=float(fraction*length),height=2.1 if x['kind']=='door' else 1.3,rooms=['main','exterior']))
    openings=assign_requests(points,openings,program['architecture'])
    if openings is None:raise PipelineError('PROPOSAL_RELATIONS','Interpolated configuration violates opening requirements')
    return validate_ir(dict(schema_version=2,meta=dict(prompt=program['prompt'],seed=seed,area_m2=program['space']['area_m2'],
        units='metres',up='Z',origin='canonical sampled room frame',pipeline_version='architecture-v1'),
        rooms=[dict(id='main',polygon=points,height=POLICY['room_height_m'],role=program['space']['kind'])],
        openings=openings,architecture=dict(walls=walls,height=POLICY['room_height_m'],door_height=2.1,window_sill=.9,window_top=2.2),
        objects=[],robots=[],provenance=dict(architecture_sampling=dict(backend='architecture',prior_sha256=bundle_hash,
        sources=[dict(source_group=r['source_group'],room_id=r['room_id']) for r in (a,b)],interpolation_weight=float(weight),
        conditioning=conditioning,configuration=a['signature'],intent_sha256=digest(program),
        metric_basis='requested_area_with_dimensionless_source_proportions',
        scale_caveat='Room-area normalization also scales apertures; absolute doorway widths are not calibrated',
        defaults=dict(wall_thickness_m=POLICY['wall_thickness_m'],height=POLICY['room_height_m'],door_height=2.1,window_sill=.9,window_top=2.2),
        semantic_function='LLM/user labels; not certified by geometric checks'),furnishing_grounding='none')))


def sample_architecture(program,seed,root,*,bundle,allow_backoff=False,robot_radius=None,access_margin=.05):
    validate_program(program)
    if program['schema_version']!=2:raise PipelineError('ARCHITECTURE_INTENT','Architecture backend requires SceneProgram v2')
    if program['architecture'].get('unsupported_requirements'):
        raise PipelineError('UNSUPPORTED_SEMANTICS','Required semantics cannot be realized by this architecture backend',
                            dict(requirements=program['architecture']['unsupported_requirements']))
    if bundle is None:raise PipelineError('PRIOR_REQUIRED','Architecture backend requires a trusted prior bundle')
    if robot_radius is not None and (not math.isfinite(robot_radius) or robot_radius<=0):raise PipelineError('ROBOT_FOOTPRINT','Radius must be positive and finite')
    if not math.isfinite(access_margin) or access_margin<0:raise PipelineError('ROBOT_FOOTPRINT','Margin must be finite and nonnegative')
    rows,mode=eligible_rows(program,bundle,allow_backoff);rng=np.random.default_rng(seed);trials=[]
    groups=defaultdict(list)
    for row in rows:groups[row['signature']].append(row)
    for attempt in range(64):
        a=rows[int(rng.integers(len(rows)))];partners=[r for r in groups[a['signature']] if r['source_group']!=a['source_group']]
        b=partners[int(rng.integers(len(partners)))];weight=float(rng.uniform(.001,.999))
        try:
            ir=realize(program,a,b,weight,seed,bundle['sha256'],mode)
            source_map={s['group']:s for s in bundle['sources']}
            ir['provenance']['architecture_sampling']['source_records']=[deepcopy(source_map[r['source_group']]) for r in (a,b)]
            ir['provenance']['architecture_sampling']['algorithm_sha256']=digest(Path(__file__).read_text())
            report=check_architecture(program,ir,bundle=bundle,robot_radius=robot_radius,access_margin=access_margin,allow_backoff=allow_backoff)
            trials.append(dict(attempt=attempt,configuration=a['signature'],sources=ir['provenance']['architecture_sampling']['sources'],
                               weight=weight,passed=report['passed'],errors=report['errors']))
            if report['passed']:
                ir['provenance']['architecture_sampling']['attempts']=attempt+1
                ir['provenance']['architecture_sampling']['sampling']='joint interpolation; first feasible; no typicality ranking'
                write_json(Path(root)/'architecture_proposals.json',trials)
                write_json(Path(root)/'architecture_validation.json',report)
                return ir
        except PipelineError as exc:trials.append(dict(attempt=attempt,passed=False,error=exc.as_dict()))
    write_json(Path(root)/'architecture_proposals.json',trials)
    raise PipelineError('ARCHITECTURE_UNSAT','Architecture proposal budget exhausted',dict(budget=64,trials=trials))
