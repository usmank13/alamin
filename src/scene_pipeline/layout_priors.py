"""Small empirical joint-prior prototype; no trained model or invented evidence."""
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .contracts import PipelineError,digest,read_json
from .reference_layout import room_features

FEATURES=('aspect','rectangularity','fixed_fixture_coverage','fixed_fixture_wall_gap','fixed_fixture_count')
# An explicit comparable subset, not an alias pretending all furniture is annotated.
FIXTURE_LABELS={'base_cabinet':'BaseCabinet','wall_cabinet':'WallCabinet','fridge':'Refrigerator'}


def verify_bundle(bundle):
    payload={k:v for k,v in bundle.items() if k!='sha256'}
    if bundle.get('schema_version')!=1 or bundle.get('kind')!='empirical_prior_bundle' or digest(payload)!=bundle.get('sha256'):
        raise PipelineError('PRIOR_HASH','Invalid prior bundle identity')
    sources={s['sha256'] for s in bundle['sources']};splits={};seen=set()
    for row in bundle['rows']:
        group=row['source_group']; split=row['split']; identity=(group,row['room_id'])
        if group not in sources:raise PipelineError('PRIOR_SOURCE','Unknown source group')
        if split not in ('train','calibration','test') or (group in splits and splits[group]!=split):
            raise PipelineError('PRIOR_LEAKAGE','A source group crosses evidence partitions')
        if identity in seen:raise PipelineError('PRIOR_DUPLICATE','Duplicate room evidence')
        splits[group]=split;seen.add(identity)
        if set(row['values'])!=set(FEATURES) or not all(math.isfinite(v) for v in row['values'].values()):
            raise PipelineError('PRIOR_FEATURES','Missing/nonfinite empirical features')
        v=row['values']
        if v['aspect']<1 or not 0<v['rectangularity']<=1.000001 or not 0<=v['fixed_fixture_coverage']<=1.000001 or v['fixed_fixture_wall_gap']<0 or v['fixed_fixture_count']<0:
            raise PipelineError('PRIOR_FEATURES','Out-of-domain empirical features')
    return bundle


def condition(bundle,role,*,allow_backoff=False):
    verify_bundle(bundle)
    rows=[r for r in bundle['rows'] if r['role'].casefold()==role.casefold()]
    sources={r['source_group'] for r in rows if r['split']=='train'}
    backoff=False
    if len(sources)<3:
        if not allow_backoff:raise PipelineError('PRIOR_UNSUPPORTED','Fewer than three training source groups for requested role',dict(role=role,source_groups=len(sources)))
        rows=bundle['rows'];backoff=True
    train=[r for r in rows if r['split']=='train']
    if len({r['source_group'] for r in train})<3:raise PipelineError('PRIOR_TOO_SMALL','Need three distinct training sources')
    # Cross-domain backoff is architecture-only: it does not ground furnishing.
    features=FEATURES[:2] if backoff else FEATURES
    matrix=np.array([[r['values'][f] for f in features] for r in train])
    scale=np.maximum(np.std(matrix,axis=0),[.05,.02,.01,.01,1.][:len(features)])
    calibration=[r for r in rows if r['split']=='calibration']
    return dict(train=train,calibration=calibration,features=features,matrix=matrix,scale=scale,
                backoff='generic_architecture_only' if backoff else 'exact_room_label',role=role)


def distance(values,prior):
    v=np.array([values[f] for f in prior['features']])
    return float(np.min(np.linalg.norm((prior['matrix']-v)/prior['scale'],axis=1)))


def scene_features(ir,root):
    if len(ir['rooms'])!=1:raise PipelineError('PRIOR_DOMAIN','Prototype features require one room')
    room=ir['rooms'][0];fixtures=[]
    for o in ir['objects']:
        if o['category'] not in FIXTURE_LABELS:continue
        package=read_json(Path(root)/o['asset'])
        if package['key']!=o['asset_key']:raise PipelineError('ASSET_ID','Asset identity mismatch')
        lo,hi=np.array(package['bounds']); c,s=math.cos(o['yaw']),math.sin(o['yaw'])
        # Derive footprints from package geometry, never the generator's bounds claim.
        polygon=[[(c*x-s*y)+o['position'][0],(s*x+c*y)+o['position'][1]]
                 for x,y in [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]]
        fixtures.append(dict(polygon=polygon,labels=[FIXTURE_LABELS[o['category']]]))
    return room_features(room,fixtures)


def evaluate(ir,root,bundle,*,allow_backoff=False):
    prior=condition(bundle,ir['rooms'][0]['role'],allow_backoff=allow_backoff)
    features=scene_features(ir,root); score=distance(features,prior)
    calibration=[distance(r['values'],prior) for r in prior['calibration']]
    # A diagnostic rank, not a calibrated pass/fail promise from a tiny corpus.
    rank=(1+sum(x>=score for x in calibration))/(1+len(calibration)) if calibration else None
    return dict(schema_version=1,scope='partial_scene_spatial_evidence',features=features,
                joint_nearest_training_distance=score,calibration_tail_rank=rank,
                calibration_samples=len(calibration),training_source_groups=len({r['source_group'] for r in prior['train']}),
                conditioning=prior['backoff'],compared_features=list(prior['features']),prior_sha256=bundle['sha256'],
                empirical_acceptance='not_calibrated_small_feasibility_sample',full_scene_distribution_verified=False)
