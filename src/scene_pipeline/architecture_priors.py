"""Domain-neutral joint boundary/opening evidence extracted from annotated plans."""
from collections import Counter
import math
from pathlib import Path

import numpy as np
from shapely.geometry import Polygon,LineString
from shapely.geometry.polygon import orient

from .contracts import PipelineError,digest,read_json,write_json
from .reference_layout import parse_svg

EXTRACTOR='architecture-joint-v1'
SIMPLIFY_REL=1e-4


def family(points):
    p=Polygon(points);v=np.asarray(points);edges=np.roll(v,-1,axis=0)-v
    crosses=[float(a[0]*b[1]-a[1]*b[0]) for a,b in zip(edges,np.roll(edges,-1,axis=0))]
    orthogonal=all(abs(float(a@b))<=.002*np.linalg.norm(a)*np.linalg.norm(b) for a,b in zip(edges,np.roll(edges,-1,axis=0)))
    if len(v)==4 and orthogonal:return 'rectangle'
    if len(v)==6 and orthogonal and sum(c<0 for c in crosses)==1:return 'l_shape'
    return 'concave' if any(c<0 for c in crosses) else 'convex'


def signature(points,openings):
    v=np.asarray(points);e=np.roll(v,-1,axis=0)-v
    turns=[1 if a[0]*b[1]-a[1]*b[0]>0 else -1 for a,b in zip(e,np.roll(e,-1,axis=0))]
    return digest(dict(family=family(points),turns=turns,openings=[(o['kind'],o['edge']) for o in openings]))


def canonical_boundary(points):
    # Reflect SVG Y before canonicalization; do not reflect learned layouts again.
    p=Polygon([(x,-y) for x,y in points]);diagonal=math.hypot(p.bounds[2]-p.bounds[0],p.bounds[3]-p.bounds[1])
    p=orient(p.simplify(diagonal*SIMPLIFY_REL,preserve_topology=True),sign=1.)
    if not p.is_valid or p.interiors or not 3<=len(p.exterior.coords)-1<=32:
        raise PipelineError('PRIOR_TOPOLOGY','Unsupported boundary topology')
    vertices=np.array(p.exterior.coords[:-1]);edges=np.roll(vertices,-1,axis=0)-vertices
    lengths=np.linalg.norm(edges,axis=1);choices=[];scale=math.sqrt(p.area)
    for i in np.flatnonzero(lengths>=lengths.max()*(1-1e-6)):
        u=edges[i]/lengths[i];matrix=np.array([[u[0],u[1]],[-u[1],u[0]]])
        rotated=np.roll(vertices,-int(i),axis=0)@matrix.T;offset=rotated.min(axis=0)
        normalized=(rotated-offset)/scale
        choices.append((tuple(np.round(normalized,8).ravel()),normalized,matrix,offset))
    _,normalized,matrix,offset=min(choices,key=lambda x:x[0])
    def project(points):return (np.array([(x,-y) for x,y in points])@matrix.T-offset)/scale
    return normalized.tolist(),project,scale


def extract_room(reference,room):
    points,project,scale=canonical_boundary(room['polygon']);p=Polygon(points)
    edges=[LineString([a,b]) for a,b in zip(points,points[1:]+points[:1])]
    walls=[Polygon(project(w['polygon'])) for w in reference['walls']]
    openings=[];excluded=[]
    for annotation in reference['openings']:
        op=Polygon(project(annotation['polygon']))
        # Topological wall intersection first, not nearest room/centroid alone.
        hosts=[w for w in walls if op.intersection(w).area>=op.area*.5]
        if not hosts:
            if op.distance(p.boundary)<.01:excluded.append(annotation['id'])
            continue
        candidates=[]
        for edge_id,edge in enumerate(edges):
            a,b=np.array(edge.coords);u=(b-a)/edge.length
            offsets=(np.array(op.exterior.coords[:-1])-a)@u
            lo,hi=float(offsets.min()),float(offsets.max())
            thickness=op.area/max(hi-lo,1e-9)
            if lo<-.001 or hi>edge.length+.001 or hi-lo<1e-5:continue
            if edge.distance(op)>.001 or op.centroid.distance(edge)>thickness*.75+.001:continue
            if not any(edge.intersection(w.buffer(.001)).length>=min((hi-lo)*.5,edge.length*.25) for w in hosts):continue
            candidates.append((edge_id,max(0.,lo/edge.length),min(1.,hi/edge.length)))
        if len(candidates)>1:
            excluded.append(annotation['id']);continue
        if not candidates:continue  # Opening belongs to another room, not this boundary.
        edge,lo,hi=candidates[0]
        openings.append(dict(kind='door' if 'Door' in annotation['labels'] else 'window',edge=edge,
                             offset=(lo+hi)/2,width=hi-lo,source_id=annotation['id']))
    if excluded:raise PipelineError('PRIOR_AMBIGUOUS_HOST','Ambiguous/unbound opening near room boundary',dict(openings=excluded))
    if not any(o['kind']=='door' for o in openings):raise PipelineError('PRIOR_NO_ENTRY','No verified door on room boundary')
    openings.sort(key=lambda o:(o['kind'],o['edge'],o['offset']))
    for i,a in enumerate(openings):
        for b in openings[i+1:]:
            if a['edge']==b['edge'] and abs(a['offset']-b['offset'])<(a['width']+b['width'])/2-1e-5:
                raise PipelineError('PRIOR_OVERLAP','Overlapping source openings')
    return dict(points=points,openings=openings,family=family(points),signature=signature(points,openings),
                role=room['labels'][1] if len(room['labels'])>1 else 'Undefined',room_id=room['id'])


def build_bundle(folders,output):
    output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS','Refusing to overwrite architecture priors')
    rows=[];sources={};excluded=[];splits={};hash_groups={};frozen=[]
    for folder in map(Path,folders):
        manifest=read_json(folder/'sources.json')
        if manifest.get('splits_frozen_before_download'):
            assignment=read_json(folder/'split_assignment.json');frozen.append(digest(assignment))
        else:assignment=None
        for item in manifest['annotations']:
            # Dataset building ID groups different quality versions of the same plan.
            group=manifest['dataset']+':'+item['archive_member'].split('/')[-2]
            split=item.get('split','train')
            if split not in ('train','calibration','test'):raise PipelineError('PRIOR_SPLIT','Unknown split')
            if split!='train' and (assignment is None or assignment.get(item['archive_member'])!=split):
                raise PipelineError('PRIOR_SPLIT','Evaluation assignment was not frozen at acquisition')
            group=hash_groups.get(item['sha256'],group);hash_groups[item['sha256']]=group
            if group in splits:
                if splits[group]!=split:raise PipelineError('PRIOR_LEAKAGE','Building/source crosses partitions')
                excluded.append(dict(source=group,reason='duplicate_source'));continue
            splits[group]=split
            source=dict(dataset=manifest['dataset'],url=manifest['source_url'],license=manifest['license'],
                        member=item['archive_member'],sha256=item['sha256'],group=group,split=split)
            sources[group]=source
            reference=parse_svg(folder/item['file'],source)
            if any(i['kind'] in ('rooms','walls','openings') for i in reference['issues']):
                excluded.append(dict(source=group,reason='incomplete_architectural_annotations',issues=reference['issues']));continue
            for room in reference['rooms']:
                if 'Outdoor' in room['labels']:continue
                try:rows.append(dict(**extract_room(reference,room),source_group=group,split=split))
                except PipelineError as exc:excluded.append(dict(source=group,room=room['id'],reason=exc.as_dict()))
    bundle=dict(schema_version=2,kind='architecture_prior_bundle',extractor=EXTRACTOR,rows=rows,sources=list(sources.values()),
                split_assignment_hashes=frozen,normalization='unit interior area; rotation canonicalized; no reflection augmentation',
                metric_scale_verified=False,extraction_tolerances=dict(simplify_relative=SIMPLIFY_REL,host_geometry_unit_area=.001))
    bundle['sha256']=digest(bundle)
    report=dict(rows=len(rows),source_groups=len(sources),splits=dict(Counter(r['split'] for r in rows)),excluded=excluded,
                limitations=['Source labels do not certify semantic function','Scale unverified','Noncommercial source corpus',
                             'Building-ID and exact-file dedup only; near-duplicate buildings may remain'])
    write_json(output/'priors.json',bundle);write_json(output/'audit.json',report)
    return report


def verify_bundle(bundle):
    if bundle.get('schema_version')!=2 or bundle.get('kind')!='architecture_prior_bundle' or bundle.get('sha256')!=digest({k:v for k,v in bundle.items() if k!='sha256'}):
        raise PipelineError('PRIOR_HASH','Invalid architecture prior identity')
    sources={s['group']:s for s in bundle['sources']};seen=set()
    if len(sources)!=len(bundle['sources']):raise PipelineError('PRIOR_DUPLICATE','Duplicate source group')
    for r in bundle['rows']:
        if r['source_group'] not in sources:raise PipelineError('PRIOR_SOURCE','Unknown source group')
        if r['split'] not in ('train','calibration','test') or sources[r['source_group']]['split']!=r['split']:
            raise PipelineError('PRIOR_LEAKAGE','Source partition mismatch')
        identity=(r['source_group'],r['room_id'])
        if identity in seen:raise PipelineError('PRIOR_DUPLICATE','Duplicate room')
        seen.add(identity)
        p=Polygon(r['points'])
        if not p.is_valid or p.interiors or abs(p.area-1)>1e-5 or not p.exterior.is_ccw:
            raise PipelineError('PRIOR_GEOMETRY','Invalid normalized room geometry')
        for o in r['openings']:
            if o['kind'] not in ('door','window') or not 0<=o['edge']<len(r['points']) or not 0<o['width']<=1 or not o['width']/2-1e-6<=o['offset']<=1-o['width']/2+1e-6:
                raise PipelineError('PRIOR_GEOMETRY','Invalid normalized opening')
        if r['signature']!=signature(r['points'],r['openings']) or r['family']!=family(r['points']):
            raise PipelineError('PRIOR_SIGNATURE','Inconsistent topology signature')
    return bundle
