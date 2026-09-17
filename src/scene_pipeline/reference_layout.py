"""Source-backed 2D layout ingestion. No LLM, inferred metres or category relabeling."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union

from .contracts import PipelineError, digest, read_json, write_json

NS = '{http://www.w3.org/2000/svg}'
EXTRACTOR = 'cubicasa-svg-v1'
COMPARABLE_FIXTURES = {'BaseCabinet','WallCabinet','Refrigerator'}


def transform(text):
    """SVG affine transforms, in the source's matrix multiplication order."""
    result = np.eye(3)
    rest = re.sub(r'([A-Za-z]+)\s*\(([^)]*)\)', '', text)
    if rest.strip(' ,\t\r\n'):
        raise PipelineError('SVG_TRANSFORM', f'Unsupported transform {text}')
    for name, raw in re.findall(r'([A-Za-z]+)\s*\(([^)]*)\)', text):
        v = [float(x) for x in re.split(r'[\s,]+', raw.strip()) if x]
        m = np.eye(3)
        if name == 'matrix' and len(v) == 6:
            a,b,c,d,e,f = v; m = np.array([[a,c,e],[b,d,f],[0,0,1.]])
        elif name == 'translate' and len(v) in (1,2):
            m[:2,2] = [v[0], v[1] if len(v)==2 else 0]
        elif name == 'scale' and len(v) in (1,2):
            m[0,0],m[1,1] = v[0],v[-1]
        elif name == 'rotate' and len(v) in (1,3):
            a = math.radians(v[0]); m[:2,:2] = [[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]]
            if len(v)==3: m[:2,2] = np.array(v[1:])-m[:2,:2]@v[1:]
        else:
            raise PipelineError('SVG_TRANSFORM', f'Unsupported transform {name}')
        result = result@m
    return result


def parse_svg(path, source):
    """Keep source coordinates, labels and missingness; skip invalid polygons visibly."""
    path = Path(path); payload = path.read_bytes()
    sha = hashlib.sha256(payload).hexdigest()
    if sha != source['sha256']:
        raise PipelineError('SOURCE_HASH', 'Annotation does not match acquisition manifest')
    if b'<!DOCTYPE' in payload or b'<!ENTITY' in payload:
        raise PipelineError('SVG_UNSAFE', 'DTD/entities are not accepted')
    root = ET.fromstring(payload)
    records = dict(rooms=[], walls=[], openings=[], fixtures=[]); issues=[]

    def walk(element, matrix):
        matrix = matrix@transform(element.get('transform',''))
        classes = element.get('class','').split()
        kind = next((k for tag,k in [('Space','rooms'),('Wall','walls'),('Door','openings'),
                                    ('Window','openings'),('FixedFurniture','fixtures')] if tag in classes),None)
        if kind:
            polygon = element.find(NS+'polygon')
            local = matrix
            if polygon is None and kind=='fixtures':
                boundary = next((x for x in element if 'BoundaryPolygon' in x.get('class','').split()),None)
                if boundary is not None:
                    polygon = boundary.find(NS+'polygon'); local = matrix@transform(boundary.get('transform',''))
            record_id = f'{kind}_{len(records[kind])}'
            if polygon is None:
                issues.append(dict(code='MISSING_POLYGON',kind=kind,classes=classes))
            else:
                raw = [float(x) for x in re.split(r'[\s,]+', polygon.get('points','').strip()) if x]
                if len(raw)<6 or len(raw)%2:
                    issues.append(dict(code='DEGENERATE_POLYGON',kind=kind,classes=classes))
                    for child in element: walk(child,matrix)
                    return
                pts = np.array(raw).reshape(-1,2)
                local = local@transform(polygon.get('transform',''))
                points = np.c_[pts,np.ones(len(pts))]@local.T
                shape = Polygon(points[:,:2])
                if not shape.is_valid or shape.area<=0:
                    issues.append(dict(code='INVALID_POLYGON',kind=kind,classes=classes))
                else:
                    records[kind].append(dict(id=record_id,source_id=element.get('id'),labels=classes,
                                              polygon=list(map(list,shape.exterior.coords[:-1]))))
        for child in element:
            walk(child,matrix)
    walk(root,np.eye(3))
    if not records['rooms'] or not records['walls']:
        raise PipelineError('REFERENCE_EMPTY', 'No usable rooms or walls')
    result=dict(schema_version=1,kind='reference_layout',extractor=EXTRACTOR,source=source,
                units='svg_units',metric_scale=None,**records,issues=issues)
    result['reference_sha256']=digest(result)
    return result


def room_features(room, fixtures):
    """Dimensionless joint features; fixture statistics cover annotated fixed fixtures only."""
    p=Polygon(room['polygon']); x0,y0,x1,y1=p.bounds
    w,d=x1-x0,y1-y0; diagonal=math.hypot(w,d)
    shapes=[Polygon(f['polygon']) for f in fixtures if COMPARABLE_FIXTURES.intersection(f['labels']) and p.covers(Polygon(f['polygon']).representative_point())]
    return dict(aspect=max(w,d)/min(w,d),rectangularity=p.area/(w*d),
                fixed_fixture_coverage=unary_union(shapes).intersection(p).area/p.area if shapes else 0.,
                fixed_fixture_wall_gap=sum(g.distance(p.boundary)/diagonal for g in shapes)/len(shapes) if shapes else 0.,
                fixed_fixture_count=len(shapes))


def audit_sample(folder, output):
    folder=Path(folder); output=Path(output)
    if output.exists(): raise PipelineError('OUTPUT_EXISTS','Refusing to overwrite corpus audit')
    manifest=read_json(folder/'sources.json'); references=[]; failures=[]; features=[]
    for entry in manifest['annotations']:
        source=dict(dataset=manifest['dataset'],url=manifest['source_url'],license=manifest['license'],
                    member=entry['archive_member'],sha256=entry['sha256'])
        try:
            r=parse_svg(folder/entry['file'],source)
        except (PipelineError,ValueError,ET.ParseError) as exc:
            failures.append(dict(file=entry['file'],error=str(exc))); continue
        references.append(r)
        for room in r['rooms']:
            if 'Outdoor' in room['labels']: continue
            features.append(dict(source_group=source['sha256'],room_id=room['id'],
                                 role=room['labels'][1] if len(room['labels'])>1 else 'Undefined',
                                 values=room_features(room,r['fixtures'])))
    # Exact source duplicates are one group. No rooms from the same source cross splits.
    groups=sorted(set(x['source_group'] for x in features))
    assignments={g:('test' if i%5==0 else 'calibration' if i%5==1 else 'train') for i,g in enumerate(groups)}
    for row in features: row['split']=assignments[row['source_group']]
    bundle=dict(schema_version=1,kind='empirical_prior_bundle',extractor=EXTRACTOR,
                evidence_scope='dimensionless_architecture_and_common_fixed_fixture_subset',
                fixture_labels=sorted(COMPARABLE_FIXTURES),
                metric_scale_verified=False,selection=manifest['selection'],
                sources=[r['source'] for r in references],rows=features)
    bundle['sha256']=digest(bundle)
    report=dict(source_count=len(references),room_count=len(features),failures=failures,
                skipped_annotations=sum(len(r['issues']) for r in references),
                roles=dict(Counter(f['role'] for f in features)),splits=dict(Counter(f['split'] for f in features)),
                license=manifest['license'],metric_scale_verified=False,
                limitations=['Small non-representative sample','No verified absolute scale',
                             'Fixed-fixture annotations are not complete furniture inventories',
                             'No commercial-workspace coverage claim'])
    write_json(output/'priors.json',bundle); write_json(output/'audit.json',report)
    for i,r in enumerate(references): write_json(output/'references'/f'{i:03d}.json',r)
    return report


def metric_geometry(reference, *, area_m2=None, metres_per_unit=None):
    """Explicit reconstruction scale: an adaptation is never a measured source scale."""
    if reference.get('kind')!='reference_layout' or reference.get('schema_version')!=1 or reference.get('units')!='svg_units':
        raise PipelineError('REFERENCE_SCHEMA','Unsupported reference contract')
    if reference.get('reference_sha256')!=digest({k:v for k,v in reference.items() if k!='reference_sha256'}):
        raise PipelineError('REFERENCE_HASH','Reference geometry does not match its normalized identity')
    if (area_m2 is None)==(metres_per_unit is None):
        raise PipelineError('REFERENCE_SCALE','Supply exactly one of target area or metres per source unit')
    rooms=[r for r in reference['rooms'] if 'Outdoor' not in r['labels']]
    if not rooms:raise PipelineError('REFERENCE_EMPTY','No interior rooms')
    floor=unary_union([Polygon(r['polygon']) for r in rooms])
    value=area_m2 if area_m2 is not None else metres_per_unit
    if not math.isfinite(value) or value<=0: raise PipelineError('REFERENCE_SCALE','Scale must be positive and finite')
    scale=math.sqrt(area_m2/floor.area) if area_m2 is not None else metres_per_unit
    x0,y0,x1,y1=floor.bounds
    def points(record): return [[(x-x0)*scale,(y1-y)*scale] for x,y in record['polygon']]
    geometry={k:[{**r,'polygon':points(r)} for r in (rooms if k=='rooms' else reference[k])]
              for k in ('rooms','walls','openings','fixtures')}
    evidence=dict(kind='target_area_normalization' if area_m2 is not None else 'user_supplied_scale',
                  metres_per_svg_unit=scale,source_metric_verified=False,target_area_m2=area_m2,
                  origin='source interior bounding-box lower-left; SVG Y reflected')
    return geometry,evidence
