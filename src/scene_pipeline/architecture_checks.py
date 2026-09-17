"""Independent architecture acceptance: geometry, explicit intent and source lineage."""
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon,LineString,Point
from shapely.ops import unary_union
from shapely import set_precision

from .contracts import PipelineError,validate_program,validate_ir,digest,read_json
from .architecture import compiled_footprints


def relationship(points,openings,relation):
    groups=[[o for o in openings if o.get('request_id')==r] for r in relation['openings']]
    if not all(groups):return False
    if relation['kind']=='same_wall':return len({o['host_edge'] for g in groups for o in g})==1
    if relation['kind']=='opposite_wall':
        for a in groups[0]:
            for b in groups[1]:
                def direction(o):
                    i=o['host_edge'];v=np.array(points[(i+1)%len(points)])-points[i]
                    return v/np.linalg.norm(v)
                if float(direction(a)@direction(b))>-.999:return False
        return True
    return False


def check_architecture(program,ir,*,bundle=None,robot_radius=None,access_margin=.05,model=None,data=None,allow_backoff=False,asset_root=None):
    if program.get('schema_version')!=2 or ir.get('schema_version')!=2:
        raise PipelineError('ARCHITECTURE_INTENT','Architecture checks require a v2 program and SceneIR')
    validate_program(program)
    if not math.isfinite(access_margin) or access_margin<0 or robot_radius is not None and (not math.isfinite(robot_radius) or robot_radius<=0):
        raise PipelineError('ROBOT_FOOTPRINT','Robot radius must be positive and margin nonnegative, both finite')
    validate_ir(ir);errors=[];preferences=[]
    if program['architecture'].get('unsupported_requirements'):
        errors.append(dict(code='UNSUPPORTED_SEMANTICS',requirements=program['architecture']['unsupported_requirements']))
    if len(ir['rooms'])!=1:return dict(passed=False,errors=[dict(code='MULTI_ROOM_UNSUPPORTED')])
    points=ir['rooms'][0]['polygon'];room=Polygon(points);architecture=ir['architecture']
    walls=unary_union([Polygon(w['polygon']) for w in architecture['walls']])
    edges=[LineString([a,b]) for a,b in zip(points,points[1:]+points[:1])]
    if abs(room.area-program['space']['area_m2'])>1e-6*program['space']['area_m2']:errors.append(dict(code='AREA_MISMATCH'))
    if not room.exterior.is_ccw:errors.append(dict(code='BOUNDARY_WINDING'))
    from .architecture_priors import family
    wanted=program['space']['shape'];actual=family(points)
    if wanted!='sampled' and not (wanted==actual or wanted=='concave' and actual=='l_shape'):
        errors.append(dict(code='OUTLINE_REQUIREMENT'))
    if walls.intersection(room).area>1e-6:errors.append(dict(code='WALL_INTRUSION'))
    from .registry import POLICY
    defaults=dict(height=POLICY['room_height_m'],door_height=2.1,window_sill=.9,window_top=2.2)
    if any(abs(architecture[k]-v)>1e-8 for k,v in defaults.items()):errors.append(dict(code='UNSOURCED_HEIGHTS'))
    expected_walls=room.buffer(POLICY['wall_thickness_m'],join_style=2).difference(room)
    if walls.symmetric_difference(expected_walls).area>1e-5:errors.append(dict(code='WALL_ENVELOPE'))
    measured=[]
    for o in ir['openings']:
        p=Polygon(o['polygon']);matches=[]
        for i,edge in enumerate(edges):
            overlap=p.buffer(1e-7).intersection(edge).length
            if overlap>1e-5:matches.append((i,overlap))
        if len(matches)!=1 or not walls.buffer(1e-6).covers(p):
            errors.append(dict(code='OPENING_HOST',opening=o['id']));continue
        edge_id,width=matches[0];edge=edges[edge_id]
        a,b=np.array(edge.coords);u=(b-a)/edge.length
        coordinates=(np.array(p.exterior.coords[:-1])-a)@u
        offset=float((coordinates.max()+coordinates.min())/(2*edge.length))
        outward=np.array([u[1],-u[0]])*POLICY['wall_thickness_m']
        left=a+u*coordinates.min();right=a+u*coordinates.max()
        expected_opening=Polygon([left,right,right+outward,left+outward])
        if p.symmetric_difference(expected_opening).area>1e-6:errors.append(dict(code='OPENING_SHAPE',opening=o['id']))
        if abs(o['height']-(2.1 if o['kind']=='door' else 1.3))>1e-8:errors.append(dict(code='OPENING_HEIGHT',opening=o['id']))
        measured.append(dict(**{k:v for k,v in o.items() if k!='host_edge'},host_edge=edge_id,
                             measured_offset=offset,measured_width=width/edge.length))
        if edge_id!=o.get('host_edge') or abs(width-o['width'])>1e-5:errors.append(dict(code='OPENING_METADATA',opening=o['id']))
        if o.get('rooms')!=['main','exterior']:errors.append(dict(code='OPENING_ADJACENCY',opening=o['id']))
    for i,a in enumerate(ir['openings']):
        for b in ir['openings'][i+1:]:
            if Polygon(a['polygon']).intersection(Polygon(b['polygon'])).area>1e-8:
                errors.append(dict(code='OPENING_OVERLAP',openings=[a['id'],b['id']]))
    requests=program['architecture']['openings']
    for kind in ('door','window'):
        hard=[r for r in requests if r['kind']==kind and r['required']]
        if hard and sum(o['kind']==kind for o in measured)!=sum(r['count'] for r in hard):
            errors.append(dict(code='OPENING_COUNT',kind=kind))
    for r in requests:
        selected=[o for o in measured if o.get('request_id')==r['id']]
        ok=len(selected)==r['count'] and all(o['kind']==r['kind'] and o.get('role')==r['role'] for o in selected)
        if not ok:
            (errors if r['required'] else preferences).append(dict(code='OPENING_REQUEST',request=r['id']))
    for r in program['architecture']['relationships']:
        if not relationship(points,measured,r):
            (errors if r['required'] else preferences).append(dict(code='OPENING_RELATION',relation=r))
    evidence=ir['provenance'].get('architecture_sampling',{})
    if bundle is not None:
        from .architecture_priors import verify_bundle
        verify_bundle(bundle)
        lookup={(r['source_group'],r['room_id']):r for r in bundle['rows'] if r['split']=='train'}
        keys=[(r.get('source_group'),r.get('room_id')) for r in evidence.get('sources',[])]
        alpha=evidence.get('interpolation_weight',-1)
        if evidence.get('prior_sha256')!=bundle['sha256'] or evidence.get('intent_sha256')!=digest(program) or len(keys)!=2 or any(k not in lookup for k in keys) or not 0<=alpha<=1:
            errors.append(dict(code='EVIDENCE_LINEAGE'))
        else:
            a,b=[lookup[k] for k in keys]
            if 'source_records' in evidence:
                source_map={s['group']:s for s in bundle['sources']}
                if evidence['source_records']!=[source_map[r['source_group']] for r in (a,b)]:errors.append(dict(code='SOURCE_METADATA'))
            mode=evidence.get('conditioning')
            valid_mode=mode=='exact_label' or mode=='generic_architecture_backoff' and allow_backoff
            if not valid_mode or mode=='exact_label' and any(r['role'].casefold()!=program['space']['kind'].casefold() for r in (a,b)):
                errors.append(dict(code='EVIDENCE_CONDITIONING'))
            supporting={r['source_group'] for r in lookup.values() if r['signature']==a['signature'] and
                        (mode=='generic_architecture_backoff' or r['role'].casefold()==program['space']['kind'].casefold())}
            if len(supporting)<3:errors.append(dict(code='EVIDENCE_SUPPORT'))
            if a['source_group']==b['source_group'] or a['signature']!=b['signature']:
                errors.append(dict(code='EVIDENCE_CONFIGURATION'))
            else:
                expected=(1-alpha)*np.array(a['points'])+alpha*np.array(b['points'])
                expected*=math.sqrt(program['space']['area_m2']/Polygon(expected).area)
                if np.shape(points)!=expected.shape or not np.allclose(points,expected,atol=1e-7,rtol=0):errors.append(dict(code='INVENTED_BOUNDARY'))
                if len(measured)!=len(a['openings']):errors.append(dict(code='INVENTED_OPENINGS'))
                else:
                    for o,x,y in zip(measured,a['openings'],b['openings']):
                        if o['kind']!=x['kind'] or o['host_edge']!=x['edge'] or abs(o['measured_offset']-((1-alpha)*x['offset']+alpha*y['offset']))>1e-6 or abs(o['measured_width']-((1-alpha)*x['width']+alpha*y['width']))>1e-6:
                            errors.append(dict(code='INVENTED_OPENING',opening=o['id']))
    access=dict(status='unverified_no_robot_footprint')
    if robot_radius is not None:
        radius=robot_radius+access_margin
        free=room.buffer(-radius)
        # Optional furnishings reserve their complete conservative swept volumes.
        obstacles=[]
        if ir['objects'] and asset_root is None:errors.append(dict(code='OBJECT_GEOMETRY_REQUIRED'))
        elif ir['objects']:
            for o in ir['objects']:
                package=read_json(Path(asset_root)/o['asset']);lo,hi=np.array(package['bounds'])
                if o['position'][2]+lo[2]>=.5:continue
                if package['affordances']:lo[1]-=max(package['dimensions'][:2])
                c,s=math.cos(o['yaw']),math.sin(o['yaw'])
                obstacle=Polygon([(c*x-s*y+o['position'][0],s*x+c*y+o['position'][1]) for x,y in
                    [(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])]])
                obstacles.append(obstacle.buffer(radius))
        if obstacles:free=free.difference(unary_union(obstacles))
        components=list(free.geoms) if free.geom_type=='MultiPolygon' else [free]
        entries=[]
        required_ids={r['id'] for r in requests if r['required'] and r['kind']=='door'}
        doors=[o for o in measured if o['kind']=='door' and (not required_ids or o.get('request_id') in required_ids)]
        for o in doors:
            edge=edges[o['host_edge']];a,b=np.array(edge.coords);u=(b-a)/edge.length;n=np.array([-u[1],u[0]])
            mid=a+(b-a)*o['measured_offset'];probe=Point(mid+n*(radius+1e-4))
            component=next((i for i,c in enumerate(components) if c.covers(probe)),None)
            if o['width']<2*radius or component is None:errors.append(dict(code='ENTRY_CLEARANCE',opening=o['id']))
            entries.append(component)
        connected=bool(entries) and None not in entries and len(set(entries))==1
        if not connected:errors.append(dict(code='ENTRY_CONNECTIVITY'))
        access=dict(status='checked',connected=connected,radius_m=robot_radius,margin_m=access_margin,
                    required_entries=len(entries),model='disc; optional object swept-AABB obstacles; not manipulation reachability')
    compiled=None
    if model is not None:
        floor=unary_union([room]+[Polygon(o['polygon']) for o in ir['openings'] if o['kind']=='door'])
        checks={}
        for name,expected,observed in [('floor',floor,compiled_footprints(model,data,'floor_reference'))]+[
            (f'walls_z_{z}',walls.difference(unary_union([Polygon(o['polygon']) for o in ir['openings']
                if o['kind']=='door' and z<architecture['door_height'] or o['kind']=='window' and architecture['window_sill']<z<architecture['window_top']])),
             compiled_footprints(model,data,'wall_reference',z=z)) for z in (.4,1.2,2.5)]:
            expected=set_precision(expected,1e-5);diff=expected.symmetric_difference(observed).area
            boundary=0. if expected.is_empty and observed.is_empty else expected.boundary.hausdorff_distance(observed.boundary)
            ok=math.isfinite(boundary) and boundary<=.001 and diff<=1e-4*max(1,expected.area)
            checks[name]=dict(passed=ok,symmetric_difference_m2=diff,boundary_error_m=boundary if math.isfinite(boundary) else None)
            if not ok:errors.append(dict(code='COMPILED_GEOMETRY',section=name))
        compiled=checks
    return dict(schema_version=1,passed=not errors,errors=errors,unsatisfied_preferences=preferences,access=access,
                compiled_geometry=compiled,semantic_function='unverified_labels_only',distribution_acceptance='evaluated_separately',
                checks=['boundary','wall_envelope','opening_hosts','counts','relations','source_parameters'])
