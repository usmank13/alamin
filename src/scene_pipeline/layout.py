"""Seeded conservative placement; all dimensions come from resolved packages."""
import math
from copy import deepcopy
from pathlib import Path

import numpy as np

from .contracts import PipelineError, validate_program, validate_ir, digest
from .registry import POLICY, CATALOG, fingerprint, class_id
from .provenance import classify
from .assets import instantiate
from .spatial import contains


def intersects(a,b,margin=0.):
    return all(a[0][i] < b[1][i]+margin and b[0][i] < a[1][i]+margin for i in range(3))


def footprint(package,position,yaw,sweep=True):
    lo,hi=np.array(package['bounds'])
    if sweep and package['affordances']:
        # Conservative front clearance; detailed 3D sweep validates the compiled result.
        lo[1]-=max(package['dimensions'][:2])
    corners=np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
    c,s=math.cos(yaw),math.sin(yaw)
    R=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    world=corners@R.T+position
    return [world.min(axis=0).tolist(),world.max(axis=0).tolist()]


def floorplan(space, *, aspect=None):
    area=space['area_m2']; aspect=POLICY['aspect'] if aspect is None else aspect
    fraction=.85 if space['shape']=='l_shape' else 1.
    w=math.sqrt(area*aspect/fraction); d=area/(fraction*w)
    outline=[[0,0],[w,0],[w,d],[0,d]]
    if space['shape']=='l_shape':
        outline=[[0,0],[w,0],[w,d*.5],[w*.7,d*.5],[w*.7,d],[0,d]]
    rooms=[dict(id='main',polygon=outline,height=POLICY['room_height_m'],role=space['kind'])]
    if space['annexes']:
        if space['shape']!='rectangle':
            raise PipelineError('FLOORPLAN_DOMAIN','Annexes currently require a rectangular main footprint')
        split=d*.78
        rooms[0]['polygon']=[[0,0],[w,0],[w,split],[0,split]]
        for n in range(space['annexes']):
            x0=n*w/space['annexes']; x1=(n+1)*w/space['annexes']
            rooms.append(dict(id=f'annex_{n}',polygon=[[x0,split],[x1,split],[x1,d],[x0,d]],height=POLICY['room_height_m'],role='store'))
    return rooms,w,d


def solve(program, seed, root, *, vendor=Path('vendor/robocasa_native'), aspect=None, architecture=None):
    validate_program(program)
    root=Path(root); rng=np.random.default_rng(seed)
    if architecture is not None:
        rooms=deepcopy(architecture['rooms']);vertices=np.array(rooms[0]['polygon'])
        w,d=vertices.max(axis=0)
    else:rooms,w,d=floorplan(program['space'],aspect=aspect)
    # Entrance threshold defines (0,0), X points into room. Local layout uses
    # +Y inward initially and is transformed at finalization below.
    openings=[dict(id='entrance',position=[w/2,0,0],width=POLICY['door_clear_width_m'],height=2.1,rooms=['main','exterior'])]
    for room in rooms[1:]:
        x0,y0=room['polygon'][0]; x1,_=room['polygon'][1]
        openings.append(dict(id=room['id']+'_door',position=[(x0+x1)/2,y0,0],width=.95,height=2.1,rooms=['main',room['id']]))
    if architecture is not None:openings=deepcopy(architecture['openings'])
    objects=[]; occupied=[]; dropped=[]; package_map={}
    polygon=rooms[0]['polygon']; main_d=max(p[1] for p in polygon)
    # Keep approach corridors at openings clear even before doors are inserted.
    for opening in openings:
        x,y,_=opening['position']
        if architecture is None:occupied.append([[x-.6,y-.15,0],[x+.6,y+1.1,2.1]])
        else:
            # Generic wall-local opening reservation; no room-role interpretation.
            edge=opening['host_edge'];a=np.array(polygon[edge]);b=np.array(polygon[(edge+1)%len(polygon)])
            tangent=(b-a)/np.linalg.norm(b-a);inward=np.array([-tangent[1],tangent[0]])
            center=np.array([x,y]);half=opening['width']/2+.1
            corners=np.array([center+tangent*s+inward*t for s in (-half,half) for t in (-.15,1.1)])
            z0=0. if opening['kind']=='door' else architecture['architecture']['window_sill']
            occupied.append([[*corners.min(axis=0),z0],[*corners.max(axis=0),opening['height']+z0]])
    requests=[]
    for request in program['objects']:
        for number in range(request['count']):
            requests.append({**request,'instance':f"{request['id']}_{number}"})
    requests.sort(key=lambda x:(CATALOG.get(x['category'],{}).get('placement')=='support',CATALOG.get(x['category'],{}).get('dynamic',False)))
    # Zones become geometry: with two or more labels each owns a seeded wall midpoint and
    # membership is a soft cost. One label means no anchor, so single-zone layouts are unchanged.
    labels=sorted({r['zone'] for r in program['objects']});edges=list(zip(polygon,polygon[1:]+polygon[:1]))
    order=rng.permutation(len(edges)) if len(labels)>1 else []
    zones={label:(np.array(edges[order[i%len(edges)]][0])+np.array(edges[order[i%len(edges)]][1]))/2 for i,label in enumerate(labels)} if len(labels)>1 else {}
    boundary=None
    if architecture is not None:
        from shapely.geometry import Polygon as _Polygon
        boundary=_Polygon(polygon).boundary
    def wall_gap(pos):
        from shapely.geometry import Point
        return boundary.distance(Point(pos[:2])) if boundary is not None else min(pos[0],w-pos[0],pos[1],main_d-pos[1])
    for request in requests:
        try:
            folder,package=instantiate(request['category'],root/'assets',vendor=vendor,family=request.get('family'),
                                       dimensions=request.get('dimensions_m'),dimension_basis=request.get('dimension_basis'))
        except PipelineError as exc:
            if request['required']:
                raise
            dropped.append(dict(id=request['instance'],reason=exc.as_dict())); continue
        package_map[package['key']]=package
        lo,hi=np.array(package['bounds']); width,depth,height=hi-lo
        candidates=[]
        config=CATALOG.get(request['category']) or package.get('layout',{})
        # Soft intent guides placement. Relations name request ids; partners are the
        # already-placed instances of those requests (including earlier instances of this one).
        partners=[]
        for r in program['relations']:
            if request['id'] in r['objects']:
                others=[o for o in objects if o['id'].rsplit('_',1)[0] in r['objects']]
                if others: partners.append((r['kind'],others,2. if r['required'] else 1.))
        row_partners=[o for kind,others,_ in partners if kind=='in_row' for o in others]
        if config.get('placement')=='support':
            for support in objects:
                parent=package_map[support['asset_key']]
                for surface in parent['supports']:
                    surface_z=support['position'][2]+surface['center'][2]
                    if 'support_height_m' in config and not config['support_height_m'][0]<=surface_z<=config['support_height_m'][1]:continue
                    for dx in np.arange(-surface['size'][0]/2-lo[0]+.005,surface['size'][0]/2-hi[0]-.005+1e-8,max(width+.025,.10)):
                        for dy in np.arange(-surface['size'][1]/2-lo[1]+.005,surface['size'][1]/2-hi[1]-.005+1e-8,max(depth+.025,.10)):
                            c,s=math.cos(support['yaw']),math.sin(support['yaw'])
                            candidates.append(([support['position'][0]+c*dx-s*dy,support['position'][1]+s*dx+c*dy,
                                                surface_z-lo[2]+.001],support['yaw'],support['id']))
        elif request['category']=='door':
            if architecture is not None:
                raise PipelineError('DOOR_ASSET_FIT_UNSUPPORTED','Opening generation does not resize or rig stock door assets')
            # Explicit door requests receive actual openings; no floating doorway props.
            used=sum(o['category']=='door' for o in objects)
            if used<len(openings):
                candidates=[(openings[used]['position'],0.,None)]
        elif architecture is not None:
            z=config.get('mount_height_m',0.)-lo[2]
            local=np.array([[x,y] for x in (lo[0],hi[0]) for y in (lo[1],hi[1])])
            for a,b in zip(polygon,polygon[1:]+polygon[:1]):
                a,b=np.array(a),np.array(b);length=np.linalg.norm(b-a);tangent=(b-a)/length
                inward=np.array([-tangent[1],tangent[0]]);yaw=math.atan2(tangent[1],tangent[0])+math.pi
                c,s=math.cos(yaw),math.sin(yaw);rotated=local@np.array([[c,s],[-s,c]])
                for distance in np.arange(width/2+.1,length-width/2-.05,.12):
                    pos=a+tangent*distance+inward*(.08-float((rotated@inward).min()))
                    candidates.append(([*pos,z],yaw,None))
            if not config.get('mount_height_m'):
                low=vertices.min(axis=0)
                for x in np.arange(low[0]+.5,w-.5,.4):
                    for y in np.arange(low[1]+.5,d-.5,.4):
                        for yaw in (0.,math.pi/2):candidates.append(([float(x),float(y),z],yaw,None))
        else:
            z=config.get('mount_height_m',0.)-lo[2]
            for x in np.arange(.2+width/2,w-width/2-.1,.12):
                candidates.append(([float(x),main_d-hi[1]-.08,z],0.,None))
                candidates.append(([float(x),hi[1]+.08,z],math.pi,None))
            for y in np.arange(.25+width/2,main_d-width/2-.1,.12):
                candidates.append(([hi[1]+.08,float(y),z],math.pi/2,None))
                candidates.append(([w-hi[1]-.08,float(y),z],-math.pi/2,None))
            if config.get('placement')=='freestanding' or request['category'] in ('counter','shelf'):
                for x in np.arange(1.5,w-1.5,.4):
                    for y in np.arange(1.5,main_d-1.5,.4):
                        candidates.append(([float(x),float(y),z],0.,None))
        if config.get('placement')!='support':
            # Rows are exact by construction: flush beside each placed row partner, same facing.
            for o in row_partners:
                pw=package_map[o['asset_key']]['dimensions'][0];c,s=math.cos(o['yaw']),math.sin(o['yaw'])
                for side in (-1,1):
                    shift=side*(pw/2+width/2+POLICY['gap_m'])
                    candidates.append(([o['position'][0]+c*shift,o['position'][1]+s*shift,o['position'][2]],o['yaw'],None))
        # Seeded ordering within candidate space provides reproducible variants.
        rng.shuffle(candidates)
        if config.get('placement')=='freestanding' and architecture is None:
            candidates.sort(key=lambda c:min(c[0][0],w-c[0][0],c[0][1],main_d-c[0][1])<1.4)
        anchor=zones.get(request['zone'])
        def cost(pos,yaw):
            xy=np.array(pos[:2]);total=0.
            for kind,others,weight in partners:
                nearest=min(np.linalg.norm(xy-np.array(o['position'][:2])) for o in others)
                if kind=='near':total+=weight*max(0.,nearest-.6)
                elif kind=='under':total+=weight*nearest
                elif kind=='against_wall':total+=weight*wall_gap(pos)
                elif kind=='in_row':
                    total+=weight*min(min(abs(xy[0]-o['position'][0]),abs(xy[1]-o['position'][1])) for o in others)
                    total+=.3*weight*min(abs((yaw-o['yaw']+math.pi)%(2*math.pi)-math.pi) for o in others)
            if anchor is not None:total+=.5*float(np.linalg.norm(xy-anchor))
            return total
        if partners or anchor is not None:
            candidates.sort(key=lambda c:cost(c[0],c[1]))  # stable: seeded order breaks ties
        chosen=None
        wall_required=any(r['required'] and r['kind']=='against_wall' and request['id'] in r['objects'] for r in program['relations'])
        for pos,yaw,parent in candidates:
            volume=footprint(package,pos,yaw)
            if architecture is not None:
                from shapely.geometry import Polygon,box
                polygon_shape=Polygon(polygon);volume_shape=box(*volume[0][:2],*volume[1][:2])
                wall_distance=volume_shape.distance(polygon_shape.boundary)
            else:wall_distance=min(volume[0][0],w-volume[1][0],volume[0][1],main_d-volume[1][1])
            if wall_required and wall_distance>=.2:
                continue
            if request['category']=='door':
                chosen=(pos,yaw,parent,volume); break
            if architecture is not None and not polygon_shape.covers(volume_shape):continue
            if architecture is None and not all(contains(polygon,x,y) for x in (volume[0][0],volume[1][0]) for y in (volume[0][1],volume[1][1])):
                continue
            # Parent support contact is permitted; other objects retain 3D clearance.
            collision=False
            for index,other in enumerate(occupied):
                if parent and index>=len(openings) and objects[index-len(openings)]['id']==parent:
                    continue
                if intersects(volume,other,.002):
                    collision=True; break
            if not collision:
                chosen=(pos,yaw,parent,volume); break
        if chosen is None:
            if request['required']:
                raise PipelineError('LAYOUT_UNSAT',f"No valid placement for {request['instance']}",dict(object=request['instance']))
            dropped.append(dict(id=request['instance'],reason='no collision-free placement')); continue
        pos,yaw,parent,volume=chosen
        objects.append(dict(id=request['instance'],category=request['category'],class_id=class_id(request['category']),
                            asset=str(folder.relative_to(root)/'asset.json'),asset_key=package['key'],dimensions=package['dimensions'],
                            position=list(pos),yaw=float(yaw),zone=request['zone'],support_parent=parent,
                            dynamic=config.get('dynamic',False),initial_state={},bounds=volume,source_classification=classify(package)))
        occupied.append(volume)
    # Relations are evaluated, never silently claimed satisfied.
    relations=[]
    for relation in program['relations']:
        group=[o for o in objects if o['id'].rsplit('_',1)[0] in relation['objects']]
        passed=False
        if relation['kind']=='against_wall':
            if architecture is not None:
                from shapely.geometry import Polygon,box
                passed=bool(group) and all(box(*o['bounds'][0][:2],*o['bounds'][1][:2]).distance(Polygon(polygon).boundary)<.2 for o in group)
            else:passed=bool(group) and all(min(o['bounds'][0][0],w-o['bounds'][1][0],o['bounds'][0][1],main_d-o['bounds'][1][1])<.2 for o in group)
        elif relation['kind']=='near':
            passed=len(group)>=2 and max(np.linalg.norm(np.array(a['position'][:2])-b['position'][:2]) for a in group for b in group)<2.
        elif relation['kind']=='in_row':
            passed=len(group)>=2 and min(np.ptp([o['position'][0] for o in group]),np.ptp([o['position'][1] for o in group]))<.15
        elif relation['kind']=='under' and len(group)==2:
            a,b=group; passed=a['bounds'][1][2]<=b['bounds'][0][2]+.01 and np.linalg.norm(np.array(a['position'][:2])-b['position'][:2])<.3
        relations.append(dict(constraint=relation,satisfied=bool(passed)))
        if relation['required'] and not passed:
            raise PipelineError('RELATION_UNSAT','Required relation unsatisfied',relation)
    if architecture is not None:
        result=deepcopy(architecture);result['objects']=objects
        result['provenance'].update(program_sha256=digest(program),registry_sha256=fingerprint(),dropped=dropped,relations=relations,
                                    furnishing_grounding='heuristic_on_sampled_architecture_not_empirical')
        return validate_ir(result)
    # Rotate layout into contract frame: entrance origin, +X into room.
    def point(p):
        return [p[1], w/2-p[0], *p[2:]]
    for room in rooms:
        room['polygon']=[point(p) for p in room['polygon']]
    for opening in openings:
        opening['position']=point(opening['position'])
    for o in objects:
        o['position']=point(o['position']); o['yaw']-=math.pi/2
        o['bounds']=footprint(package_map[o['asset_key']],o['position'],o['yaw'])
    return validate_ir(dict(schema_version=1,meta=dict(prompt=program['prompt'],seed=seed,area_m2=program['space']['area_m2'],units='metres',up='Z',origin='entrance threshold',pipeline_version='0.1.0'),
                            rooms=rooms,openings=openings,objects=objects,robots=[],
                            provenance=dict(program_sha256=digest(program),registry_sha256=fingerprint(),dropped=dropped,relations=relations)))
