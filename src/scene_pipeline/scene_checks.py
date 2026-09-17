"""Independent scene checks against immutable intent and resolved package geometry."""
import math
from pathlib import Path
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from .contracts import read_json,validate_ir
from .scene_intent import check_inventory


def planar(points,instance):
    c,s=math.cos(instance['yaw']),math.sin(instance['yaw'])
    return Polygon([(c*x-s*y+instance['position'][0],s*x+c*y+instance['position'][1]) for x,y in points])


def check_scene(intent,ir,root):
    validate_ir(ir);errors=check_inventory(intent,ir);packages={};geometry={}
    floor=unary_union([Polygon(r['polygon']) for r in ir['rooms']]);by_id={o['id']:o for o in ir['objects']}
    for o in ir['objects']:
        package=read_json(Path(root)/o['asset']);packages[o['id']]=package
        if package['key']!=o['asset_key'] or package['category']!=o['category']:
            errors.append(dict(code='ASSET_IDENTITY',object=o['id']))
        if not np.allclose(o['dimensions'],package['dimensions'],rtol=0,atol=1e-8):
            errors.append(dict(code='INVENTED_DIMENSIONS',object=o['id']))
        lo,hi=np.array(package['bounds'])
        footprint=planar([(lo[0],lo[1]),(hi[0],lo[1]),(hi[0],hi[1]),(lo[0],hi[1])],o)
        geometry[o['id']]=(footprint,o['position'][2]+lo[2])
        if o['category']!='door' and not floor.buffer(1e-6).covers(footprint):
            errors.append(dict(code='OUTSIDE_ROOMS',object=o['id']))
        if o['position'][2]+lo[2]<-.002:errors.append(dict(code='BELOW_FLOOR',object=o['id']))
    for o in ir['objects']:
        parent=o['support_parent']
        if not parent:continue
        if parent not in by_id or parent==o['id']:
            errors.append(dict(code='SUPPORT_REFERENCE',object=o['id']));continue
        owner=by_id[parent];footprint,bottom=geometry[o['id']];supported=False
        for surface in packages[parent]['supports']:
            x,y,z=surface['center'];w,d=surface['size'][:2]
            polygon=planar([(x-w/2,y-d/2),(x+w/2,y-d/2),(x+w/2,y+d/2),(x-w/2,y+d/2)],owner)
            if polygon.buffer(.002).covers(footprint) and abs(bottom-(owner['position'][2]+z))<=.01:
                supported=True;break
        if not supported:errors.append(dict(code='UNSUPPORTED_OBJECT',object=o['id']))
    # Recompute required relations; do not accept provenance['relations'] as proof.
    for relation in intent['relations']:
        if not relation['required']:continue
        group=[o for o in ir['objects'] if o['id'].rsplit('_',1)[0] in relation['objects']]
        kind=relation['kind'];satisfied=False
        if kind=='against_wall':
            satisfied=bool(group) and all(geometry[o['id']][0].distance(floor.boundary)<.2 for o in group)
        elif kind=='near':
            satisfied=len(group)>=2 and max(math.dist(a['position'][:2],b['position'][:2]) for a in group for b in group)<2.
        elif kind=='in_row':
            satisfied=len(group)>=2 and min(np.ptp([o['position'][0] for o in group]),np.ptp([o['position'][1] for o in group]))<.15
        elif kind=='under' and len(group)==2:
            a,b=group
            top=a['position'][2]+packages[a['id']]['bounds'][1][2]
            satisfied=top<=geometry[b['id']][1]+.01 and math.dist(a['position'][:2],b['position'][:2])<.3
        if not satisfied:errors.append(dict(code='REQUIRED_RELATION',relation=relation))
    return dict(schema_version=1,passed=not errors,errors=errors,
                scope=['required_inventory','required_relations','asset_identity','dimensions','room_containment','floor','support'],
                unverified=['natural_language_interpretation','functional_zone_geometry','robot_connectivity',
                            'full_scene_distribution'],
                note='Physics/collisions remain separate; no LLM approval used.')
