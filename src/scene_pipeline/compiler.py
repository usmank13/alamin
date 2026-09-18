"""SceneIR -> MuJoCo specification. No generator dependencies in sim_harness."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assets import vec, pair
from .contracts import read_json, write_json, validate_ir
from .registry import POLICY, MATERIALS
from .provenance import classify,inventory
from .contracts import PipelineError
from . import fal


def materials(asset,appearance):
    """Scene-level finishes. Flat placeholders always exist; with appearance.materials.source=='fal' every cached PATINA set
    becomes one layered PBR material (rgb/normal/roughness/metallic, metric repeat) that attached geoms are rebound to."""
    tint=appearance.get('tint',[1.,1.,1.]);options=appearance.get('materials',{});mode=options.get('source','flat');seed=int(options.get('seed',0))
    if mode not in ('flat','fal'):raise PipelineError('MATERIAL_SOURCE',f'Unknown material source {mode}')
    assets_bytes={};realized={};fallback=[]
    for finish,spec in MATERIALS.items():
        spec={**spec,**options.get('overrides',{}).get(finish,{})}
        ET.SubElement(asset,'material',name=f'finish_{finish}',rgba=vec([spec['rgba'][i]*tint[i] for i in range(3)]+[spec['rgba'][3]]),
                      shininess=str(spec['shininess']),reflectance=str(spec['reflectance']))
        if mode!='fal' or 'prompt' not in spec:continue
        model,payload=fal.patina_request(finish,seed,prompt=spec['prompt']);entry=fal.cached(model,payload)
        if entry is None:fallback.append(finish);continue
        layers=[]
        for map_name,role in fal.PATINA_ROLES.items():
            path=entry/f'{map_name}.png'
            if not path.exists():continue
            texture=f'pbr_{finish}_{role}';assets_bytes[texture+'.png']=path.read_bytes()
            ET.SubElement(asset,'texture',name=texture,type='2d',file=texture+'.png');layers.append((texture,role))
        layered=ET.SubElement(asset,'material',name=f'pbr_{finish}',rgba=vec([*tint,1.]),texuniform='true',texrepeat=vec([1/spec['tile_m']]*2),
                              shininess=str(spec['shininess']),reflectance=str(spec['reflectance']))
        for texture,role in layers:ET.SubElement(layered,'layer',texture=texture,role=role)
        meta=read_json(entry/'meta.json')
        realized[finish]=dict(model=meta['model'],request_id=meta.get('request_id'),seed=seed,tile_m=spec['tile_m'],prompt=payload['prompt'],files=meta.get('files',{}))
    if 'floor_tile' not in realized:
        # A flat floor keeps the checker so previews retain structure; its repeat follows the appearance knob.
        ET.SubElement(asset,'texture',name='floor_checker',type='2d',builtin='checker',width='128',height='128',rgb1='.32 .34 .36',rgb2='.40 .42 .44')
        floor=asset.find("material[@name='finish_floor_tile']");floor.set('texture','floor_checker');floor.set('rgba',vec([*tint,1.]))
        floor.set('texrepeat',vec([appearance.get('texture_repeat',10.)]*2))
    return assets_bytes,dict(source=mode,seed=seed,realized=realized,fallback=fallback)


def compile_scene(ir, root):
    validate_ir(ir); root=Path(root)
    xml=ET.Element('mujoco',model='generated_scene')
    ET.SubElement(xml,'compiler',angle='radian',inertiagrouprange='3 3')
    opt=ET.SubElement(xml,'option',timestep='.002',integrator='implicitfast',cone='elliptic',noslip_iterations='1')
    # Keep native robot parent filtering. Independent geometric checks still
    # inspect articulated fixture intersections without relying on this filter.
    visual=ET.SubElement(xml,'visual'); ET.SubElement(visual,'global',offwidth='640',offheight='480')
    asset=ET.SubElement(xml,'asset')
    appearance=ir['meta'].get('appearance',{})
    assets_bytes,materials_report=materials(asset,appearance)
    world=ET.SubElement(xml,'worldbody')
    vertices=np.array([p for room in ir['rooms'] for p in room['polygon']]); low=vertices.min(axis=0); high=vertices.max(axis=0)
    ET.SubElement(world,'light',pos=vec([*(.5*(low+high)),5]),dir='0 0 -1',directional='true',
                  diffuse=vec([.7*appearance.get('light_multiplier',1.)]*3))
    # Floor is finite and follows each room polygon (orthogonal L-shape split).
    edges=set()
    if 'architecture' in ir:
        from .architecture import compile_architecture
        compile_architecture(ir,asset,world)
    for room in ([] if 'architecture' in ir else ir['rooms']):
        polygon=room['polygon']; points=np.array(polygon)
        xs=sorted(set(points[:,0])); ys=sorted(set(points[:,1]))
        from .spatial import contains
        for x0,x1 in zip(xs[:-1],xs[1:]):
            for y0,y1 in zip(ys[:-1],ys[1:]):
                if contains(polygon,(x0+x1)/2,(y0+y1)/2):
                    pair(world,f"floor_{room['id']}_{x0:g}_{y0:g}",[(x1-x0)/2,(y1-y0)/2,.05],[(x0+x1)/2,(y0+y1)/2,-.05],material='finish_floor_tile')
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            edge=tuple(sorted((tuple(a),tuple(b))))
            if edge in edges:
                continue
            edges.add(edge)
            a,b=np.array(a),np.array(b); delta=b-a; length=np.linalg.norm(delta); direction=delta/length
            cuts=[(0.,length)]
            for opening in ir['openings']:
                p=np.array(opening['position'][:2]); t=float((p-a)@direction)
                if -.001<t<length+.001 and np.linalg.norm(p-(a+t*direction))<.001:
                    half=opening['width']/2+.015
                    new=[]
                    for left,right in cuts:
                        if t+half<=left or t-half>=right:
                            new.append((left,right)); continue
                        if left<t-half: new.append((left,t-half))
                        if t+half<right: new.append((t+half,right))
                    cuts=new
            for left,right in cuts:
                center=a+direction*(left+right)/2
                halfxy=abs(direction)*(right-left)/2+(1-abs(direction))*POLICY['wall_thickness_m']/2
                # Offset outward prevents consuming the declared interior dimensions.
                name=f'wall_{len(edges)}_{left:g}'
                pair(world,name,[*halfxy,room['height']/2],[*center,room['height']/2],material='finish_wall_paint')
    spec=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'),assets=assets_bytes)
    manifest={'schema_version':1,'instances':{},'class_names':{},'geom_instances':{},'joint_instances':{},
              'layout_provenance':ir['provenance']}
    children=[]
    for instance in ir['objects']:
        info=read_json(root/instance['asset'])
        child=mujoco.MjSpec.from_zip(str((root/instance['asset']).parent/info['model']))
        children.append((instance,info,child))
    # MuJoCo warns when asset-free specs are attached after a file-backed spec.
    # Attach asset-free templates first; semantic identity never depends on IDs.
    for instance,info,child in sorted(children,key=lambda item:bool(item[2].assets)):
        classification=classify(info)
        existing=manifest['class_names'].get(str(instance['class_id']))
        if existing is not None and existing!=instance['category']:
            raise PipelineError('SEMANTIC_ID_COLLISION','Two categories share a class ID',dict(categories=[existing,instance['category']],class_id=instance['class_id']))
        if classification['allowed_use']=='visual_only':
            if instance['dynamic'] or list(child.joints):
                raise PipelineError('DECORATION_PHYSICS','Generated visual-only assets cannot have dynamics or articulation')
            for geom in child.geoms:
                if geom.contype or geom.conaffinity:
                    raise PipelineError('DECORATION_PHYSICS','Generated visual-only assets cannot supply contact geometry')
        # Global scene inertia policy cannot reinterpret retrieved group numbers.
        source_model=child.compile()
        for body,compiled in zip(child.bodies,range(source_model.nbody)):
            if compiled and source_model.body_mass[compiled]>0:
                body.mass=float(source_model.body_mass[compiled]); body.ipos=source_model.body_ipos[compiled]
                body.iquat=source_model.body_iquat[compiled]; body.inertia=source_model.body_inertia[compiled]
                body.explicitinertial=True
        yaw=instance['yaw']; frame=spec.worldbody.add_frame(pos=instance['position'],quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)])
        spec.attach(child,prefix=instance['id']+'/',frame=frame)
        manifest['instances'][instance['id']]={**instance,'affordances':info['affordances'],'provenance':info['provenance'],'source_classification':classification}
        manifest['class_names'][str(instance['class_id'])]=instance['category']
    # Scene-level materials: rebind every `finish_*` placeholder, compiler geoms and attached assets alike, to the realized PBR set.
    rebound=0
    for geom in spec.geoms:
        finish=geom.material.rsplit('/',1)[-1]
        if finish.startswith('finish_') and finish[7:] in materials_report['realized']:geom.material=f'pbr_{finish[7:]}';rebound+=1
    manifest['materials']={**materials_report,'rebound_geoms':rebound,
                           'note':'MuJoCo material texture layers (rgb/normal/roughness/metallic) with metric repeat; Cycles reads them from this compiled model. Flat finishes are declared engineering colours.'}
    model=spec.compile(); data=mujoco.MjData(model); mujoco.mj_forward(model,data)
    for geom in range(model.ngeom):
        prefix=model.geom(geom).name.split('/')[0]
        if prefix in manifest['instances']: manifest['geom_instances'][str(geom)]=prefix
    for j in range(model.njnt):
        prefix=model.joint(j).name.split('/')[0]
        if prefix in manifest['instances']: manifest['joint_instances'][model.joint(j).name]=prefix
    spec.add_key(name='harness_initial',qpos=data.qpos,ctrl=data.ctrl)
    spec.add_text(name='semantic_manifest',data=__import__('json').dumps(manifest))
    spec.to_zip(str(root/'scene.mjz'))
    write_json(root/'manifest.json',manifest)
    inventory(manifest,root/'provenance.json')
    return spec,model,data
