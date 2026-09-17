"""SceneIR -> MuJoCo specification. No generator dependencies in sim_harness."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .assets import vec, pair
from .contracts import read_json, write_json, validate_ir
from .registry import POLICY
from .provenance import classify,inventory
from .contracts import PipelineError


def compile_scene(ir, root):
    validate_ir(ir); root=Path(root)
    xml=ET.Element('mujoco',model='generated_scene')
    ET.SubElement(xml,'compiler',angle='radian',inertiagrouprange='3 3')
    opt=ET.SubElement(xml,'option',timestep='.002',integrator='implicitfast',cone='elliptic',noslip_iterations='1')
    # Keep native robot parent filtering. Independent geometric checks still
    # inspect articulated fixture intersections without relying on this filter.
    visual=ET.SubElement(xml,'visual'); ET.SubElement(visual,'global',offwidth='640',offheight='480')
    asset=ET.SubElement(xml,'asset')
    ET.SubElement(asset,'texture',name='floor_texture',type='2d',builtin='checker',width='128',height='128',rgb1='.32 .34 .36',rgb2='.40 .42 .44')
    ET.SubElement(asset,'material',name='floor_material',texture='floor_texture',texrepeat='10 10',reflectance='.1')
    ET.SubElement(asset,'material',name='wall_material',rgba='.75 .74 .7 1')
    world=ET.SubElement(xml,'worldbody')
    appearance=ir['meta'].get('appearance',{})
    tint=appearance.get('tint',[1.,1.,1.])
    for material in asset.findall('material'):
        if material.get('rgba'):
            rgba=[float(x) for x in material.get('rgba').split()]
            material.set('rgba',vec([rgba[i]*tint[i] for i in range(3)]+[rgba[3]]))
        if material.get('texrepeat'):
            repeat=appearance.get('texture_repeat',10.)
            material.set('texrepeat',vec([repeat,repeat]))
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
                    pair(world,f"floor_{room['id']}_{x0:g}_{y0:g}",[(x1-x0)/2,(y1-y0)/2,.05],[(x0+x1)/2,(y0+y1)/2,-.05],material='floor_material')
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
                pair(world,name,[*halfxy,room['height']/2],[*center,room['height']/2],material='wall_material')
    spec=mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'))
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
