"""Conservative SDF rigid-prop importer. Source meshes are data, never plugins/code.

Static use is explicit. Moving joints, includes and relative frame graphs are
rejected rather than silently flattened. Concave collision meshes are decomposed,
not replaced by the visual mesh's bounding box.
"""
from io import BytesIO
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .assets import bounds,vec
from .contracts import PipelineError,read_json,write_json
from .registry import POLICY

VERSION='sdf-rigid-v4'


def xml(path):
    raw=Path(path).read_bytes()
    if len(raw)>32*1024*1024 or b'<!DOCTYPE' in raw.upper() or b'<!ENTITY' in raw.upper():
        raise PipelineError('ASSET_XML','DTD/entities and oversized XML are not supported')
    return ET.fromstring(raw)


def numbers(text,count,default=None):
    result=np.asarray([float(x) for x in text.split()] if text else default,dtype=float)
    if result.shape!=(count,) or not np.isfinite(result).all():raise PipelineError('SDF_VALUE','Invalid finite vector')
    return result


def pose(element):
    p=element.find('pose')
    if p is not None and any(p.get(k) for k in ('relative_to','frame','rotation_format','degrees')):
        raise PipelineError('SDF_FRAME','Only local xyz/radian-RPY poses are supported')
    values=numbers(p.text if p is not None else None,6,[0]*6)
    result=np.eye(4);result[:3,:3]=Rotation.from_euler('xyz',values[3:]).as_matrix();result[:3,3]=values[:3]
    return result


def local_file(uri,root,base=None):
    """Resolve only files inside this model; never fetch file:// or model:// URIs."""
    root=Path(root).resolve();base=Path(base or root)
    if uri.startswith('model://'):
        parts=uri[len('model://'):].split('/')
        if parts[0]!=root.name and parts[0]!=read_json(root/'source.json')['candidate']['folder'].rsplit('/',1)[-1]:
            raise PipelineError('SDF_DEPENDENCY','Cross-model dependency requires a separate explicit import',uri)
        path=root/('/'.join(parts[1:]))
    elif uri.startswith('file://models/'):
        parts=uri[len('file://models/'):].split('/')
        folder=read_json(root/'source.json')['candidate']['folder'].rsplit('/',1)[-1]
        if parts[0]!=folder:raise PipelineError('SDF_DEPENDENCY','External model file reference',uri)
        path=root/('/'.join(parts[1:]))
    elif '://' in uri or uri.startswith('/'):
        raise PipelineError('SDF_PATH','External/absolute resource reference forbidden',uri)
    else:path=base/uri
    if not path.resolve().is_relative_to(root):raise PipelineError('SDF_PATH','Resource outside model package',uri)
    if not path.exists() and Path(uri).name==uri:
        # Gazebo searches the model's resource directories. Resolve an unqualified
        # texture name only when it identifies exactly one file in this package.
        matches=[p for p in root.rglob('*') if p.is_file() and p.name==uri]
        if len(matches)==1:path=matches[0]
    if not path.resolve().is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise PipelineError('SDF_PATH','Resource missing or outside model package',uri)
    return path.resolve()


class PackageResolver(trimesh.resolvers.Resolver):
    def __init__(self,root,base):self.root=Path(root);self.base=Path(base)
    def get(self,key):return local_file(key,self.root,self.base).read_bytes()
    def keys(self):return [str(p.relative_to(self.root)) for p in self.root.rglob('*') if p.is_file()]
    def write(self,name,data):raise PipelineError('SDF_PATH','Mesh resource resolver is read-only')
    def namespaced(self,namespace):
        path=(self.base/namespace).resolve()
        if not path.is_relative_to(self.root.resolve()):raise PipelineError('SDF_PATH','Resource namespace outside package')
        return PackageResolver(self.root,path)


def mesh_parts(path,root):
    factor=1.;up='Z_UP'
    if path.suffix.lower()=='.dae':
        document=xml(path);ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
        unit=document.find('c:asset/c:unit',ns)
        if unit is not None:factor=float(unit.get('meter','1'))
        up=document.findtext('c:asset/c:up_axis','Z_UP',ns)
        for reference in document.findall('.//c:library_images/c:image/c:init_from',ns):
            local_file(reference.text,root,path.parent)
    # OBJ material references must also be local before trimesh reads any resources.
    if path.suffix.lower()=='.obj':
        for line in path.read_text().splitlines():
            if line.startswith('mtllib '):
                mtl=local_file(line[7:].strip(),root,path.parent)
                for material_line in mtl.read_text().splitlines():
                    if material_line.strip().startswith(('map_','bump ')):
                        local_file(material_line.split()[-1],root,mtl.parent)
    if path.suffix.lower() not in ('.dae','.obj','.stl'):raise PipelineError('SDF_MESH','Unsupported mesh format')
    scene=trimesh.load_scene(path,process=False,resolver=PackageResolver(root,path.parent),ignore_broken=False)
    if len(scene.geometry)>256:raise PipelineError('ASSET_BUDGET','Too many mesh parts')
    if up not in ('Z_UP','Y_UP','X_UP') or not np.isfinite(factor) or factor<=0:raise PipelineError('SDF_UNITS','Unsupported COLLADA axes/units')
    # Match Gazebo Classic ColladaLoader: apply node transforms and metre scale,
    # but not an additional up_axis rotation. SDF/model exports already encode
    # their frame correction. Applying both rotates real hospital props twice.
    transform=np.eye(4);transform[:3,:3]*=factor
    parts=[]
    for node in scene.graph.nodes_geometry:
        matrix,name=scene.graph[node];mesh=scene.geometry[name].copy()
        if len(mesh.faces)>250000 or not np.isfinite(mesh.vertices).all():raise PipelineError('ASSET_BUDGET','Invalid/oversized mesh')
        mesh.apply_transform(transform@matrix);parts.append(mesh)
    if not parts:raise PipelineError('SDF_MESH','Empty mesh')
    return parts


def convex_parts(mesh):
    import coacd
    coacd.set_log_level('error')
    # Preserve disconnected convex boards/rails exactly (important for pallet gaps).
    mesh=trimesh.Trimesh(mesh.vertices,mesh.faces,process=True)
    pieces=mesh.split(only_watertight=False)
    result=[]
    for piece in pieces:
        piece=trimesh.Trimesh(piece.vertices,piece.faces,process=True)
        if len(piece.faces)<4:raise PipelineError('SDF_COLLISION','Degenerate collision surface')
        if piece.is_convex:
            result.append(piece.convex_hull);continue
        if len(piece.faces)>50000:raise PipelineError('ASSET_BUDGET','Collision mesh exceeds decomposition budget')
        hulls=coacd.run_coacd(coacd.Mesh(np.asarray(piece.vertices),np.asarray(piece.faces)),threshold=.04,
            max_convex_hull=32,preprocess_resolution=30,resolution=1000,mcts_nodes=10,mcts_iterations=30,mcts_max_depth=3,seed=0)
        result.extend(trimesh.Trimesh(v,f,process=False) for v,f in hulls)
        if len(result)>128:raise PipelineError('ASSET_BUDGET','Too many collision hulls')
    if not result:raise PipelineError('SDF_COLLISION','No collision hulls')
    return result


def import_sdf(source,output,*,mode='static'):
    source=Path(source);output=Path(output)
    if mode!='static':raise PipelineError('ASSET_CAPABILITY','SDF adapter currently supports explicit static rigid props; no invented articulation')
    config=xml(source/'model.config')
    candidates=[p.text.strip() for p in config.findall('sdf') if p.text]
    if not candidates:raise PipelineError('SDF_CONFIG','model.config has no SDF entry')
    document=xml(local_file(candidates[-1],source))
    for tag in ('plugin','include','frame','actor','sensor','world'):
        if document.find('.//'+tag) is not None:raise PipelineError('SDF_UNSUPPORTED',f'{tag} is outside rigid-prop import scope')
    models=document.findall('model')
    if len(models)!=1 or models[0].find('model') is not None:raise PipelineError('SDF_UNSUPPORTED','Expected one non-nested model')
    model=models[0]
    if model.findall('joint'):raise PipelineError('SDF_JOINT','Joint-bearing models need an articulated adapter; never silently freeze joints')
    links=model.findall('link')
    if len(links)!=1:raise PipelineError('SDF_LINKS','First SDF adapter requires one rigid link')
    top=ET.Element('mujoco',model='retrieved_prop');ET.SubElement(top,'compiler',angle='radian',inertiagrouprange='3 3')
    ET.SubElement(top,'option',timestep='.002',integrator='implicitfast')
    assets=ET.SubElement(top,'asset');world=ET.SubElement(top,'worldbody');body=ET.SubElement(world,'body',name='root')
    resources={};warnings=[];counts=dict(visual_parts=0,source_collisions=0,collision_hulls=0);support_candidates=[]
    def emit_mesh(mesh,collision,transform,color=None):
        mesh=mesh.copy();mesh.apply_transform(transform)
        name=f'mesh_{len(assets)}'
        attrs=dict(name=name,inertia='convex' if collision else 'shell',vertex=vec(mesh.vertices.reshape(-1)),face=' '.join(map(str,mesh.faces.reshape(-1))))
        material=None
        if not collision and mesh.visual.kind=='texture' and mesh.visual.uv is not None:
            visual=mesh.visual.material
            image=getattr(visual,'baseColorTexture',None)
            if image is None:image=getattr(visual,'image',None)
            if image is not None:
                if image.width*image.height>16_777_216:raise PipelineError('ASSET_BUDGET','Texture too large')
                filename=name+'.png';stream=BytesIO();image.convert('RGB').save(stream,format='PNG');resources[filename]=stream.getvalue()
                ET.SubElement(assets,'texture',name=name+'_tex',type='2d',file=filename)
                material=name+'_mat';ET.SubElement(assets,'material',name=material,texture=name+'_tex')
                attrs['texcoord']=vec(np.asarray(mesh.visual.uv).reshape(-1))
            if color is None:
                value=getattr(visual,'baseColorFactor',None)
                if value is None:value=getattr(visual,'diffuse',None)
                if value is not None:color=np.asarray(value)/255 if np.max(value)>1 else np.asarray(value)
        if color is None:color=[.65,.67,.69,1.]
        ET.SubElement(assets,'mesh',**attrs)
        common=dict(name=name,type='mesh',mesh=name,group='3' if collision else '2',
                    contype='1' if collision else '0',conaffinity='1' if collision else '0',
                    density=str(POLICY['retrieved_shell_proxy_kg_m3'] if collision else 0),rgba=vec([.8,.3,.1,0] if collision else color))
        if material:common['material']=material
        ET.SubElement(body,'geom',**common)
        counts['collision_hulls' if collision else 'visual_parts']+=1
    link=links[0];base=pose(model)@pose(link)
    for collision,tag in ((False,'visual'),(True,'collision')):
        for element in link.findall(tag):
            geometry=element.find('geometry')
            if geometry is None or len(geometry)!=1:raise PipelineError('SDF_GEOMETRY','Expected one geometry per element')
            shape=geometry[0];transform=base@pose(element)
            diffuse=element.findtext('material/diffuse');color=numbers(diffuse,4) if diffuse else None
            if element.find('material/script') is not None:warnings.append('Gazebo material scripts are not executed; diffuse/mesh texture used where available')
            if collision:counts['source_collisions']+=1
            if shape.tag=='mesh':
                if shape.find('submesh') is not None:raise PipelineError('SDF_UNSUPPORTED','Mesh sub-selection unsupported')
                filename=local_file(shape.findtext('uri',''),source)
                scale=numbers(shape.findtext('scale'),3,[1,1,1])
                if np.any(scale<=0):raise PipelineError('SDF_SCALE','Mesh scales must be positive')
                for mesh in mesh_parts(filename,source):
                    mesh.apply_scale(scale)
                    pieces=convex_parts(mesh) if collision else [mesh]
                    for piece in pieces:emit_mesh(piece,collision,transform,color)
            else:
                if shape.tag=='box':size=numbers(shape.findtext('size'),3)/2;kind='box'
                elif shape.tag=='cylinder':size=np.array([float(shape.findtext('radius')),float(shape.findtext('length'))/2]);kind='cylinder'
                elif shape.tag=='sphere':size=np.array([float(shape.findtext('radius'))]);kind='sphere'
                else:raise PipelineError('SDF_GEOMETRY',f'Unsupported geometry: {shape.tag}')
                if np.any(size<=0) or not np.isfinite(size).all():raise PipelineError('SDF_VALUE','Invalid primitive size')
                quat=Rotation.from_matrix(transform[:3,:3]).as_quat()[[3,0,1,2]]
                ET.SubElement(body,'geom',name=f'{tag}_{len(body)}',type=kind,size=vec(size),pos=vec(transform[:3,3]),quat=vec(quat),
                    group='3' if collision else '2',contype='1' if collision else '0',conaffinity='1' if collision else '0',
                    density=str(POLICY['retrieved_shell_proxy_kg_m3'] if collision else 0),rgba=vec([.8,.3,.1,0] if collision else (color if color is not None else [.65,.67,.69,1])))
                counts['collision_hulls' if collision else 'visual_parts']+=1
                if collision and kind in ('box','cylinder') and np.allclose(transform[:3,:3],np.eye(3),atol=1e-6):
                    half=size[:2] if kind=='box' else np.array([size[0],size[0]])/math.sqrt(2)
                    z=transform[2,3]+size[-1]
                    support_candidates.append(dict(center=[*transform[:2,3],z],size=(2*half-.02).tolist()))
    if not counts['visual_parts'] or not counts['source_collisions']:raise PipelineError('SDF_GEOMETRY','Both source visuals and collisions required')
    spec=mujoco.MjSpec.from_string(ET.tostring(top,encoding='unicode'),assets=resources)
    compiled=spec.compile();data=mujoco.MjData(compiled);mujoco.mj_forward(compiled,data)
    vlo,vhi=bounds(compiled,data);clo,chi=bounds(compiled,data,collision=True)
    lo=np.minimum(vlo,clo);hi=np.maximum(vhi,chi)
    if np.min(hi-lo)<.001 or np.max(hi-lo)>30:raise PipelineError('SDF_SCALE','Asset extent outside import sanity bounds')
    shift=np.array([-(lo[0]+hi[0])/2,-(lo[1]+hi[1])/2,-lo[2]])
    spec.body('root').pos=shift
    # Only verified horizontal source primitives supply rectangular support regions.
    supports=[];normalized=spec.compile();state=mujoco.MjData(normalized);mujoco.mj_forward(normalized,state)
    # MuJoCo rays skip transparent geoms; expose only the collision group in
    # this temporary query model, without changing the saved visual materials.
    normalized.geom_rgba[normalized.geom_group==3,3]=1
    groups=np.array([0,0,0,1,0,0],dtype=np.uint8);geom_id=np.zeros(1,dtype=np.int32)
    for surface in support_candidates:
        if min(surface['size'])<=.02:continue
        surface['center']=(np.array(surface['center'])+shift).tolist()
        x,y,z=surface['center'];w,d=surface['size'];clear=True;top=float(hi[2]+shift[2]+.1)
        for dx in (-w/2,0,w/2):
            for dy in (-d/2,0,d/2):
                distance=mujoco.mj_ray(normalized,state,np.array([x+dx,y+dy,top]),np.array([0.,0.,-1.]),groups,True,-1,geom_id)
                clear &= distance>=0 and abs(top-distance-z)<.002
        if clear:supports.append(surface)
    output.mkdir(parents=True,exist_ok=True);spec.to_zip(str(output/'asset.mjz'))
    result=dict(bounds=[(lo+shift).tolist(),(hi+shift).tolist()],dimensions=(hi-lo).tolist(),affordances=[],supports=supports,
        adapter_version=VERSION,counts={**counts,'textures':len(resources)},source_static=model.findtext('static','false'),mode=mode,
        collision_policy='source primitives / connected convex components / bounded CoACD decomposition of source collision meshes',
        coordinate_policy='Gazebo Classic: COLLADA node transforms and metre scale; SDF poses; no extra up_axis rotation',
        normalized_translation_m=shift.tolist(),warnings=sorted(set(warnings)),
        limitations=['Static rigid prop only; no rolling, lifting or actuation claimed','Source SDF/COLLADA scale is not a measured product dimension',
                     'Approximate source collider decomposition; cavities/contact-rich suitability need task checks',
                     'No support regions inferred from visual meshes','Inertia uses declared engineering density; source mass is not calibrated'])
    write_json(output/'import.json',result)
    return result


def main():
    import json,sys,resource
    resource.setrlimit(resource.RLIMIT_CPU,(120,120))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    req=json.load(sys.stdin)
    try:import_sdf(req['source'],req['output'],mode=req['mode'])
    except Exception as exc:
        write_json(Path(req['output'])/'failure.json',exc.as_dict() if isinstance(exc,PipelineError) else dict(code='SDF_IMPORT',message=str(exc)))
        raise SystemExit(1)


if __name__=='__main__':main()
