"""URDF scene package and independent PyBullet articulation verification."""
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from .assets import vec
from .contracts import PipelineError,write_json,read_json
from .validation import load


def rotation(quat):
    return Rotation.from_quat(np.asarray(quat)[[1,2,3,0]])


def origin(element,pos,quat=(1,0,0,0)):
    ET.SubElement(element,'origin',xyz=vec(pos),rpy=vec(rotation(quat).as_euler('xyz')))


def export_urdf(root,output=None):
    root=Path(root);output=Path(output) if output else root/'urdf'
    output.mkdir(parents=True,exist_ok=True);model=load(root)
    if model.neq or model.ntendon:
        raise PipelineError('EXPORT_UNSUPPORTED','Equality constraints/tendons require an explicit export policy')
    free_roots=[];anchors=np.zeros((model.nbody,3));joint_for_body={}
    for b in range(1,model.nbody):
        ids=np.flatnonzero(model.jnt_bodyid==b)
        if len(ids)>1: raise PipelineError('EXPORT_UNSUPPORTED','Multiple joints per body require intermediate links')
        if len(ids):
            j=int(ids[0]);kind=int(model.jnt_type[j]);joint_for_body[b]=j
            if kind==0:
                if model.body_parentid[b]!=0: raise PipelineError('EXPORT_UNSUPPORTED','Nested free body')
                free_roots.append(b)
            elif kind in (2,3): anchors[b]=model.jnt_pos[j]
            else: raise PipelineError('EXPORT_UNSUPPORTED','Ball joints not supported by this URDF adapter')
    packages=[]
    reference_data=mujoco.MjData(model);mujoco.mj_forward(model,reference_data)
    def descendants(base):
        result=[base]
        for b in range(base+1,model.nbody):
            if int(model.body_parentid[b]) in result: result.append(b)
        return result
    free_set=set(sum([descendants(b) for b in free_roots],[]))
    groups=[(0,[b for b in range(model.nbody) if b not in free_set])]+[(b,descendants(b)) for b in free_roots]
    for base,bodies in groups:
        xml=ET.Element('robot',name=f'scene_{base}'); joints=[]
        for b in bodies:
            link=ET.SubElement(xml,'link',name=f'body_{b}')
            inertial=ET.SubElement(link,'inertial');origin(inertial,model.body_ipos[b]-anchors[b],model.body_iquat[b])
            ET.SubElement(inertial,'mass',value=str(float(model.body_mass[b])))
            inertia=model.body_inertia[b]
            ET.SubElement(inertial,'inertia',ixx=str(inertia[0]),iyy=str(inertia[1]),izz=str(inertia[2]),ixy='0',ixz='0',iyz='0')
            for i in np.flatnonzero(model.geom_bodyid==b):
                contact=bool(model.geom_contype[i] or model.geom_conaffinity[i]);kind=int(model.geom_type[i])
                if not contact and model.geom_rgba[i,3]==0: continue
                node=ET.SubElement(link,'collision' if contact else 'visual',name=f'geom_{i}')
                origin(node,model.geom_pos[i]-anchors[b],model.geom_quat[i]);geometry=ET.SubElement(node,'geometry');size=model.geom_size[i]
                if kind==6: ET.SubElement(geometry,'box',size=vec(2*size))
                elif kind==2: ET.SubElement(geometry,'sphere',radius=str(size[0]))
                elif kind==5: ET.SubElement(geometry,'cylinder',radius=str(size[0]),length=str(2*size[1]))
                elif kind==7:
                    mesh=int(model.geom_dataid[i]);v=int(model.mesh_vertadr[mesh]);nv=int(model.mesh_vertnum[mesh]);f=int(model.mesh_faceadr[mesh]);nf=int(model.mesh_facenum[mesh])
                    import trimesh
                    filename=f'mesh_{mesh}.obj'
                    trimesh.Trimesh(vertices=model.mesh_vert[v:v+nv],faces=model.mesh_face[f:f+nf],process=False).export(output/filename)
                    ET.SubElement(geometry,'mesh',filename=filename,scale='1 1 1')
                else: raise PipelineError('EXPORT_UNSUPPORTED',f'Unsupported geom type {kind}')
                if not contact:
                    material=ET.SubElement(node,'material',name=f'material_{i}')
                    mid=int(model.geom_matid[i]);color=model.mat_rgba[mid] if mid>=0 else model.geom_rgba[i]
                    ET.SubElement(material,'color',rgba=vec(color))
            if b==base: continue
            parent=int(model.body_parentid[b]);j=joint_for_body.get(b)
            kind='fixed' if j is None else ('prismatic' if int(model.jnt_type[j])==2 else ('revolute' if model.jnt_limited[j] else 'continuous'))
            node=ET.SubElement(xml,'joint',name=f'joint_{b}',type=kind)
            ET.SubElement(node,'parent',link=f'body_{parent}');ET.SubElement(node,'child',link=f'body_{b}')
            pos=model.body_pos[b]+rotation(model.body_quat[b]).apply(anchors[b])-anchors[parent]
            origin(node,pos,model.body_quat[b])
            if j is not None:
                ET.SubElement(node,'axis',xyz=vec(model.jnt_axis[j]))
                reference=model.qpos0[model.jnt_qposadr[j]]
                if model.jnt_limited[j]:
                    ET.SubElement(node,'limit',lower=str(model.jnt_range[j,0]-reference),upper=str(model.jnt_range[j,1]-reference),effort='1000',velocity='1')
                adr=model.jnt_dofadr[j]
                ET.SubElement(node,'dynamics',damping=str(model.dof_damping[adr]),friction=str(model.dof_frictionloss[adr]))
                joints.append(dict(name=f'joint_{b}',source=model.joint(j).name,source_id=j,body=b,
                                   limits=(model.jnt_range[j]-reference).tolist() if model.jnt_limited[j] else [-math.pi,math.pi],axis=model.jnt_axis[j].tolist()))
        filename=f'scene_{base}.urdf';ET.indent(xml);ET.ElementTree(xml).write(output/filename,encoding='unicode')
        packages.append(dict(file=filename,fixed=base==0,position=model.body_pos[base].tolist(),quaternion_xyzw=model.body_quat[base][[1,2,3,0]].tolist(),
                             joints=joints,body_ids=bodies,
                             reference_bodies={f'body_{b}':dict(mass=float(model.body_mass[b]),inertia=model.body_inertia[b].tolist(),
                                                              com=reference_data.xipos[b].tolist()) for b in bodies if b!=0}))
    metadata=dict(schema_version=1,packages=packages,source_manifest=read_json(root/'manifest.json'),
                  fidelity=dict(preserved=['metric geometry','hinge/slide frames and limits','mass and inertia','dynamic clutter as separate roots','semantic sidecar'],
                                approximated=['joint friction','effort/velocity limits use export defaults'],
                                omitted=['textures/PBR','MuJoCo solver/contact masks','springs/armature','actuators','sensor rigs']))
    import shutil
    metadata['asset_licenses']={}
    for instance in read_json(root/'ir.json')['objects']:
        package=(root/instance['asset']).parent;licenses=package/'licenses'
        if licenses.exists():
            relative=Path('licenses')/instance['asset_key']
            shutil.copytree(licenses,output/relative,dirs_exist_ok=True)
            metadata['asset_licenses'][instance['asset_key']]=str(relative)
    write_json(output/'package.json',metadata)
    return output


def verify_urdf(output):
    import pybullet as p
    started=time.perf_counter();output=Path(output).resolve();manifest=read_json(output/'package.json');client=p.connect(p.DIRECT)
    records=[];loaded=[];physical=[]
    try:
        p.setGravity(0,0,-9.81,physicsClientId=client);p.setTimeStep(.002,physicsClientId=client)
        for package in manifest['packages']:
            body=p.loadURDF(str(output/package['file']),basePosition=package['position'],baseOrientation=package['quaternion_xyzw'],useFixedBase=package['fixed'],
                            flags=p.URDF_USE_INERTIA_FROM_FILE,physicsClientId=client)
            mapping={p.getJointInfo(body,j,physicsClientId=client)[1].decode():j for j in range(p.getNumJoints(body,physicsClientId=client))}
            links={p.getJointInfo(body,j,physicsClientId=client)[12].decode():j for j in range(p.getNumJoints(body,physicsClientId=client))}
            links[f"body_{package['body_ids'][0]}"]=-1
            for name,ref in package.get('reference_bodies',{}).items():
                link=links[name];info=p.getDynamicsInfo(body,link,physicsClientId=client)
                com=p.getBasePositionAndOrientation(body,physicsClientId=client)[0] if link==-1 else p.getLinkState(body,link,physicsClientId=client)[0]
                mass_error=abs(info[0]-ref['mass']);inertia_error=float(np.max(abs(np.array(info[2])-ref['inertia'])))
                com_error=float(np.linalg.norm(np.array(com)-ref['com']))
                physical.append(dict(body=name,mass_error_kg=mass_error,inertia_error_kg_m2=inertia_error,com_error_m=com_error,
                                     passed=mass_error<1e-6 and inertia_error<1e-6 and com_error<1e-6))
            loaded.append((package,body,mapping))
        # Load the entire scene before exercising articulation, including clutter.
        for package,body,mapping in loaded:
            for item in package['joints']:
                j=mapping[item['name']];errors=[]
                for target in item['limits']:
                    # Independent runtime applies joint motor forces, not kinematic resets.
                    p.setJointMotorControl2(body,j,p.POSITION_CONTROL,targetPosition=target,force=1000,maxVelocity=3,physicsClientId=client)
                    for _ in range(1500): p.stepSimulation(physicsClientId=client)
                    errors.append(abs(p.getJointState(body,j,physicsClientId=client)[0]-target))
                records.append(dict(joint=item['source'],endpoint_errors=errors,passed=max(errors)<.005))
        report=dict(engine='PyBullet DIRECT',seconds=time.perf_counter()-started,passed=all(x['passed'] for x in records+physical) and bool(loaded),joints=records,physical_roundtrip=physical,
                    verification_scope='load_and_articulated_dynamics' if records else 'static_package_load_only',
                    articulation_verified=bool(records) and all(x['passed'] for x in records),
                    caveats=['Endpoint dynamics are not a cross-engine force/contact equivalence proof','See package.json fidelity losses'])
        write_json(output/'verification.json',report);return report
    finally:
        p.disconnect(client)
