"""Offline Inicio bundle import and portable MuJoCo agricultural scenes.

The simulation side never imports Inicio, Blender or L-Py. Geometry and source
metadata cross that boundary only through the versioned, checksummed bundle.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec
from .contracts import PipelineError, read_json, write_json


PROFILE = dict(schema_version=1, calibrated=False, chassis_mass_kg=70., wheel_mass_kg=2.,
               chassis_diaginertia_kg_m2=[20., 8., 24.], friction=[.8, .005, .0001],
               velocity_gain=30., max_wheel_torque_nm=35., max_forward_mps=1., max_yaw_rate_rps=1.,
               provenance={'geometry': 'Inicio Element CAD and RAPTOR_30 manifest',
                           'dynamics': 'Prototype estimates, not measured vehicle parameters'},
               limitations=['Fixed payload deck; no suspension', 'Rigid wheels and terrain',
                            'Open-loop skid steering: commanded twist differs from achieved motion',
                            'No arms, deformable soil or plant contact'])


def vector(value):
    return ' '.join(f'{float(v):.9g}' for v in value)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def checked_path(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()) or not path.is_file():
        raise PipelineError('AGRICULTURE_ASSET', f'Missing or nonportable bundle asset: {relative}')
    return path


def load_bundle(root):
    root = Path(root)
    bundle = read_json(root / 'bundle.json')
    if (bundle.get('schema_version') != 1 or bundle.get('kind') != 'inicio_sim_bundle'
            or bundle.get('units') != 'metres' or bundle.get('up_axis') != 'Z'):
        raise PipelineError('AGRICULTURE_VERSION', 'Expected Inicio simulation bundle v1, metres, Z-up')
    files = bundle.get('sha256', {})
    for relative, expected in files.items():
        if sha256(checked_path(root, relative)) != expected:
            raise PipelineError('AGRICULTURE_CHECKSUM', relative)
    meshes = [m for i in bundle['instances'] for m in i['meshes']]
    meshes += bundle['robot']['meshes']
    meshes += [m for w in bundle['robot']['wheels'] for m in w['meshes']]
    required = {bundle['terrain']['file'], *(m['file'] for m in meshes)}
    if not required <= files.keys():
        raise PipelineError('AGRICULTURE_CHECKSUM', 'Every referenced asset must have a checksum')
    if bundle['robot']['type'] != 'RAPTOR_30' or len(bundle['robot']['wheels']) != 4:
        raise PipelineError('AGRICULTURE_ROBOT', 'Expected four-wheel RAPTOR_30')
    ids = [i['id'] for i in bundle['instances']]
    if len(ids) != len(set(ids)):
        raise PipelineError('AGRICULTURE_ID', 'Duplicate instance identity')
    return bundle


def mesh_bytes(path):
    """MuJoCo MSH keeps portable meshes binary instead of inflating MJCF XML."""
    with np.load(path, allow_pickle=False) as arrays:
        vertices = np.asarray(arrays['vertices'], dtype='<f4')
        faces = np.asarray(arrays['faces'], dtype='<i4')
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3
            or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices)):
        raise PipelineError('AGRICULTURE_MESH', str(path))
    # Leaves can be perfectly planar. A sub-micron backing surface gives the
    # mesh compiler a convex hull without introducing a plant collision shape.
    centered = vertices - vertices.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    if singular[-1] < 1e-7:
        count = len(vertices)
        vertices = np.concatenate([vertices, vertices + axes[-1] * 1e-6]).astype('<f4')
        faces = np.concatenate([faces, faces[:, ::-1] + count]).astype('<i4')
    return struct.pack('<4i', len(vertices), 0, 0, len(faces)) + vertices.tobytes() + faces.tobytes()


def add_meshes(asset, body, meshes, root, assets, prefix, geom_labels=None, instance_id=None):
    for index, mesh in enumerate(meshes):
        content = mesh_bytes(checked_path(root, mesh['file']))
        filename = hashlib.sha256(content).hexdigest() + '.msh'
        name = f'{prefix}_{index}'
        assets[filename] = content
        ET.SubElement(asset, 'mesh', name=name, file=filename, inertia='shell', maxhullvert='64')
        ET.SubElement(body, 'geom', name=name, type='mesh', mesh=name, rgba=vector(mesh['rgba']),
                      contype='0', conaffinity='0', mass='0', group='2')
        if geom_labels is not None:
            geom_labels[name] = dict(instance=instance_id, class_id=mesh.get('class_id', 0))


def build_robot(bundle, root, profile=None):
    profile = {**PROFILE, **(profile or {})}
    robot = bundle['robot']
    xml = ET.Element('mujoco', model='element_raptor30')
    ET.SubElement(xml, 'compiler', angle='radian')
    asset = ET.SubElement(xml, 'asset'); assets = {}
    body = ET.SubElement(ET.SubElement(xml, 'worldbody'), 'body', name='chassis')
    ET.SubElement(body, 'freejoint', name='root')
    ET.SubElement(body, 'inertial', pos='0 0 -.18', mass=str(profile['chassis_mass_kg']),
                  diaginertia=vector(profile['chassis_diaginertia_kg_m2']))
    ET.SubElement(body, 'geom', name='chassis_collision', type='box', size='.45 .77 .08',
                  pos='0 0 .02', group='3')
    for side in (-1, 1):
        ET.SubElement(body, 'geom', name=f'rail_{side}', type='box', size='.36 .035 .23',
                      pos=vector([0, side*.76, -.34]), group='3')
    add_meshes(asset, body, robot['meshes'], root, assets, 'visual')
    actuator = ET.SubElement(xml, 'actuator')
    for wheel in robot['wheels']:
        name = wheel['id']
        child = ET.SubElement(body, 'body', name=name, pos=vector(wheel['center_m']))
        ET.SubElement(child, 'joint', name=name, axis='0 1 0', damping='.1')
        ET.SubElement(child, 'geom', name=name+'_collision', type='cylinder',
                      size=vector([wheel['radius_m'], wheel['half_width_m']]),
                      quat=vector([math.sqrt(.5), math.sqrt(.5), 0, 0]),
                      mass=str(profile['wheel_mass_kg']), group='3', condim='3',
                      friction=vector(profile['friction']))
        add_meshes(asset, child, wheel['meshes'], root, assets, name+'_visual')
        ET.SubElement(actuator, 'velocity', name=name, joint=name, kv=str(profile['velocity_gain']),
                      forcelimited='true', forcerange=vector([-profile['max_wheel_torque_nm'], profile['max_wheel_torque_nm']]))
    sensor = ET.SubElement(xml, 'sensor')
    ET.SubElement(body, 'site', name='imu', pos='0 0 0', size='.002', group='4')
    for kind in ('gyro', 'accelerometer'):
        ET.SubElement(sensor, kind, name=kind, site='imu')
    for name, camera in robot['cameras'].items():
        optics = camera['optics']
        fovy = math.degrees(2*math.atan(optics['sensor_height_mm']/(2*optics['lens_mm'])))
        ET.SubElement(body, 'camera', name=name, pos=vector(camera['position_m']),
                      quat=vector(camera['quaternion_wxyz']), fovy=str(fovy))
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'), assets=assets)
    spec.compile()
    profile.update(track_width_m=robot['track_width_m'], wheels=robot['wheels'], cameras=robot['cameras'],
                   root_ground_height_m=max(w['radius_m']-w['center_m'][2] for w in robot['wheels']))
    return spec, profile


def terrain_data(bundle, root):
    with np.load(checked_path(root, bundle['terrain']['file']), allow_pickle=False) as arrays:
        z = arrays['elevation_m'].astype(np.float64)
    bounds = np.asarray(bundle['terrain']['bounds_m'], dtype=float)
    if (z.ndim != 2 or min(z.shape) < 2 or not np.isfinite(z).all()
            or bounds.shape != (2,2) or not np.isfinite(bounds).all() or not (bounds[1]>bounds[0]).all()):
        raise PipelineError('AGRICULTURE_TERRAIN', 'Invalid heightfield or bounds')
    return z, bounds


def compile_environment(bundle, root, props=()):
    z, bounds = terrain_data(bundle, root)
    low = float(z.min()); height = max(.001, float(z.max()-low))
    xml = ET.Element('mujoco', model='agricultural_field')
    ET.SubElement(xml, 'compiler', angle='radian')
    ET.SubElement(xml, 'option', timestep='.002', integrator='implicitfast', cone='elliptic')
    visual = ET.SubElement(xml, 'visual')
    ET.SubElement(visual, 'global', offwidth='1280', offheight='720')
    ET.SubElement(visual, 'map', znear='.001', zfar='20')
    ET.SubElement(visual, 'headlight', ambient='.35 .35 .35', diffuse='.5 .5 .5')
    asset = ET.SubElement(xml, 'asset'); assets = {}
    # Binary heightfields are indexed [y, x], increasing from the lower bounds.
    assets['terrain.bin'] = struct.pack('<2i', *z.shape) + ((z-low)/height).astype('<f4').tobytes()
    ET.SubElement(asset, 'hfield', name='terrain', file='terrain.bin',
                  size=vector([*((bounds[1]-bounds[0])/2), height, .1]))
    world = ET.SubElement(xml, 'worldbody')
    ET.SubElement(world, 'light', pos='0 0 6', dir='.3 .2 -1', directional='true', diffuse='.7 .7 .65')
    ET.SubElement(world, 'geom', name='terrain', type='hfield', hfield='terrain',
                  pos=vector([*((bounds[1]+bounds[0])/2), low]), rgba='.28 .19 .11 1', friction='.8 .005 .0001')
    labels = {'terrain':dict(instance=None, class_id=0)}
    instances = {}
    for item in bundle['instances']:
        identity = item['id']
        instances[identity] = {k:v for k,v in item.items() if k!='meshes'}
        add_meshes(asset, world, item['meshes'], root, assets, identity, labels, identity)
        if item.get('rock'):
            low_m = np.min([m['bounds_m'][0] for m in item['meshes']],axis=0)
            high_m = np.max([m['bounds_m'][1] for m in item['meshes']],axis=0)
            if max(high_m-low_m) >= .04:
                ET.SubElement(world,'geom',name=identity+'_collision',type='ellipsoid',
                              pos=vector((high_m+low_m)/2),size=vector(np.maximum(.001,(high_m-low_m)/2)),group='3')
    spec = mujoco.MjSpec.from_string(ET.tostring(xml,encoding='unicode'),assets=assets)
    for item in props:
        name = item['name']
        if not __import__('re').fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name) or name in instances:
            raise PipelineError('AGRICULTURE_PROP', f'Invalid or duplicate prop name: {name}')
        child = _read_spec(item['model'])
        for key in list(child.keys): child.delete(key)
        child.nkey=0
        pos = np.asarray(item.get('position_m',[0,0,0]), dtype=float)
        quat = np.asarray(item.get('quaternion_wxyz',[1,0,0,0]), dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all() or quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat),1):
            raise PipelineError('AGRICULTURE_PROP','Invalid prop transform')
        # Assign names even to anonymous geometry so semantic identity survives attach.
        for i, geom in enumerate(child.geoms):
            geom.name = f'geom_{i}'
            labels[name+'/'+geom.name] = dict(instance=name,class_id=9)
        spec.attach(child,prefix=name+'/',frame=spec.worldbody.add_frame(pos=pos,quat=quat))
        instances[name] = dict(id=name, category='obstruction', position_m=pos.tolist(), source_sha256=sha256(item['model']))
    return spec, dict(schema_version=1, domain='agriculture', instances=instances, geom_labels=labels,
                      class_names=bundle['class_names'], rows=bundle['rows'], terrain=bundle['terrain'])


def compose_robot(environment, robot, profile, spawn, bounds):
    xy = np.asarray(spawn.get('position_m',[0,0]),dtype=float)
    yaw = float(spawn.get('yaw_rad',0))
    if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
        raise PipelineError('ROBOT_PLACEMENT','Spawn requires two finite XY coordinates and a finite yaw')
    rotation = np.array([[math.cos(yaw),-math.sin(yaw)],[math.sin(yaw),math.cos(yaw)]])
    footprint = robot_footprint(profile) @ rotation.T + xy
    if (footprint < bounds[0]).any() or (footprint > bounds[1]).any():
        raise PipelineError('ROBOT_PLACEMENT','Element footprint lies outside terrain bounds')
    spec = environment.copy()
    # Ray queries use the collision surface, including user-supplied obstacles.
    model = spec.compile(); data = mujoco.MjData(model); mujoco.mj_forward(model,data)
    heights = []
    for wheel in profile['wheels']:
        for dx in (-wheel['radius_m'],0,wheel['radius_m']):
            for dy in (-wheel['half_width_m'],0,wheel['half_width_m']):
                point = rotation @ (np.asarray(wheel['center_m'][:2])+[dx,dy]) + xy
                ray = np.array([*point,100.]); geom = np.array([-1],dtype=np.int32)
                distance = mujoco.mj_ray(model,data,ray,np.array([0.,0.,-1.]),np.array([1,1,0,1,0,0],dtype=np.uint8),1,-1,geom)
                if distance < 0:
                    raise PipelineError('ROBOT_PLACEMENT','No ground beneath wheel')
                if model.geom(int(geom[0])).name != 'terrain':
                    raise PipelineError('ROBOT_PLACEMENT','Obstacle beneath wheel; choose a clear spawn')
                heights.append(100-distance+wheel['radius_m']-wheel['center_m'][2])
    height = max(heights)+.015
    for key in list(spec.keys):spec.delete(key)
    spec.nkey=0
    child = robot.copy()
    for key in list(child.keys):child.delete(key)
    child.nkey=0
    spec.attach(child,prefix='element/',frame=spec.worldbody.add_frame(pos=[*xy,height],quat=[math.cos(yaw/2),0,0,math.sin(yaw/2)]))
    model=spec.compile(); data=mujoco.MjData(model)
    for name, value in profile.get('initial_joint_positions', {}).items():
        data.joint('element/'+name).qpos[0] = value
    for name, value in profile.get('initial_controls', {}).items():
        data.ctrl[model.actuator('element/'+name).id] = value
    mujoco.mj_forward(model,data)
    for contact in data.contact:
        if contact.dist < -.001:
            names=[model.geom(int(g)).name for g in contact.geom]
            raise PipelineError('ROBOT_PLACEMENT',f'Initial collision {names} ({contact.dist:.4f} m); choose a clear spawn')
    return spec,model,data


def robot_footprint(profile):
    low, high = profile.get('footprint_xy_m', [[-.65,-1.], [.65,1.]])
    return np.array([[low[0],low[1]], [low[0],high[1]], [high[0],low[1]], [high[0],high[1]]])


def compile_bundle(bundle_root, output, *, simulation=None):
    """Compile an existing bundle; independent of the producer checkout."""
    bundle_root=Path(bundle_root); output=Path(output)
    if output.exists():raise PipelineError('OUTPUT_EXISTS',str(output))
    bundle=load_bundle(bundle_root)
    simulation=simulation if simulation is not None else bundle['input_config'].get('simulation',{})
    props=[]
    for prop in simulation.get('props',[]):
        path=checked_path(bundle_root,prop['model'])
        if prop['model'] not in bundle['sha256']:
            raise PipelineError('AGRICULTURE_CHECKSUM','Prop models must be packaged and checksummed')
        props.append({**prop,'model':str(path)})
    environment,manifest=compile_environment(bundle,bundle_root,props)
    robot,profile=build_robot(bundle,bundle_root)
    bounds=np.asarray(bundle['terrain']['bounds_m'])
    spawn=simulation.get('spawn',{})
    spec,model,data=compose_robot(environment,robot,profile,spawn,bounds)
    output.mkdir(parents=True)
    shutil.copytree(bundle_root,output/'bundle')
    export_mujoco(environment,mujoco.MjData(environment.compile()),output/'environment.mjz')
    export_mujoco(robot,mujoco.MjData(robot.compile()),output/'robot.mjz')
    export_mujoco(spec,data,output/'scene.mjz')
    write_json(output/'robot_profile.json',profile)
    write_json(output/'manifest.json',manifest)
    write_json(output/'generation.json',dict(domain='agriculture',spawn=spawn,provenance=bundle['provenance'],
                                            limitations=bundle['limitations'],robot_dynamics=profile['provenance']))
    report=dict(passed=True,domain='agriculture',plants=sum(i['category'] in ('crop','weed','grass') for i in bundle['instances']),
                rows=len(bundle['rows']),cameras=len(profile['cameras']),geoms=model.ngeom,
                bundle_sha256=sha256(bundle_root/'bundle.json'),output=str(output))
    write_json(output/'validation.json',report)
    preview(output,model,data)
    return report


def generate(config, inicio_root, blender, output, *, timeout=900):
    """Run the producer in its Blender environment, then import its bundle."""
    import tempfile
    output=Path(output); config=Path(config).resolve(); inicio_root=Path(inicio_root).resolve()
    if output.exists():raise PipelineError('OUTPUT_EXISTS',str(output))
    exporter=inicio_root/'src/tools/export_sim_bundle.py'
    if not exporter.is_file():raise PipelineError('INICIO_EXPORTER',f'Missing {exporter}; install the Inicio simulation exporter')
    executable=shutil.which(str(blender))
    if not executable:raise PipelineError('BLENDER_MISSING',str(blender))
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='inicio-export-',dir=output.parent) as work:
        work=Path(work); log=work/'export.log'
        command=[executable,'--background','--factory-startup','--python-exit-code','1','--python',str(exporter),
                 '--','--config',str(config),'--output',str(work/'bundle'),'--inicio-root',str(inicio_root)]
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'}
        try:
            with log.open('w') as stream:
                process=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,env=env)
            if process.returncode:raise PipelineError('INICIO_EXPORT_FAILED',log.read_text()[-6000:])
            bundle=read_json(work/'bundle/bundle.json')
            simulation=bundle['input_config'].get('simulation',{})
            # Alamin props can include XML-relative meshes and textures. Pack
            # those into MJZ before handing the bundle to the offline importer.
            for index,prop in enumerate(simulation.get('props',[])):
                prop_spec=_read_spec((config.parent/prop['model']).resolve())
                relative=f'props/{index}.mjz';destination=work/'bundle'/relative
                prop_model=prop_spec.compile()
                export_mujoco(prop_spec,mujoco.MjData(prop_model),destination)
                prop['model']=relative
                bundle['sha256'][relative]=sha256(destination)
            write_json(work/'bundle/bundle.json',bundle)
            result=compile_bundle(work/'bundle',output,simulation=simulation)
        except subprocess.TimeoutExpired as exc:
            raise PipelineError('INICIO_TIMEOUT',f'Exporter exceeded {timeout}s') from exc
        finally:
            if log.exists():
                # Preserve the producer log even when generation fails.
                log_dest=output/'inicio_export.log' if output.exists() else output.with_suffix('.export.log')
                shutil.copyfile(log,log_dest)
    return result


def validate(root):
    """Check stored assets and load the compiled scene without a producer."""
    root=Path(root);bundle=load_bundle(root/'bundle')
    from sim_harness.scene import load_scene
    model,data=load_scene(root/'scene.mjz')
    profile=read_json(root/'robot_profile.json')
    rotation=data.body('element/chassis').xmat.reshape(3,3)
    difference=data.geom('element/front_left_collision').xpos-data.geom('element/front_right_collision').xpos
    actual_track=abs((rotation.T@difference)[1])
    valid=bool(np.isfinite(data.qpos).all() and np.isclose(actual_track,profile['track_width_m']))
    valid &= all(model.camera('element/'+name).id>=0 for name in profile['cameras'])
    report=dict(passed=valid,domain='agriculture',track_width_m=float(actual_track),
                cameras=len(profile['cameras']),geoms=model.ngeom,bundle_sha256=sha256(root/'bundle/bundle.json'))
    if profile.get('arms'):
        report.update(robot_type=profile['robot_type'],arms=len(profile['arms']),
                      suspension_joints=len(profile['suspension']),actuators=model.nu)
    write_json(root/'validation.json',report)
    return report


def preview(root, model=None, data=None):
    from .inspection import scene_page
    from PIL import Image
    root=Path(root)
    if model is None:
        from sim_harness.scene import load_scene
        model,data=load_scene(root/'scene.mjz')
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.azimuth=135;camera.elevation=-35;camera.distance*=.72
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    with mujoco.Renderer(model,height=360,width=640) as renderer:
        renderer.update_scene(data,camera,scene_option=option)
        Image.fromarray(renderer.render()).save(root/'preview.png')
    scene_page(root)
