"""Import the Onshape Gen 3 export into the portable agricultural simulator.

This is an adapter for this export's named assemblies, not a general URDF
loader. CAD geometry and pivots are authoritative; dynamics and optical mounts
are explicit estimates because the export does not contain those parameters.
"""
import copy
import math
from pathlib import Path
import shutil
import struct
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec
from .agriculture import PROFILE, compose_robot, preview, sha256, vector
from .contracts import PipelineError, read_json, write_json


WHEELS = {'wheel_simple': 'front_right', 'wheel_simple_1': 'rear_right',
          'wheel_simple_2': 'rear_left', 'wheel_simple_3': 'front_left'}
ARMS = [('arm_j1_2', 'arm_j2_1'), ('arm_j1', 'arm_j2_2'),
        ('arm_j1_1', 'arm_j2'), ('arm_j1_3', 'arm_j2_3')]
NAMES = {'rover_simple': 'chassis', 'side_assy_simple': 'rocker_left',
         'side_assy_simple_1': 'rocker_right', **WHEELS,
         **{link: f'arm_{i+1}_{axis}' for i, pair in enumerate(ARMS)
            for link, axis in zip(pair, ('yaw', 'pitch'))}}


def origin(element):
    if element is None:
        return np.zeros(3), Rotation.identity()
    return (np.fromstring(element.get('xyz', '0 0 0'), sep=' '),
            Rotation.from_euler('xyz', np.fromstring(element.get('rpy', '0 0 0'), sep=' ')))


def pose(position, rotation):
    return dict(pos=vector(position), quat=vector(rotation.as_quat(scalar_first=True)))


def initialize(model, data, profile, prefix=''):
    for name, value in profile.get('initial_joint_positions', {}).items():
        data.joint(prefix+name).qpos[0] = value
    for name, value in profile.get('initial_controls', {}).items():
        data.ctrl[model.actuator(prefix+name).id] = value
    mujoco.mj_forward(model, data)


def camera_layout(layout, inicio_root, mounts):
    optics = dict(sensor_width_mm=36., sensor_height_mm=20.25, lens_mm=18.97,
                  optics_id='D405', depth_range_m=[.07, 1.02])
    if layout == 'URDF':
        return {f'gen3_{i+1}': dict(position_m=[-.18, float(mount[1]), -.299],
                    quaternion_wxyz=[math.sqrt(.5), 0, 0, -math.sqrt(.5)], optics=optics,
                    calibration='Estimated optical mount; URDF has no camera frame')
                for i, mount in enumerate(mounts)}, None
    if layout not in ('RAPTOR_22', 'RAPTOR_30') or inicio_root is None:
        raise PipelineError('GEN3_LAYOUT', 'Inicio presets require --inicio-root and RAPTOR_22 or RAPTOR_30')
    path = Path(inicio_root)/'assets/robots'/f'{layout}.json'
    record = read_json(path)
    cameras = {name: dict(position_m=(np.asarray(c['loc_bu'])/10).tolist(),
                         quaternion_wxyz=Rotation.from_euler('xyz', c['euler_deg'], degrees=True).as_quat(scalar_first=True).tolist(),
                         optics=optics, calibration='Inicio pose; lateral payload relocation is an assembly assumption')
               for name, c in record['cameras'].items()}
    if len(cameras) != 4:
        raise PipelineError('GEN3_LAYOUT', 'Expected four camera/payload positions')
    cameras = dict(sorted(cameras.items(), key=lambda item: -item[1]['position_m'][1]))
    return cameras, dict(file=str(path), sha256=sha256(path))


def payload_configuration(config):
    """An explicit list defines the installed payloads, not just enabled motors."""
    import re
    config=read_json(config) if isinstance(config,(str,Path)) else copy.deepcopy(config)
    if not isinstance(config,dict) or set(config)!={'payloads'} or not isinstance(config['payloads'],list):
        raise PipelineError('GEN3_PAYLOADS','Expected {"payloads": [{"id": "arm_left", "lateral_m": 0.45}, ...]}')
    ids=set()
    for payload in config['payloads']:
        if not isinstance(payload,dict) or set(payload)!={'id','lateral_m'}:
            raise PipelineError('GEN3_PAYLOADS','Each payload requires exactly id and lateral_m')
        name=payload['id'];y=payload['lateral_m']
        if not isinstance(name,str) or not re.fullmatch(r'arm_[a-z0-9_]+',name) or name in ids:
            raise PipelineError('GEN3_PAYLOADS','Payload IDs must be unique arm_ names using lowercase letters, digits and underscores')
        if isinstance(y,bool) or not isinstance(y,(int,float)) or not math.isfinite(y):
            raise PipelineError('GEN3_PAYLOADS','lateral_m must be finite chassis-frame metres (+Y left)')
        ids.add(name)
    return config


def assemble_payloads(xml,profile,config):
    """Instance the corrected CAD payload; translate camera and native frame together."""
    chassis=xml.find("./worldbody/body[@name='chassis']")
    actuator=xml.find('actuator')
    template=profile['arms'][0]
    body=copy.deepcopy(chassis.find(f"body[@name='{template['yaw_joint']}']"))
    camera=copy.deepcopy(chassis.find(f"camera[@name='{template['camera']}']"))
    motors=[copy.deepcopy(actuator.find(f"position[@name='{template[key]}']")) for key in ('yaw_joint','pitch_joint')]
    initial={key:(profile['initial_joint_positions'][template[key]],profile['initial_controls'][template[key]])
             for key in ('yaw_joint','pitch_joint')}
    original_arms=copy.deepcopy(profile['arms'])
    optics=copy.deepcopy(profile['cameras'][template['camera']])
    for arm in original_arms:
        chassis.remove(chassis.find(f"body[@name='{arm['yaw_joint']}']"))
        for key in ('yaw_joint','pitch_joint'):
            actuator.remove(actuator.find(f"position[@name='{arm[key]}']"))
            profile['initial_joint_positions'].pop(arm[key]);profile['initial_controls'].pop(arm[key])
    for element in list(chassis.findall('camera')):chassis.remove(element)
    profile['arms']=[];profile['cameras']={}
    for payload in config['payloads']:
        name=payload['id'];delta=payload['lateral_m']-template['position_chassis_m'][1]
        camera_name='gen3_'+name.removeprefix('arm_')
        rename=lambda value: name+value[len(template['id']):] if value.startswith(template['id']+'_') else value
        instance=copy.deepcopy(body)
        for node in instance.iter():
            if 'name' in node.attrib:node.set('name',rename(node.get('name')))
        position=np.fromstring(instance.get('pos'),sep=' ');position[1]+=delta
        instance.set('pos',vector(position));chassis.append(instance)
        for motor,key in zip(motors,('yaw_joint','pitch_joint')):
            motor=copy.deepcopy(motor)
            for attr in ('name','joint'):motor.set(attr,rename(motor.get(attr)))
            actuator.append(motor)
            profile['initial_joint_positions'][motor.get('joint')]=initial[key][0]
            profile['initial_controls'][motor.get('name')]=initial[key][1]
        cam=copy.deepcopy(camera);cam.set('name',camera_name)
        position=np.fromstring(cam.get('pos'),sep=' ');position[1]+=delta;cam.set('pos',vector(position));chassis.append(cam)
        profile['cameras'][camera_name]={**copy.deepcopy(optics),'position_m':position.tolist()}
        arm=copy.deepcopy(template)
        arm.update(id=name,camera=camera_name)
        for key in ('yaw_joint','pitch_joint','tool_site'):arm[key]=rename(arm[key])
        arm['position_chassis_m'][1]+=delta
        arm['source_payload']=template['id']
        profile['arms'].append(arm)
    profile['payload_configuration']=copy.deepcopy(config)
    profile['provenance']['payload_configuration']=copy.deepcopy(config)
    profile['provenance']['payload_template']='Corrected '+template['id']+' CAD arm/camera assembly; lateral translation only'
    # Conservatively enclose the entire payload sweep, not just its home pose.
    low,high=profile['footprint_xy_m']
    for arm in profile['arms']:
        low[1]=min(low[1],arm['position_chassis_m'][1]-.65)
        high[1]=max(high[1],arm['position_chassis_m'][1]+.65)


def build_robot(urdf, *, layout='URDF', inicio_root=None, mesh_faces=30000, payloads=None):
    """Return an articulated MjSpec and reproducible assembly/dynamics profile."""
    import trimesh
    payloads=payload_configuration(payloads) if payloads is not None else None
    urdf = Path(urdf).expanduser().resolve()
    if urdf.is_dir():
        candidates = list(urdf.rglob('*.urdf'))
        if len(candidates) != 1:
            raise PipelineError('GEN3_URDF', 'Expected exactly one URDF in export directory')
        urdf = candidates[0]
    if mesh_faces < 1000:
        raise PipelineError('GEN3_MESH', 'mesh_faces must be at least 1000')
    source = ET.parse(urdf).getroot()
    links = {link.get('name'): link for link in source.findall('link')}
    joints = source.findall('joint')
    if set(links) != set(NAMES) | {'root'} or len(joints) != 15:
        raise PipelineError('GEN3_URDF', 'Unsupported assembly; expected Gen 3 chassis, two rockers, four wheels and four 2-DOF arms')
    by_child = {j.find('child').get('link'): j for j in joints}
    mounts = [origin(by_child[yaw].find('origin'))[0] for yaw, _ in ARMS]
    cameras, layout_source = camera_layout(layout, inicio_root, mounts)
    xml = ET.Element('mujoco', model='element_gen3')
    ET.SubElement(xml, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(xml, 'option', timestep='.002', integrator='implicitfast')
    asset = ET.SubElement(xml, 'asset'); assets = {}; mesh_info = {}
    world = ET.SubElement(xml, 'worldbody')
    actuator = ET.SubElement(xml, 'actuator')
    profile = copy.deepcopy(PROFILE)
    profile.pop('chassis_diaginertia_kg_m2')
    profile['wheel_mass_kg']=5.
    profile.update(robot_type='GEN3', layout=layout, initial_joint_positions={}, initial_controls={},
                   arms=[], suspension=[], cameras=cameras, footprint_xy_m=[[-1.02, -1.08], [1.02, 1.08]],
                   provenance=dict(geometry=str(urdf), urdf_sha256=sha256(urdf), inicio_layout=layout_source,
                       dynamics='Estimated per-part masses; CAD inertia shape scaled by mass ratio. Estimated joint limits, gains, damping and collision proxies.'),
                   limitations=['Uncalibrated masses, inertia, friction, limits and actuator response',
                       'Passive rocker suspension with estimated spring and damping; rigid tires and soil',
                       'Approximate collision shapes; plants are visual and cannot be removed by tools',
                       'Camera optical mounts are not calibrated to this CAD',
                       'Inicio presets relocate arm pivots and optical frames; camera support brackets remain part of the exported chassis mesh',
                       'Arm motor commands are simulated; physical weeding efficacy is not modeled'],
                   corrections=['arm_j2_1 uses the identical arm_j2 reference link inertial/visual frame to remove its baked 0.604232 rad pose',
                                'Left rocker starts at +0.0698087 rad to cancel the exported assembly tilt'])
    # These are assumptions, not density conversions of the tiny exported masses.
    masses = dict(chassis=70., rocker=8., wheel=5., yaw=.6, pitch=.55)
    profile['estimated_part_masses_kg'] = masses
    bodies = {}

    def add_link(link_name, parent):
        name = NAMES[link_name]
        joint = by_child[link_name]
        pos, rot = origin(joint.find('origin'))
        if name.startswith('arm_') and name.endswith('_yaw') and layout != 'URDF':
            pos[1] = list(cameras.values())[int(name.split('_')[1])-1]['position_m'][1]
        body = ET.SubElement(parent, 'body', name=name, **pose(pos, rot)); bodies[name] = body
        part = ('wheel' if link_name in WHEELS else 'rocker' if name.startswith('rocker')
                else name.rsplit('_', 1)[-1] if name.startswith('arm') else 'chassis')
        link = links['arm_j2'] if link_name == 'arm_j2_1' else links[link_name]
        inertial = link.find('inertial')
        ipos, irot = origin(inertial.find('origin'))
        tensor = inertial.find('inertia').attrib
        inertia = np.array([[float(tensor['i'+a+b if 'i'+a+b in tensor else 'i'+b+a]) for b in 'xyz'] for a in 'xyz'])
        inertia = irot.as_matrix() @ inertia @ irot.as_matrix().T
        inertia *= masses[part]/float(inertial.find('mass').get('value'))
        ET.SubElement(body, 'inertial', pos=vector(ipos), mass=str(masses[part]),
                      fullinertia=vector([*inertia.diagonal(), inertia[0,1], inertia[0,2], inertia[1,2]]))
        if part == 'chassis':
            ET.SubElement(body, 'freejoint', name='root')
            ET.SubElement(body, 'geom', name='chassis_collision', type='box', size='.54 .94 .04', pos='0 0 -.045', group='3')
        else:
            attrs = dict(name=name, axis=joint.find('axis').get('xyz'), damping='.1', armature='.002')
            if part == 'rocker':
                home = .0698087 if name == 'rocker_left' else 0.
                attrs.update(range=vector([home-.22, home+.22]), stiffness='800', springref=str(home), damping='60', armature='.05')
                profile['initial_joint_positions'][name] = home
                profile['suspension'].append(dict(joint=name, home_rad=home, range_rad=[home-.22, home+.22],
                                                  stiffness_nm_rad=800., damping_nm_s_rad=60.))
                # Keep proxies in the visual assembly frame, including the CAD tilt.
                _, vr = origin(link.find('visual/origin'))
                for label, a, b, radius in [('upright', [0,-.052,-.05], [0,-.052,-.50], .055),
                    ('rear', [0,-.052,-.50], [-.40,-.052,-.6964], .045),
                    ('front', [0,-.052,-.50], [.40,-.052,-.6964], .045)]:
                    ET.SubElement(body, 'geom', name=name+'_'+label+'_collision', type='capsule',
                                  fromto=vector([*vr.apply(a), *vr.apply(b)]), size=str(radius), group='3')
            elif part == 'wheel':
                ET.SubElement(body, 'geom', name=name+'_collision', type='cylinder',
                    size='.20748 .050795', pos='0 -.020052 0', quat=vector([math.sqrt(.5), math.sqrt(.5), 0, 0]),
                    group='3', friction=vector(profile['friction']))
                ET.SubElement(actuator, 'velocity', name=name, joint=name, kv='30', forcerange='-35 35')
            else:
                limits = [-.65, .65] if part == 'yaw' else [-1.15, .15]
                home = 0. if part == 'yaw' else -.65
                attrs.update(range=vector(limits), damping='1', armature='.01')
                profile['initial_joint_positions'][name] = home
                profile['initial_controls'][name] = home
                ET.SubElement(actuator, 'position', name=name, joint=name, kp='100', kv='5',
                              ctrlrange=vector(limits), forcerange='-12 12')
                if part == 'yaw':
                    ET.SubElement(body, 'geom', name=name+'_collision', type='box', size='.065 .045 .055', pos='0 .055 -.065', group='3')
                else:
                    ET.SubElement(body, 'geom', name=name+'_tube_collision', type='capsule', size='.012',
                                  fromto='0 -.035 0 -.55 -.035 0', group='3')
                    ET.SubElement(body, 'geom', name=name+'_tool_collision', type='box', size='.02 .045 .025', pos='-.572 -.035 -.018', group='3')
                    ET.SubElement(body, 'site', name=name+'_tip', pos='-.592 -.035 -.018', size='.008', rgba='1 0 0 1', group='4')
            ET.SubElement(body, 'joint', **attrs)
        visual = link.find('visual'); mesh = visual.find('geometry/mesh')
        filename = Path(mesh.get('filename')).name
        path = urdf.parent.parent/'meshes'/filename
        if filename not in mesh_info:
            scene = trimesh.load(path, force='scene')
            pieces = []
            for node in scene.graph.nodes_geometry:
                transform, geometry = scene.graph[node]
                piece = scene.geometry[geometry].copy(); piece.apply_transform(transform); pieces.append(piece)
            combined = trimesh.util.concatenate(pieces)
            original_faces = len(combined.faces)
            if original_faces > mesh_faces:
                # glTF duplicates vertices at CAD face/normal seams. Without
                # welding them the decimator erases independent face interiors,
                # leaving a shredded shell instead of a closed-looking solid.
                combined.merge_vertices()
                combined = combined.simplify_quadric_decimation(face_count=mesh_faces)
            vertices = np.asarray(combined.vertices, dtype='<f4'); faces = np.asarray(combined.faces, dtype='<i4')
            key = Path(filename).stem
            assets[key+'.msh'] = struct.pack('<4i', len(vertices), 0, 0, len(faces))+vertices.tobytes()+faces.tobytes()
            ET.SubElement(asset, 'mesh', name=key, file=key+'.msh', inertia='shell', maxhullvert='64')
            mesh_info[filename] = dict(sha256=sha256(path), source_faces=original_faces, faces=len(faces))
        vp, vr = origin(visual.find('origin'))
        ET.SubElement(body, 'geom', name=name+'_visual', type='mesh', mesh=Path(filename).stem,
                      rgba=visual.find('material/color').get('rgba'), group='2', contype='0', conaffinity='0', mass='0', **pose(vp, vr))
        for child_name, child_joint in by_child.items():
            if child_joint.find('parent').get('link') == link_name:
                add_link(child_name, body)

    add_link('rover_simple', world)
    chassis = bodies['chassis']
    ET.SubElement(chassis, 'site', name='imu', size='.002', group='4')
    sensors = ET.SubElement(xml, 'sensor')
    for kind in ('gyro', 'accelerometer'):
        ET.SubElement(sensors, kind, name=kind, site='imu')
    for i, (name, camera) in enumerate(cameras.items()):
        fovy = math.degrees(2*math.atan(camera['optics']['sensor_height_mm']/(2*camera['optics']['lens_mm'])))
        ET.SubElement(chassis, 'camera', name=name, pos=vector(camera['position_m']), quat=vector(camera['quaternion_wxyz']), fovy=str(fovy))
        yaw_name, pitch_name = f'arm_{i+1}_yaw', f'arm_{i+1}_pitch'
        mount = np.fromstring(bodies[yaw_name].get('pos'), sep=' ')
        profile['arms'].append(dict(id=f'arm_{i+1}', camera=name, yaw_joint=yaw_name, pitch_joint=pitch_name,
            tool_site=pitch_name+'_tip', position_chassis_m=[float(mount[0]), float(mount[1]), -.4513928],
            quaternion_chassis_wxyz=[0, 0, 0, 1], motor_to_joint_sign=[-1, -1],
            motor_limits_rad=[[-.65, .65], [-.15, 1.15]], home_motor_rad=[0, .65]))
    if payloads is not None:assemble_payloads(xml,profile,payloads)
    spec = mujoco.MjSpec.from_string(ET.tostring(xml, encoding='unicode'), assets=assets)
    model = spec.compile(); data = mujoco.MjData(model); initialize(model, data, profile)
    profile['wheels'] = []
    for name in WHEELS.values():
        gid = model.geom(name+'_collision').id
        jid = model.joint(name).id
        profile['wheels'].append(dict(id=name, center_m=data.geom_xpos[gid].tolist(), radius_m=.20748,
            half_width_m=.050795, drive_sign=float(np.sign(data.xaxis[jid][1]))))
    centers = {w['id']: w['center_m'] for w in profile['wheels']}
    profile['track_width_m'] = abs(centers['front_left'][1]-centers['front_right'][1])
    profile['root_ground_height_m'] = max(w['radius_m']-w['center_m'][2] for w in profile['wheels'])
    profile['provenance']['meshes'] = mesh_info
    profile['source_joint_map'] = {joint.get('name'): ('root' if joint.find('child').get('link')=='rover_simple'
                                 else NAMES[joint.find('child').get('link')]) for joint in joints}
    if payloads is not None:
        profile['source_joint_map']={key:value for key,value in profile['source_joint_map'].items() if not value.startswith('arm_')}
        profile['payload_joint_map']={arm['id']:{axis:arm[axis+'_joint'] for axis in ('yaw','pitch')} for arm in profile['arms']}
    return spec, profile


def replace_robot(scene, urdf, output, *, layout='URDF', inicio_root=None, mesh_faces=30000, payloads=None):
    """Create a new agricultural scene, retaining the existing portable field."""
    scene = Path(scene); output = Path(output)
    if output.exists():
        raise PipelineError('OUTPUT_EXISTS', str(output))
    generation = read_json(scene/'generation.json')
    if generation.get('domain') != 'agriculture':
        raise PipelineError('AGRICULTURE_SCENE', 'Gen 3 composition requires an agricultural field scene')
    manifest = read_json(scene/'manifest.json')
    robot, profile = build_robot(urdf, layout=layout, inicio_root=inicio_root, mesh_faces=mesh_faces, payloads=payloads)
    benchmark=manifest.get('weeding_benchmark')
    if benchmark and not set(benchmark.get('arms',[benchmark.get('arm')]))<={a['id'] for a in profile['arms']}:
        raise PipelineError('GEN3_PAYLOADS','Existing benchmark references removed payloads; create a new weeding scene from the reconfigured field')
    spec, model, data = compose_robot(_read_spec(scene/'environment.mjz'), robot, profile,
                                    generation['spawn'], np.asarray(manifest['terrain']['bounds_m']))
    output.mkdir(parents=True)
    shutil.copytree(scene/'bundle', output/'bundle')
    shutil.copyfile(scene/'environment.mjz', output/'environment.mjz')
    robot_model = robot.compile(); robot_data = mujoco.MjData(robot_model); initialize(robot_model, robot_data, profile)
    export_mujoco(robot, robot_data, output/'robot.mjz')
    export_mujoco(spec, data, output/'scene.mjz')
    generation.update(robot_dynamics=profile['provenance'], robot_type='GEN3', robot_layout=layout)
    write_json(output/'generation.json', generation)
    write_json(output/'manifest.json', manifest)
    write_json(output/'robot_profile.json', profile)
    report = dict(passed=True, robot_type='GEN3', layout=layout, arms=len(profile['arms']), suspension_joints=2,
                  driven_wheels=4, cameras=len(profile['cameras']), nq=model.nq, nv=model.nv,
                  actuators=model.nu, output=str(output), calibrated=False)
    write_json(output/'validation.json', report)
    robot_preview(output, model, data)
    preview(output, model, data)
    return report


def robot_preview(output, model, data):
    """Close view of the assembled CAD, supplementing the whole-field preview."""
    from PIL import Image
    camera=mujoco.MjvCamera()
    camera.lookat[:]=data.body('element/chassis').xpos+[0,0,-.35]
    camera.distance=3.3;camera.azimuth=155;camera.elevation=-12
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    with mujoco.Renderer(model,height=540,width=960) as renderer:
        renderer.update_scene(data,camera,scene_option=option)
        Image.fromarray(renderer.render()).save(Path(output)/'robot.png')
