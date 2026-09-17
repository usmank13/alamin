"""P5 pilot: mesh URDF import, CoACD, uniform vs protected-part rescaling.

Deliberately limited to fixed/revolute/prismatic trees with boxes and triangle meshes.
Unsupported URDF features fail explicitly. Preserved parts require a name match;
no claim of automatic semantic segmentation or grasp feasibility.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import coacd
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import trimesh

from .physics import emit, validate


def xyz(element, key, default='0 0 0'):
    return np.fromstring(element.get(key, default) if element is not None else default, sep=' ')


def read_urdf(path, collision_source='collision'):
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    parts, joints = [], []
    link_names = [x.attrib['name'] for x in root.findall('link')]
    joint_names = [x.attrib['name'] for x in root.findall('joint')]
    if len(set(link_names)) != len(link_names) or len(set(joint_names)) != len(joint_names):
        raise ValueError('Duplicate link or joint names in source URDF')
    for node in root.findall('joint'):
        if any(node.find(kind).attrib['link'] not in link_names for kind in ('parent', 'child')):
            raise ValueError('Source joint references a nonexistent link')
    if root.find('.//mimic') is not None or root.find('.//transmission') is not None:
        raise ValueError('Pilot does not implement mimic/transmission semantics')
    for link in root.findall('link'):
        # Use collision geometry as the decomposed source; original render styling is out of scope.
        elements = link.findall(collision_source)
        if not elements:
            raise ValueError(f'No {collision_source} geometry for {link.attrib["name"]}')
        for index, element in enumerate(elements):
            geom = element.find('geometry')
            mesh_node = geom.find('mesh')
            box_node = geom.find('box')
            if mesh_node is not None:
                filename = mesh_node.attrib['filename']
                if '://' in filename:
                    raise ValueError('Resolve package:// paths before the pilot')
                mesh = trimesh.load(path.parent / filename, force='mesh', process=True)
                mesh.apply_scale(xyz(mesh_node, 'scale', '1 1 1'))
                name = filename
            elif box_node is not None:
                mesh = trimesh.creation.box(extents=xyz(box_node, 'size'))
                name = f'{link.attrib["name"]}_box{index}'
            else:
                raise ValueError('Pilot supports mesh/box geometry only')
            origin = element.find('origin')
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_euler('xyz', xyz(origin, 'rpy')).as_matrix()
            transform[:3, 3] = xyz(origin, 'xyz')
            mesh.apply_transform(transform)
            parts.append(dict(link=link.attrib['name'], name=name, mesh=mesh))
    for node in root.findall('joint'):
        typ = node.attrib['type']
        if typ not in ('fixed', 'revolute', 'prismatic'):
            raise ValueError(f'Unsupported joint type: {typ}')
        origin, limit = node.find('origin'), node.find('limit')
        joints.append(dict(name=node.attrib['name'], parent=node.find('parent').attrib['link'],
                           child=node.find('child').attrib['link'], pos=xyz(origin, 'xyz').tolist(),
                           rpy=xyz(origin, 'rpy').tolist(), axis=xyz(node.find('axis'), 'xyz', '1 0 0').tolist(),
                           range=[float(limit.get('lower')), float(limit.get('upper'))] if typ != 'fixed' else [0, 0],
                           type={'fixed': 'fixed', 'revolute': 'hinge', 'prismatic': 'slide'}[typ]))
    return parts, joints


def decompose(mesh, cache):
    key = hashlib.sha256(mesh.vertices.tobytes()+mesh.faces.tobytes()+b'coacd-1.0.14-seed0-pilot-v1').hexdigest()
    path = cache / f'{key}.npz'
    if path.exists():
        with np.load(path) as arrays:
            return [trimesh.Trimesh(arrays[f'v{i}'], arrays[f'f{i}'], process=False) for i in range(int(arrays['n']))]
    if mesh.is_convex:
        hulls = [mesh.convex_hull]
    else:
        coacd.set_log_level('error')
        result = coacd.run_coacd(coacd.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces)),
                                 max_convex_hull=16, seed=0, resolution=1000,
                                 mcts_iterations=20, mcts_nodes=10, preprocess_resolution=30)
        hulls = [trimesh.Trimesh(v, f, process=False) for v, f in result]
    if not 1 <= len(hulls) <= 16:
        raise ValueError('Collision hull budget exceeded')
    arrays = {'n': np.array(len(hulls))}
    for i, hull in enumerate(hulls):
        arrays[f'v{i}'], arrays[f'f{i}'] = hull.vertices, hull.faces
    cache.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return hulls


def closed_extents(parts, joints):
    children = {j['child']: j for j in joints}
    transforms = {}
    def transform(link):
        if link not in transforms:
            matrix = np.eye(4)
            if link in children:
                j = children[link]
                matrix[:3, :3] = Rotation.from_euler('xyz', j['rpy']).as_matrix()
                matrix[:3, 3] = j['pos']
                matrix = transform(j['parent']) @ matrix
            transforms[link] = matrix
        return transforms[link]
    vertices = np.concatenate([trimesh.transformations.transform_points(p['mesh'].vertices, transform(p['link'])) for p in parts])
    return np.ptp(vertices, axis=0)


def scaled(parts, joints, scale, route, protected='handle', mount_x=None):
    out, metrics = [], []
    for p in parts:
        q = dict(p)
        mesh = p['mesh'].copy()
        preserve = protected in p['name'].lower() or protected in p['link'].lower()
        factor = 1 if route == 'G4' and preserve else scale
        # Preserve shape about a declared mount plane, or the centroid in the
        # generic arithmetic counterexample. This is not automatic part segmentation.
        center = mesh.bounds.mean(axis=0)
        if preserve and mount_x is not None:
            center[0] = mount_x
        def transform(m):
            r = m.copy()
            r.vertices = (r.vertices-center)*factor + center*scale
            return r
        q['mesh'] = transform(mesh)
        q['collision_meshes'] = [transform(h) for h in p['collision_meshes']]
        out.append(q)
        if preserve:
            # Heuristic aperture screen: not a grasp/collision/reachability test.
            metrics.append(dict(name=p['name'], source_extents=mesh.extents.tolist(),
                                scaled_extents=q['mesh'].extents.tolist(),
                                aperture_screen_m=float(np.sort(q['mesh'].extents)[1]),
                                under_80mm=bool(np.sort(q['mesh'].extents)[1] <= .08)))
    result_joints = copy.deepcopy(joints)
    for j in result_joints:
        j['pos'] = (np.asarray(j['pos'])*scale).tolist()
        if j['type'] == 'slide':
            j['range'] = (np.asarray(j['range'])*scale).tolist()
    return out, result_joints, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('urdf', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, default=Path('outputs/prototypes/p5'))
    parser.add_argument('--collision-source', choices=['collision', 'visual'], default='collision',
                        help='visual explicitly rebuilds colliders from render geometry')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for path in args.urdf:
        start = time.perf_counter()
        parts, joints = read_urdf(path, args.collision_source)
        source_extents = closed_extents(parts, joints)
        native_probe = {}
        try:
            native = mujoco.MjModel.from_xml_path(str(path.resolve()))
            native_probe = dict(compiles=True, bodies=native.nbody, joints=native.njnt,
                                body_names=[native.body(i).name for i in range(native.nbody)],
                                mass=float(native.body_mass.sum()))
        except Exception as exc:
            native_probe = dict(compiles=False, error=str(exc))
        for p in parts:
            p['collision_meshes'] = decompose(p['mesh'], args.output / 'hull_cache')
        preprocess = time.perf_counter()-start
        panels = [p for p in parts if p['link'] == 'door_link']
        # Dataset-specific mount frame is explicit: +X is out of the panel, all origins zero.
        mount_x = max(p['mesh'].bounds[1, 0] for p in panels) if panels else None
        if mount_x is not None and any(np.any(j['rpy']) or np.any(j['pos']) for j in joints):
            raise ValueError('Mount-plane G4 pilot requires zero joint origins/rotations')
        for route in ('G3', 'G4'):
            for scale in (.7, 1., 1.4):
                case = f'{path.parent.name}_{route}_{scale}'
                output_parts, output_joints, handles = scaled(parts, joints, scale, route, mount_x=mount_x)
                xml = emit(output_parts, output_joints, args.output/case)
                report = validate(xml)
                record = dict(asset=str(path.resolve()), source_urdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                              route=route, scale=scale, validation=report, handles=handles,
                              hull_counts=[len(p['collision_meshes']) for p in parts],
                              preprocessing_seconds=preprocess, native_urdf_probe=native_probe,
                              collision_source=args.collision_source,
                              requested_extents=(source_extents*scale).tolist(),
                              actual_extents=closed_extents(output_parts, output_joints).tolist(),
                              max_relative_bbox_error=float(np.max(abs(closed_extents(output_parts, output_joints)-source_extents*scale)/(source_extents*scale))),
                              g4_method='preserve hardware about the source panel +X mounting plane; scale tangential centroid and joint origins; specific to these zero-frame cabinet URDFs',
                              grasp_test='rough AABB aperture heuristic only; not actual Panda closure')
                records.append(record)
                print(case, report['passed'], 'handle matches', len(handles), flush=True)
                (args.output/'results.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
