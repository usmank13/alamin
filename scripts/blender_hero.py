"""Path-lit hero frames for a replay, run by Blender (never imported by the simulator).

  scripts/replay_video.py RUN --export-hero DIR
  vendor/blender/blender-*/blender --background --python scripts/blender_hero.py -- DIR HDRI.hdr SKY_YAW_RAD
      [--tree TREE.gltf --bush BUSH.gltf] [--frames A B] [--samples 32]
  scripts/replay_video.py RUN --hero-frames DIR

Draws the exported scene with the sky panorama as the light, a sun lamp for crisp shadows,
and physically based materials; scanned tree and shrub models stand in for the sim's procedural ones.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector
import numpy as np

MESH, CAPSULE, PLANE = 7, 3, 0  # mjtGeom
FACES = 3_000_000  # per scanned model; thinning below this shreds leaf cards (tried 60k), so only a guard


def pose(pos, quat):
    return Matrix.Translation(Vector(pos)) @ Quaternion(quat).to_matrix().to_4x4()


def material(g, root, cache):
    key = g.get('material') or tuple(round(c, 3) for c in g['rgba'])
    if key in cache:
        return cache[key]
    m = bpy.data.materials.new(str(key))
    m.use_nodes = True
    tree = m.node_tree
    bsdf = tree.nodes['Principled BSDF']
    name = g.get('material') or ''
    bsdf.inputs['Base Color'].default_value = g['rgba']
    bsdf.inputs['Roughness'].default_value = .9 if 'texture' in g else max(.12, 1 - 1.6 * g.get('shininess', .2))
    bsdf.inputs['Metallic'].default_value = 1. if name.endswith(('robot_metal', 'robot_tool')) else 0.
    if 'texture' in g:
        image = tree.nodes.new('ShaderNodeTexImage')
        image.image = bpy.data.images.load(str(root / g['texture']), check_existing=True)
        coords, scale = tree.nodes.new('ShaderNodeTexCoord'), tree.nodes.new('ShaderNodeMapping')
        scale.inputs['Scale'].default_value = (*g['repeat'], 1)
        tree.links.new(coords.outputs['UV'], scale.inputs['Vector'])
        tree.links.new(scale.outputs['Vector'], image.inputs['Vector'])
        tree.links.new(image.outputs['Color'], bsdf.inputs['Base Color'])
        if 'ground' in name or 'grass' in name:  # relief from the photo's own light and dark
            bump = tree.nodes.new('ShaderNodeBump')
            bump.inputs['Strength'].default_value, bump.inputs['Distance'].default_value = .6, .03
            tree.links.new(image.outputs['Color'], bump.inputs['Height'])
            tree.links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    elif not name:  # plants: no two quite the same green ('tint' is one random number per plant part)
        tint, hsv = tree.nodes.new('ShaderNodeAttribute'), tree.nodes.new('ShaderNodeHueSaturation')
        tint.attribute_name = 'tint'
        ramp = tree.nodes.new('ShaderNodeMapRange')
        ramp.inputs['To Min'].default_value, ramp.inputs['To Max'].default_value = .6, 1.
        tree.links.new(tint.outputs['Fac'], ramp.inputs['Value'])
        tree.links.new(ramp.outputs['Result'], hsv.inputs['Value'])
        hsv.inputs['Hue'].default_value = .47  # a little toward yellow: leaf green, not signal green
        hsv.inputs['Color'].default_value = g['rgba']
        tree.links.new(hsv.outputs['Color'], bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = .55
    cache[key] = m
    return m


def scanned(path, height):
    """A glTF model as one object, its base at the origin, scaled to `height`."""
    before = set(bpy.data.objects)
    bpy.ops.import_scene.gltf(filepath=str(path))
    parts = [o for o in set(bpy.data.objects) - before if o.type == 'MESH']
    bpy.ops.object.select_all(action='DESELECT')
    for o in parts:
        o.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    faces = len(obj.data.polygons)
    print('SCANNED', path.name, faces, 'faces')
    if faces > FACES:  # scans carry millions of faces, and every sample redraws them into the shadow maps
        thin = obj.modifiers.new('thin', 'DECIMATE')
        thin.ratio = FACES / faces
        bpy.ops.object.modifier_apply(modifier='thin')
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    low, top = min(c.z for c in corners), max(c.z for c in corners)
    obj.data.transform(Matrix.Scale(height / (top - low), 4) @ Matrix.Translation((0, 0, -low)))
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    ap.add_argument('hdri', type=Path)
    ap.add_argument('sky_yaw', type=float)
    ap.add_argument('--tree', type=Path)
    ap.add_argument('--bush', type=Path)
    ap.add_argument('--frames', type=int, nargs=2)
    ap.add_argument('--samples', type=int, default=16)
    ap.add_argument('--fast', action='store_true', help='no ray-traced reflections')
    args = ap.parse_args(sys.argv[sys.argv.index('--') + 1:])
    root = args.root
    scene_json = json.loads((root / 'scene.json').read_text())
    arrays, track = np.load(root / 'arrays.npz'), np.load(root / 'track.npz')
    bodies = scene_json['bodies']

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module='io_scene_gltf2')
    scene = bpy.context.scene
    collection = scene.collection
    meshes, materials, moving, props = {}, {}, [], {}  # moving: (object, body, pose in body); props: body -> its objects
    stand_ins = {'oak': (args.tree, 5.8), 'bush': (args.bush, 1.4)}
    rng = np.random.default_rng(7)
    plants = {}  # material -> [vertices], [faces], [tint]: thousands of static plant parts drawn as one mesh each
    for g in scene_json['geoms']:
        body = bodies[g['body']]
        kind = body['name'].split('_')[0]
        if stand_ins.get(kind, (None,))[0]:
            continue  # a scanned model stands here instead
        if g['kind'] == MESH and body['name'] == 'world' and 'texture' not in g:
            vertices, faces, tints, count = plants.setdefault(material(g, root, materials), ([], [], [], [0]))
            world = np.array(pose(g['pos'], g['quat']))
            v = arrays[f"v{g['mesh']}"]
            vertices.append(v @ world[:3, :3].T + world[:3, 3])
            faces.append(arrays[f"f{g['mesh']}"] + count[0])
            tints.append(np.full(len(v), rng.uniform()))
            count[0] += len(v)
            continue
        if g['kind'] == MESH:
            m = g['mesh']
            if m not in meshes:
                mesh = bpy.data.meshes.new(f'mesh{m}')
                faces = arrays[f'f{m}']
                mesh.from_pydata(arrays[f'v{m}'].tolist(), [], faces.tolist())
                if f'uv{m}' in arrays:
                    layer = mesh.uv_layers.new(name='UVMap')
                    layer.data.foreach_set('uv', arrays[f'uv{m}'][arrays[f'fuv{m}'].ravel()].ravel())
                for polygon in mesh.polygons:
                    polygon.use_smooth = body['moving']  # CAD surfaces smooth, leaves faceted
                meshes[m] = mesh
            obj = bpy.data.objects.new(g['name'], meshes[m])
            collection.objects.link(obj)
        elif g['kind'] == CAPSULE:
            bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=g['size'][0], depth=2 * g['size'][1])
            obj = bpy.context.object
        elif g['kind'] == PLANE:
            bpy.ops.mesh.primitive_plane_add(size=2 * g['size'][0])
            obj = bpy.context.object
        else:
            continue
        if not obj.data.materials:  # instances share their mesh, and so its material
            obj.data.materials.append(material(g, root, materials))
        local = pose(g['pos'], g['quat'])
        obj.matrix_world = pose(body['pos'], body['quat']) @ local
        if body['moving']:
            moving.append((obj, g['body'], local))
        props.setdefault(g['body'], []).append(obj)
    for n, (mat, (vertices, faces, tints, _)) in enumerate(plants.items()):
        mesh = bpy.data.meshes.new(f'plants{n}')
        mesh.from_pydata(np.concatenate(vertices).tolist(), [], np.concatenate(faces).tolist())
        tint = np.concatenate(tints)
        mesh.color_attributes.new('tint', 'FLOAT_COLOR', 'POINT').data.foreach_set('color', np.repeat(tint, 4))
        mesh.materials.append(mat)
        collection.objects.link(bpy.data.objects.new(f'plants{n}', mesh))
    for kind, (path, height) in stand_ins.items():
        if not path:
            continue
        source = scanned(path, height)
        for index, body in enumerate(bodies):
            if body['name'].startswith(kind + '_'):
                copy = source.copy()
                collection.objects.link(copy)
                copy.matrix_world = Matrix.Translation(Vector(body['pos'])) @ Matrix.Rotation(rng.uniform(0, 2 * math.pi), 4, 'Z') @ Matrix.Scale(rng.uniform(.85, 1.15), 4)
                props.setdefault(index, []).append(copy)
        bpy.data.objects.remove(source)

    # Light: the panorama all round, plus a sun lamp where its sun is.
    world = bpy.data.worlds.new('sky')
    scene.world = world
    world.use_nodes = True
    nodes, links = world.node_tree.nodes, world.node_tree.links
    env, coords, turn = nodes.new('ShaderNodeTexEnvironment'), nodes.new('ShaderNodeTexCoord'), nodes.new('ShaderNodeMapping')
    env.image = bpy.data.images.load(str(args.hdri))
    turn.vector_type = 'POINT'
    turn.inputs['Rotation'].default_value = (0, 0, math.pi - args.sky_yaw)  # Blender puts the panorama's centre at -X
    links.new(coords.outputs['Generated'], turn.inputs['Vector'])
    links.new(turn.outputs['Vector'], env.inputs['Vector'])
    links.new(env.outputs['Color'], nodes['Background'].inputs['Color'])
    sun = bpy.data.objects.new('sun', bpy.data.lights.new('sun', 'SUN'))
    collection.objects.link(sun)
    sun.data.energy, sun.data.angle, sun.data.color = 3.5, math.radians(1.5), (1, .95, .85)
    sun.rotation_euler = Vector(scene_json['sun']).to_track_quat('-Z', 'Y').to_euler()

    camera = bpy.data.objects.new('camera', bpy.data.cameras.new('camera'))
    collection.objects.link(camera)
    scene.camera = camera
    camera.data.sensor_fit, camera.data.angle_y = 'VERTICAL', math.radians(scene_json['fovy'])
    camera.data.clip_start, camera.data.clip_end = .05, 2000
    nav = bpy.data.objects.new('nav_front', bpy.data.cameras.new('nav_front'))  # the robot's front camera, for the side panel
    collection.objects.link(nav)
    nav.data.sensor_fit, nav.data.angle_y = 'VERTICAL', math.radians(scene_json['nav_fovy'])
    nav.data.clip_start, nav.data.clip_end = .02, 2000

    scene.render.engine = 'BLENDER_EEVEE_NEXT'
    scene.eevee.taa_render_samples = args.samples
    scene.eevee.use_raytracing = not args.fast
    scene.eevee.use_shadows = True
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'JPEG'
    scene.render.image_settings.quality = 93
    (root / 'frames').mkdir(exist_ok=True)
    (root / 'nav').mkdir(exist_ok=True)

    first, last = args.frames or (0, len(track['poses']))
    for n in range(first, min(last, len(track['poses']))):
        if (root / 'frames' / f'{n:06d}.jpg').exists():
            continue  # resume an interrupted render
        if (root / 'frames' / f'{n:06d}.jpg').exists() and (root / 'nav' / f'{n:06d}.jpg').exists():
            continue  # resume an interrupted render
        poses, view = track['poses'][n], track['camera'][n]
        for obj, body, local in moving:
            obj.matrix_world = pose(poses[body, :3], poses[body, 3:]) @ local
        eye, forward, up = Vector(view[:3]), Vector(view[3:6]), Vector(view[6:])
        right = forward.cross(up)
        camera.matrix_world = Matrix(((right.x, up.x, -forward.x, eye.x), (right.y, up.y, -forward.y, eye.y),
                                      (right.z, up.z, -forward.z, eye.z), (0, 0, 0, 1)))  # a camera looks down its -Z
        ghosted = set(json.loads(str(track['ghosted'][n])))
        look = track['nav'][n]  # MuJoCo cameras look down -Z with +Y up too
        nav.matrix_world = Matrix((*(tuple(look[3 + 3 * r:6 + 3 * r]) + (look[r],) for r in range(3)), (0, 0, 0, 1)))
        for folder, eye_object, size, hidden in (('frames', camera, scene_json['size'], ghosted), ('nav', nav, scene_json['nav_size'], ())):
            for body, objects in props.items():
                if not bodies[body]['moving']:
                    for obj in objects:
                        obj.hide_render = body in hidden  # scenery between the chase camera and the rover steps aside
            scene.camera = eye_object
            scene.render.resolution_x, scene.render.resolution_y = size
            scene.render.filepath = str(root / folder / f'{n:06d}.jpg')
            bpy.ops.render.render(write_still=True)

if __name__ == '__main__':
    main()
