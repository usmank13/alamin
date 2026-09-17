"""Executed by Blender's Python, never imported by the simulator."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix,Vector


def main():
    args=sys.argv[sys.argv.index('--')+1:]; path=Path(args[0]); recipe=json.loads(path.read_text())
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    for g in recipe['geoms']:
        size=g['size'];kind=g['kind']
        if kind==7:
            mesh=bpy.data.meshes.new(g['name']);mesh.from_pydata(g['vertices'],[],g['faces']);mesh.update()
            obj=bpy.data.objects.new(g['name'],mesh);bpy.context.collection.objects.link(obj)
            if g.get('uv'):
                uv=mesh.uv_layers.new(name='UVMap')
                for poly,indices in zip(mesh.polygons,g['face_uv']):
                    for loop,index in zip(poly.loop_indices,indices):
                        if index>=0: uv.data[loop].uv=g['uv'][index]
        elif kind==6:
            bpy.ops.mesh.primitive_cube_add(size=2);obj=bpy.context.object;obj.scale=size
        elif kind==2:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=24,ring_count=12,radius=size[0]);obj=bpy.context.object
        elif kind in (3,5):
            bpy.ops.mesh.primitive_cylinder_add(vertices=32,radius=size[0],depth=2*size[1]);obj=bpy.context.object
            if kind==3: raise RuntimeError('Capsules need explicit conversion; cannot silently render as cylinders')
        else:
            raise RuntimeError(f'Unsupported render geometry {kind}')
        obj.name=g['name'];obj.location=g['position'];obj.rotation_mode='QUATERNION';obj.rotation_quaternion=Matrix(g['rotation']).to_quaternion()
        material=bpy.data.materials.new(g['name']);material.use_nodes=True
        node=material.node_tree.nodes.get('Principled BSDF');m=g['material'];rgba=m.get('rgba',g['rgba'])
        node.inputs['Base Color'].default_value=rgba;node.inputs['Roughness'].default_value=max(.05,m.get('roughness',.6));node.inputs['Metallic'].default_value=m.get('metallic',0)
        if m.get('texture'):
            tex=material.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path.parent/m['texture']))
            if not g.get('uv'):
                tex.projection='BOX';tex.projection_blend=.1
                coords=material.node_tree.nodes.new('ShaderNodeTexCoord');material.node_tree.links.new(coords.outputs['Generated'],tex.inputs['Vector'])
            material.node_tree.links.new(tex.outputs['Color'],node.inputs['Base Color'])
        obj.data.materials.append(material)
    points=[p for r in recipe['rooms'] for p in r['polygon']]
    low=[min(p[i] for p in points) for i in range(2)];high=[max(p[i] for p in points) for i in range(2)]
    center=Vector(((low[0]+high[0])/2,(low[1]+high[1])/2,0.8));span=max(high[i]-low[i] for i in range(2))
    # Multi-room/reference inspection uses a whole-scene view, not a camera inside a wall.
    overview=recipe.get('camera_view')=='overview'
    location=(center.x-span*.6,center.y-span*.6,span*1.3+2.8) if overview else (low[0]+.45,low[1]+.45,2.55)
    bpy.ops.object.camera_add(location=location);camera=bpy.context.object
    camera.rotation_euler=(center-camera.location).to_track_quat('-Z','Y').to_euler();camera.data.lens=18
    if overview:camera.data.type='ORTHO';camera.data.ortho_scale=span*1.65
    scene=bpy.context.scene;scene.camera=camera
    for x in (.25,.75):
        for y in (.25,.75):
            bpy.ops.object.light_add(type='AREA',location=(low[0]+x*(high[0]-low[0]),low[1]+y*(high[1]-low[1]),2.65))
            bpy.context.object.data.energy=350*recipe.get('appearance',{}).get('light_multiplier',1);bpy.context.object.data.shape='DISK';bpy.context.object.data.size=2
    scene.world.color=(.15,.15,.15)
    scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=int(args[1]);scene.cycles.use_denoising=True;scene.cycles.seed=recipe['seed']
    scene.render.resolution_x=int(args[2]);scene.render.resolution_y=int(args[2]);scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.filepath=str(path.parent/'cycles.png')
    bpy.ops.wm.save_as_mainfile(filepath=str(path.parent/'scene.blend'))
    bpy.ops.render.render(write_still=True)


if __name__=='__main__': main()
