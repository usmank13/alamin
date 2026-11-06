#!/usr/bin/env python3
"""Build the annotated aerial reference as a portable, offline MuJoCo farm.

MUJOCO_GL=egl .venv/bin/python scripts/build_reference_farm.py --output outputs/reference_farm
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from scene_pipeline.agriculture import compile_environment, compose_robot, load_bundle, sha256
from scene_pipeline.navigation import add_cameras
from scene_pipeline.inspection import scene_page
from sim_harness.export import export_mujoco
from sim_harness.scene import _read_spec, load_scene
from build_two_fields import add_ground_visual, check_scene, vegetation

SCALE = .12
IMAGE_HEIGHT = 1049
BOUNDS = np.array([[0., 0.], [731*SCALE, IMAGE_HEIGHT*SCALE]])
ROAD_WIDTH = 4.2
SEED = 41


def metric(points):
    result = np.array(points, dtype=float)
    result[..., 0] *= SCALE
    result[..., 1] = (IMAGE_HEIGHT-result[..., 1])*SCALE
    return result


PLOTS = [dict(id='east', polygon_m=metric([[377,86],[665,89],[624,824],[374,824]]).tolist(),
              condition='harvested / cultivated soil', row_direction='north-south'),
         dict(id='west', polygon_m=metric([[109,306],[233,305],[226,830],[108,828]]).tolist(),
              condition='green row crop with a soybean / weed work patch', row_direction='north-south'),
         dict(id='south', polygon_m=metric([[370,874],[594,872],[590,1049],[368,1049]]).tolist(),
              condition='partial cultivated field with green east rows', row_direction='north-south')]
ROAD_POINTS = metric([[696,72],[642,852],[0,848]])
SPAWN = dict(position_m=ROAD_POINTS[0].tolist(), yaw_rad=math.atan2(*(ROAD_POINTS[1]-ROAD_POINTS[0])[::-1]))
DESTINATIONS = [dict(id='east', position_m=[59., 68.], yaw_rad=math.pi/2),
                dict(id='west', position_m=[20., 45.], yaw_rad=math.pi/2),
                dict(id='south', position_m=[54., 10.], yaw_rad=-math.pi/2)]


def polygon_mask(x, y, polygon):
    """Convex reference polygons, accepting either winding."""
    p = np.asarray(polygon)
    cross = [(p[(i+1)%len(p), 0]-a[0])*(y-a[1]) -
             (p[(i+1)%len(p), 1]-a[1])*(x-a[0]) for i, a in enumerate(p)]
    return np.logical_or(np.logical_and.reduce([c >= 0 for c in cross]),
                         np.logical_and.reduce([c <= 0 for c in cross]))


def road_distance(x, y):
    x, y = np.broadcast_arrays(x, y)
    best = np.full(x.shape, np.inf)
    for a, b in zip(ROAD_POINTS[:-1], ROAD_POINTS[1:]):
        v = b-a
        t = np.clip(((x-a[0])*v[0]+(y-a[1])*v[1])/np.dot(v,v), 0, 1)
        best = np.minimum(best, np.hypot(x-a[0]-t*v[0], y-a[1]-t*v[1]))
    return best


def surface(x, y):
    x, y = np.broadcast_arrays(x, y)
    z = .008*np.sin(x*.45)*np.sin(y*.31)
    # Keep the NE parking/spawn area level while retaining shallow road ruts.
    distance = road_distance(x, y)
    rut = -.012*np.exp(-((distance-.83)/.2)**2)
    height = (z+rut)*np.clip((np.hypot(x-SPAWN['position_m'][0], y-SPAWN['position_m'][1])-3)/3, 0, 1)
    return np.where(abs(height)<1e-8, 0., height)


def route_points(points, spacing=.2):
    chunks = [np.linspace(a,b,max(2,math.ceil(np.linalg.norm(np.array(b)-a)/spacing)+1))[:-1]
              for a,b in zip(points[:-1],points[1:])]
    return np.concatenate([*chunks, [points[-1]]])


def routes():
    corner = ROAD_POINTS[1].tolist()
    return {d['id']: route_points([SPAWN['position_m'], corner,
            [d['position_m'][0], corner[1]], d['position_m']]) for d in DESTINATIONS}


def landscape_texture(path):
    rng = np.random.default_rng(SEED)
    x,y = np.meshgrid(np.linspace(*BOUNDS[:,0],1462), np.linspace(*BOUNDS[::-1,1],2098))
    noise = rng.normal(size=x.shape)
    broad = gaussian_filter(noise, 22); broad /= broad.std()
    texture = noise*4+broad*5
    soil = np.array([166.,148.,119.])+(texture+4*np.sin(x*31)+2*np.cos(x*7))[...,None]
    rgb = np.broadcast_to([94.,112.,60.], (*x.shape,3)).copy()+texture[...,None]
    # Match the unannotated field strips and long north-south cultivation marks.
    bare = (x<6) | ((x>29)&(x<43)&(y>27)) | (y<21) | (x>45)
    rgb[bare] = soil[bare]
    crop = ((x>12.6)&(x<28)&(y>27)) | ((x>30)&(x<41)&(y>28)) | ((x<32)&(y<21)) | ((x>63)&(x<71)&(y<21))
    row = np.maximum(0,np.cos(2*np.pi*x/.9))**3
    leafy = np.array([83.,107.,55.])+(texture+row*20)[...,None]
    rgb[crop] = leafy[crop]
    # The west work alley and south headland remain open for the full robot.
    alley = (abs(x-20)<1.7)&(y>26)&(y<49)
    rgb[alley] = (soil*.72)[alley]
    for plot in (PLOTS[0], PLOTS[2]):
        mask = polygon_mask(x,y,plot['polygon_m']) & ~crop
        rgb[mask] = soil[mask]
    woodland = ((x>6)&(x<12)) | (x>82)
    rgb[woodland] = (np.array([49.,70.,31.])+texture[...,None])[woodland]
    dist = road_distance(x,y)
    weight = np.clip((ROAD_WIDTH/2+.10*np.sin(4*y+3*x)-dist)/.22,0,1)
    tracks = np.exp(-((dist-.83)/.25)**2)
    dirt = np.array([157.,145.,116.])+(texture+tracks*22)[...,None]
    rgb = rgb*(1-weight[...,None])+dirt*weight[...,None]
    Image.fromarray(np.uint8(np.clip(rgb,0,255))).save(path)


def add_mesh(spec, name, verts, faces, color):
    spec.add_mesh(name=name, uservert=np.asarray(verts).ravel(), userface=np.asarray(faces).ravel(),
                  inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, maxhullvert=32)
    return dict(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=name, rgba=color,
                contype=0, conaffinity=0, group=2, mass=0)


def plant_mesh(spec, name, height, color):
    rng = np.random.default_rng(SEED)
    verts,faces = [],[]
    for i in range(22):
        angle=i*2.4
        z=height*(.25+.7*i/22)
        p=np.array([.11*np.cos(angle),.11*np.sin(angle),z])
        u=np.array([.12*np.cos(angle),.12*np.sin(angle),.025])
        v=np.array([-.055*np.sin(angle),.055*np.cos(angle),.015])
        start=len(verts)
        verts.extend([p-u,p+v,p+u,p-v,p+[0,0,rng.uniform(.015,.03)]])
        faces.extend([[start+j,start+(j+1)%4,start+4] for j in range(4)])
    return add_mesh(spec,name,verts,faces,color)


def add_crops(spec, manifest, bundle):
    """Detailed semantic work patch, plus inexpensive textured background rows."""
    meshes = {'crop':plant_mesh(spec,'soybean_simple',.33,[.22,.38,.12,1]),
              'weed':plant_mesh(spec,'weed_simple',.22,[.34,.43,.14,1])}
    rotation=np.array([[0,-1],[1,0]])
    centre=np.array([20.,52.])
    for item in bundle['instances']:
        xy=rotation@np.array(item['position_m'][:2])+centre
        name='west_'+item['id']
        record=copy.deepcopy({k:v for k,v in item.items() if k!='meshes'})
        record.update(id=name,position_m=[*xy,float(surface(*xy))],plot_id='west')
        if record.get('row_id'): record['row_id']='west_'+record['row_id']
        manifest['instances'][name]=record
        spec.worldbody.add_geom(name=name,pos=record['position_m'],**meshes[item['category']])
        manifest['geom_labels'][name]=dict(instance=name,class_id=2 if item['category']=='crop' else 1)
    for row in bundle['rows']:
        manifest['rows'].append(dict(id='west_'+row['id'],plot_id='west',
            endpoints_m=(np.array(row['endpoints_m'])@rotation.T+centre).tolist()))
    # Background rows are long low foliage ribbons, not thousands of physics bodies.
    verts,faces=[],[]
    strips=[(13.,27.,28.,124.),(30.,41.,28.,124.),(1.,31.,1.,20.),(63.,70.,1.,20.)]
    for xmin,xmax,ymin,ymax in strips:
        for x in np.arange(xmin,xmax,.9):
            for y in np.arange(ymin,ymax,1.4):
                if xmin==13. and ((abs(x-20)<1.8 and y<49) or (abs(x-20)<2.8 and 49<y<55)):
                    continue
                h=.18+.07*math.sin(x*7+y*3)
                start=len(verts)
                verts.extend([[x-.17,y,0],[x+.17,y,0],[x,y,h],
                              [x-.17,min(y+1.4,ymax),0],[x+.17,min(y+1.4,ymax),0],[x,min(y+1.4,ymax),h]])
                faces.extend([[start,start+2,start+3],[start+2,start+5,start+3],
                              [start+1,start+4,start+2],[start+2,start+4,start+5]])
    spec.worldbody.add_geom(name='background_rows', **add_mesh(spec,'background_rows',verts,faces,[.29,.38,.15,1]))
    manifest['geom_labels']['background_rows']=dict(instance=None,class_id=2)


def add_woodland(spec, manifest):
    rng=np.random.default_rng(SEED)
    for kind, positions in [('tree',[(9+rng.uniform(-.5,.5),y) for y in np.arange(32,124,7)] +
                              [(87.1+rng.uniform(-.2,.2),y) for y in np.arange(5,125,6)]),
                             ('hedge',[(9.7,y) for y in np.arange(29,124,2.1)] +
                              [(43.,y) for y in np.arange(29,124,3.3)])]:
        template=vegetation(kind=='tree', SEED)
        # Copy each template mesh once; every vegetation instance shares it.
        for mesh in template.meshes:
            spec.add_mesh(name=kind+'/'+mesh.name,uservert=mesh.uservert,userface=mesh.userface,
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL,maxhullvert=32)
        geoms=list(template.geoms)
        originals=[dict(type=g.type,meshname=(kind+'/'+g.meshname if g.meshname else ''),rgba=g.rgba.copy(),size=g.size.copy(),
                        pos=g.pos.copy(),quat=g.quat.copy(),fromto=g.fromto.copy(),
                        contype=g.contype,conaffinity=g.conaffinity,group=g.group) for g in geoms]
        for i,(x,y) in enumerate(positions):
            # Keep leaf crowns and solid vegetation away from spawn and all road edges.
            if road_distance(x,y)<(2.8 if kind=='tree' else 4) or np.linalg.norm(np.array([x,y])-SPAWN['position_m'])<6:
                continue
            name=f'{kind}_{i}'
            shift=np.array([x,y,float(surface(x,y))])
            manifest['instances'][name]=dict(id=name,category='obstruction',position_m=shift.tolist())
            for j,values in enumerate(originals):
                values={**values,'pos':values['pos']+shift}
                if values['type']==mujoco.mjtGeom.mjGEOM_CAPSULE:
                    values['fromto']=values['fromto']+np.tile(shift,2)
                    del values['pos'];del values['quat']
                else:
                    del values['fromto']
                geom=spec.worldbody.add_geom(name=f'{name}_{j}',**values)
                manifest['geom_labels'][geom.name]=dict(instance=name,class_id=9)


def previews(out, model, data):
    option=mujoco.MjvOption();option.geomgroup[3:]=0
    with mujoco.Renderer(model,height=1049,width=731,max_geom=10000) as renderer:
        camera=mujoco.MjvCamera();camera.lookat[:]=[*BOUNDS[1]/2,0]
        camera.distance=160;camera.azimuth=90;camera.elevation=-90;camera.orthographic=1
        model.vis.global_.fovy=math.degrees(2*math.atan(128/(2*camera.distance)))
        renderer.update_scene(data,camera,scene_option=option)
        Image.fromarray(renderer.render()).save(out/'topdown.png')
    model.vis.global_.fovy=45
    with mujoco.Renderer(model,height=720,width=1280,max_geom=10000) as renderer:
        camera=mujoco.MjvCamera();camera.lookat[:]=[46,66,0]
        camera.distance=151;camera.azimuth=90;camera.elevation=-62
        renderer.update_scene(data,camera,scene_option=option)
        Image.fromarray(renderer.render()).save(out/'overview.png')
        renderer.update_scene(data,'element/nav_front',scene_option=option)
        Image.fromarray(renderer.render()).save(out/'nav_front.png')
    shutil.copy(out/'overview.png',out/'preview.png')
    annotated=Image.open(out/'topdown.png');draw=ImageDraw.Draw(annotated)
    def pixels(points):
        p=np.array(points);return [tuple(v) for v in np.column_stack(((p[:,0]-BOUNDS[1,0]/2)*1049/128+731/2,
                                                        (BOUNDS[1,1]/2-p[:,1])*1049/128+1049/2))]
    for plot in PLOTS:
        p=pixels(plot['polygon_m']);draw.line(p+[p[0]],fill='#ffbf39',width=3)
        draw.text(p[0],plot['id'].upper(),fill='white',stroke_width=2,stroke_fill='#26331e')
    p=pixels([SPAWN['position_m']])[0]
    draw.ellipse([p[0]-11,p[1]-11,p[0]+11,p[1]+11],outline='#ffe457',width=3)
    draw.text((p[0]-58,p[1]-27),'START',fill='white',stroke_width=2,stroke_fill='#26331e')
    annotated.save(out/'layout.png')


def build(source,out):
    source,out=Path(source),Path(out)
    bundle=load_bundle(source/'bundle')
    profile=json.loads((source/'robot_profile.json').read_text())
    out.mkdir(parents=True,exist_ok=False)
    xs,ys=np.linspace(*BOUNDS[:,0],221),np.linspace(*BOUNDS[:,1],316)
    x,y=np.meshgrid(xs,ys);z=surface(x,y)
    np.savez_compressed(out/'terrain.npz',elevation_m=z)
    empty=dict(terrain=dict(file='terrain.npz',bounds_m=BOUNDS.tolist()),instances=[],rows=[],class_names=bundle['class_names'])
    spec,manifest=compile_environment(empty,out)
    spec.visual.global_.offheight=1100
    landscape_texture(out/'ground.png');add_ground_visual(spec,xs,ys,z,out/'ground.png')
    add_crops(spec,manifest,bundle);add_woodland(spec,manifest)
    spec.add_texture(name='sky',type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,rgb1=[.48,.65,.81],rgb2=[.87,.89,.84],width=256,height=1536)
    spec.visual.quality.shadowsize=2048
    shutil.copy(source/'robot.mjz',out/'robot.mjz')
    (out/'robot_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    robot=_read_spec(out/'robot.mjz');add_cameras(robot,profile,dict(cameras=['front','rear']))
    composed,model,data=compose_robot(spec,robot,profile,SPAWN,BOUNDS)
    paths=routes()
    report=check_scene(model,data,profile,np.concatenate(list(paths.values())))
    export_mujoco(composed,data,out/'scene.mjz');spec.to_zip(str(out/'environment.mjz'))
    portable,restored=load_scene(out/'scene.mjz')
    np.testing.assert_allclose(restored.qpos,data.qpos,atol=1e-5)
    assert portable.ngeom==model.ngeom
    del portable,restored
    manifest.update(plots=PLOTS,road=dict(width_m=ROAD_WIDTH,centreline_m=ROAD_POINTS.tolist()),
                    work_patch=dict(plot_id='west',centre_m=[20,52],entry_m=[20,48],row_direction='north-south'))
    mission=dict(start=SPAWN,goal={**DESTINATIONS[0], 'tolerance_m':.7},destinations=DESTINATIONS,
                 road_width_m=ROAD_WIDTH,reference_waypoints_m=paths['east'][::10].tolist()+[paths['east'][-1].tolist()],
                 routes_m={name:path[::10].tolist()+[path[-1].tolist()] for name,path in paths.items()},
                 instruction='Start in the northeast clearing. Follow the east dirt track south to the cross-track, turn west, enter the large east field through its south headland, and drive north to the goal. The west soybean work patch and partial south field are alternative destinations.')
    generation=dict(domain='agriculture',spawn=SPAWN,provenance=dict(seed=SEED,source=str(source.resolve()),
                    source_bundle_sha256=sha256(source/'bundle/bundle.json'),image_scale_m_per_pixel=SCALE,
                    reference_image_pixels=[731,1049],paid_generation=False),
                    limitations=['Image-relative reconstruction; 0.12 m/pixel is an assumed scale, not survey data.',
                    'Rigid ground and uncalibrated GEN3 dynamics; foliage and crops are visual-only.',
                    'Background rows use low-poly strips; the west work patch has individually labeled crop and weed meshes.',
                    'Routes are geometric clearance checks, not a completed autonomous drive.'])
    report.update(plots=3,rows=len(manifest['rows']),crop_count=sum(i['category']=='crop' for i in manifest['instances'].values()),
                  weed_count=sum(i['category']=='weed' for i in manifest['instances'].values()),
                  geoms=model.ngeom,meshes=model.nmesh,archive_reload_passed=True,road_width_m=ROAD_WIDTH,
                  destinations_checked=list(paths),world_bounds_m=BOUNDS.tolist())
    drive=dict(seconds=650,controller=dict(perception='none',waypoints=[dict(xy=p) for p in mission['reference_waypoints_m']],
                parameters=dict(auto_speed_max_ms=.3,arrival_distance=.2)))
    for name,value in [('manifest',manifest),('generation',generation),('mission',mission),('validation',report),('reference_drive',drive)]:
        (out/f'{name}.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
    previews(out,model,data);scene_page(out)
    print(f'Built {out}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=REPO/'outputs/soybean30_early_gen3')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.source,args.output)
