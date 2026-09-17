"""Actual MuJoCo previews and a geometry-faithful Cycles bridge."""
import os
os.environ.setdefault('MUJOCO_GL','osmesa')
os.environ.setdefault('LP_NUM_THREADS','1')
import math
from pathlib import Path
import shutil
import subprocess
import time

import mujoco
import numpy as np
from PIL import Image,ImageDraw

from .contracts import PipelineError,write_json,read_json
from .validation import load


def preview(root, *, cutaway=True):
    root=Path(root);model=load(root);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    ir=read_json(root/'ir.json')
    points=np.array([p for r in ir['rooms'] for p in r['polygon']])
    low,high=points.min(axis=0),points.max(axis=0)
    camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    camera.lookat[:]=[*(low+high)/2,.6];camera.distance=float(np.linalg.norm(high-low)*1.1)
    camera.azimuth=130;camera.elevation=-55
    opt=mujoco.MjvOption();opt.geomgroup[3]=0;opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=0  # sensor rays are not scene content
    model.vis.quality.shadowsize=1024
    model.vis.headlight.ambient[:]=.5
    # Diagnostic cutaway is labeled; simulation/physics geometry is unchanged.
    rgba=model.geom_rgba.copy()
    for i in range(model.ngeom):
        if cutaway and model.geom(i).name.startswith('wall_'): model.geom_rgba[i,3]=0
    with mujoco.Renderer(model,height=480,width=640) as renderer:
        frames=[]
        for fraction in np.r_[np.linspace(0,1,10),np.linspace(1,0,10)]:
            mujoco.mj_resetData(model,data)
            for j in range(model.njnt):
                if int(model.jnt_type[j]) in (2,3) and model.jnt_limited[j]:
                    data.qpos[model.jnt_qposadr[j]]=fraction*model.jnt_range[j,np.argmax(abs(model.jnt_range[j]))]
            mujoco.mj_forward(model,data);renderer.update_scene(data,camera,scene_option=opt)
            image=Image.fromarray(renderer.render());ImageDraw.Draw(image).text((12,12),'MuJoCo cutaway / kinematic inspection' if cutaway else 'MuJoCo source architecture reconstruction',fill='white')
            frames.append(image)
        frames[0].save(root/'preview.png')
        frames[0].save(root/'articulation.gif',save_all=True,append_images=frames[1:],duration=120,loop=0)
        manifest=read_json(root/'manifest.json')
        palette={'procedural':[.2,.6,.95,1.],'retrieved':[.95,.55,.15,1.],'generated':[.75,.25,.85,1.]}
        for i in range(model.ngeom):
            entry=manifest['instances'].get(model.geom(i).name.split('/')[0],{})
            source=entry.get('source_classification',{})
            if source.get('origin') in palette and not (model.geom_contype[i] or model.geom_conaffinity[i]):
                model.geom_matid[i]=-1;model.geom_rgba[i]=palette[source['origin']]
        mujoco.mj_resetData(model,data);mujoco.mj_forward(model,data);renderer.update_scene(data,camera,scene_option=opt)
        source_view=Image.fromarray(renderer.render())
        ImageDraw.Draw(source_view).text((12,12),'Sources: blue=procedural / orange=retrieved / purple=generated',fill='white')
        source_view.save(root/'provenance.png')
    model.geom_rgba[:]=rgba
    report=read_json(root/'validation.json') if (root/'validation.json').exists() else {}
    import html
    rows=''.join('<tr>'+''.join('<td>'+html.escape(str(value))+'</td>' for value in [key,entry['category'],entry.get('source_classification',{}).get('origin','legacy/unknown'),entry.get('source_classification',{}).get('provider',''),entry.get('source_classification',{}).get('allowed_use','')])+'</tr>' for key,entry in manifest['instances'].items())
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Scene inspection</title><style>body{font-family:sans-serif;max-width:1300px;margin:2em auto}td,th{padding:.4em;text-align:left}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>Scene inspection</h1><p>Cutaway views; joint animation is kinematic, not robot manipulation.</p><img src="preview.png"><img src="articulation.gif"><h2>Object provenance</h2><img src="provenance.png"><p>Origin does not certify contact-rich suitability.</p><table><tr><th>Instance</th><th>Category</th><th>Origin</th><th>Provider</th><th>Allowed use</th></tr>'+rows+'</table><p><a href="ir.json">SceneIR</a> | <a href="manifest.json">Semantics</a> | <a href="provenance.json">Provenance JSON</a> | <a href="validation.json">Validation</a> | <a href="render/cycles.png">Cycles still</a></p><pre>'+html.escape(str(report))+'</pre>')


def render_recipe(root, *, view='auto'):
    if view not in ('auto','interior','overview'):raise PipelineError('CAMERA_VIEW','Unknown render view')
    root=Path(root);model=load(root);data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    output=root/'render'; output.mkdir(exist_ok=True)
    textures={}
    for t in range(model.ntex):
        h,w,c=map(int,(model.tex_height[t],model.tex_width[t],model.tex_nchannel[t]))
        start=int(model.tex_adr[t]); pixels=model.tex_data[start:start+h*w*c].reshape(h,w,c)
        path=f'texture_{t}.png';Image.fromarray(pixels[...,0] if c==1 else pixels).save(output/path);textures[t]=path
    # Texture roles are the sync contract: the layers MuJoCo compiled become Principled BSDF inputs, nothing else.
    roles={int(r):r.name.replace('mjTEXROLE_','').lower() for r in mujoco.mjtTextureRole.__members__.values() if r.name.startswith('mjTEXROLE_')}
    materials={}
    for mid in range(model.nmat):
        maps={roles[i]:textures[int(t)] for i,t in enumerate(model.mat_texid[mid]) if t>=0}
        materials[model.material(mid).name]=dict(rgba=model.mat_rgba[mid].tolist(),roughness=float(1-model.mat_shininess[mid]),
                                                metallic=float(model.mat_reflectance[mid]),textures=maps,repeat=model.mat_texrepeat[mid].tolist(),
                                                texuniform=bool(model.mat_texuniform[mid]))
    geoms=[]
    for i in range(model.ngeom):
        if model.geom_contype[i] or model.geom_conaffinity[i] or model.geom_rgba[i,3]==0: continue
        kind=int(model.geom_type[i]);mid=int(model.geom_matid[i])
        g=dict(name=model.geom(i).name,kind=kind,size=model.geom_size[i].tolist(),position=data.geom_xpos[i].tolist(),
               rotation=data.geom_xmat[i].reshape(3,3).tolist(),rgba=model.geom_rgba[i].tolist(),material=model.material(mid).name if mid>=0 else None)
        if kind==int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh=int(model.geom_dataid[i]);v=int(model.mesh_vertadr[mesh]);n=int(model.mesh_vertnum[mesh]);f=int(model.mesh_faceadr[mesh]);nf=int(model.mesh_facenum[mesh])
            g['vertices']=model.mesh_vert[v:v+n].tolist();g['faces']=model.mesh_face[f:f+nf].tolist()
            ta=int(model.mesh_texcoordadr[mesh]);tn=int(model.mesh_texcoordnum[mesh])
            if ta>=0 and tn>0:
                g['uv']=model.mesh_texcoord[ta:ta+tn].tolist();g['face_uv']=model.mesh_facetexcoord[f:f+nf].tolist()
        geoms.append(g)
    ir=read_json(root/'ir.json')
    recipe=dict(schema_version=1,source='SceneIR compiled geometry at initial state',scene_ir_sha256=__import__('hashlib').sha256((root/'ir.json').read_bytes()).hexdigest(),
                geoms=geoms,materials=materials,rooms=ir['rooms'],seed=ir['meta']['seed'],appearance=ir['meta'].get('appearance',{}),
                camera_view=('overview' if len(ir['rooms'])>1 or 'architecture' in ir else 'interior') if view=='auto' else view)
    write_json(output/'recipe.json',recipe)
    return output/'recipe.json'


def cycles(root, *, blender=None, samples=32, resolution=1024,view='auto'):
    recipe=render_recipe(root,view=view)
    executable=blender or shutil.which('blender')
    if not executable:
        choices=sorted(Path('vendor/blender').glob('blender-*/blender'))
        executable=str(choices[-1].resolve()) if choices else None
    if not executable:
        raise PipelineError('MISSING_BLENDER','Install Blender or run scripts/fetch_blender.py; preview is not a path-traced substitute')
    script=Path(__file__).with_name('cycles_worker.py')
    start=time.perf_counter()
    result=subprocess.run([executable,'--background','--factory-startup','--python',str(script.resolve()),'--',str(recipe.resolve()),str(samples),str(resolution)],capture_output=True,text=True,timeout=900)
    (recipe.parent/'blender.log').write_text(result.stdout+'\n'+result.stderr)
    if result.returncode or not (recipe.parent/'cycles.png').exists():
        raise PipelineError('CYCLES_FAILED','See render/blender.log')
    write_json(recipe.parent/'report.json',dict(engine='Cycles',device='CPU',samples=samples,resolution=resolution,seconds=time.perf_counter()-start,
                                               geometry_source='same compiled SceneIR',materials='MuJoCo material texture layers (rgb/normal/roughness/metallic) mapped one-to-one onto Principled BSDF from the same compiled model; flat finishes use the MuJoCo shininess/reflectance scalars',diffusion=False))
    return recipe.parent/'cycles.png'
