"""PBR materials and generated decor: cache, compiler layers, rebinding, recipe roles, static collision placement. Network-free."""
import io
import json
import zipfile

import numpy as np
import pytest
pytest.importorskip('jsonschema')
import mujoco
from PIL import Image

from scene_pipeline import fal
from scene_pipeline.compiler import compile_scene
from scene_pipeline.contracts import PipelineError,write_json
from scene_pipeline.dsl import load,parse
from scene_pipeline.layout import solve
from scene_pipeline.render import render_recipe


def png(color,n=8):
    b=io.BytesIO();Image.fromarray(np.full((n,n,3),color,np.uint8)).save(b,'PNG');return b.getvalue()


class FakeFal:
    """The queue protocol without a network: every job completes on its first poll."""
    def __init__(self,glb=None):self.calls=[];self.glb=glb;self.n=0
    def http(self,method,url,body=None):
        self.calls.append((method,url))
        if method=='POST':
            self.n+=1;return dict(request_id=f'req{self.n}',status_url=f'{url}/status/{self.n}',response_url=f'{url}/response/{self.n}')
        if '/status/' in url:return dict(status='COMPLETED')
        if fal.PATINA in url:return dict(images=[dict(url=f'https://cdn.test/{m}.png',map_type=m) for m in fal.PATINA_ROLES],seed=0)
        return dict(model_glb=dict(url='https://cdn.test/model.glb',file_name='model.glb'),model_urls=dict(glb=dict(url='https://cdn.test/model.glb'),fbx=dict(url='https://cdn.test/model.fbx')))
    def download(self,url):
        assert not url.endswith('.fbx'),'only png/glb files are stored'
        return self.glb if url.endswith('.glb') else png([200,120,60] if 'basecolor' in url else [128,128,255])


@pytest.fixture
def cache(tmp_path,monkeypatch):
    monkeypatch.setattr(fal,'ROOT',tmp_path/'fal');monkeypatch.setenv('FAL_KEY','test-key');return tmp_path/'fal'


def test_prefetch_caches_ledgers_and_caps(cache):
    fake=FakeFal();jobs=fal.material_jobs(0)
    records=fal.prefetch(jobs,http=fake.http,download=fake.download,interval=0)
    assert all(not r.get('error') and not r['cached'] and r['cost_usd']==fal.PRICE_USD[fal.PATINA] for r in records)
    entry=fal.cached(*fal.patina_request('paint',0))
    assert entry and all((entry/f'{m}.png').exists() for m in fal.PATINA_ROLES) and json.loads((entry/'meta.json').read_text())['request_id']
    again=fal.prefetch(jobs,http=fake.http,download=fake.download,interval=0)
    assert all(r['cached'] and r['cost_usd']==0 for r in again) and len(fake.calls)==3*len(jobs)  # submit, status, response; no calls on hits
    model,payload=fal.patina_request('paint',7)
    capped=fal.prefetch([dict(kind='material',finish='paint',model=model,payload=payload)],budget_usd=.01,http=fake.http,download=fake.download,interval=0)
    assert capped[0]['error']['code']=='BUDGET_EXHAUSTED' and fal.cached(model,payload) is None


def test_prefetch_requires_key_only_for_uncached_jobs(cache,monkeypatch):
    fake=FakeFal();fal.prefetch(fal.material_jobs(0)[:1],http=fake.http,download=fake.download,interval=0)
    monkeypatch.delenv('FAL_KEY')
    monkeypatch.delenv('FAL_API_KEY',raising=False)
    assert fal.prefetch(fal.material_jobs(0)[:1],http=fake.http,download=fake.download)[0]['cached']
    with pytest.raises(PipelineError,match='FAL_KEY'):fal.prefetch(fal.material_jobs(0)[1:2],http=fake.http,download=fake.download)


def test_api_key_alias_and_scene_local_floor_override(cache,tmp_path,monkeypatch):
    monkeypatch.delenv('FAL_KEY');monkeypatch.setenv('FAL_API_KEY','test-alias')
    prompt='seamless natural oak plank flooring'
    model,payload=fal.patina_request('floor_tile',prompt=prompt)
    fake=FakeFal();fal.prefetch([dict(kind='material',finish='floor_tile',model=model,payload=payload)],http=fake.http,download=fake.download,interval=0)
    ir=solve(load('examples/cafe_program.py'),1,tmp_path)
    ir['meta']['appearance']={'materials':dict(source='fal',overrides={'floor_tile':dict(prompt=prompt,tile_m=2.)})}
    write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    report=json.loads((tmp_path/'manifest.json').read_text())['materials']
    assert report['realized']['floor_tile']['prompt']==prompt
    assert report['realized']['floor_tile']['tile_m']==2.
    assert fal.cached(*fal.patina_request('floor_tile')) is None
    compiled=mujoco.MjSpec.from_zip(str(tmp_path/'scene.mjz')).compile()
    assert compiled.mat_texrepeat[compiled.material('pbr_floor_tile').id].tolist()==pytest.approx([.5,.5])


def test_compiler_realizes_pbr_layers_once_and_degrades_per_finish(cache,tmp_path):
    fake=FakeFal();fal.prefetch([j for j in fal.material_jobs(0) if j['finish']!='wall_paint'],http=fake.http,download=fake.download,interval=0)
    ir=solve(load('examples/cafe_program.py'),1,tmp_path)
    ir['meta']['appearance']={'materials':{'source':'fal','seed':0},'tint':[1.,.5,1.]}
    write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    report=json.loads((tmp_path/'manifest.json').read_text())['materials']
    assert set(report['realized'])=={'floor_tile','paint','stainless','wood'} and report['fallback']==['wall_paint'] and report['rebound_geoms']>0
    assert json.loads((tmp_path/'provenance.json').read_text())['materials']['source']=='fal'
    model=mujoco.MjSpec.from_zip(str(tmp_path/'scene.mjz')).compile();R=mujoco.mjtTextureRole
    mid=model.material('pbr_paint').id
    assert all(model.mat_texid[mid,int(r)]>=0 for r in (R.mjTEXROLE_RGB,R.mjTEXROLE_NORMAL,R.mjTEXROLE_ROUGHNESS,R.mjTEXROLE_METALLIC))
    assert model.mat_texuniform[mid] and model.mat_texrepeat[mid,0]==pytest.approx(2.) and model.ntex==16  # four maps x four realized finishes; no checker
    assert model.mat_rgba[mid].tolist()==pytest.approx([1.,.5,1.,1.])  # tint multiplies the rgb layer, as in the flat path
    bound={model.material(model.geom_matid[i]).name for i in range(model.ngeom) if model.geom_matid[i]>=0}
    assert {'pbr_paint','pbr_floor_tile','pbr_stainless','finish_wall_paint'}<=bound and not any(b.endswith('/finish_paint') or b.endswith('/finish_stainless') for b in bound)
    assert zipfile.ZipFile(tmp_path/'scene.mjz').namelist().count('pbr_paint_rgb.png')==1
    recipe=json.loads(render_recipe(tmp_path).read_text())
    assert set(recipe['materials']['pbr_paint']['textures'])=={'rgb','normal','roughness','metallic'} and recipe['materials']['pbr_paint']['texuniform']
    assert any(g['material']=='pbr_paint' for g in recipe['geoms'])


def test_flat_mode_keeps_placeholders_and_checker(tmp_path):
    ir=solve(load('examples/cafe_program.py'),1,tmp_path);write_json(tmp_path/'ir.json',ir);compile_scene(ir,tmp_path)
    report=json.loads((tmp_path/'manifest.json').read_text())['materials']
    assert report['source']=='flat' and report['realized']=={} and report['rebound_geoms']==0
    model=mujoco.MjSpec.from_zip(str(tmp_path/'scene.mjz')).compile()
    assert model.ntex==1 and model.material('finish_floor_tile').id>=0


def test_cycles_mesh_uvs_preserve_source_atlas_orientation(tmp_path):
    # Asymmetric UVs catch the vertical flip introduced when MuJoCo compiles an
    # OBJ. Blender must recover the source mapping alongside the original image.
    source_uv=np.array([[.1,.2],[.8,.3],[.7,.9],[.2,.6]])
    obj='v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n'
    obj+=''.join(f'vt {u} {v}\n' for u,v in source_uv)
    obj+='f 1/1 3/3 2/2\nf 1/1 2/2 4/4\nf 1/1 4/4 3/3\nf 2/2 3/3 4/4\n'
    pixels=np.array([[[255,0,0],[0,255,0]],[[0,0,255],[255,255,0]]],dtype=np.uint8)
    image=io.BytesIO();Image.fromarray(pixels).save(image,'PNG')
    spec=mujoco.MjSpec.from_string('''<mujoco><asset>
      <mesh name="mesh" file="mesh.obj"/><texture name="atlas" type="2d" file="atlas.png"/>
      <material name="surface" texture="atlas"/></asset><worldbody>
      <geom name="prop" type="mesh" mesh="mesh" material="surface" contype="0" conaffinity="0"/>
      </worldbody></mujoco>''',assets={'mesh.obj':obj.encode(),'atlas.png':image.getvalue()})
    spec.to_zip(str(tmp_path/'scene.mjz'))
    write_json(tmp_path/'ir.json',dict(rooms=[],meta=dict(seed=0)))
    recipe=json.loads(render_recipe(tmp_path).read_text())
    np.testing.assert_allclose(recipe['geoms'][0]['uv'],source_uv,atol=1e-7)
    texture=tmp_path/'render'/recipe['materials']['surface']['textures']['rgb']
    np.testing.assert_array_equal(np.array(Image.open(texture)),pixels)


def test_generated_decor_has_static_collision_and_is_placed_on_supports(cache,tmp_path):
    import trimesh
    box=trimesh.creation.box(extents=[.5,1.,.3])  # genuine glTF Y-up: height is the 1.0 m Y axis
    box.visual=trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices),2)),image=Image.fromarray(np.full((4,4,3),90,np.uint8)))
    fake=FakeFal(glb=box.export(file_type='glb'))
    program=parse('scene("a prep table with mugs and a kettle",space(area_m2=40),[place("table","prep_table"),place("mugs","mug",count=2,required=False),place("kettle","kettle",required=False)])')
    ir=solve(program,3,tmp_path/'cold')  # nothing cached: optional decor is dropped with a reason and the scene still solves
    assert not any(o['category'] in ('mug','kettle') for o in ir['objects'])
    assert {d['reason']['code'] for d in ir['provenance']['dropped']}=={'GENERATED_ASSET_UNAVAILABLE'}
    fal.prefetch(fal.decor_jobs(program),http=fake.http,download=fake.download,interval=0)
    ir=solve(program,3,tmp_path/'warm')
    mugs=[o for o in ir['objects'] if o['category']=='mug']
    assert len(mugs)==2 and all(o['support_parent']=='table_0' for o in mugs) and not any(o['dynamic'] for o in mugs)
    assert mugs[0]['dimensions'][2]==pytest.approx(.105,abs=1e-6) and mugs[0]['source_classification']['allowed_use']=='static_collision'
    assert mugs[0]['dimensions'][:2]==pytest.approx([.0525,.0315],abs=1e-6)
    write_json(tmp_path/'warm'/'ir.json',ir);compile_scene(ir,tmp_path/'warm')
    inventory=json.loads((tmp_path/'warm'/'provenance.json').read_text());assert inventory['counts']['generated']==3
    manifest=json.loads((tmp_path/'warm'/'manifest.json').read_text())
    assert manifest['instances']['mugs_0']['provenance']['request_id'] and manifest['instances']['mugs_0']['provenance']['hashes']
    model=mujoco.MjSpec.from_zip(str(tmp_path/'warm'/'scene.mjz')).compile()
    decor=[i for i in range(model.ngeom) if model.geom(i).name.startswith('mugs_0/')]
    assert len(decor)==2 and all(model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH for i in decor)
    visual=model.geom('mugs_0/surface_visual').id;collider=model.geom('mugs_0/surface_collision').id
    assert not model.geom_contype[visual] and not model.geom_conaffinity[visual]
    assert model.geom_contype[collider]==model.geom_conaffinity[collider]==1
    assert model.geom_group[collider]==3 and model.geom_rgba[collider,3]==0
    assert model.mesh_texcoordnum[model.geom_dataid[visual]]>0
    assert model.body_mass[model.geom_bodyid[collider]]>0
    provenance=manifest['instances']['mugs_0']['provenance']
    assert provenance['physical_use']=='static_collision' and not provenance['calibrated']
    assert not manifest['instances']['mugs_0']['source_classification']['contact_rich_certified']

    # A moving probe representing an arm link must contact and stop at the prop,
    # after the full asset -> scene -> MJZ round trip, not just expose a mask bit.
    spec=mujoco.MjSpec.from_zip(str(tmp_path/'warm'/'scene.mjz'))
    center=np.array(mugs[0]['position'])+np.array([0,0,mugs[0]['dimensions'][2]/2])
    # Approach along the mug's local X axis (layout may rotate it).
    yaw=mugs[0]['yaw'];axis=np.array([np.cos(yaw),np.sin(yaw),0.])
    distance=mugs[0]['dimensions'][0]/2+.04
    probe=spec.worldbody.add_body(name='arm_probe',pos=center-axis*distance)
    probe.add_joint(name='probe_slide',type=mujoco.mjtJoint.mjJNT_SLIDE,axis=axis)
    probe.add_geom(name='probe_tip',type=mujoco.mjtGeom.mjGEOM_SPHERE,size=[.01,0,0],mass=.1)
    model=spec.compile();data=mujoco.MjData(model)
    data.joint('probe_slide').qvel[0]=.5
    tip=model.geom('probe_tip').id;collider=model.geom('mugs_0/surface_collision').id
    contacted=False;furthest=0.
    for _ in range(300):
        mujoco.mj_step(model,data)
        contacted |= any({c.geom1,c.geom2}=={tip,collider} for c in data.contact)
        furthest=max(furthest,float(data.joint('probe_slide').qpos[0]))
    assert contacted
    assert furthest<.035  # .03 m to contact plus soft-contact tolerance; cannot pass through

    from scene_pipeline.portability import export_urdf
    import xml.etree.ElementTree as ET
    export_urdf(tmp_path/'warm')
    assert any(node.find('geometry/mesh') is not None
               for path in (tmp_path/'warm'/'urdf').glob('*.urdf')
               for node in ET.parse(path).iter('collision'))


@pytest.mark.parametrize('category',['tea_towel','jar'])
def test_novel_generated_category_registers_and_replays_without_fal(cache,tmp_path,monkeypatch,category):
    import trimesh
    from scene_pipeline.orchestrator import generate
    from scene_pipeline.asset_library import verify,registered_assets
    from scene_pipeline.contracts import read_json,validate_program
    program=load('examples/cafe_program.py')
    request=dict(prompt='a '+category.replace('_',' '),size_m=.2,placement='support',physical_use='static_collision')
    program['objects'].append(dict(id='decor',category=category,count=2,zone='main',required=True,generated_request=request))
    fake=FakeFal(glb=trimesh.creation.box(extents=[.2,.03,.15]).export(file_type='glb'))
    jobs=fal.decor_jobs(program)
    assert len(jobs)==1
    fal.prefetch(jobs,http=fake.http,download=fake.download,interval=0)
    store=tmp_path/'library';scene=tmp_path/'scene'
    assert generate(program['prompt'],1,scene,program=program,clutter='fal',asset_store=store,preview=False)['passed']
    resolved=read_json(scene/'program.json');item=resolved['objects'][-1]
    assert 'generated_request' not in item and item['asset_ref']
    asset=verify(store/'packages'/item['asset_ref'])
    assert asset['capabilities']['static_collision'] and not asset['capabilities']['visual_only']
    assert not asset['capabilities']['articulated'] and not asset['capabilities']['dynamic']
    assert asset['provenance']['size_basis']=='agent_estimate_not_measured'
    assert registered_assets(category.replace('_',' '),store)[0]['asset_ref']==item['asset_ref']
    assert all(o['source_classification']['allowed_use']=='static_collision' for o in read_json(scene/'ir.json')['objects'] if o['category']==category)
    metrics=[m for m in read_json(scene/'validation.json')['metrics'] if m['category']==category]
    assert metrics and all(m['mass_ok'] is None and m['mass_check_scope']=='unverified_generated_static_collision_proxy' for m in metrics)
    monkeypatch.setattr(fal,'_http',lambda *a,**k:pytest.fail('A registered asset must replay offline'))
    assert generate(program['prompt'],1,tmp_path/'replay',program=resolved,asset_store=store,preview=False)['passed']
    bad={**item,'generated_request':request}
    resolved['objects'][-1]=bad
    with pytest.raises(PipelineError,match='asset request/reference'):validate_program(resolved)


def test_generated_request_cannot_claim_contact_capability():
    program=load('examples/cafe_program.py')
    program['objects'].append(dict(id='novel',category='novel',count=1,zone='main',required=True,
        generated_request=dict(prompt='a prop',size_m=.2,placement='support',physical_use='contact_rich')))
    from scene_pipeline.contracts import validate_program
    with pytest.raises(PipelineError,match='visual_only'):validate_program(program)


def test_supported_object_can_turn_to_fit(cache,tmp_path,monkeypatch):
    import math
    import trimesh
    from scene_pipeline.asset_library import register_generated
    request=dict(prompt='a long rectangular visual prop',size_m=1.,placement='support',physical_use='visual_only')
    fake=FakeFal(glb=trimesh.creation.box(extents=[.55,.1,1.]).export(file_type='glb'))
    fal.prefetch(fal.decor_jobs({'objects':[dict(category='long_prop',generated_request=request)]}),
                 http=fake.http,download=fake.download,interval=0)
    store=tmp_path/'library';monkeypatch.setenv('SCENE_PIPELINE_ASSET_STORE',str(store))
    _,asset=register_generated('long_prop',request,store=store)
    program=parse('scene("A work surface and a long prop",space(area_m2=30),[place("prop","long_prop",asset_ref="'+asset['key']+'"),place("table","prep_table")])')
    ir=solve(program,23,tmp_path/'scene');by_id={o['id']:o for o in ir['objects']}
    prop,table=by_id['prop_0'],by_id['table_0']
    assert prop['support_parent']=='table_0'
    assert abs(math.sin(prop['yaw']-table['yaw']))==pytest.approx(1.)
