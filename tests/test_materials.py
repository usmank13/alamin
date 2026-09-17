"""PBR materials and generated decor: cache, compiler layers, rebinding, recipe roles, visual-only placement. Network-free."""
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
    assert fal.prefetch(fal.material_jobs(0)[:1],http=fake.http,download=fake.download)[0]['cached']
    with pytest.raises(PipelineError,match='FAL_KEY'):fal.prefetch(fal.material_jobs(0)[1:2],http=fake.http,download=fake.download)


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


def test_generated_decor_is_visual_only_and_placed_on_supports(cache,tmp_path):
    import trimesh
    box=trimesh.creation.box(extents=[.5,.3,1.])  # trimesh writes/reads glTF Y-up; the 1.0 m Z axis must survive the round trip
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
    assert mugs[0]['dimensions'][2]==pytest.approx(.105,abs=1e-6) and mugs[0]['source_classification']['allowed_use']=='visual_only'
    write_json(tmp_path/'warm'/'ir.json',ir);compile_scene(ir,tmp_path/'warm')
    inventory=json.loads((tmp_path/'warm'/'provenance.json').read_text());assert inventory['counts']['generated']==3
    manifest=json.loads((tmp_path/'warm'/'manifest.json').read_text())
    assert manifest['instances']['mugs_0']['provenance']['request_id'] and manifest['instances']['mugs_0']['provenance']['hashes']
    model=mujoco.MjSpec.from_zip(str(tmp_path/'warm'/'scene.mjz')).compile()
    decor=[i for i in range(model.ngeom) if model.geom(i).name.startswith('mugs_0/')]
    assert decor and all(model.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH and not (model.geom_contype[i] or model.geom_conaffinity[i]) for i in decor)
    assert model.mesh_texcoordnum[model.geom_dataid[decor[0]]]>0
