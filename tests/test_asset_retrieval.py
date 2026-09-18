"""Offline fixtures exercise the same importer and reference boundary as real downloads."""
from copy import deepcopy
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

from scene_pipeline.contracts import PipelineError,read_json,write_json,validate_program,digest
from scene_pipeline.sdf_import import import_sdf,local_file
from scene_pipeline import asset_library as library

SDF='''<sdf version="1.6"><model name="fixture"><link name="link"><pose>2 3 .25 0 0 0</pose>
<visual name="v"><geometry><box><size>1 2 .5</size></box></geometry></visual>
<collision name="c"><geometry><box><size>1 2 .5</size></box></geometry></collision>
</link><static>true</static></model></sdf>'''


@pytest.fixture
def source(tmp_path):
    root=tmp_path/'fixture';root.mkdir()
    (root/'model.config').write_text('<model><name>Fixture</name><sdf version="1.6">model.sdf</sdf></model>')
    (root/'model.sdf').write_text(SDF);(root/'REPOSITORY_LICENSE').write_text('Test fixture authored in this repository')
    record=dict(candidate=dict(candidate_id='test:fixture',folder='fixture',source_url='https://example.test/fixture',
        source=dict(repo='test/fixtures',revision='a'*40,license='test fixture')),sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir()})
    write_json(root/'source.json',record)
    return root,record


def test_sdf_native_dimensions_and_centering(source,tmp_path):
    root,_=source;report=import_sdf(root,tmp_path/'import')
    assert np.allclose(report['dimensions'],[1,2,.5])
    assert np.allclose(report['bounds'],[[-.5,-1,0],[.5,1,.5]])
    model=mujoco.MjSpec.from_zip(str(tmp_path/'import/asset.mjz')).compile()
    assert model.njnt==0 and model.ngeom==2 and report['mode']=='static'
    assert report['supports'][0]['center']==[0,0,.5]


def test_scene_rejects_asset_above_ceiling(package,tmp_path):
    from scene_pipeline.dsl import parse
    from scene_pipeline.layout import solve
    from scene_pipeline.scene_intent import freeze
    from scene_pipeline.scene_checks import check_scene
    _,folder,asset=package
    import shutil
    shutil.copytree(folder,tmp_path/'scene/assets'/asset['key'])
    p=parse('scene("A workroom",space(),[place("prop","test_prop",asset_ref="'+asset['key']+'")])')
    ir=solve(p,1,tmp_path/'scene');ir['rooms'][0]['height']=.2
    report=check_scene(freeze(p,authority='supplied_program'),ir,tmp_path/'scene')
    assert any(e['code']=='ABOVE_CEILING' for e in report['errors'])


@pytest.mark.parametrize('extra',['<plugin name="p" filename="danger.so"/>','<include><uri>https://example.test</uri></include>',
                                '<joint name="j" type="revolute"/>','<frame name="x"/>'])
def test_unsupported_features_not_silently_stripped(source,tmp_path,extra):
    root,_=source;(root/'model.sdf').write_text(SDF.replace('</model>',extra+'</model>'))
    with pytest.raises(PipelineError):import_sdf(root,tmp_path/'import')


@pytest.mark.parametrize('uri',['/etc/passwd','file:///etc/passwd','../../outside','https://example.test/mesh.obj','model://other/mesh.dae'])
def test_mesh_paths_cannot_escape(source,uri):
    with pytest.raises(PipelineError):local_file(uri,source[0])


def test_package_resource_resolution_is_local_and_unambiguous(source):
    from scene_pipeline.sdf_import import PackageResolver
    root,_=source;(root/'materials/textures').mkdir(parents=True);(root/'meshes').mkdir()
    (root/'materials/textures/test.png').write_bytes(b'test')
    resolver=PackageResolver(root,root/'meshes')
    assert resolver.get('../materials/textures/test.png')==b'test'
    assert resolver.get('test.png')==b'test'
    (root/'test.png').write_bytes(b'ambiguous')
    with pytest.raises(PipelineError):resolver.get('test.png')
    with pytest.raises(PipelineError):resolver.get('../../../outside')


def test_coplanar_visual_mesh_can_import_without_fake_collision_volume(source,tmp_path):
    root,_=source
    (root/'flat.obj').write_text('v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3\nf 1 3 4\n')
    sdf=SDF.replace('<visual name="v"><geometry><box><size>1 2 .5</size></box></geometry></visual>',
                    '<visual name="v"><geometry><mesh><uri>flat.obj</uri></mesh></geometry></visual>')
    (root/'model.sdf').write_text(sdf)
    report=import_sdf(root,tmp_path/'import')
    assert report['counts']['visual_parts']==1 and report['counts']['source_collisions']==1


def test_classic_collada_unit_and_axis_convention(source):
    pytest.importorskip('collada');pytest.importorskip('networkx')
    import trimesh
    import xml.etree.ElementTree as ET
    from scene_pipeline.sdf_import import mesh_parts
    root,_=source
    document=ET.fromstring(trimesh.exchange.dae.export_collada(trimesh.creation.box(extents=[1,2,3])))
    ns={'c':'http://www.collada.org/2005/11/COLLADASchema'}
    asset=document.find('c:asset',ns)
    unit=asset.find('c:unit',ns)
    if unit is None:unit=ET.SubElement(asset,'{'+ns['c']+'}unit')
    unit.set('meter','.01')
    up=document.find('c:asset/c:up_axis',ns);up.text='Y_UP'
    path=root/'mesh.dae';path.write_bytes(ET.tostring(document))
    assert np.allclose(mesh_parts(path,root)[0].extents,[.01,.02,.03])
    up.text='Z_UP';path.write_bytes(ET.tostring(document))
    assert np.allclose(mesh_parts(path,root)[0].extents,[.01,.02,.03])


def test_source_xml_entities_rejected(source,tmp_path):
    root,_=source;(root/'model.sdf').write_text('<!DOCTYPE sdf [<!ENTITY x SYSTEM "file:///etc/passwd">]>'+SDF)
    with pytest.raises(PipelineError,match='entities'):import_sdf(root,tmp_path/'import')


def test_claimed_articulation_cannot_be_invented(source,tmp_path):
    with pytest.raises(PipelineError,match='no invented articulation'):import_sdf(source[0],tmp_path/'import',mode='articulated')


@pytest.fixture
def package(source,tmp_path,monkeypatch):
    monkeypatch.setattr(library,'download',lambda *a,**kw:source)
    store=tmp_path/'library'
    folder,asset=library.acquire('test:fixture','test_prop',store=store,preview=False)
    return store,folder,asset


def test_acquire_seals_and_retains_licenses(package,tmp_path):
    store,folder,asset=package
    assert library.verify(folder)['state']=='verified'
    assert (folder/'licenses/REPOSITORY_LICENSE').exists()
    assert asset['capabilities']['articulated'] is False
    copy,info=library.materialize(asset['key'],tmp_path/'scene/assets',store)
    assert info['key']==asset['key'] and (copy/'asset.mjz').read_bytes()==(folder/'asset.mjz').read_bytes()
    (copy/'asset.mjz').write_bytes(b'corrupt')
    with pytest.raises(PipelineError,match='changed'):library.verify(copy)


def test_reference_in_scene_and_replay_without_original_store(package,tmp_path,monkeypatch):
    from test_architecture_generation import make_bundle
    from scene_pipeline.orchestrator import generate
    from scene_pipeline.dataset import collect_variants,Recorder
    from scene_pipeline import flows
    store,folder,asset=package
    program=read_json('examples/architecture_hall.json');program['space']['kind']='workroom'
    program['objects']=[dict(id='prop',category='test_prop',count=1,zone='main',required=True,asset_ref=asset['key'])]
    first=tmp_path/'scene'
    result=generate(program['prompt'],1,first,program=program,preview=False,layout_backend='architecture',priors=make_bundle(),asset_store=store)
    assert result['passed'] and result['usage']['calls']==0
    ir=read_json(first/'ir.json');assert ir['objects'][0]['source_classification']['origin']=='retrieved'
    assert read_json(first/'asset_resolutions.json')['prop']['dimensions_m']==asset['dimensions']
    monkeypatch.setenv('SCENE_PIPELINE_ASSET_STORE',str(tmp_path/'empty_library'))
    def fake_run(scene,flow,output,**kwargs):
        Path(output).mkdir();rec=Recorder(Path(output)/'data.h5',{});rec.append('state',0.,value=1);rec.close()
        return dict(passed=True)
    monkeypatch.setattr(flows,'run',fake_run)
    report=collect_variants(first,tmp_path/'variants',variants=1,seconds=.1)
    assert report['passed'],report


def test_category_and_dimension_override_rejected(package,tmp_path):
    store,folder,asset=package
    item=dict(id='prop',category='invented',count=1,zone='main',required=True,asset_ref=asset['key'])
    with pytest.raises(PipelineError,match='category differs'):library.resolve_request(item,tmp_path/'out',store=store)
    program=read_json('examples/architecture_hall.json');program['objects']=[{**item,'family':'box'}]
    with pytest.raises(PipelineError,match='native asset dimensions'):validate_program(program)


def test_supplied_query_does_not_start_nested_agent(tmp_path,monkeypatch):
    monkeypatch.setattr(library,'search',lambda *a:[dict(candidate_id='test:fixture')])
    with pytest.raises(PipelineError,match='supplied programs never start an agent'):
        library.resolve_request(dict(id='prop',category='prop',asset_request=dict(query='fixture')),tmp_path)


def test_agent_can_reject_all_retrieval_candidates(tmp_path,monkeypatch):
    monkeypatch.setattr(library,'search',lambda *a:[dict(candidate_id='test:cabinet',label='Kitchen cabinet')])
    def select(instruction,schema,*args,**kwargs):
        assert 'none' in schema['properties']['candidate_id']['enum']
        return dict(candidate_id='none',reason='A cabinet is not a stove')
    monkeypatch.setattr(library,'request',select)
    monkeypatch.setattr(library,'acquire',lambda *a,**k:pytest.fail('Rejected candidate must not be downloaded'))
    with pytest.raises(PipelineError,match='No credible semantic match'):
        library.resolve_request(dict(id='stove',category='stove',asset_request=dict(query='kitchen stove')),tmp_path,allow_agent=True)


def test_pinned_catalog_and_blob_download(tmp_path):
    from scene_pipeline.asset_catalog import index_sources,download,search
    files={'fixture/model.config':b'<model><sdf>model.sdf</sdf></model>', 'fixture/model.sdf':SDF.encode(),'LICENSE':b'test license'}
    tree=dict(truncated=False,tree=[dict(path=k,type='blob',mode='100644',size=len(v),sha=hashlib.sha1(f'blob {len(v)}\0'.encode()+v).hexdigest()) for k,v in files.items()])
    sources=[dict(id='test',repo='test/fixtures',revision='a'*40,prefix='',license_file='LICENSE',license='test')]
    import json
    def fetch(url,**kw):
        if 'api.github.com' in url:return json.dumps(tree).encode()
        return files[url.split('/'+'a'*40+'/')[1]]
    index_sources(tmp_path,sources,fetcher=fetch)
    assert search('fixture',tmp_path)[0]['candidate_id']=='test:fixture'
    root,record=download('test:fixture',tmp_path,fetcher=fetch)
    assert (root/'model.sdf').read_bytes()==SDF.encode()
    (root/'model.sdf').write_text('changed')
    with pytest.raises(PipelineError,match='changed'):download('test:fixture',tmp_path,fetcher=fetch)
