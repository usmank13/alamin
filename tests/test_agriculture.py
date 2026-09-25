import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','osmesa')

import h5py
import mujoco
import numpy as np
import pytest

from scene_pipeline.agriculture import (build_robot, compile_bundle, compile_environment,
    compose_robot, load_bundle, sha256)
from scene_pipeline.agriculture_drive import drive_config, run, wheel_targets
from scene_pipeline.contracts import PipelineError, write_json
from sim_harness.scene import load_scene


@pytest.fixture
def bundle(tmp_path):
    root=tmp_path/'source';root.mkdir();(root/'meshes').mkdir()
    np.savez_compressed(root/'terrain.npz',elevation_m=np.zeros((41,61),dtype=np.float32))
    # A visible crop under the first camera with separate semantic identity.
    points=np.array([[-.4,.4,0],[.1,.4,0],[-.2,.9,0],[-.2,.62,.25]])
    faces=np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]])
    np.savez_compressed(root/'meshes/plant.npz',vertices=points,faces=faces)
    mesh=dict(file='meshes/plant.npz',rgba=[.1,.8,.1,1],class_id=2,bounds_m=[points.min(0).tolist(),points.max(0).tolist()])
    wheels=[]
    for x,fore in ((.4155,'front'),(-.4155,'rear')):
        for y,side in ((.8382,'left'),(-.8382,'right')):
            wheels.append(dict(id=f'{fore}_{side}',center_m=[x,y,-.6667],radius_m=.1714,half_width_m=.06,meshes=[]))
    cameras={f'raptor_30_{i+1}':dict(position_m=[-.18,y,-.299],quaternion_wxyz=[1,0,0,0],
                                  optics=dict(sensor_width_mm=36,sensor_height_mm=20.25,lens_mm=18.97))
             for i,y in enumerate((.622,.1397,-.1397,-.622))}
    record=dict(schema_version=1,kind='inicio_sim_bundle',units='metres',up_axis='Z',
                robot=dict(type='RAPTOR_30',track_width_m=1.6764,wheels=wheels,cameras=cameras,meshes=[]),
                terrain=dict(file='terrain.npz',bounds_m=[[-3,-2],[3,2]]),
                instances=[dict(id='plant_0',category='crop',species='soybean',growth_stage=.3,
                                meshes=[mesh],position_m=[-.2,.62,0])],
                rows=[dict(id='soybean_row_0',endpoints_m=[[-2,.62],[2,.62]])],
                class_names={'0':'background','2':'crop_foliage','4':'occlusion'},
                input_config={},resolved_config={},provenance={'inicio_revision':'fixture'},limitations=[])
    record['sha256']={str(p.relative_to(root)):sha256(p) for p in root.rglob('*.npz')}
    write_json(root/'bundle.json',record)
    return root,record


def composed(bundle, spawn=None):
    root,record=bundle
    environment,_=compile_environment(record,root)
    robot,profile=build_robot(record,root)
    spec,model,data=compose_robot(environment,robot,profile,spawn or {},np.asarray(record['terrain']['bounds_m']))
    return spec,model,data,profile


@pytest.mark.parametrize('v,w',[(.25,0),(-.25,0),(0,.35)])
def test_wheels_move_physical_base_and_stop(bundle,v,w):
    _,model,data,profile=composed(bundle)
    mujoco.mj_step(model,data,nstep=500)
    initial=data.body('element/chassis').xpos.copy()
    target,_=wheel_targets(profile,v,w)
    data.ctrl[:]=target
    mujoco.mj_step(model,data,nstep=1000)
    position=data.body('element/chassis').xpos.copy()
    if v:
        assert (position[0]-initial[0])*np.sign(v)>.15
        assert abs(position[1]-initial[1])<.03
    else:
        rotation=data.body('element/chassis').xmat.reshape(3,3)
        # Skid steering has slip; this prototype checks direction and a visible
        # turn, not calibrated chassis yaw-rate tracking.
        assert np.arctan2(rotation[1,0],rotation[0,0])>np.radians(5)
    data.ctrl[:]=0
    mujoco.mj_step(model,data,nstep=1000)
    assert np.linalg.norm(data.qvel[:3])<.025
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)


def test_asymmetric_heightfield_matches_ray_queries(bundle):
    root,record=bundle
    z=np.arange(15,dtype=np.float32).reshape(3,5)*.01-.12
    np.savez_compressed(root/'terrain.npz',elevation_m=z)
    record['terrain']['bounds_m']=[[-2,-1],[2,1]]
    spec,_=compile_environment(record,root)
    model=spec.compile();data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    for x,y,expected in ((-1,-.5,-.085),(1,.5,-.015),(0,0,-.05)):
        distance=mujoco.mj_ray(model,data,np.array([x,y,2.]),np.array([0.,0.,-1.]),
                               np.array([1,0,0,0,0,0],dtype=np.uint8),1,-1,np.array([-1],dtype=np.int32))
        assert 2-distance==pytest.approx(expected,abs=1e-6)


def test_portable_capture_masks_intrinsics_and_replay_state(bundle,tmp_path):
    root,_=bundle
    scene=tmp_path/'scene';compile_bundle(root,scene)
    # Neither producer assets nor the exporter are needed after compilation.
    root.rename(tmp_path/'unavailable')
    output=tmp_path/'capture'
    result=run(scene,output,config={'seconds':.2,'commands':[{'time':0,'forward_mps':0,'yaw_rate_rps':0}],
                                   'camera':{'width':160,'height':90}},video=False)
    assert result['passed']
    rig=json.loads((output/'rig.json').read_text())
    with h5py.File(output/'data.h5') as f:
        assert len(f['state/time'])==21
        for name in rig['cameras']:
            camera=f['camera_'+name]
            np.testing.assert_allclose(camera['time'][:],[0,.1,.2],atol=1e-9)
            assert camera['rgb'].shape==(3,90,160,3)
            assert np.isfinite(camera['depth'][:]).all()
            assert set(np.unique(camera['semantic'][:]))<={0,2,4}
        first=f['camera_raptor_30_1']
        crop=first['semantic'][0]==2
        assert crop.sum()>20
        assert (first['instance'][0][crop]==1).all()
        # Crop is within 0.6m, rather than Inicio's tenfold Blender units.
        assert first['depth'][0][crop].max()<.65
        model,data=load_scene(output/'rollout_scene.mjz')
        data.qpos[:]=f['state/qpos'][-1];data.qvel[:]=f['state/qvel'][-1]
        mujoco.mj_forward(model,data)
        np.testing.assert_allclose(data.body('element/chassis').xpos,f['state/chassis_position'][-1],atol=1e-8)
    fx=rig['cameras']['raptor_30_1']['K'][0][0]
    assert fx==pytest.approx(18.97/36*160)
    # Post-rollout replay can reuse the compiled field without loading another
    # potentially large copy. Removing the archive makes that guarantee real.
    if __import__('shutil').which('ffmpeg'):
        from scene_pipeline.replay import video
        (output/'rollout_scene.mjz').rename(output/'unavailable_scene.mjz')
        replay=video(output,model=model)
        assert (output/'replay.mp4').stat().st_size>0
        assert replay['source_sha256']==sha256(output/'data.h5')


def test_bundle_integrity_and_placement_failures(bundle):
    root,record=bundle
    load_bundle(root)
    with pytest.raises(PipelineError,match='outside terrain'):
        composed(bundle,{'position_m':[2.8,0]})
    with (root/'terrain.npz').open('ab') as f:f.write(b'corruption')
    with pytest.raises(PipelineError) as exc:load_bundle(root)
    assert exc.value.code=='AGRICULTURE_CHECKSUM'
    record['sha256']={'../outside.npz':'x'};write_json(root/'bundle.json',record)
    with pytest.raises(PipelineError) as exc:load_bundle(root)
    assert exc.value.code=='AGRICULTURE_ASSET'


@pytest.mark.parametrize('config',[{'seconds':float('nan')},{'seconds':.011},
    {'commands':[{}]}, {'commands':{}}, [],
    {'commands':[dict(time=1,forward_mps=0,yaw_rate_rps=0)]},
    {'commands':[dict(time=0,forward_mps=float('inf'),yaw_rate_rps=0)]},
    {'camera':{'width':320,'height':240}}])
def test_invalid_drive_configs(config):
    with pytest.raises(PipelineError):drive_config(config)


def test_commands_clipped_and_capture_does_not_change_physics(bundle,tmp_path):
    root,_=bundle;scene=tmp_path/'scene';compile_bundle(root,scene)
    config={'seconds':.12,'camera':{'width':64,'height':36},'commands':[dict(time=0,forward_mps=10,yaw_rate_rps=-10)]}
    for tier in ('full','state'):
        run(scene,tmp_path/tier,config=config,tier=tier,video=False)
    with h5py.File(tmp_path/'full/data.h5') as a,h5py.File(tmp_path/'state/data.h5') as b:
        np.testing.assert_array_equal(a['state/qpos'][:],b['state/qpos'][:])
        np.testing.assert_allclose(a['commands/applied_twist'][0],[1,-1])


def test_render_recipe_has_field_bounds_and_contact_surface(bundle,tmp_path):
    from scene_pipeline.render import render_recipe
    root,_=bundle;scene=tmp_path/'scene';compile_bundle(root,scene)
    recipe=json.loads(render_recipe(scene).read_text())
    assert recipe['bounds_m']==[[-3,-2],[3,2]]
    assert 'rooms' not in recipe
    terrain=next(g for g in recipe['geoms'] if g['name']=='terrain')
    assert len(terrain['vertices'])==41*61
    assert len(terrain['faces'])==40*60*2
    assert all('collision' not in g['name'] for g in recipe['geoms'])


def test_packaged_prop_composes_and_survives_bundle_import(bundle,tmp_path):
    root,record=bundle
    xml='<mujoco><worldbody><body name="crate" pos="2 0 .15"><geom type="box" size=".15 .15 .15"/></body></worldbody></mujoco>'
    prop=mujoco.MjSpec.from_string(xml)
    (root/'props').mkdir();prop.to_zip(str(root/'props/crate.mjz'))
    record['sha256']['props/crate.mjz']=sha256(root/'props/crate.mjz')
    record['input_config']['simulation']={'props':[dict(name='crate',model='props/crate.mjz')]}
    write_json(root/'bundle.json',record)
    scene=tmp_path/'scene';compile_bundle(root,scene)
    manifest=json.loads((scene/'manifest.json').read_text())
    assert manifest['geom_labels']['crate/geom_0']==dict(instance='crate',class_id=9)
    second=tmp_path/'second';compile_bundle(scene/'bundle',second)
    model,_=load_scene(second/'scene.mjz')
    assert model.geom('crate/geom_0').id>=0


def test_wall_budget_produces_inspectable_failed_rollout(bundle,tmp_path):
    root,_=bundle;scene=tmp_path/'scene';compile_bundle(root,scene)
    report=run(scene,tmp_path/'capture',tier='state',video=False,timeout=1e-12)
    assert not report['passed']
    assert report['failure']=='Wall-clock budget exceeded'
    assert (tmp_path/'capture/index.html').exists()


def test_large_camera_frames_have_independent_compression_chunks(tmp_path):
    from scene_pipeline.dataset import Recorder,load_stream,inspect
    path=tmp_path/'capture.h5';recorder=Recorder(path,{})
    frame=np.random.default_rng(2).integers(0,256,(360,640,3),dtype=np.uint8)
    try:
        recorder.append('camera',0.,rgb=frame)
        recorder.append('camera',.1,rgb=255-frame)
    finally:recorder.close()
    with h5py.File(path) as f:
        assert f['camera/rgb'].chunks[0]==1
    np.testing.assert_array_equal(load_stream(path,'camera',1,2)['rgb'][0],255-frame)
    assert inspect(path)['camera']['samples']==2
