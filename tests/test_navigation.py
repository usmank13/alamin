import json
import os
from pathlib import Path
import subprocess
import time

import h5py
import mujoco
import numpy as np
from PIL import Image
import pytest
from scipy.spatial.transform import Rotation

from scene_pipeline.agriculture import compile_bundle
from scene_pipeline.agriculture_drive import camera_rig,run
from scene_pipeline.contracts import PipelineError
from scene_pipeline.navigation import add_cameras
from scene_pipeline.rhizome import RhizomeController,validate_config
from test_agriculture import bundle
from test_gen3 import robot


def test_nav_camera_axes_and_intrinsics_are_separate_from_crop_capture():
    spec=mujoco.MjSpec();spec.worldbody.add_body(name='chassis')
    profile=dict(cameras={},root_ground_height_m=.9)
    result=add_cameras(spec,profile,dict(cameras=['front','rear']))
    rig=camera_rig(result,640,360)
    assert not profile['cameras']
    for name,sign in [('nav_front',1),('nav_rear',-1)]:
        camera=spec.camera(name)
        q=camera.quat;R=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        ray=-R[:,2]
        assert ray[0]*sign>.9 and ray[2]<-.3
        assert rig[name]['K']==[[400,0,320],[0,400,240],[0,0,1]]
        assert (rig[name]['width'],rig[name]['height'])==(640,480)


def test_nav_config_has_no_required_route_and_rejects_bad_options():
    base=dict(root='/tmp/rhizome',nav_model=dict(package='/tmp/package'))
    assert validate_config(base)['waypoints']==[]
    with pytest.raises(PipelineError,match='replaces route waypoints'):
        validate_config({**base,'waypoints':[dict(xy=[1,0])]})
    for value in [dict(use_map=True),dict(arrangement=4),dict(cameras=['crop']),dict(cameras=[]),dict(device='automatic')]:
        with pytest.raises(PipelineError):validate_config({**base,'nav_model':{**base['nav_model'],**value}})


@pytest.fixture
def nav_config():
    repo=Path(__file__).resolve().parents[1]
    root=Path(os.environ.get('RHIZOME_ROOT',repo.parent/'rhizome'))
    host=Path(os.environ.get('RHIZOME_HOST_BUILD',repo/'outputs/rhizome_host_nav/host'))
    python=Path(os.environ.get('RHIZOME_INFERENCE_PYTHON',repo.parent/'ml-aigen-tools/venv312/bin/python'))
    package=Path(os.environ.get('RHIZOME_NAV_MODEL',repo/'outputs/nav_obstacle_package/nav_grid_4_nav'))
    if not (host/'build_manifest.json').exists() or not python.exists() or not (package/'model.ts').exists():
        pytest.skip('Requires the optional Rhizome host build, inference Python and nav package')
    return dict(root=str(root),host_build=str(host),python=str(python),nav_model=dict(package=str(package),cameras=['front']))


def reference_inputs(package):
    return dict(front_rgb=np.asarray(Image.open(Path(package)/'selftest/image.png').convert('RGB')),
        front_K=np.array([[400,0,320],[0,400,240],[0,0,1.]]),
        front_rotation_robot=Rotation.from_euler('xyz',[-1.952,0,-1.57]).as_matrix(),
        front_position_robot=np.array([.40949,0,-.089478]),front_height_m=np.array(.8144))


def test_native_navigation_moves_and_stops_on_missing_or_expired_frames(nav_config,tmp_path):
    # Rhizome's own input-dependent TorchScript fixture isolates driving gates
    # from learned prediction accuracy; the real package is tested separately.
    stub=tmp_path/'stub'
    code="import sys;sys.path[:0]=[sys.argv[1]+'/src',sys.argv[1]+'/test'];from nav_model_stub import write_stub_package;write_stub_package(sys.argv[2],with_selftest=True)"
    result=subprocess.run([nav_config['python'],'-c',code,nav_config['root'],str(stub)],capture_output=True,text=True,timeout=60)
    assert result.returncode==0,result.stderr
    nav_config['nav_model']['package']=str(stub)
    controller=RhizomeController(nav_config,tmp_path/'run',timeout=90)
    arrays=reference_inputs(stub)
    try:
        for tick in range(21):
            t=tick*.01;step=controller.step(t,[0,0,.9],[1,0,0,0],0)
            if tick%10==0:observation=controller.nav_infer(t,**arrays)
        assert observation['driving'] and len(observation['prediction']['rows_fitted'])==2
        assert step['twist'][0]>0
        time.sleep(.6)  # Wall time must not expire simulated frames/goals.
        assert controller.step(.21,[0,0,.9],[1,0,0,0],0)['twist'][0]>0
        for tick in range(22,72):step=controller.step(tick*.01,[0,0,.9],[1,0,0,0],0)
        assert step['twist']==[0,0]
        observation=controller.nav_infer(.71,**arrays)
        assert observation['driving']
        assert not controller.nav_infer(.71)['driving']
        assert controller.step(.72,[0,0,.9],[1,0,0,0],0)['twist']==[0,0]
    finally:controller.close()


def test_real_nav_model_capture_has_no_oracle_inputs(bundle,nav_config,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Vision navigation used scene truth')
    monkeypatch.setattr('scene_pipeline.rhizome.visible_detections',forbidden)
    monkeypatch.setattr('scene_pipeline.rhizome.visible_row_geometry',forbidden)
    monkeypatch.setattr('scene_pipeline.agriculture_drive.shadow_row_profile',forbidden)
    source,_=bundle;scene=tmp_path/'scene';compile_bundle(source,scene)
    output=tmp_path/'capture'
    report=run(scene,output,config=dict(seconds=.2,controller=nav_config),video=False,timeout=120)
    assert report['passed'],report['failure']
    assert report['rhizome']['navigation']['frames']==3
    assert report['rhizome']['oracle_detections']==0
    assert report['rhizome']['provenance']['navigation']['selftest_max_error']<1e-3
    with h5py.File(output/'data.h5') as data:
        assert data['camera_nav_front/rgb'].shape==(3,480,640,3)
        assert 'depth' not in data['camera_nav_front']
        assert len(data['nav_model/time'])==3
    assert (output/'nav_front_neural.png').exists()
    assert not (output/'rhizome/nav_input.npz').exists()


@pytest.mark.parametrize('count',[3,4])
def test_navigation_and_payload_ownership_and_stop(robot,nav_config,tmp_path,count):
    from scene_pipeline.rhizome import RhizomeMultiController,bind_physical_arms
    profile=robot[1]
    cfg,arms=bind_physical_arms({**nav_config,'perception':'oracle_visible',
        'physical_arms':[a['id'] for a in profile['arms'][:count]]},profile)
    controller=RhizomeMultiController(cfg,tmp_path,arms,timeout=120)
    feedback={name:[0,0] for name in arms};arrays=reference_inputs(nav_config['nav_model']['package'])
    try:
        assert controller.nav_worker.info['navigation_enabled']
        assert all(not w.info['navigation_enabled'] for w in controller.workers.values())
        first=controller.step(0,[0,0,.9],[1,0,0,0],0,feedback)
        assert first['twist']==[0,0]
        assert all(a['navigation_held'] and a['commanded']==[0,0] for a in first['arms'].values())
        controller.nav_infer(0,**arrays)
        for tick in range(1,21):
            step=controller.step(tick*.01,[0,0,.9],[1,0,0,0],0,feedback)
            if tick==10:controller.nav_infer(.1,**arrays)
        assert step['twist'][0]>0 and all(not a['navigation_held'] for a in step['arms'].values())
        controller.nav_infer(.2)
        step=controller.step(.21,[0,0,.9],[1,0,0,0],0,feedback)
        assert step['twist']==[0,0]
        assert all(a['navigation_held'] and a['commanded']==[0,0] for a in step['arms'].values())
        controller.nav_worker.process.kill();controller.nav_worker.process.wait()
        with pytest.raises(PipelineError,match='navigation'):
            controller.step(.22,[0,0,.9],[1,0,0,0],0,feedback)
        assert all(w.process.poll() is not None for w in controller.workers.values())
    finally:controller.close()
