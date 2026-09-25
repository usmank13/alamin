import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest

from scene_pipeline.agriculture import compile_bundle
from scene_pipeline.agriculture_drive import run
from scene_pipeline.rhizome import RhizomeController,validate_config
from scene_pipeline.rhizome_perception import frame_transforms
from test_agriculture import bundle
from test_rhizome import rhizome_root


@pytest.fixture
def neural_config(rhizome_root):
    python=Path(os.environ.get('RHIZOME_INFERENCE_PYTHON',Path(rhizome_root).parent/'ml-aigen-tools/venv312/bin/python'))
    package=Path(os.environ.get('RHIZOME_MODEL','/data/models/edge_packages/multicrop_td_wzmle4cf_grid'))
    if not python.exists() or not any((package/name).exists() for name in ('model.ts','core.ts')):
        pytest.skip('Needs RHIZOME_INFERENCE_PYTHON and RHIZOME_MODEL')
    return dict(root=rhizome_root,python=str(python),waypoints=[dict(xy=[1,0])],perception='neural',
        neural=dict(package=str(package),crop_species='soybean',device='cpu',
                    target_positions_chassis_m={f'raptor_30_{i+1}':[-.18,y,-.3] for i,y in enumerate((.622,.1397,-.1397,-.622))}),
        shadow_arm=dict(camera='raptor_30_1',position_chassis_m=[-.18,.622,-.3]))


def test_camera_and_target_transform_roundtrip():
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    origin=np.array([2.,3.,.8]);mount=np.array([.1,.6,-.3])
    cam=origin+rotation@mount
    rc,tc,rt,tt=frame_transforms(rotation,cam,rotation,origin,mount)
    assert tc.shape==tt.shape==(3,)  # GridModel translations are vectors, not columns.
    assert np.allclose(rt@(rc@np.array([0,0,.5])+tc)+tt,[0,0,-.5])
    target_point=np.array([.1,-.2,-.4])
    world=rotation@target_point+cam
    assert np.allclose(rt@world+tt,target_point)


def test_neural_requires_explicit_model_and_matching_arm_mount():
    base=dict(root='/tmp/rhizome',waypoints=[dict(xy=[0,0])],perception='neural')
    with pytest.raises(ValueError):validate_config(base)
    neural=dict(package='/tmp/model',crop_species='soybean',target_positions_chassis_m={'cam':[0,0,0]})
    with pytest.raises(ValueError):
        validate_config({**base,'neural':neural,'shadow_arm':dict(camera='cam',position_chassis_m=[0,1,0])})


def test_production_core_matches_package_reference_and_depth_units(neural_config):
    # Execute under the selected inference environment; Alamin stays free of
    # Torch/OpenCV dependencies. No substitute neural implementation is used.
    code='''
import sys,os,numpy as np,torch,yaml
sys.path[:0]=[os.environ.get('RHIZOME_HOST_BUILD',sys.argv[1]+'/build/host')+'/py',sys.argv[1]+'/src',sys.argv[3]]
from crop_inference_engine.model.runtime import ModelPipeline
from rhizome_perception import camera_inputs
root=sys.argv[2]
cfg=yaml.safe_load(open(root+'/config.yml'))
torch.set_num_threads(2)
model=ModelPipeline(root,cfg,'cpu',core_runtime='torch',post_runtime='torch')
reference=np.load(root+'/reference.npz')
with torch.inference_mode():
    outputs=model.core(*(torch.from_numpy(reference[k]) for k in ('image','depth','crop_vector')))
for name,output in zip(('seg_softmax','z','g','seg_logits'),outputs):
    np.testing.assert_allclose(output.numpy(),reference[name],atol=5e-4,rtol=5e-4)
rgb=np.zeros((2,4,3),np.uint8);rgb[:,:,0]=255
depth=np.array([[.1,.5,1.,np.nan],[0,-1,np.inf,7]],np.float32)
rgba,raw,K=camera_inputs(rgb,depth,np.array([[4,0,2],[0,2,1],[0,0,1]]),4,2)
assert list(rgba[0,0])==[255,0,0,255]
np.testing.assert_array_equal(raw,[[1000,5000,10000,0],[0,0,0,0]])
rgba,raw,K=camera_inputs(rgb,depth,np.array([[4,0,2],[0,2,1],[0,0,1]]),8,4)
np.testing.assert_array_equal(K,[[8,0,4],[0,4,2],[0,0,1]])
'''
    result=subprocess.run([neural_config['python'],'-c',code,neural_config['root'],neural_config['neural']['package'],
                           str(Path(__file__).resolve().parents[1]/'src/scene_pipeline')],capture_output=True,text=True,timeout=90)
    assert result.returncode==0,result.stdout+result.stderr


def test_neural_capture_has_no_oracle_path(bundle,neural_config,tmp_path,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Neural inference accessed oracle data')
    monkeypatch.setattr('scene_pipeline.rhizome.visible_detections',forbidden)
    monkeypatch.setattr('scene_pipeline.rhizome.visible_row_geometry',forbidden)
    monkeypatch.setattr('scene_pipeline.agriculture_drive.shadow_row_profile',forbidden)
    source,_=bundle;scene=tmp_path/'scene';compile_bundle(source,scene)
    output=tmp_path/'capture'
    report=run(scene,output,config=dict(seconds=.2,camera=dict(width=64,height=36),controller=neural_config),
               video=False,timeout=120)
    assert report['passed'],report['failure']
    assert report['rhizome']['oracle_detections']==0
    assert not (output/'rhizome/camera_input.npz').exists()
    messages=[json.loads(line) for line in (output/'rhizome/messages.jsonl').read_text().splitlines()]
    observations=[m for m in messages if m['op']=='infer']
    assert len(observations)==12
    for observation in observations:
        assert observation['output']['row_profile'] is not None
        assert observation['output']['prediction']['foliage']['present']
        assert observation['output']['tracks']['foliage']==observation['output']['prediction']['foliage']
        assert observation['output']['prediction']['camera_timestamp']['secs']==1
        assert observation['output']['tracks']['camera_timestamp']==observation['output']['prediction']['camera_timestamp']
        assert all(d['keypoint']['kp']['frame']=='odom' for d in observation['output']['tracks']['detections'])
    with h5py.File(output/'data.h5') as data:
        for i in range(1,5):
            group=data[f'rhizome_raptor_30_{i}']
            assert 'neural_detection_count' in group and 'oracle_detection_count' not in group
            assert len(group['time'])==3
    assert len(list(output.glob('*_neural.png')))==4


def test_missing_model_is_inspectable_failure(bundle,neural_config,tmp_path):
    source,_=bundle;scene=tmp_path/'scene';compile_bundle(source,scene)
    neural_config['neural']['package']=str(tmp_path/'absent')
    output=tmp_path/'failure'
    report=run(scene,output,config=dict(seconds=.01,camera=dict(width=64,height=36),controller=neural_config),video=False)
    assert not report['passed'] and 'FileNotFoundError' in report['failure']
    assert report['rhizome']['oracle_detections']==0
    assert (output/'index.html').exists()
