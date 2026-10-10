import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest

from scene_pipeline.contracts import PipelineError
from scene_pipeline.navigation import raw_depth,robot_dimensions
from scene_pipeline.rhizome import RhizomeController,validate_config
from test_navigation import reference_inputs


def test_depth_units_and_invalid_returns():
    result=raw_depth(np.array([[0,.5,1.2345,6.5535,7.,np.inf,np.nan,-1]]))
    assert result.dtype==np.uint16
    assert result.tolist()==[[0,5000,12345,65535,0,0,0,0]]


def test_obstacle_config_requires_depth_and_booleans():
    base=dict(root='/tmp/rhizome',nav_model=dict(package='/tmp/model',obstacle_detection=True))
    with pytest.raises(PipelineError,match='requires'):validate_config(base)
    for key in ('depth','obstacle_detection'):
        with pytest.raises(PipelineError,match='boolean'):
            validate_config({**base,'nav_model':{**base['nav_model'],'depth':True,key:'true'}})


def test_envelope_uses_installed_robot_geometry():
    geometry=robot_dimensions(dict(footprint_xy_m=[[-1,-1.2],[.8,1.1]],track_width_m=1.7,
        wheels=[dict(center_m=[x,0,0],radius_m=.2) for x in (-.4,.4)]))
    assert geometry==dict(robot=dict(length=2.,width=2.4),drive=dict(trackwidth=1.7,wheelbase=.8),wheel=dict(radius=.2))


@pytest.fixture
def obstacle_config(tmp_path):
    repo=Path(__file__).resolve().parents[1]
    root=Path(os.environ.get('RHIZOME_OBSTACLE_ROOT',repo.parent/'rhizome-nav-grid'))
    host=Path(os.environ.get('RHIZOME_OBSTACLE_BUILD',repo/'outputs/rhizome_host_obstacle/host'))
    python=repo.parent/'ml-aigen-tools/venv312/bin/python'
    if not (host/'build_manifest.json').exists() or not (root/'src/nav_model/nav_obstacle.py').exists():
        pytest.skip('Requires the optional obstacle-grid Rhizome checkout and matching host build')
    stub=tmp_path/'stub'
    code="import sys;sys.path[:0]=[sys.argv[1]+'/src',sys.argv[1]+'/test'];from nav_model_stub import write_stub_package;write_stub_package(sys.argv[2],with_selftest=True)"
    result=subprocess.run([str(python),'-c',code,str(root),str(stub)],capture_output=True,text=True,timeout=60,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stderr
    return dict(root=str(root),host_build=str(host),python=str(python),nav_model=dict(package=str(stub),
        depth=True,obstacle_detection=True,cameras=['front']))


@pytest.mark.parametrize('enabled',[True,False])
def test_native_obstacle_debounce_stop_and_resume(obstacle_config,tmp_path,enabled):
    obstacle_config['nav_model']['obstacle_detection']=enabled
    arrays=reference_inputs(obstacle_config['nav_model']['package'])
    arrays['front_depth']=np.zeros((480,640),np.uint16)
    controller=RhizomeController(obstacle_config,tmp_path/'run',timeout=120)
    tick=0
    def capture():
        nonlocal tick
        for _ in range(10):
            step=controller.step(tick*.01,[0,0,.9],[1,0,0,0],0)
            tick+=1
        return controller.nav_infer((tick-1)*.01,**arrays),step
    try:
        for _ in range(3):observation,step=capture()
        assert observation['driving'] and step['twist'][0]>0
        arrays['front_depth'][:]=20000
        for i in range(4):
            observation,step=capture()
            assert observation['prediction']['grid_stop_fwd']
            assert observation['prediction']['obstacle_fwd']==(enabled and i==3)
        if enabled:
            assert observation['prediction']['gate']=='OBSTACLE'
            assert observation['goals']['waypoints']==[]
            assert observation['obstacle']['is_obstacle_detected_front']
            assert observation['obstacle_health']['invariant_failures']==['OBSTACLE_FRONT']
            assert any(r['service']=='create-snapshot' for r in observation['service_requests'])
            assert controller.step(tick*.01,[0,0,.9],[1,0,0,0],0)['twist']==[0,0]
            tick+=1
        else:
            assert observation['driving'] and step['twist'][0]>0
        arrays['front_depth'][:]=0
        for i in range(11):
            observation,step=capture()
            assert observation['prediction']['obstacle_fwd']==(enabled and i<10)
        assert observation['driving']
        observation,step=capture()
        assert step['twist'][0]>0
    finally:controller.close()


def test_capture_snapshot_matches_current_pose(obstacle_config,tmp_path):
    # Exercise the native time-buffered graph with nonzero translation/height
    # and changing roll/pitch: identity poses conceal a one-frame-late lookup.
    cfg=validate_config(obstacle_config)
    cfgfile=tmp_path/'config.json';cfgfile.write_text(json.dumps(cfg))
    code='''
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,sys.argv[1])
from rhizome_worker import Runtime
r=Runtime(json.loads(Path(sys.argv[2]).read_text()),Path(sys.argv[3]))
from util.transform_graph import extract_rt
mountR=Rotation.from_euler('xyz',[-1.952,0,-1.57]).as_matrix()
mountT=np.array([.40949,0,-.089478])
for tick in range(11):
    R=Rotation.from_euler('xyz',[.01*tick,-.005*tick,.03*tick]).as_matrix()
    q=Rotation.from_matrix(R).as_quat()[[3,0,1,2]]
    xyz=np.array([3+.01*tick,-2.,.9])
    r.step(dict(time=tick*.01,position=xyz.tolist(),quaternion=q.tolist(),forward_mps=0))
    if tick%10:continue
    r.navigation.infer(tick*.01,dict(front_rgb=np.zeros((480,640,3),np.uint8),
        front_depth=np.zeros((480,640),np.uint16),front_K=np.array([[400,0,320],[0,400,240],[0,0,1]]),
        front_rotation_robot=mountR,front_position_robot=mountT,front_height_m=np.array(.8144)))
    views,_=r.navigation.node._views(1+tick*.01)
    assert len(views)==1
    np.testing.assert_allclose(views[0].T_cam_odom,xyz+R@mountT,atol=1e-6)
    np.testing.assert_allclose(views[0].R_cam_odom,R@mountR,atol=1e-6)
'''
    result=subprocess.run([cfg['python'],'-c',code,str(Path(__file__).resolve().parents[1]/'src/scene_pipeline'),
        str(cfgfile),str(tmp_path/'worker')],capture_output=True,text=True,timeout=120,
        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'})
    assert result.returncode==0,result.stdout+result.stderr
