"""Native integration checks skip only when the optional checkout is absent."""
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from scene_pipeline.agriculture import compile_bundle
from scene_pipeline.agriculture_drive import drive_config, run
from scene_pipeline.contracts import PipelineError
from scene_pipeline.rhizome import RhizomeController, validate_config, visible_detections, visible_row_geometry
from test_agriculture import bundle


@pytest.fixture
def rhizome_root():
    root=Path(os.environ.get('RHIZOME_ROOT',Path(__file__).resolve().parents[2]/'rhizome'))
    if not (root/'build/host/py/pyzome').exists():
        pytest.skip('Needs RHIZOME_ROOT with a Python 3.12 host build')
    pytest.importorskip('yaml')
    return str(root)


def test_config_rejects_ambiguous_and_unimplemented_modes():
    cfg=dict(root='/tmp/rhizome',waypoints=[dict(xy=[1,0])])
    for changes in ({'waypoints':[]},{'perception':'neural'}, {'parameters':{'goals_stale_sec':0}},
                    {'waypoints':[dict(xy=[float('nan'),0])]}, {'shadow_arm':{}},
                    {'parameters':{'auto_speed_max_ms':-1}}):
        with pytest.raises(PipelineError):validate_config({**cfg,**changes})
    with pytest.raises(PipelineError):drive_config(dict(controller=cfg,commands=[]))
    normalized=drive_config(dict(controller=cfg))
    assert drive_config(normalized)==normalized


def test_oracle_only_reports_visible_plants():
    manifest={'instances':{'weed':dict(category='weed',position_m=[1,2,3]),
                           'crop':dict(category='crop',position_m=[2,2,3]),
                           'rock':dict(category='rock',position_m=[3,2,3])}}
    result=visible_detections(np.array([[0,1,3]]),{'weed':1,'crop':2,'rock':3},manifest)
    assert len(result)==1
    assert result[0]['class_label']=='WEED'
    assert result[0]['keypoint']['kp']==dict(x=1,y=2,z=3,frame='odom')


@pytest.mark.parametrize('angle',[0,.7])
def test_oracle_keeps_interleaved_rows_separate_and_ids_stable(angle):
    rotation=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    def xyz(x,y):return [*(rotation@np.array([x,y])),.03]
    manifest=dict(instances={},rows=[]);ids={}
    for row,y in enumerate([-.381,.381]):
        name=f'row_{row}'
        manifest['rows'].append(dict(id=name,endpoints_m=[xyz(-1,y)[:2],xyz(1,y)[:2]]))
        for col,x in enumerate([-.1,.02,.15]):
            plant=f'{row}_{col}';ids[plant]=len(ids)+1
            manifest['instances'][plant]=dict(category='crop',row_id=name,position_m=xyz(x,y))
    snapshot=visible_row_geometry(np.array([list(ids.values())]),ids,manifest)
    assert snapshot['present'] and len(snapshot['rows'])==2
    for row,y in zip(snapshot['rows'],[-.381,.381]):
        for key in ['anchor','begin','end']:
            p=row[key];assert p['frame']=='odom'
            assert (rotation.T@np.array([p['x'],p['y']]))[1]==pytest.approx(y)
        np.testing.assert_allclose([row['direction']['x'],row['direction']['y']],rotation[:,0])
    visible=np.array([[ids['1_0'],ids['1_2']]])
    assert [r['id'] for r in visible_row_geometry(visible,ids,manifest)['rows']]==[1]
    assert visible_row_geometry(np.zeros((1,1)),ids,manifest)['rows']==[]


@pytest.mark.parametrize('goal,direction,expected', [([1,0],'FORWARD',1),([-1,0],'REVERSE',-1)])
def test_native_controller_uses_feedback_and_stops(rhizome_root,tmp_path,goal,direction,expected):
    cfg=dict(root=rhizome_root,waypoints=[dict(xy=goal,direction=direction)])
    controller=RhizomeController(cfg,tmp_path)
    try:
        assert controller.step(0,[0,0,.84],[1,0,0,0],0)['twist'][0]*expected>0
        for tick in range(1,11):result=controller.step(tick*.01,[*goal,.84],[1,0,0,0],0)
        assert result['twist']==[0,0]
        assert controller.info['modules_sha256']
    finally:controller.close()
    assert controller.process.poll() is not None


@pytest.mark.parametrize('yaw',[0,np.pi/2])
def test_native_tracker_and_shadow_planner_select_weed(rhizome_root,tmp_path,yaw):
    cfg=dict(root=rhizome_root,waypoints=[dict(xy=[1,0])],perception='oracle_visible',
             shadow_arm=dict(camera='raptor_30_1',position_chassis_m=[-.18,.622,-.3]))
    controller=RhizomeController(cfg,tmp_path)
    ids=[];targets=[];pitch=[]
    try:
        for tick in range(111):
            t=tick*.01
            result=controller.step(t,[tick*.002*np.cos(yaw),tick*.002*np.sin(yaw),.84],
                                   [np.cos(yaw/2),0,0,np.sin(yaw/2)],.2)
            targets.append(result['arm']['target_id']);pitch.append(result['arm']['commanded'][1])
            if tick%10==0:
                point=dict(x=.2*np.cos(yaw)-.62*np.sin(yaw),y=.2*np.sin(yaw)+.62*np.cos(yaw),z=0,frame='odom')
                detections=[dict(class_label='WEED',score=1,keypoint=dict(kp=point,size=.02,vis=1),
                                 rect=dict(center=point,lx=.02,ly=.02))]
                row=[dict(x=0,y=y,z=-.54,frame='arm_shadow') for y in (-.3,0,.3)]
                tracks=controller.observe(t,'raptor_30_1',detections,row)['tracks']['detections']
                ids.extend(d['track_id'] for d in tracks)
                for detection in tracks:
                    assert detection['keypoint']['kp']['x']==pytest.approx(point['x'],abs=1e-5)
                    assert detection['keypoint']['kp']['frame']=='odom'
        assert len(ids)>4 and len(set(ids))==1
        assert ids[0] in targets
        assert max(pitch)>.1
    finally:controller.close()


@pytest.mark.parametrize('weed_y,should_strike',[(0,True),(.38,False)])
def test_native_separate_rows_allow_lane_strike_but_protect_crop(rhizome_root,tmp_path,weed_y,should_strike):
    manifest=dict(instances={'weed':dict(category='weed',position_m=[.2,weed_y,0])},rows=[])
    for number,y in enumerate([-.381,.381]):
        name=f'row_{number}'
        manifest['rows'].append(dict(id=name,endpoints_m=[[-1,y],[1,y]]))
        for i,x in enumerate([-.2,0,.2,.4,.6]):
            manifest['instances'][f'{name}_{i}']=dict(category='crop',row_id=name,position_m=[x,y,0])
    ids={name:i+1 for i,name in enumerate(manifest['instances'])}
    image=np.array([list(ids.values())])
    detections=visible_detections(image,ids,manifest)
    foliage=visible_row_geometry(image,ids,manifest)
    cfg=dict(root=rhizome_root,waypoints=[dict(xy=[1,0])],perception='oracle_visible',
             targeting_diagnostics=True,shadow_arm=dict(camera='cam',position_chassis_m=[-.18,0,-.3]))
    controller=RhizomeController(cfg,tmp_path);states=[]
    try:
        for tick in range(301):
            t=tick*.01
            result=controller.step(t,[t*.2,0,.84],[1,0,0,0],.2)
            states.append(result['arm']['state'])
            if tick%10==0:
                profile=[dict(x=0,y=y,z=-.54,frame='arm_shadow') for y in (-.5,0,.5)]
                tracked=controller.observe(t,'cam',detections,profile,foliage)['tracks']
                assert tracked['foliage']['rows']==foliage['rows']
                if tick>=20:
                    assert sum(d['class_label']=='CROP' for d in tracked['detections'])==10
            if tick>0:
                geometry=result['arm']['foliage']
                assert geometry['usable'] and geometry['row_boundaries']==2
        assert ('TargetingState.STRIKING' in states)==should_strike
        # A stalled camera must expire geometry on simulated time, not host time.
        for tick in range(301,2501):
            expired=controller.step(tick*.01,[tick*.002,0,.84],[1,0,0,0],.2)
        assert not expired['arm']['foliage']['usable']
    finally:controller.close()


def test_native_rollout_moves_mujoco_and_records_stop(bundle,rhizome_root,tmp_path):
    source,_=bundle;scene=tmp_path/'scene';compile_bundle(source,scene)
    cfg=dict(seconds=5,controller=dict(root=rhizome_root,waypoints=[dict(xy=[.65,0])]))
    output=tmp_path/'rollout'
    report=run(scene,output,config=cfg,tier='state',video=False)
    assert report['passed'] and report['rhizome']['goal_reached']
    assert report['displacement_m']>.3
    with h5py.File(output/'data.h5') as data:
        assert np.max(data['commands/requested_twist'][:,0])>0
        assert np.allclose(data['commands/requested_twist'][-1],[0,0])
        assert len(data['state/time'])==501
    assert (output/'rhizome/messages.jsonl').exists()


def test_worker_exit_is_failure_with_inspectable_capture(bundle,rhizome_root,tmp_path,monkeypatch):
    source,_=bundle;scene=tmp_path/'scene';compile_bundle(source,scene)
    original=RhizomeController.step
    def terminate(self,*args):
        self.process.kill();self.process.wait()
        return original(self,*args)
    monkeypatch.setattr(RhizomeController,'step',terminate)
    output=tmp_path/'failed'
    report=run(scene,output,config=dict(seconds=.2,controller=dict(root=rhizome_root,waypoints=[dict(xy=[1,0])])),tier='state',video=False)
    assert not report['passed'] and 'RHIZOME_WORKER' in report['failure']
    assert report['simulated_seconds']==0
    assert (output/'index.html').exists()
    assert json.loads((output/'report.json').read_text())['passed'] is False


def test_detection_expiry_uses_simulation_time_not_inference_wall_time(rhizome_root,tmp_path):
    import time
    import yaml
    # Keep native binaries/configs, shortening only the registry TTL to make
    # a real wall-clock delay cheap to test. No simulator-specific TTL override.
    original=Path(rhizome_root);root=tmp_path/'rhizome_source';(root/'src/physics').mkdir(parents=True)
    for name in ('build','test'):(root/name).symlink_to(original/name,target_is_directory=True)
    for name in ('nav','vision','base_control'):(root/'src'/name).symlink_to(original/'src'/name,target_is_directory=True)
    params=list(yaml.safe_load_all((original/'src/physics/physics_0.node.yml').read_text()))
    params[1]['environment']['detection_timeout']=.3
    (root/'src/physics/physics_0.node.yml').write_text(yaml.safe_dump_all(params))
    output=tmp_path/'run';output.mkdir()
    controller=RhizomeController(dict(root=str(root),waypoints=[dict(xy=[1,0])],perception='oracle_visible',
        shadow_arm=dict(camera='cam',position_chassis_m=[-.18,.622,-.3])),output)
    point=dict(x=.5,y=.622,z=0,frame='odom')
    detections=[dict(class_label='WEED',score=1,keypoint=dict(kp=point,size=.02,vis=1),rect=dict(center=point,lx=.02,ly=.02))]
    try:
        for tick in range(21):
            result=controller.step(tick*.01,[0,0,.84],[1,0,0,0],0)
            if tick%10==0:controller.observe(tick*.01,'cam',detections,[])
        time.sleep(.4)  # Longer than the configured TTL, without advancing simulation.
        assert controller.step(.21,[0,0,.84],[1,0,0,0],0)['arm']['detections']==1
        for tick in range(22,52):result=controller.step(tick*.01,[0,0,.84],[1,0,0,0],0)
        assert result['arm']['detections']==0
    finally:controller.close()


def test_multi_worker_startup_failure_closes_started_workers(monkeypatch,tmp_path):
    from scene_pipeline import rhizome
    created=[]
    class Worker:
        def __init__(self,*args,worker_name,**kwargs):
            if worker_name=='arm_2':raise PipelineError('RHIZOME_WORKER','startup failed')
            self.closed=False;created.append(self)
        def close(self):self.closed=True
    monkeypatch.setattr(rhizome,'RhizomeController',Worker)
    arms={name:dict(camera='camera_'+name,position_chassis_m=[0,0,0],quaternion_chassis_wxyz=[1,0,0,0],
                   motor_limits_rad=[[-.5,.5],[0,1]]) for name in ['arm_1','arm_2']}
    with pytest.raises(PipelineError,match='startup failed'):
        rhizome.RhizomeMultiController(dict(physical_arms=list(arms)),tmp_path,arms)
    assert len(created)==1 and created[0].closed
