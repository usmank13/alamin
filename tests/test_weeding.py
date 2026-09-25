import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scene_pipeline.agriculture_drive import drive_config
from scene_pipeline.contracts import PipelineError,write_json
from scene_pipeline.weeding import evaluate,segment_box_distance,swept_clearance


def test_segment_box_distance_handles_crossing_parallel_and_corner():
    half=np.ones(3)
    assert segment_box_distance([-2,0,0],[2,0,0],half)==0
    assert segment_box_distance([2,-2,0],[2,2,0],half)==pytest.approx(1)
    assert segment_box_distance([2,2,2],[3,3,3],half)==pytest.approx(np.sqrt(3))
    assert segment_box_distance([0,0,0],[0,0,0],half)==0


def test_sweep_catches_translation_and_rotation_between_samples():
    plant=dict(position_m=[0,0,0],height_m=.1,radius_m=.005)
    half=np.array([.01,.01,.01]);eye=np.eye(3)
    assert swept_clearance(np.array([-.1,0,.05]),eye,np.array([.1,0,.05]),eye,half,plant)<0
    assert swept_clearance(np.array([-.1,.03,.05]),eye,np.array([.1,.03,.05]),eye,half,plant)>0
    # A long narrow blade rotates through the plant, missing at both endpoints.
    plant['position_m']=[.19,0,0]
    a=Rotation.from_euler('z',-45,degrees=True).as_matrix();b=Rotation.from_euler('z',45,degrees=True).as_matrix()
    assert swept_clearance(np.array([0,0,.05]),a,np.array([0,0,.05]),b,np.array([.2,.005,.01]),plant)<0
    # Crossing the XY projection above the plant is not a strike.
    assert swept_clearance(np.array([-.1,0,.4]),eye,np.array([.3,0,.4]),eye,half,plant)>0


def recording(root,*,selected=True,crop_y=.2,high=False,initial_contact=False):
    root.mkdir();(root/'rhizome').mkdir()
    arm=dict(id='arm_1',camera='cam',position_chassis_m=[0,0,0],quaternion_chassis_wxyz=[1,0,0,0])
    benchmark=dict(arm='arm_1',camera='cam',association_radius_m=.06,sweep_tolerance_m=.001,
        plants=[dict(id='weed',category='weed',position_m=[0,0,0],radius_m=.015,height_m=.04),
                dict(id='crop',category='crop',position_m=[0,crop_y,0],radius_m=.03,height_m=.15)])
    point=dict(x=0,y=0,z=0,frame='odom')
    detection=dict(class_label='WEED',keypoint=dict(kp=point),track_id=7)
    events=[dict(simulation_time=0,op='observe',input=dict(camera='cam',detections=[detection]),output=dict(tracks=dict(detections=[detection])))]
    for t in [0,.01,.02,.03]:
        locked=selected and t>=.01
        diag=dict(track_id=7,class_label='WEED',x=0,y=0,z=0,in_reach_circle=True,in_reach_triangle=True,
                  strike_ready=locked,is_locked_target=locked,is_valid_candidate=True,tool_protection_radius=0)
        events.append(dict(simulation_time=t,op='step',input=dict(position=[0,0,.5],quaternion=[1,0,0,0]),
            output=dict(arm=dict(state='LOCKED' if locked else 'SEARCHING',target_id=7 if locked else -1,
                                commanded=[0,.7],measured=[0,.6],diagnostics=[{**diag,'z':-.5}]))))
    (root/'rhizome/messages.jsonl').write_text('\n'.join(json.dumps(e) for e in sorted(events,key=lambda x:x['simulation_time']))+'\n')
    times=np.arange(16)*.002
    x=np.interp(times,[0,.01,.02,.03],[-.15,-.15,.15,.15])
    if initial_contact:x[:]=0
    with h5py.File(root/'data.h5','w') as f:
        g=f.create_group('weeding_tool');g['time']=times
        g['position_world']=np.column_stack([x,np.zeros(16),np.full(16,.4 if high else .025)])
        g['rotation_world']=np.tile(np.eye(3),(16,1,1));g['half_size_m']=np.tile([.02,.02,.02],(16,1))
    return {'weeding_benchmark':benchmark},arm


def test_ordered_acceptance_is_independent_of_planner_success(tmp_path):
    for mode,options in [('hit',{}),('miss',{'high':True}),('crop_contact',{'crop_y':0}),
                         ('incidental',{'selected':False}),('already_touching',{'initial_contact':True})]:
        root=tmp_path/mode;manifest,arm=recording(root,**options)
        result=evaluate(root,manifest,arm,simulation_passed=True)
        assert result['passed']==(mode=='hit')
        assert result['stages']['detection'] and result['stages']['tracking']
        if mode=='miss':assert result['plants']['weed']['min_clearance_m']>.1
        if mode=='crop_contact':assert not result['stages']['crop_avoided']
        if mode=='already_touching':assert result['plants']['weed']['first_contact_at']==0
        assert (root/'weeding.html').exists()
        assert '__WEEDING_DATA__' not in (root/'weeding.html').read_text()


def test_failed_simulation_cannot_pass_contact_acceptance(tmp_path):
    root=tmp_path/'run';manifest,arm=recording(root)
    assert not evaluate(root,manifest,arm,simulation_passed=False)['passed']


def test_selected_camera_capture_validation():
    assert drive_config({'camera':{'names':['cam']}})['camera']['names']==['cam']
    for names in [[],['cam','cam'],'cam',[None]]:
        with pytest.raises(PipelineError):drive_config({'camera':{'names':names}})


@pytest.mark.parametrize('failure',[None,'miss_second_weed','other_arms_crop'])
def test_multi_acceptance_requires_each_arm_and_protects_all_crops(tmp_path,failure):
    import copy
    from scene_pipeline.weeding import evaluate_multi
    root=tmp_path/'run';manifest,arm=recording(root)
    benchmark=manifest['weeding_benchmark'];benchmark.pop('arm');benchmark.pop('camera')
    benchmark['arms']=['arm_1','arm_2']
    for p in benchmark['plants']:p['arm']='arm_1'
    other=[]
    for p in benchmark['plants']:
        p=copy.deepcopy(p);p['id']+='2';p['arm']='arm_2';p['position_m'][1]+=1;other.append(p)
    if failure=='other_arms_crop':other[1]['position_m'][1]=0
    benchmark['plants']+=other
    first=root/'rhizome/arm_1';first.mkdir()
    messages=(root/'rhizome/messages.jsonl').read_text();(first/'messages.jsonl').write_text(messages)
    second=root/'rhizome/arm_2';second.mkdir();events=[]
    for line in messages.splitlines():
        e=json.loads(line)
        if e['op']=='step':e['input']['position'][1]+=1
        else:
            e['input']['camera']='cam2'
            for d in e['input']['detections']+e['output']['tracks']['detections']:d['keypoint']['kp']['y']+=1
        events.append(json.dumps(e))
    (second/'messages.jsonl').write_text('\n'.join(events)+'\n')
    with h5py.File(root/'data.h5','a') as f:
        f.copy('weeding_tool','weeding_tool_arm_1');f.copy('weeding_tool','weeding_tool_arm_2')
        positions=f['weeding_tool_arm_2/position_world'][:];positions[:,1]+=1
        if failure=='miss_second_weed':positions[:,2]+=.4
        f['weeding_tool_arm_2/position_world'][:]=positions
    result=evaluate_multi(root,manifest,dict(arm_1=arm,arm_2={**arm,'id':'arm_2','camera':'cam2'}),simulation_passed=True)
    assert result['passed']==(failure is None)
    if failure=='miss_second_weed':
        assert result['arms']['arm_1']['passed']
        assert not result['arms']['arm_2']['stages']['weed_contact']
    if failure=='other_arms_crop':assert not result['arms']['arm_1']['stages']['crop_avoided']
    assert 'arm_2_weeding.html' in (root/'weeding.html').read_text()
    assert 'arm_1_weeding_events.jsonl' in (root/'arm_1_weeding.html').read_text()


@pytest.mark.parametrize('options,ordered,crop_contact',[
    ({},True,False),({'high':True},False,False),({'selected':False},False,False),
    ({'initial_contact':True},False,False),({'crop_y':0},True,True)])
def test_field_diagnostics_preserve_order_and_independent_crop_contacts(tmp_path,options,ordered,crop_contact):
    from scene_pipeline.weeding_field import analyze
    root=tmp_path/'run';manifest,arm=recording(root,**options)
    plants=manifest['weeding_benchmark']['plants']
    # Many distant plants exercise broad-phase exclusion without changing contacts.
    plants=plants+[dict(id='distant_'+str(i),category='weed',position_m=[2+i,3,0]) for i in range(20)]
    write_json(root/'manifest.json',dict(instances={p['id']:p for p in plants}))
    write_json(root/'rig.json',dict(profile=dict(arms=[arm])))
    write_json(root/'drive_config.json',dict(controller=dict(perception='oracle_visible',physical_arm='arm_1')))
    result=analyze(root)
    assert bool(result['unique_ordered_weed_contacts'])==ordered
    assert bool(result['unique_contacted_crops'])==crop_contact
    assert not any(p.startswith('distant') for p in result['unique_contacted_weeds'])
    assert result['arms']['arm_1']['detected_weeds']==['weed']
    assert result['arms']['arm_1']['selected_weeds']==([] if options.get('selected') is False else ['weed'])
    assert (root/'field_weeding.html').exists()


def test_neural_field_diagnostics_transform_predictions_before_association(tmp_path):
    from scene_pipeline.weeding_field import analyze
    root=tmp_path/'run';manifest,arm=recording(root)
    events=[json.loads(line) for line in (root/'rhizome/messages.jsonl').read_text().splitlines()]
    observation=events[0];observation['op']='infer'
    prediction=json.loads(json.dumps(observation['input']['detections']))
    prediction[0]['keypoint']['kp']['z']=-.5
    observation['output']['prediction']=dict(detections=prediction)
    del observation['input']['detections']
    events.sort(key=lambda e:(e['simulation_time'],e['op']!='step'))
    (root/'rhizome/messages.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
    write_json(root/'manifest.json',dict(instances={p['id']:p for p in manifest['weeding_benchmark']['plants']}))
    write_json(root/'rig.json',dict(profile=dict(arms=[arm])))
    write_json(root/'drive_config.json',dict(controller=dict(perception='neural',physical_arm='arm_1')))
    result=analyze(root)
    assert result['unique_ordered_weed_contacts']==['weed']
    assert result['arms']['arm_1']['prediction_samples']==1
