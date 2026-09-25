"""Gen 3 CAD integration; local asset checks skip when the export is absent."""
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL','osmesa')

import mujoco
import numpy as np
import pytest

from scene_pipeline.agriculture import compile_environment, compose_robot
from scene_pipeline.agriculture_drive import arm_schedule, wheel_targets
from scene_pipeline.contracts import PipelineError
from scene_pipeline.gen3 import build_robot, camera_layout
from scene_pipeline.rhizome import RhizomeController, bind_physical_arm
from scene_pipeline.rhizome_perception import frame_transforms
from sim_harness.export import export_mujoco
from sim_harness.scene import load_scene
from test_agriculture import bundle
from test_rhizome import rhizome_root


@pytest.fixture(scope='module')
def robot():
    path=Path(os.environ.get('GEN3_URDF','/data/URDF EXPORT GEN 3.0'))
    if not path.exists():pytest.skip('Requires the local Gen 3 CAD export (GEN3_URDF)')
    return build_robot(path)


def composed(robot,bundle):
    root,record=bundle
    environment,_=compile_environment(record,root)
    spec,profile=robot
    return (*compose_robot(environment,spec,profile,{},np.asarray(record['terrain']['bounds_m'])),profile)


def test_assembly_symmetry_camera_pairs_and_portability(robot,bundle,tmp_path):
    spec,model,data,profile=composed(robot,bundle)
    assert model.nu==12 and model.nq==21
    assert all(model.joint('element/'+name).id>=0 for name in profile['source_joint_map'].values())
    assert len(profile['arms'])==4 and len(profile['suspension'])==2
    tips=np.array([data.site('element/'+a['tool_site']).xpos for a in profile['arms']])
    np.testing.assert_allclose(tips[:,[0,2]],np.tile(tips[0,[0,2]],(4,1)),atol=1e-6)
    np.testing.assert_allclose(np.diff(tips[:,1]),[-.3302,-.3048,-.3302],atol=1e-6)
    centers=np.array([w['center_m'] for w in profile['wheels']])
    assert np.ptp(centers[:,2])<1e-6
    np.testing.assert_allclose(np.abs(centers[:,1]),.837859,atol=2e-6)
    for arm in profile['arms']:
        assert profile['cameras'][arm['camera']]['position_m'][1]==pytest.approx(arm['position_chassis_m'][1])
    export_mujoco(spec,data,tmp_path/'portable.mjz')
    other,state=load_scene(tmp_path/'portable.mjz')
    np.testing.assert_allclose(state.qpos,data.qpos,atol=1e-6)
    np.testing.assert_allclose(state.ctrl,data.ctrl)
    assert other.nmesh==model.nmesh


@pytest.mark.parametrize('forward,yaw',[(.25,0),(-.25,0),(0,.4)])
def test_gen3_wheel_axes_drive_and_turn(robot,bundle,forward,yaw):
    _,model,data,profile=composed(robot,bundle)
    mujoco.mj_step(model,data,nstep=500)
    start=data.body('element/chassis').xpos.copy()
    targets,_=wheel_targets(profile,forward,yaw)
    for wheel,value in zip(profile['wheels'],targets):
        data.ctrl[model.actuator('element/'+wheel['id']).id]=value
    mujoco.mj_step(model,data,nstep=1000)
    if forward:
        assert (data.body('element/chassis').xpos[0]-start[0])*np.sign(forward)>.2
    else:
        r=data.body('element/chassis').xmat.reshape(3,3)
        assert np.arctan2(r[1,0],r[0,0])>.08
    assert not any(w.number for w in data.warning)


def test_all_arms_actuate_independently_and_rockers_are_passive(robot,bundle):
    _,model,data,profile=composed(robot,bundle)
    for index,arm in enumerate(profile['arms']):
        data.ctrl[model.actuator('element/'+arm['pitch_joint']).id]=-.2
        data.ctrl[model.actuator('element/'+arm['yaw_joint']).id]=(-1)**index*.2
    mujoco.mj_step(model,data,nstep=1000)
    for index,arm in enumerate(profile['arms']):
        assert data.joint('element/'+arm['pitch_joint']).qpos[0]==pytest.approx(-.2,abs=.04)
        assert data.joint('element/'+arm['yaw_joint']).qpos[0]==pytest.approx((-1)**index*.2,abs=.04)
    for rocker in profile['suspension']:
        jid=model.joint('element/'+rocker['joint']).id
        assert jid not in model.actuator_trnid[:,0]
        before=data.joint('element/'+rocker['joint']).qpos[0]
        dof=model.jnt_dofadr[jid]
        data.qfrc_applied[dof]=25
        mujoco.mj_step(model,data,nstep=100)
        assert abs(data.joint('element/'+rocker['joint']).qpos[0]-before)>.005
        data.qfrc_applied[dof]=0
    assert not any(w.number for w in data.warning)


def test_rearward_target_frame_and_presets(tmp_path):
    rotation=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    rc,tc,rt,tt=frame_transforms(rotation,[2,3,.5],rotation,[2,3,.9],[-.345,.48,-.45],[0,0,0,1])
    world=np.array([2,3,.9])+rotation@np.array([-.745,.38,-.65])
    np.testing.assert_allclose(rt@world+tt,[.4,.1,-.2],atol=1e-12)
    folder=tmp_path/'assets/robots';folder.mkdir(parents=True)
    record={'cameras':{f'raptor_22_{i}':{'loc_bu':[-1.8,y,-2.99],'euler_deg':[0,0,-90]} for i,y in enumerate([3.683,1.905,-1.905,-3.683])}}
    (folder/'RAPTOR_22.json').write_text(json.dumps(record))
    cameras,source=camera_layout('RAPTOR_22',tmp_path,[])
    assert source['sha256']
    np.testing.assert_allclose([c['position_m'][1] for c in cameras.values()],[.3683,.1905,-.1905,-.3683])


def test_native_arm_uses_actual_mujoco_feedback(robot,bundle,rhizome_root,tmp_path):
    _,model,data,profile=composed(robot,bundle)
    cfg,arm=bind_physical_arm(dict(root=rhizome_root,waypoints=[dict(xy=[0,0])],
                                  perception='oracle_visible',physical_arm='arm_1'),profile)
    controller=RhizomeController(cfg,tmp_path)
    commands=[];measured=[];targets=[]
    try:
        for tick in range(101):
            joints=['element/'+arm[key] for key in ('yaw_joint','pitch_joint')]
            feedback=-np.array([data.joint(name).qpos[0] for name in joints])
            result=controller.step(tick*.01,data.body('element/chassis').xpos,
                                   data.body('element/chassis').xquat,0,arm_measured=feedback)['arm']
            np.testing.assert_allclose(result['measured'],feedback,atol=1e-6)
            commands.append(result['commanded']);measured.append(feedback);targets.append(result['target_id'])
            for name,value in zip(joints,result['commanded']):data.ctrl[model.actuator(name).id]=-value
            if tick%10==0:
                point=dict(x=-.60,y=.4826,z=0,frame='odom')
                detections=[dict(class_label='WEED',score=1,keypoint=dict(kp=point,size=.02,vis=1),rect=dict(center=point,lx=.02,ly=.02))]
                controller.observe(tick*.01,arm['camera'],detections,[dict(x=0,y=y,z=-.47,frame='arm_shadow') for y in [-.3,0,.3]])
            mujoco.mj_step(model,data,nstep=5)
        assert np.ptp(np.array(measured)[:,1])>.05
        assert any(t>=0 for t in targets)
        assert not any(w.number for w in data.warning)
    finally:controller.close()


def test_invalid_arm_schedules_and_mounts(robot):
    _,profile=robot
    for commands in ({'absent':[]},{'arm_1':[]},{'arm_1':[dict(time=0,yaw_rad=2,pitch_rad=.5)]},
                     {'arm_1':[dict(time=.001,yaw_rad=0,pitch_rad=.5)]}):
        with pytest.raises(PipelineError):arm_schedule({'arm_commands':commands},profile)
    with pytest.raises(PipelineError):
        bind_physical_arm(dict(physical_arm='missing'),profile)


@pytest.mark.parametrize('custom',[False,True])
def test_multi_arm_native_feedback_and_camera_isolation(robot,bundle,rhizome_root,tmp_path,custom):
    from scene_pipeline.rhizome import RhizomeMultiController,bind_physical_arms
    if custom:
        layout=Path(__file__).parents[1]/'examples/agriculture/payloads_three.json'
        robot=build_robot(robot[1]['provenance']['geometry'],payloads=layout,mesh_faces=1000)
    _,model,data,profile=composed(robot,bundle)
    cfg,arms=bind_physical_arms(dict(root=rhizome_root,waypoints=[dict(xy=[0,0])],
        perception='oracle_visible',physical_arms=[a['id'] for a in profile['arms']],targeting_diagnostics=True),profile)
    controller=RhizomeMultiController(cfg,tmp_path,arms)
    targets={name:[] for name in arms}
    names=list(arms);active=names[::2];empty=names[1::2]
    try:
        assert [w.info['navigation_enabled'] for w in controller.workers.values()]==[True]+[False]*(len(arms)-1)
        for tick in range(101):
            feedback={name:np.asarray(a['motor_to_joint_sign'])*[data.joint('element/'+a[k]).qpos[0] for k in ('yaw_joint','pitch_joint')] for name,a in arms.items()}
            result=controller.step(tick*.01,data.body('element/chassis').xpos,data.body('element/chassis').xquat,0,feedback)
            for name,a in arms.items():
                native=result['arms'][name];targets[name].append(native['target_id'])
                np.testing.assert_allclose(native['measured'],feedback[name],atol=1e-6)
                for axis,key in enumerate(('yaw_joint','pitch_joint')):
                    data.ctrl[model.actuator('element/'+a[key]).id]=native['commanded'][axis]*a['motor_to_joint_sign'][axis]
                if tick%10==0:
                    # Only two cameras see a weed. A broadcast/routing leak activates the other arms.
                    point=dict(x=-.60,y=a['position_chassis_m'][1],z=0,frame='odom')
                    detections=[] if name in empty else [dict(class_label='WEED',score=1,
                        keypoint=dict(kp=point,size=.02,vis=1),rect=dict(center=point,lx=.02,ly=.02))]
                    tracks=controller.observe(tick*.01,a['camera'],detections,
                        [dict(x=0,y=y,z=-.47,frame='arm_shadow') for y in [-.3,0,.3]])['tracks']['detections']
                    for d in tracks:assert d['keypoint']['kp']['y']==pytest.approx(point['y'],abs=1e-5)
            mujoco.mj_step(model,data,nstep=5)
        assert all(any(t>=0 for t in targets[name]) for name in active)
        assert all(set(targets[name])=={-1} for name in empty)
        assert not any(w.number for w in data.warning)
        # A worker death must close all remaining native processes.
        controller.workers[names[1]].process.kill();controller.workers[names[1]].process.wait()
        with pytest.raises(PipelineError,match=names[1]):
            controller.step(1.01,data.body('element/chassis').xpos,data.body('element/chassis').xquat,0,feedback)
        assert all(w.process.poll() is not None for w in controller.workers.values())
    finally:controller.close()


def test_multi_arm_binding_rejects_invalid_or_ambiguous_ownership(robot):
    from scene_pipeline.rhizome import bind_physical_arms,validate_config
    import copy
    _,profile=robot
    base=dict(root='/tmp/rhizome',waypoints=[dict(xy=[0,0])],perception='oracle_visible',physical_arms=['arm_1','arm_2'])
    for change in ({'physical_arms':[]},{'physical_arms':['arm_1','arm_1']},{'physical_arm':'arm_1'},
                   {'shadow_arm':{}},{'perception':'none'},{'physical_arms':['missing']}):
        with pytest.raises(PipelineError):bind_physical_arms({**base,**change},profile)
    duplicate=copy.deepcopy(profile);duplicate['arms'][1]['camera']=duplicate['arms'][0]['camera']
    with pytest.raises(PipelineError,match='distinct camera'):bind_physical_arms(base,duplicate)
    config,arms=bind_physical_arms({**base,'perception':'neural','neural':dict(package='/tmp/model',crop_species='SOYBEAN')},profile)
    assert set(arms)=={'arm_1','arm_2'}
    assert config['neural']['target_positions_chassis_m']['gen3_2']==arms['arm_2']['position_chassis_m']
    assert validate_config(config)==config


@pytest.mark.parametrize('layout_file',['payloads_three.json','payloads_four_offset.json'])
def test_configurable_payload_assembly_frames_and_roundtrip(robot,layout_file,tmp_path):
    from scene_pipeline.gen3 import initialize
    from scene_pipeline.rhizome import bind_physical_arms
    _,baseline=robot
    config=json.loads((Path(__file__).parents[1]/'examples/agriculture'/layout_file).read_text())
    spec,profile=build_robot(baseline['provenance']['geometry'],payloads=config,mesh_faces=1000)
    model=spec.compile();data=mujoco.MjData(model);initialize(model,data,profile)
    count=len(config['payloads'])
    assert model.nu==4+2*count and model.nq==13+2*count
    assert len(profile['arms'])==len(profile['cameras'])==count
    assert profile['payload_configuration']==config
    baseline_model=robot[0].compile();baseline_data=mujoco.MjData(baseline_model);initialize(baseline_model,baseline_data,baseline)
    base=baseline['arms'][0]
    for arm,placement in zip(profile['arms'],config['payloads']):
        assert arm['id']==placement['id']
        delta=np.array([0,placement['lateral_m']-base['position_chassis_m'][1],0])
        np.testing.assert_allclose(arm['position_chassis_m'],np.asarray(base['position_chassis_m'])+delta)
        np.testing.assert_allclose(data.body(arm['yaw_joint']).xpos,baseline_data.body(base['yaw_joint']).xpos+delta,atol=1e-8)
        np.testing.assert_allclose(data.site(arm['tool_site']).xpos,baseline_data.site(base['tool_site']).xpos+delta,atol=1e-8)
        np.testing.assert_allclose(data.cam(arm['camera']).xpos,baseline_data.cam(base['camera']).xpos+delta,atol=1e-8)
        cfg,bound=bind_physical_arms(
            dict(root='/tmp/rhizome',waypoints=[dict(xy=[0,0])],physical_arms=[a['id'] for a in profile['arms']],
                 perception='neural',neural=dict(package='/tmp/model',crop_species='SOYBEAN')),profile)
        assert cfg['neural']['target_positions_chassis_m'][arm['camera']]==arm['position_chassis_m']
    # Removed arms have no residual joints, actuators or cameras in the model.
    assert {model.camera(i).name for i in range(model.ncam)}==set(profile['cameras'])
    assert {model.actuator(i).name for i in range(model.nu)}=={w['id'] for w in profile['wheels']}|{a[k] for a in profile['arms'] for k in ('yaw_joint','pitch_joint')}
    export_mujoco(spec,data,tmp_path/'configured.mjz')
    reloaded,state=load_scene(tmp_path/'configured.mjz')
    assert reloaded.nu==model.nu and reloaded.ncam==count
    np.testing.assert_allclose(state.qpos,data.qpos,atol=1e-6)
    np.testing.assert_allclose(state.ctrl,data.ctrl)


def test_payload_config_rejects_ambiguous_and_nonfinite_layouts():
    from scene_pipeline.gen3 import payload_configuration
    valid=dict(payloads=[dict(id='arm_left',lateral_m=.4)])
    assert payload_configuration(valid)==valid
    for config in ([],{},dict(payloads=None),dict(payloads=[{}]),
                   dict(payloads=[dict(id='left',lateral_m=0)]),
                   dict(payloads=[dict(id='arm_left',lateral_m=float('nan'))]),
                   dict(payloads=[dict(id='arm_left',lateral_m=True)]),
                   dict(payloads=[dict(id='arm_left',lateral_m=0,enabled=False)]),
                   dict(payloads=valid['payloads']*2)):
        with pytest.raises(PipelineError):payload_configuration(config)


@pytest.mark.parametrize('count',[0,5])
def test_payload_count_is_not_tied_to_export_count(robot,count):
    config=dict(payloads=[dict(id=f'arm_{i+1}',lateral_m=.6-i*.3) for i in range(count)])
    spec,profile=build_robot(robot[1]['provenance']['geometry'],payloads=config,mesh_faces=1000)
    model=spec.compile()
    assert len(profile['arms'])==model.ncam==count
    assert model.nu==4+2*count and model.nq==13+2*count
    assert set(profile['initial_controls'])=={a[k] for a in profile['arms'] for k in ('yaw_joint','pitch_joint')}
