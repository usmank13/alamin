"""Behavior-level regressions for generation, adaptation and independent export."""
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
pytest.importorskip('jsonschema')
pytest.importorskip('h5py')
import mujoco

from scene_pipeline.contracts import PipelineError,digest,write_json
from scene_pipeline.dsl import parse,load
from scene_pipeline.evidence import measurement
from scene_pipeline.assets import template,scale_native,scale_for_constraints,instantiate
from scene_pipeline.layout import solve
from scene_pipeline.compiler import compile_scene
from scene_pipeline.validation import probe,validate_scene,geometric_overlap


@pytest.fixture
def generated(tmp_path):
    program=load('examples/cafe_program.py')
    ir=solve(program,1,tmp_path)
    write_json(tmp_path/'program.json',program);write_json(tmp_path/'ir.json',ir)
    compile_scene(ir,tmp_path)
    return tmp_path


def test_source_field_binding():
    source='<p>Product ABC</p><table><tr><td>Product Width</td><td>12.12 in</td></tr><tr><td>Packaging Width</td><td>20 in</td></tr></table>'
    result=measurement(source,identity='ABC',label='Product Width',number='12.12',unit='in',url='https://example.test')
    assert result['value_m']==pytest.approx(.307848)
    for label,num,unit in [('Product Width','20','in'),('Product Width','12.12','cm')]:
        with pytest.raises(PipelineError):measurement(source,identity='ABC',label=label,number=num,unit=unit,url='https://example.test')


@pytest.mark.parametrize('text',['__import__("os").system("echo unsafe")','[x for x in range(10)]','open("/tmp/should-not-exist","w")'])
def test_dsl_never_executes_python(text):
    with pytest.raises(PipelineError):parse(text)


def test_uniform_scale_constraints():
    assert scale_for_constraints([1,2,3],{'width':(.9,1.1),'depth':(1.8,2.2)})==1
    with pytest.raises(PipelineError):scale_for_constraints([1,2,3],{'width':(1,1.1),'depth':(3,4)})


def test_scalar_scaling_and_inertia():
    spec,_,_=template('drawer',[.6,.6,.9]);m=spec.compile()
    s=scale_native(spec,.75);n=s.compile()
    np.testing.assert_allclose(n.body_mass,m.body_mass*.75**3)
    np.testing.assert_allclose(n.body_inertia,m.body_inertia*.75**5,atol=1e-12)
    np.testing.assert_allclose(n.jnt_range,m.jnt_range*.75)
    np.testing.assert_array_equal(n.geom_contype,m.geom_contype)


def test_source_required_not_silently_dropped(tmp_path):
    p=load('examples/cafe_program.py');p['objects'][0]['category']='unknown_appliance'
    with pytest.raises(PipelineError,match='approved route'):solve(p,1,tmp_path)


def test_large_fixtures_are_placed_before_small_fillers(tmp_path):
    p=parse('scene("A workroom",space(area_m2=30),[place("small","base_cabinet"),place("large","prep_table")])')
    ir=solve(p,23,tmp_path)
    assert ir['objects'][0]['id']=='large_0'


def test_layout_repeatable(tmp_path):
    p=load('examples/cafe_program.py');a=solve(p,7,tmp_path/'a');b=solve(p,7,tmp_path/'b')
    assert digest(a)==digest(b)
    assert a['meta']['origin']=='entrance threshold'


def test_scene_settle_and_all_scalar_sweeps(generated):
    r=validate_scene(generated)
    assert r['passed'],r
    assert len(r['sweeps'])==5  # Regression: NumPy scalar/pybind enum tuple membership.
    assert r['articulated_objects']==5
    assert all(x['method'].startswith('21') for x in r['sweeps'])


def test_mask_independent_overlap_detection():
    m=mujoco.MjModel.from_xml_string('''<mujoco><worldbody><geom name="fixed" type="box" size=".1 .1 .1"/>
      <body><joint type="slide" axis="1 0 0"/><geom name="moving" type="box" pos=".15 0 0" size=".1 .1 .1" contype="2" conaffinity="2"/></body>
      </worldbody></mujoco>''')
    d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    assert len(d.contact)==0
    amount,pair=geometric_overlap(m,d)
    assert amount>.04 and pair==['fixed','moving']


def test_export_load_after_relocation(generated,tmp_path):
    pytest.importorskip('pybullet')
    from scene_pipeline.portability import export_urdf,verify_urdf
    path=export_urdf(generated)
    moved=tmp_path/'relocated';shutil.copytree(path,moved)
    # Prove dependencies resolve without referring back to the export directory.
    path.rename(tmp_path/'original_hidden')
    report=verify_urdf(moved)
    assert report['passed'] and len(report['joints'])==5


def test_dataset_clock_and_loader(tmp_path):
    from scene_pipeline.dataset import Recorder,inspect,load_stream
    p=tmp_path/'record.h5';r=Recorder(p,{'noise':'declared'})
    for tick in (0,5,10):r.append('state',tick*.002,qpos=np.array([1.,2.]))
    r.close();assert inspect(p)['state']['samples']==3
    stream=load_stream(p,'state',1,3)
    np.testing.assert_allclose(stream['time'],[.01,.02])
    assert stream['qpos'].shape==(2,2)


def test_unnamed_collision_geometry_ownership():
    from scene_pipeline.flows import belongs_to
    m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="arm/link"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    assert m.geom(0).name==''
    assert belongs_to(m,0,'arm/')
    assert not belongs_to(m,0,'drawer/')


def test_initial_closed_state_does_not_prove_return():
    from scene_pipeline.flows import DrawerController
    controller=DrawerController.__new__(DrawerController)
    controller.target={'id':'drawer'};controller.affordance={'joint':'slide'}
    controller.range=np.array([0.,.39]);controller.min=0.;controller.max=.39
    controller.contacts=100;controller.completed=True;controller.final_q=.008
    assert not controller.report()['passed']
    controller.final_q=.001
    assert controller.report()['passed']


def test_semantic_snapshot_and_fixed_taxonomy(generated):
    from scene_pipeline.semantics import snapshot
    from scene_pipeline.registry import CLASS_IDS
    m=mujoco.MjSpec.from_zip(str(generated/'scene.mjz')).compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    manifest=json.loads((generated/'manifest.json').read_text());s=snapshot(m,d,manifest)
    assert set(s['objects'])==set(manifest['instances'])
    for entry in s['objects'].values():
        assert entry['class_id']==CLASS_IDS[entry['category']]
        assert np.array(entry['bbox']).shape==(2,3)


def test_capture_tier_does_not_change_observations(generated,tmp_path):
    if not Path('vendor/mujoco_menagerie').exists():pytest.skip('Stock robot checkout required')
    import subprocess
    import sys
    from scene_pipeline.dataset import load_stream
    # Backend selection must precede MuJoCo import, as in the public CLI.
    for tier in ('state','full'):
        result=subprocess.run([sys.executable,'-m','scene_pipeline.cli','run',str(generated),'--flow','mapping',
                               '--output',str(tmp_path/tier),'--seconds','.1','--tier',tier,'--seed','42'],capture_output=True,text=True)
        assert result.returncode in (0,1),result.stderr
    for stream in ('state','imu','range','odometry'):
        a=load_stream(tmp_path/'state/data.h5',stream);b=load_stream(tmp_path/'full/data.h5',stream)
        for key in a:np.testing.assert_array_equal(a[key],b[key])


def test_artifact_collision_refused(tmp_path):
    from scene_pipeline.orchestrator import generate
    with pytest.raises(PipelineError,match='overwrite'):generate('x',0,tmp_path,program=load('examples/cafe_program.py'))


def test_robot_native_sensor_rig(generated):
    if not Path('vendor/mujoco_menagerie').exists():pytest.skip('Stock robot checkout required')
    from scene_pipeline.flows import robot_spec
    _,m,d,_,rig,_=robot_spec(generated,'mapping')
    assert m.nsensor==74
    assert d.sensor('base/rig_ray_0').data.shape==(1,)
    assert rig['rates_hz']['range']==20
    assert rig['camera']['K'][0][0]>0


def test_semantic_mask_is_not_geom_id(generated):
    from scene_pipeline.sensors import semantic_masks
    m=mujoco.MjSpec.from_zip(str(generated/'scene.mjz')).compile()
    manifest=json.loads((generated/'manifest.json').read_text())
    gid=next(i for i in range(m.ngeom) if m.geom(i).name.startswith('cabinet_0/'))
    raw=np.array([[[gid,int(mujoco.mjtObj.mjOBJ_GEOM)],[-1,-1]]],dtype=np.int32)
    instances,classes=semantic_masks(raw,m,manifest)
    assert instances[0,0]>0 and classes[0,0]==manifest['instances']['cabinet_0']['class_id']
    assert instances[0,1]==0


def test_navigate_smoke(generated,tmp_path):
    if not Path('vendor/mujoco_menagerie').exists():pytest.skip('Stock robot checkout required')
    import subprocess
    import sys
    from scene_pipeline.dataset import load_stream
    command=[sys.executable,'-m','scene_pipeline.cli','run',str(generated),'--flow','navigate','--seconds','.1','--seed','42','--output',str(tmp_path/'nav')]
    result=subprocess.run(command+['--goal','go to the storage shelf'],capture_output=True,text=True)
    assert result.returncode in (0,1),result.stderr
    report=json.loads((tmp_path/'nav'/'report.json').read_text())
    assert report['goal']['category']=='shelf' and report['policy']=='planner'
    assert {'true_distance_m','estimated_distance_m','declared_arrival','reached','chunks','time_to_goal_s'}<=set(report)
    assert 'video' in report or 'video_error' in report
    assert load_stream(tmp_path/'nav'/'data.h5','action')['chunk'].shape[1:]==(25,3)
    missing=subprocess.run(command+['--goal','go to the fridge','--output',str(tmp_path/'none')],capture_output=True,text=True)
    assert missing.returncode==2 and 'UNKNOWN_GOAL' in missing.stderr


def test_variant_factors_randomize_clutter():
    from scene_pipeline.dataset import randomize_program
    program=load('examples/cafe_program.py')
    a,fa=randomize_program(program,np.random.default_rng(1));b,fb=randomize_program(program,np.random.default_rng(2))
    assert fa!=fb and set(fa)=={'goods'} and all(v>=1 for v in fa.values())
    assert program['objects'][-1]['count']==8  # the source program is untouched


def test_cost_table_from_artifacts(tmp_path):
    from scene_pipeline.costs import table
    env=tmp_path/'env';(env/'attempts'/'0').mkdir(parents=True)
    write_json(env/'cost.json',dict(status='validated',seconds=7.,api_spend_usd=None,attempts=[dict(layout_seconds=1.,compile_seconds=.5,validation_seconds=2.)]))
    (env/'attempts'/'0'/'trajectory.jsonl').write_text('{"type":"turn.started"}\n{"type":"turn.completed","usage":{"input_tokens":1000,"output_tokens":500}}\n')
    flow=tmp_path/'nav';flow.mkdir();write_json(flow/'report.json',dict(flow='navigate',wall_seconds=3.,simulated_seconds=60.,passed=False))
    result=table([env,flow],tmp_path/'costs.md',usd_in=2.,usd_out=8.)
    assert result['total_wall_s']==pytest.approx(10.)
    agent=next(r for r in result['rows'] if r['stage']=='generate/agent+io')
    assert agent['wall_s']==pytest.approx(3.5) and agent['tokens']=={'input':1000,'output':500} and agent['api_spend_usd']==pytest.approx(.006)
    assert 'flow/navigate' in (tmp_path/'costs.md').read_text()
