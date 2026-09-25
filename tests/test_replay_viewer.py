import os
os.environ.setdefault('MUJOCO_GL','osmesa')

import h5py
import mujoco
import numpy as np
import pytest

from scene_pipeline.contracts import PipelineError
from scene_pipeline.replay_viewer import Playback, RecordedStates


def test_playback_pause_seek_step_speed_and_loop():
    player=Playback([2.,2.1,2.4,3.],paused=True,speed=2.)
    player.advance(10);assert player.position==2.
    player.paused=False;player.advance(.15)
    assert player.position==pytest.approx(2.3)
    assert player.index==1  # Irregular samples: never display a future pose.
    player.step(1);assert player.position==2.4 and player.paused
    player.step(-1);assert player.position==2.1
    player.seek(99);assert player.index==3
    player.paused=False;player.advance(.1)
    assert player.position==pytest.approx(2.2)
    player.loop=False;player.advance(10)
    assert player.position==3. and player.paused
    player.seek(-99);assert player.index==0
    single=Playback([5.]);single.advance(1)
    assert single.index==0 and single.paused


@pytest.mark.parametrize('times,speed',[([],1),([0,0],1),([1,0],1),([float('nan')],1),([0],float('inf')),([0],0)])
def test_playback_rejects_invalid_clock(times,speed):
    with pytest.raises(PipelineError):Playback(times,speed=speed)


def test_replay_applies_saved_states_without_advancing_physics(tmp_path,monkeypatch):
    model=mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
        <body><joint name="hinge"/><geom type="capsule" size=".05" fromto="0 0 0 0 0 .4"/></body>
        </worldbody><actuator><motor joint="hinge"/></actuator></mujoco>''')
    values=dict(time=[0.,.1],qpos=[[.2],[.7]],qvel=[[.3],[-.5]],ctrl=[[1.],[2.]])
    with h5py.File(tmp_path/'data.h5','w') as file:
        state=file.create_group('state')
        for key,value in values.items():state[key]=value
    recorded=RecordedStates(tmp_path,model);data=mujoco.MjData(model)
    def forbidden(*args,**kwargs):raise AssertionError('Replay must never step physics')
    monkeypatch.setattr(mujoco,'mj_step',forbidden)
    for index in [1,0,1]:
        recorded.apply(model,data,index)
        for key in ['qpos','qvel','ctrl']:np.testing.assert_array_equal(getattr(data,key),values[key][index])
        assert data.time==values['time'][index]
        np.testing.assert_allclose(data.xquat[1],[np.cos(values['qpos'][index][0]/2),0,0,np.sin(values['qpos'][index][0]/2)])
    with h5py.File(tmp_path/'data.h5','a') as file:
        del file['state/qpos'];file['state/qpos']=[[0,0],[0,0]]
    with pytest.raises(PipelineError,match='qpos'):RecordedStates(tmp_path,model)
