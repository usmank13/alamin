"""Interactive viewing of saved states. Never advances physics or runs control."""
from pathlib import Path
from queue import SimpleQueue, Empty
import math
import threading
import time

import h5py
import mujoco
import numpy as np

from .contracts import PipelineError


class Playback:
    def __init__(self, times, *, paused=False, speed=1.):
        self.times=np.asarray(times,dtype=float)
        if (self.times.ndim!=1 or not len(self.times) or not np.isfinite(self.times).all()
                or np.any(np.diff(self.times)<=0)):
            raise PipelineError('REPLAY_STATE','Expected finite, strictly increasing recorded times')
        if not math.isfinite(speed) or not .0625<=speed<=16:
            raise PipelineError('REPLAY_RATE','Playback speed must be between 0.0625 and 16')
        self.position=float(self.times[0]);self.paused=paused;self.speed=speed;self.loop=True

    @property
    def index(self):
        return max(0,min(len(self.times)-1,int(np.searchsorted(self.times,self.position,side='right'))-1))

    def seek(self, position):
        self.position=float(np.clip(position,self.times[0],self.times[-1]))

    def step(self, offset):
        self.paused=True
        self.position=float(self.times[np.clip(self.index+offset,0,len(self.times)-1)])

    def advance(self, elapsed):
        if self.paused:return
        position=self.position+max(0.,elapsed)*self.speed
        duration=self.times[-1]-self.times[0]
        if position>self.times[-1]:
            if self.loop and duration>0:position=self.times[0]+(position-self.times[0])%duration
            else:self.paused=True
        self.seek(position)


class RecordedStates:
    def __init__(self, root, model):
        with h5py.File(Path(root)/'data.h5','r') as recording:
            state=recording['state']
            self.times=state['time'][:]
            self.values={key:state[key][:] for key in ('qpos','qvel','ctrl','act','mocap_pos','mocap_quat') if key in state}
        for key,shape in (('qpos',(model.nq,)),('qvel',(model.nv,)),('ctrl',(model.nu,)),
                          ('act',(model.na,)),('mocap_pos',(model.nmocap,3)),('mocap_quat',(model.nmocap,4))):
            if key not in self.values:
                if key in ('qpos','qvel'):raise PipelineError('REPLAY_STATE',f'Missing {key}')
                continue
            value=self.values[key]
            if value.shape!=(len(self.times),*shape) or not np.isfinite(value).all():
                raise PipelineError('REPLAY_STATE',f'Invalid recorded {key} for the bundled model')

    def apply(self, model, data, index):
        for key,value in self.values.items():getattr(data,key)[:]=value[index]
        data.time=float(self.times[index])
        mujoco.mj_forward(model,data)


def interactive(root, *, paused=False, speed=1., seconds=None):
    if seconds is not None and (not math.isfinite(seconds) or seconds<=0):
        raise PipelineError('REPLAY_DURATION','Viewer seconds must be positive and finite')
    # Validate rate before compiling potentially large agricultural fields.
    Playback([0.],speed=speed)
    import glfw
    import mujoco.viewer

    root=Path(root)
    model=mujoco.MjSpec.from_zip(str(root/'rollout_scene.mjz')).compile()
    data=mujoco.MjData(model);states=RecordedStates(root,model)
    playback=Playback(states.times,paused=paused,speed=speed)
    states.apply(model,data,0)
    chassis=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'element/chassis')
    follow=chassis>=0
    commands=SimpleQueue()
    existing_threads=set(threading.enumerate())
    viewer_threads=set()
    print('Recorded replay: Space play/pause; Left/Right seek 1s; comma/period step; '
          'Home restart; [/] slower/faster; F follow; L loop. Drag to orbit; right-drag pan; scroll zoom.',flush=True)
    try:
        with mujoco.viewer.launch_passive(model,data,key_callback=commands.put) as viewer:
            viewer_threads=set(threading.enumerate())-existing_threads
            with viewer.lock():
                mujoco.mjv_defaultFreeCamera(model,viewer.cam)
                viewer.opt.geomgroup[3]=0
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=0
                if follow:
                    viewer.cam.lookat[:]=data.xpos[chassis]+[0,0,-.35]
                    viewer.cam.distance=3.3;viewer.cam.azimuth=155;viewer.cam.elevation=-18
            previous_position=data.xpos[chassis].copy() if follow else None
            start=last=time.monotonic()
            while viewer.is_running():
                now=time.monotonic()
                if seconds is not None and now-start>=seconds:break
                playback.advance(now-last);last=now
                while True:
                    try:key=commands.get_nowait()
                    except Empty:break
                    if key==glfw.KEY_SPACE:playback.paused=not playback.paused
                    elif key==glfw.KEY_LEFT:playback.seek(playback.position-1.)
                    elif key==glfw.KEY_RIGHT:playback.seek(playback.position+1.)
                    elif key==glfw.KEY_COMMA:playback.step(-1)
                    elif key==glfw.KEY_PERIOD:playback.step(1)
                    elif key==glfw.KEY_HOME:playback.seek(states.times[0])
                    elif key==glfw.KEY_LEFT_BRACKET:playback.speed=max(.0625,playback.speed/2)
                    elif key==glfw.KEY_RIGHT_BRACKET:playback.speed=min(16.,playback.speed*2)
                    elif key==glfw.KEY_L:playback.loop=not playback.loop
                    elif key==glfw.KEY_F and chassis>=0:follow=not follow
                with viewer.lock():
                    states.apply(model,data,playback.index)
                    if chassis>=0:
                        position=data.xpos[chassis].copy()
                        if follow and viewer.cam.type==mujoco.mjtCamera.mjCAMERA_FREE:
                            # Preserve the user's orbit, zoom and pan offset.
                            viewer.cam.lookat[:]+=position-previous_position
                        previous_position=position
                viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_100,mujoco.mjtGridPos.mjGRID_TOPLEFT,
                    f'Recorded replay | {"PAUSED" if playback.paused else "PLAYING"} | '
                    f'{data.time:.2f}/{states.times[-1]:.2f}s | {playback.speed:g}x\n'
                    f'Space: pause  Arrows: seek  ,/.: step  Home: restart\n'
                    f'[/]: speed  F: follow ({"on" if follow else "off"})  L: loop ({"on" if playback.loop else "off"})',''))
                viewer.sync()
                time.sleep(max(0.,1/60-(time.monotonic()-now)))
    finally:
        # launch_passive.close signals exit; join before GLFW's atexit teardown.
        for thread in viewer_threads:
            thread.join(timeout=10)
            if thread.is_alive():raise RuntimeError('MuJoCo viewer thread did not shut down')
    return dict(viewer='recorded states',rollout=str(root),frames=len(states.times),final_time=float(data.time))
