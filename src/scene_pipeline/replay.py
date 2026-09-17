"""Video of recorded physical states; no new task trajectory is synthesized."""
import hashlib
from pathlib import Path
import shutil
import subprocess

import h5py
import mujoco
import numpy as np
from PIL import Image,ImageDraw

from .contracts import PipelineError,read_json,write_json


def video(root,fps=5):
    root=Path(root);destination=root/'replay.mp4'
    if destination.exists(): raise PipelineError('OUTPUT_EXISTS',str(destination))
    executable=shutil.which('ffmpeg')
    if not executable: raise PipelineError('MISSING_FFMPEG','Install ffmpeg to encode recorded-state videos')
    if not 1<=fps<=30: raise PipelineError('VIDEO_RATE','Expected 1..30 fps')
    model=mujoco.MjSpec.from_zip(str(root/'rollout_scene.mjz')).compile();data=mujoco.MjData(model)
    report=read_json(root/'report.json');camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    option=mujoco.MjvOption();option.geomgroup[3]=0;model.vis.quality.shadowsize=1024
    # Observer-only cutaway: recorded state and physical geometry are unchanged.
    for i in range(model.ngeom):
        if model.geom(i).name.startswith('wall_'):model.geom_rgba[i,3]=0
    with h5py.File(root/'data.h5','r') as recording:
        state=recording['state'];times=state['time'][:]
        data.qpos[:]=state['qpos'][0];mujoco.mj_forward(model,data)
        if report['flow']=='interaction':
            bodies=[i for i in range(model.nbody) if model.body(i).name.startswith(report['target']+'/')]
            rotation=data.xmat[bodies[0]].reshape(3,3)
            camera.lookat[:]=data.xpos[bodies[0]]+rotation@np.array([0,-.35,.6])
            camera.distance=2.8;camera.azimuth=float(np.degrees(np.arctan2(rotation[1,0],rotation[0,0]))+135);camera.elevation=-25
        else:
            camera.distance*=1.3;camera.elevation=-60
        command=[executable,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r',str(fps),
                 '-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
        with (root/'replay.stderr.log').open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            try:
                with mujoco.Renderer(model,height=360,width=480) as renderer:
                    for number,t in enumerate(np.arange(times[0],times[-1]+1e-6,1/fps)):
                        index=min(int(np.searchsorted(times,t)),len(times)-1)
                        data.qpos[:]=state['qpos'][index];data.qvel[:]=state['qvel'][index];data.time=float(times[index]);mujoco.mj_forward(model,data)
                        renderer.update_scene(data,camera,scene_option=option)
                        frame=Image.fromarray(renderer.render())
                        ImageDraw.Draw(frame).text((8,8),f'Physics replay / cutaway | {data.time:.2f}s | pass: {report["passed"]}',fill='white')
                        if number==0:frame.save(root/'replay.png')
                        process.stdin.write(np.asarray(frame).tobytes())
                process.stdin.close()
                if process.wait(timeout=30):raise PipelineError('VIDEO_FAILED','See replay.stderr.log')
            finally:
                if process.poll() is None:process.kill();process.wait()
    with (root/'data.h5').open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(video=str(destination),fps=fps,source_sha256=source_hash,source='Recorded live-physics qpos, not scripted joint sweeps',view='observer cutaway; walls hidden only for visualization',task_passed=report['passed'])
    write_json(root/'replay.json',result)
    return result
