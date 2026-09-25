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


def video(root,fps=5,*,model=None):
    root=Path(root);destination=root/'replay.mp4'
    if destination.exists(): raise PipelineError('OUTPUT_EXISTS',str(destination))
    executable=shutil.which('ffmpeg')
    if not executable: raise PipelineError('MISSING_FFMPEG','Install ffmpeg to encode recorded-state videos')
    if not 1<=fps<=30: raise PipelineError('VIDEO_RATE','Expected 1..30 fps')
    # Agricultural captures can retain a large compiled plant field. Reuse it
    # when called directly after a rollout instead of holding a second copy.
    if model is None:model=mujoco.MjSpec.from_zip(str(root/'rollout_scene.mjz')).compile()
    data=mujoco.MjData(model)
    report=read_json(root/'report.json');camera=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(model,camera)
    follow_robot=(report['flow']=='agriculture-drive' and (root/'rig.json').exists()
                  and bool(read_json(root/'rig.json')['profile'].get('arms')))
    option=mujoco.MjvOption();option.geomgroup[3]=0;option.flags[mujoco.mjtVisFlag.mjVIS_RANGEFINDER]=0;model.vis.quality.shadowsize=1024
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
        elif report['flow']=='agriculture-drive':
            bounds=np.asarray(read_json(root/'manifest.json')['terrain']['bounds_m'])
            camera.lookat[:]=[*bounds.mean(axis=0),.3]
            camera.distance=float(np.linalg.norm(bounds[1]-bounds[0])*.9)
            camera.azimuth=135;camera.elevation=-40
            if follow_robot:
                camera.distance=3.3;camera.azimuth=155;camera.elevation=-12
        else:
            # Imported meshes can inflate MuJoCo's default model extent. Frame
            # the actual finite room floor so the robot and aisles stay visible.
            floors=[i for i in range(model.ngeom) if model.geom(i).name.startswith('floor_')]
            if floors:
                corners=[]
                for i in floors:
                    rotation=data.geom_xmat[i].reshape(3,3)
                    center=data.geom_xpos[i]+rotation@model.geom_aabb[i,:3]
                    half=np.abs(rotation)@model.geom_aabb[i,3:]
                    corners.extend([center-half,center+half])
                low,high=np.min(corners,axis=0),np.max(corners,axis=0)
                camera.lookat[:]=[*(low[:2]+high[:2])/2,.6]
                camera.distance=float(np.linalg.norm(high[:2]-low[:2])*1.1)
            else:camera.distance*=1.3
            camera.azimuth=130;camera.elevation=-60
        command=[executable,'-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s','480x360','-r',str(fps),
                 '-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
        with (root/'replay.stderr.log').open('w') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            try:
                with mujoco.Renderer(model,height=360,width=480) as renderer:
                    for number,t in enumerate(np.arange(times[0],times[-1]+1e-6,1/fps)):
                        index=min(int(np.searchsorted(times,t)),len(times)-1)
                        data.qpos[:]=state['qpos'][index];data.qvel[:]=state['qvel'][index];data.time=float(times[index]);mujoco.mj_forward(model,data)
                        if follow_robot:camera.lookat[:]=data.body('element/chassis').xpos+[0,0,-.35]
                        renderer.update_scene(data,camera,scene_option=option)
                        frame=Image.fromarray(renderer.render())
                        view='robot' if follow_robot else 'field' if report['flow']=='agriculture-drive' else 'cutaway'
                        ImageDraw.Draw(frame).text((8,8),f'Physics replay / {view} | {data.time:.2f}s | pass: {report["passed"]}',fill='white')
                        if number==0:frame.save(root/'replay.png')
                        process.stdin.write(np.asarray(frame).tobytes())
                process.stdin.close()
                if process.wait(timeout=30):raise PipelineError('VIDEO_FAILED','See replay.stderr.log')
            finally:
                if process.poll() is None:process.kill();process.wait()
    with (root/'data.h5').open('rb') as stream:source_hash=hashlib.file_digest(stream,'sha256').hexdigest()
    result=dict(video=str(destination),fps=fps,source_sha256=source_hash,source='Recorded live-physics qpos, not scripted joint sweeps',view='robot follow' if follow_robot else 'field overview' if report['flow']=='agriculture-drive' else 'observer cutaway; walls hidden only for visualization',task_passed=report['passed'])
    write_json(root/'replay.json',result)
    return result
