"""Navigation camera assembly and recorded prediction overlays."""
import copy
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation


def raw_depth(depth_m):
    """MuJoCo optical-axis metres to Rhizome GRAY16 (0.1 mm); zero=no return."""
    depth=np.asarray(depth_m)
    valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
    return np.rint(np.where(valid,depth,0)*10000).astype(np.uint16)


def robot_dimensions(profile):
    """Rhizome envelope dimensions from this robot's collision footprint."""
    bounds=np.asarray(profile['footprint_xy_m'])
    wheels=profile['wheels']
    return dict(robot=dict(length=float(2*np.max(np.abs(bounds[:,0]))),
                           width=float(2*np.max(np.abs(bounds[:,1])))),
                drive=dict(trackwidth=profile['track_width_m'],
                           wheelbase=float(np.ptp([w['center_m'][0] for w in wheels]))),
                wheel=dict(radius=max(w['radius_m'] for w in wheels)))


def add_cameras(robot, profile, config):
    profile=copy.deepcopy(profile)
    body=robot.body('chassis')
    for position in config['cameras']:
        name='nav_'+position
        mount=config.get('mounts',{}).get(position,{})
        xyz=mount.get('position_m',[.40949 if position=='front' else -.40949,0,-.089478])
        # Rhizome's optical calibration; MuJoCo uses right/up/backward axes.
        angles=mount.get('euler_xyz_rad',[-1.952,0,-1.57 if position=='front' else 1.57])
        optical=Rotation.from_euler('xyz',angles).as_matrix()
        xyzw=Rotation.from_matrix(optical@np.diag([1,-1,-1])).as_quat()
        quaternion=xyzw[[3,0,1,2]]
        fy=float(mount.get('fy_px',400.));fx=float(mount.get('fx_px',fy))
        if not math.isclose(fx,fy):raise ValueError('Navigation camera currently requires square pixels')
        height=float(mount.get('height_m',profile['root_ground_height_m']+xyz[2]))
        if height<=0:raise ValueError('Navigation mounting height must be positive')
        body.add_camera(name=name,pos=xyz,quat=quaternion,fovy=math.degrees(2*math.atan(240/fy)))
        profile['cameras'][name]=dict(position_m=list(xyz),quaternion_wxyz=quaternion.tolist(),
            optics=dict(lens_mm=fx,sensor_width_mm=640.,sensor_height_mm=480.),
            width=640,height=480,cam_height_m=height,role='navigation',
            calibration='Rhizome nominal nav optical pose; estimated square-pixel intrinsics and fixed CAD mounting height')
    return profile


def overlay(rgb, prediction, position):
    image=Image.fromarray(rgb);draw=ImageDraw.Draw(image)
    view=prediction.get(position,{})
    for line,prob in zip(view.get('rows_image',[]),view.get('row_vis_prob',[])):
        if prob>=.5:
            draw.line([(line['x1'],line['y1']),(line['x2'],line['y2'])],fill='#00ff66',width=3)
    x,y=view.get('vp_u_px',0),view.get('vp_v_px',0)
    draw.ellipse((x-5,y-5,x+5,y+5),outline='yellow',width=2)
    draw.rectangle((0,0,440,26),fill='black')
    draw.text((6,6),f"{position}: {prediction.get('gate','NO_FRAME')} / {prediction.get('steer_source','ROWS')}",fill='white')
    if 'obstacle_enabled' in prediction:
        blocked=prediction.get('obstacle_fwd' if position=='front' else 'obstacle_rev',False)
        grid=prediction.get('grid_stop_fwd' if position=='front' else 'grid_stop_rev',False)
        draw.rectangle((0,28,640,50),fill='black')
        draw.text((6,32),f"Obstacle {'ON' if prediction['obstacle_enabled'] else 'OFF'} | blocked: {blocked} | grid: {grid} | head: {view.get('obstacle_prob',0):.3f}",fill='red' if blocked else 'white')
        if blocked:draw.rectangle((1,1,image.width-2,image.height-2),outline='red',width=4)
    return image


def prediction_video(root):
    """Overlay saved model outputs on their actual captured images; no inference."""
    import json
    from pathlib import Path
    import subprocess
    import h5py
    from .contracts import PipelineError
    root=Path(root)
    config=json.loads((root/'drive_config.json').read_text())['controller']
    cameras=config['nav_model']['cameras']
    path=root/'rhizome'
    if config.get('physical_arms'):path=path/'navigation'
    messages=[json.loads(line) for line in (path/'messages.jsonl').read_text().splitlines()]
    captures=[r for r in messages if r['op']=='nav_infer']
    if not captures:return
    with h5py.File(root/'data.h5') as data,(root/'navigation_video.stderr.log').open('w') as log:
        streams=[data['camera_nav_'+p] for p in cameras]
        for stream in streams:
            if len(stream['time'])!=len(captures):raise PipelineError('NAV_VIDEO','Camera/prediction counts differ')
            if not np.allclose(stream['time'][:],[r['simulation_time'] for r in captures]):
                raise PipelineError('NAV_VIDEO','Camera/prediction capture times differ')
        width=640*len(cameras);height=480
        process=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{width}x{height}',
            '-framerate','10','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',
            str(root/'navigation.mp4')],stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
        try:
            for index,record in enumerate(captures):
                canvas=Image.new('RGB',(width,height))
                for col,(position,stream) in enumerate(zip(cameras,streams)):
                    canvas.paste(overlay(stream['rgb'][index],record['output']['prediction'],position),(col*640,0))
                draw=ImageDraw.Draw(canvas);draw.rectangle((0,height-24,width,height),fill='black')
                draw.text((6,height-18),f"{record['simulation_time']:.2f}s | driving: {record['output']['driving']}",fill='white')
                process.stdin.write(np.asarray(canvas).tobytes())
            process.stdin.close()
            if process.wait(timeout=120):raise PipelineError('NAV_VIDEO','ffmpeg failed; see navigation_video.stderr.log')
        finally:
            if process.poll() is None:process.kill();process.wait()
