"""Camera adaptation for Rhizome's production GridModel and message builder.

This module runs in the optional inference interpreter. It never receives plant
identities, semantic masks, generator keypoints or terrain ground truth.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def camera_inputs(rgb, depth_m, K, width, height):
    import cv2
    if rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or depth_m.shape!=rgb.shape[:2]:
        raise ValueError('Expected uint8 RGB and aligned metric depth')
    source_h,source_w=depth_m.shape
    intrinsics=np.asarray(K,dtype=float).copy()
    intrinsics[0,:]*=width/source_w;intrinsics[1,:]*=height/source_h
    color=cv2.resize(rgb,(width,height),interpolation=cv2.INTER_LINEAR)
    color=cv2.cvtColor(color,cv2.COLOR_RGB2RGBA)
    depth=cv2.resize(depth_m,(width,height),interpolation=cv2.INTER_NEAREST)
    # Rhizome's package encodes depth_max_m = depth_norm / 10000.
    # Invalid/out-of-range rays use the sensor's zero-invalid convention.
    valid=np.isfinite(depth)&(depth>0)&(depth<=6.5535)
    raw=np.rint(np.where(valid,depth,0)*10000).astype(np.uint16)
    return color,raw,intrinsics


def quaternion_rotation(quaternion):
    w,x,y,z=quaternion
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def frame_transforms(camera_rotation_world,camera_position_world,chassis_rotation_world,
                     chassis_position_world,target_position_chassis,target_quaternion_chassis=(1,0,0,0)):
    # MuJoCo: right, up, backward. RealSense optical: right, down, forward.
    camera_rotation=np.asarray(camera_rotation_world)@np.diag([1.,-1.,-1.])
    chassis_rotation=np.asarray(chassis_rotation_world)
    target_rotation=chassis_rotation@quaternion_rotation(target_quaternion_chassis)
    target_position=np.asarray(chassis_position_world)+chassis_rotation@np.asarray(target_position_chassis)
    return (camera_rotation,np.asarray(camera_position_world).reshape(3),
            target_rotation.T,(-target_rotation.T@target_position).reshape(3))


class NeuralPerception:
    def __init__(self,config,core):
        import torch
        import yaml
        from crop_inference_engine.model.model import Model
        from crop_inference_engine.msg.publisher import CropMessageBuilder
        self.torch,self.p,self.Model,self.Builder=torch,core,Model,CropMessageBuilder
        self.config=config;self.models={};self.calibrations={}
        root=Path(config['package'])
        package=yaml.safe_load((root/'config.yml').read_text())
        if package.get('variant')!='grid' or package.get('publish_frame')!='arm':
            raise ValueError('This adapter requires a grid package publishing in arm frame')
        if config.get('require_geometry') and not package.get('geometry'):
            raise ValueError('Native weeding requires a geometry-capable grid postprocessor; '
                             'upgrade the package with examples/agriculture/upgrade_grid_geometry.py')
        if not any((root/name).is_file() for name in ('model.ts','core.ts')):
            raise FileNotFoundError(root/'core.ts')
        for name in ('post.ts',):
            if not (root/name).is_file():raise FileNotFoundError(root/name)
        if config['device']=='cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable in the inference worker')
        if config['crop_species'] not in package['inference']['crop_species']:
            raise ValueError('crop_species is absent from the package conditioning layout')
        if config['input_width']<=package['preprocess']['left_crop_px']:
            raise ValueError('Model input width must exceed the package left crop')
        self.info=dict(package=str(root),package_config=package,device=config['device'],runtime='torch',
                       torch_version=torch.__version__,crop_species=config['crop_species'],
                       files_sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                     for name in ('config.yml','model.ts','core.ts','post.ts','grid_split_manifest.json','upgrade.json') if (root/name).exists()},
                       input_size=[config['input_width'],config['input_height']],
                       depth_units_m=.0001,depth_invalid=0,left_crop_px=package['preprocess']['left_crop_px'],
                       target_positions_chassis_m=config['target_positions_chassis_m'],
                       target_quaternions_chassis_wxyz=config.get('target_quaternions_chassis_wxyz',{}),
                       target_axes='Configured arm frame; identity for legacy chassis-aligned mounts',
                       row_profile='model prediction only; no terrain fallback',
                       observation='rendered RGB and ideal metric depth; no synthetic detections or masks')

    def infer(self,camera,capture_ns,arrays):
        color,depth,K=camera_inputs(arrays['rgb'],arrays['depth'],arrays['K'],
                                   self.config['input_width'],self.config['input_height'])
        if camera not in self.models:
            model=self.Model.load(self.config['package'],60,device=self.config['device'],runtime='torch')
            model.set_crop_species(self.config['crop_species'])
            model.set_intrinsics(self.p.CamIntrinsicsMsg(dict(params=dict(
                fx=K[0,0],fy=K[1,1],cx=K[0,2],cy=K[1,2],
                frame_width=color.shape[1],frame_height=color.shape[0]))))
            if hasattr(model.postprocessor,'set_row_profile_tube_length_m_'):
                model.postprocessor.set_row_profile_tube_length_m_(self.config['tube_length_m'])
            builder=self.Builder();builder.set_frame(self.p.Frame.Frame('inference_'+camera))
            self.models[camera]=(model,builder)
            self.calibrations[camera]=K
        if not np.array_equal(K,self.calibrations[camera]):raise ValueError('Camera intrinsics changed during a rollout')
        model,builder=self.models[camera]
        matrices=frame_transforms(arrays['camera_rotation_world'],arrays['camera_position_world'],
                                  arrays['chassis_rotation_world'],arrays['chassis_position_world'],
                                  self.config['target_positions_chassis_m'][camera],
                                  self.config.get('target_quaternions_chassis_wxyz',{}).get(camera,[1,0,0,0]))
        transforms=[self.torch.tensor(a,dtype=self.torch.float32,device=self.config['device']) for a in matrices]
        model.begin_frame(capture_ns)
        started=time.perf_counter()
        prediction=model(color,depth,*transforms)
        inf,row,density=builder.build(prediction,capture_ns,capture_ns,0,copy=True,
                                    geometry=model.geometry_snapshot())
        elapsed=time.perf_counter()-started
        # Preserve simulated capture time; wall latency is a separate diagnostic.
        encoded=json.loads(str(inf))
        json.dumps(encoded,allow_nan=False)
        if row is not None:json.dumps(json.loads(str(row)),allow_nan=False)
        return inf,row,density,dict(inference_wall_seconds=elapsed,
            prediction=encoded,row_profile=json.loads(str(row)) if row is not None else None,
            density=json.loads(str(density)) if density is not None else None,
            resized_K=K.tolist(),input_size=[color.shape[1],color.shape[0]])
