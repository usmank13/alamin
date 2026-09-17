"""Explicit sensor rigs and seeded observation noise, separate from ground truth."""
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

NOISE={'version':'synthetic-v2','calibrated':False,'range_sigma_m':.01,'range_dropout':.01,
       'range_missing':'NaN: missing measurement, do not update map','range_no_return_m':8.,
       'gyro_sigma_rad_s':.002,'accelerometer_sigma_m_s2':.03,'joint_sigma':1e-4,
       'rgb_sigma_8bit':2.,'depth_sigma_m':.005,'force_sigma_n':.05,'torque_sigma_nm':.005}
RATES={'state':100,'imu':100,'wrench':100,'range':20,'camera':10}


def attach_rig(spec,body_name,prefix,*,lidar=False,wrench_body=None):
    body=spec.body(body_name)
    body.add_site(name=prefix+'imu_site',pos=[0,0,.12],size=[.003,0,0],rgba=[0,0,0,0])
    for name,kind in [('accelerometer',mujoco.mjtSensor.mjSENS_ACCELEROMETER),('gyro',mujoco.mjtSensor.mjSENS_GYRO)]:
        spec.add_sensor(name=prefix+name,type=kind,objtype=mujoco.mjtObj.mjOBJ_SITE,objname=prefix+'imu_site')
    camera_R=np.array([[0,0,-1],[-1,0,0],[0,1,0]])
    quat=Rotation.from_matrix(camera_R).as_quat()[[3,0,1,2]]
    body.add_camera(name=prefix+'rgbd',pos=[.04,0,.18],quat=quat,fovy=60)
    if lidar:
        for i,angle in enumerate(np.linspace(-math.pi,math.pi,72,endpoint=False)):
            direction=np.array([math.cos(angle),math.sin(angle),0.])
            q=np.zeros(4);mujoco.mju_quatZ2Vec(q,direction)
            name=prefix+f'ray_{i}'
            body.add_site(name=name,pos=[0,0,.2],quat=q,size=[.001,0,0],rgba=[0,0,0,0])
            spec.add_sensor(name=name,type=mujoco.mjtSensor.mjSENS_RANGEFINDER,objtype=mujoco.mjtObj.mjOBJ_SITE,objname=name,intprm=[1,0,0])
    if wrench_body:
        spec.body(wrench_body).add_site(name=prefix+'wrench_site',pos=[0,0,0],size=[.002,0,0],rgba=[0,0,0,0])
        for name,kind in [('force',mujoco.mjtSensor.mjSENS_FORCE),('torque',mujoco.mjtSensor.mjSENS_TORQUE)]:
            spec.add_sensor(name=prefix+name,type=kind,objtype=mujoco.mjtObj.mjOBJ_SITE,objname=prefix+'wrench_site')
    focal=120/math.tan(math.radians(30))
    return dict(prefix=prefix,rates_hz=RATES,noise=NOISE,camera=dict(width=320,height=240,fovy_deg=60,
                K=[[focal,0,160],[0,focal,120],[0,0,1]],parent_body=body_name,position_m=[.04,0,.18],quaternion_wxyz=quat.tolist()),
                imu=dict(parent_body=body_name,position_m=[0,0,.12],quaternion_wxyz=[1,0,0,0]),
                range=dict(parent_body=body_name,position_m=[0,0,.2],angles_rad=np.linspace(-math.pi,math.pi,72,endpoint=False).tolist(),max_range_m=8.) if lidar else None)


def sensor(data,name):
    return data.sensor(name).data.copy()


def noisy_ranges(data,prefix,rng):
    raw=np.array([float(sensor(data,prefix+f'ray_{i}')[0]) for i in range(72)])
    # -1 is the native no-return flag. A max-range observation is not an obstacle.
    valid=(raw>=0)&(raw<8.)
    truth=np.where(valid,raw,8.)
    observation=np.clip(truth+rng.normal(0,NOISE['range_sigma_m'],72),0,8)
    dropped=rng.random(72)<NOISE['range_dropout']
    observation[~valid]=8.
    valid &= ~dropped
    # Missing measurement is not evidence of eight metres of free space.
    observation[dropped]=np.nan
    return truth,observation,valid


def semantic_masks(raw,model,manifest):
    instance=np.zeros(raw.shape[:2],dtype=np.int32);semantic=np.zeros_like(instance)
    instances=sorted(manifest['instances']);ids={name:i+1 for i,name in enumerate(instances)}
    for geom in np.unique(raw[:,:,0]):
        if geom<0 or geom>=model.ngeom: continue
        name=model.geom(int(geom)).name.split('/')[0]
        if name not in ids: continue
        mask=(raw[:,:,0]==geom)&(raw[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        instance[mask]=ids[name];semantic[mask]=manifest['instances'][name]['class_id']
    return instance,semantic
