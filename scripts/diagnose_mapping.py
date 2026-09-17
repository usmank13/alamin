"""Offline pose/range ablations. Oracle inputs never enter the live controller."""
import argparse
from pathlib import Path

import h5py
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from scene_pipeline.contracts import read_json,write_json
from scene_pipeline.flows import Mapper


def diagnose(scene,rollout,output):
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    root=Path(rollout);model=mujoco.MjSpec.from_zip(str(root/'rollout_scene.mjz')).compile()
    data=mujoco.MjData(model);mujoco.mj_forward(model,data)
    ir=read_json(Path(scene)/'ir.json');base=model.body('base/base').id
    results={}
    with h5py.File(root/'data.h5','r') as f:
        times=f['state/time'][:];scan_times=f['range/time'][:]
        indices=np.array([abs(times-t).argmin() for t in scan_times])
        positions=f['state/body_position'][:,base,:]
        quats=f['state/body_quaternion'][:,base,:]
        yaw=Rotation.from_quat(quats[:,[1,2,3,0]]).as_euler('xyz')[:,2]
        estimated=f['odometry/pose'][:]
        errors=np.linalg.norm(positions[:,:2]-estimated[:,:2],axis=1)
        for mode in ('estimated_pose_noisy_ranges','true_pose_noisy_ranges','true_pose_true_ranges'):
            mapper=Mapper(ir,model,data,positions[0,:2])
            for k,i in enumerate(indices):
                mapper.scan_map={}  # Disable localization to isolate integration.
                pose=f['localization/pose'][k] if 'localization' in f else estimated[i]
                mapper.pose=pose.copy() if mode.startswith('estimated') else np.array([*positions[i,:2],yaw[i]])
                ranges=f['range/truth'][k] if mode.endswith('true_ranges') else f['range/observation'][k]
                valid=ranges<8 if mode.endswith('true_ranges') else f['range/valid'][k]
                mapper.observe(ranges,valid)
            folder=output/mode;folder.mkdir()
            report=mapper.report(folder)
            report.pop('passed')  # An oracle ablation cannot pass a task gate.
            results[mode]=report
        result=dict(kind='offline_diagnostic_not_task_acceptance',source=str(root),cases=results,
                    mean_position_error_m=float(errors.mean()),max_position_error_m=float(errors.max()),
                    actual_travel_m=float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
                    caveats=['Recorded estimated pose is pre-scan-correction in older recordings',
                             'Old recordings encode missing readings as max range; cannot distinguish dropouts retroactively'])
    write_json(output/'diagnostic.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene');p.add_argument('rollout');p.add_argument('output')
    a=p.parse_args()
    print(diagnose(a.scene,a.rollout,a.output))
