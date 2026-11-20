import json
from types import SimpleNamespace

import mujoco
import numpy as np

from scene_pipeline.rhizome_e2e_camera import CropCameras, stage_crop


def test_native_crop_profiles_and_mujoco_calibration(tmp_path):
    def node(name):
        config = {} if name == 'crop_ie' else dict(
            weeding_mode='e2e', field_package='/robot/field',
            camera=dict(common=dict(shm={}), crop=dict(
                combo=dict(resolution=dict(output=dict(width=848, height=480))))),
        )
        return dict(meta=dict(includes=['camera/crop.yml']), config=config, missing_includes=[])

    task = SimpleNamespace(node=node)
    paths = {key: tmp_path/key for key in
             ('model_package', 'field_package', 'library_dir', 'socket_root', 'field_dir')}
    stage_crop(task, tmp_path, **paths)
    staged = json.loads((tmp_path/'nodes/crop_ie_0.json').read_text())
    assert 'includes' not in staged['meta']
    assert staged['config']['field_package'] == str(paths['field_package'])
    assert staged['config']['camera']['common']['shm']['combo_stream'] == str(paths['socket_root'])
    assert staged['config']['weeding_mode'] == 'e2e'

    cameras = ''.join(f'<camera name="element/c{i}" fovy="90" pos="{i} 0 0"/>' for i in range(4))
    model = mujoco.MjModel.from_xml_string(
        f'<mujoco><worldbody><body name="element/chassis">{cameras}</body></worldbody></mujoco>')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    world = SimpleNamespace(model=model, data=data,
                            profile=dict(arms=[dict(camera=f'c{i}') for i in range(4)]))
    rig = CropCameras(world, task, {1: tmp_path/'p1', 2: tmp_path/'p2'})
    intrinsics, frame = rig.calibration(2, 0)
    assert np.isclose(intrinsics['params']['fx'], 240)
    assert frame['origin']['x'] == 2
    assert np.isclose(abs(frame['origin']['qx']), 1)
