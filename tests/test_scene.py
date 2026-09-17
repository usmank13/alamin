import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim_harness.scene import compose_scene, load_scene
from sim_harness.export import export_mujoco

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "examples/lab.xml"
ROBOTS = ROOT / "examples/robots.json"


def test_scene_alone():
    model, data = load_scene(SCENE)
    assert model.nu == 0
    mujoco.mj_step(model, data, nstep=500)
    assert data.body("test_cube").xpos[2] == pytest.approx(0.84, abs=0.002)


def test_stock_robots_place_initialize_and_actuate():
    model, data = load_scene(SCENE, ROBOTS)
    assert model.nu == 12
    np.testing.assert_allclose(data.body("arm/link0").xpos, [0.3, 0.6, 0.8])
    np.testing.assert_allclose(data.body("base/base").xpos, [-0.5, -0.5, 0.03])
    assert data.joint("arm/joint4").qpos[0] == pytest.approx(-1.57079)
    mujoco.mj_step(model, data, nstep=1000)
    base_before = data.body("base/base").xpos.copy()
    arm_before = data.joint("arm/joint1").qpos.copy()
    data.actuator("arm/actuator1").ctrl[:] = 0.2
    data.actuator("base/wheel1_speed").ctrl[:] = 4
    data.actuator("base/wheel2_speed").ctrl[:] = -4
    mujoco.mj_step(model, data, nstep=1000)
    assert np.linalg.norm(data.body("base/base").xpos[:2] - base_before[:2]) > 0.03
    assert abs(data.joint("arm/joint1").qpos[0] - arm_before[0]) > 0.1
    assert np.isfinite(data.qpos).all()
    assert not any(w.number for w in data.warning)
    assert data.time == pytest.approx(4)


def test_multiple_instances_and_rotation(tmp_path):
    model_path = str(ROOT / "vendor/mujoco_menagerie/robot_soccer_kit/robot_soccer_kit.xml")
    manifest = tmp_path / "robots.json"
    manifest.write_text(json.dumps([
        {"name": "one", "model": model_path, "pos": [-1, 0, 0]},
        {"name": "two", "model": model_path, "pos": [1, 0, 0], "quat": [0, 0, 0, 1]},
    ]))
    model, data = load_scene(SCENE, manifest)
    assert model.nu == 8
    np.testing.assert_allclose(data.body("two/base").xpos, [1, 0, 0.03])
    np.testing.assert_allclose(data.body("two/base").xmat.reshape(3, 3), np.diag([-1, -1, 1]), atol=1e-8)
    data.actuator("one/wheel1_speed").ctrl[:] = 2
    assert data.actuator("two/wheel1_speed").ctrl[0] == 0


def test_duplicate_names_rejected(tmp_path):
    entry = {"name": "same", "model": str(ROOT / "vendor/mujoco_menagerie/franka_emika_panda/panda.xml")}
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="duplicate"):
        load_scene(SCENE, manifest)


@pytest.mark.parametrize("scene,robots", [(SCENE, ROBOTS),
    (ROOT / "examples/stretch_lab.xml", ROOT / "examples/stretch.json")])
def test_portable_export_preserves_articulation_assets_and_state(tmp_path, scene, robots):
    spec, model, data = compose_scene(scene, robots)
    bundle = tmp_path / "scene.mjz"
    export_mujoco(spec, data, bundle)
    restored = mujoco.MjSpec.from_zip(str(bundle)).compile()
    state = mujoco.MjData(restored)
    mujoco.mj_resetDataKeyframe(restored, state, restored.key("harness_initial").id)
    mujoco.mj_forward(restored, state)
    assert (restored.njnt, restored.nu, restored.nmesh) == (model.njnt, model.nu, model.nmesh)
    np.testing.assert_allclose(restored.jnt_range, model.jnt_range)
    assert (restored.ntendon, restored.neq, restored.ntex) == (model.ntendon, model.neq, model.ntex)
    np.testing.assert_allclose(state.qpos, data.qpos)
    np.testing.assert_allclose(state.ctrl, data.ctrl)
    np.testing.assert_allclose(state.xpos, data.xpos)
    _, replay = load_scene(bundle)
    np.testing.assert_allclose(replay.qpos, data.qpos)
    np.testing.assert_allclose(replay.ctrl, data.ctrl)
    mujoco.mj_step(restored, state, nstep=500)
    assert not any(w.number for w in state.warning)
    with pytest.raises(FileExistsError):
        export_mujoco(spec, data, bundle)


def test_stretch_drive_lift_extend_and_turn():
    model, data = load_scene(ROOT / "examples/stretch_lab.xml", ROOT / "examples/stretch.json")
    assert model.nu == 8
    np.testing.assert_allclose(data.body("stretch/base_link").xpos, [-0.8, -0.6, 0])
    mujoco.mj_step(model, data, nstep=1000)
    start = data.body("stretch/base_link").xpos.copy()
    data.actuator("stretch/forward").ctrl[:] = 0.2
    mujoco.mj_step(model, data, nstep=1000)
    distance = np.linalg.norm(data.body("stretch/base_link").xpos[:2] - start[:2])
    assert 0.05 < distance < 0.2
    data.actuator("stretch/forward").ctrl[:] = 0
    data.actuator("stretch/lift").ctrl[:] = 0.2
    data.actuator("stretch/arm_extend").ctrl[:] = 0.25
    mujoco.mj_step(model, data, nstep=2000)
    assert data.joint("stretch/joint_lift").qpos[0] == pytest.approx(0.2, abs=0.01)
    extension = [data.joint(f"stretch/joint_arm_l{i}").qpos[0] for i in range(4)]
    assert sum(extension) == pytest.approx(0.25, abs=0.01)
    assert np.ptp(extension) < 0.002  # stock coupling constraints survived attachment
    before = data.body("stretch/base_link").xmat.copy()
    data.actuator("stretch/turn").ctrl[:] = 0.2
    mujoco.mj_step(model, data, nstep=1000)
    assert np.linalg.norm(data.body("stretch/base_link").xmat - before) > 0.1
    assert data.body("stretch/base_link").xmat.reshape(3, 3)[2, 2] > 0.99
    assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    assert not any(w.number for w in data.warning)
    assert data.time == pytest.approx(10)
