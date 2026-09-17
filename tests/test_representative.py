"""Focused regressions for the source-preserving retrieval prototype."""
from pathlib import Path

import mujoco
import numpy as np
import pytest

from prototypes.p5_native import scale_native
from prototypes.visual_checks import bounds
from sim_harness.scene import _read_spec, load_scene


def test_scaling_scalar_geometry_and_slide_reference():
    spec = mujoco.MjSpec.from_string('''<mujoco><worldbody><body pos="1 2 3">
      <joint type="slide" range="0 1" springref="-.2"/>
      <geom type="box" size=".1 .2 .3" density="100"/>
    </body></worldbody></mujoco>''')
    base = spec.compile()
    model = scale_native(spec, .5).compile()
    np.testing.assert_allclose(model.body_pos[1], base.body_pos[1]*.5)
    np.testing.assert_allclose(model.geom_size, base.geom_size*.5)
    np.testing.assert_allclose(model.body_mass, base.body_mass*.5**3)
    np.testing.assert_allclose(model.body_inertia, base.body_inertia*.5**5)
    np.testing.assert_allclose(model.jnt_range, base.jnt_range*.5)
    np.testing.assert_allclose(model.qpos_spring, base.qpos_spring*.5)


def test_scaling_rejects_unsupported_features():
    spec = mujoco.MjSpec.from_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match='scalar'):
        scale_native(spec, 1)
    with pytest.raises(ValueError, match='positive'):
        scale_native(spec, 0)


@pytest.mark.parametrize('factor', [.75, 1, 1.25])
def test_compiled_mesh_rescale_roundtrip(tmp_path, factor):
    path = Path('vendor/robocasa_native/fixtures/microwaves/Microwave075/model.xml')
    if not path.exists():
        pytest.skip('Run prototypes/fetch_robocasa.py for dataset integration tests')
    spec = _read_spec(path)
    native = spec.compile()
    data = mujoco.MjData(native)
    mujoco.mj_forward(native, data)
    lo, hi = bounds(native, data)
    scaled = scale_native(spec, factor)
    model = scaled.compile()
    second = mujoco.MjData(model)
    mujoco.mj_forward(model, second)
    low, high = bounds(model, second)
    np.testing.assert_allclose(high-low, (hi-lo)*factor, rtol=1e-5)
    np.testing.assert_allclose(model.body_mass, native.body_mass*factor**3, rtol=1e-5)
    np.testing.assert_array_equal(model.geom_contype, native.geom_contype)
    np.testing.assert_array_equal(model.geom_conaffinity, native.geom_conaffinity)
    scaled.add_key(name='harness_initial', qpos=model.qpos0)
    archive = tmp_path/'scene.mjz'
    scaled.to_zip(str(archive))
    restored, _ = load_scene(archive)
    # MJCF decimal serialization/recomputed mesh inertia can differ by a few ppm.
    np.testing.assert_allclose(restored.body_mass, model.body_mass, rtol=1e-5)
