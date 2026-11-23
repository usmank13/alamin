"""Run with RHIZOME_ROOT and a RHIZOME_HOST_BUILD built with --sim-odometry."""
import json
import math
import os
import time
from pathlib import Path

import pytest

from scene_pipeline.rhizome import RhizomeController


@pytest.mark.skipif(not os.environ.get('RHIZOME_HOST_BUILD'), reason='requires a selected native build')
@pytest.mark.parametrize(('planar', 'fixed'), [(True, False), (False, False), (False, True)])
def test_native_distance_and_pitch_drift(tmp_path, planar, fixed):
    manifest = Path(os.environ['RHIZOME_HOST_BUILD'])/'build_manifest.json'
    if 'sim_odometry_hooks_sha256' not in json.loads(manifest.read_text()):
        pytest.skip('build the selected host with --sim-odometry')
    dims = dict(robot=dict(length=1., width=2.), drive=dict(trackwidth=1.6, wheelbase=.8),
                wheel=dict(radius=.2))
    cfg = dict(waypoints=[dict(xy=[10, 0])], localization_dimensions=dims,
               localization=dict(mode='odometry', params=dict(disable_tilt=planar)))
    # Truth coordinates in the RPC log must never enter odometry or LocalControl.
    pitch = math.radians(1.5)
    sample = dict(wheels=dict(front_left=2., rear_left=2., front_right=-2., rear_right=-2.),
                  attitude=[math.cos(pitch/2), 0., math.sin(pitch/2), 0.],
                  accel=[-math.sin(pitch), 0., math.cos(pitch)], gyro=[0., 0., 0.])
    controller = RhizomeController(cfg, tmp_path, timeout=60)
    try:
        previous = None
        ticks = 2000 if fixed else 300
        for k in range(ticks + 1):
            if planar and k in (90, 100, 110):
                time.sleep(.08)  # A slow camera tick must not advance the estimator.
            sample.pop('rtk_kinematics', None)
            if fixed and k % 20 == 0:
                sample['rtk_kinematics'] = dict(position=[.004*k, 0., 1.], rotation=[1., 0., 0., 0.],
                    velocity=[.4, 0., 0.], omega=[0., 0., 0.], dz=0., status='RTK_FIXED', v_acc=.014)
            result = controller.step(k / 100, [123, 0, 25], [1, 0, 0, 0], 0., sensors=sample)
            current = result['localization']
            if planar and previous is not None:
                dx = current['pose']['x'] - previous['pose']['x']
                assert dx == pytest.approx(.01*previous['twist']['v'], abs=1e-8)
            previous = current
        pose = result['localization']['pose']
        distance = ticks * .004
        assert pose['x'] == pytest.approx(distance, abs=.02)
        if fixed:
            assert abs(pose['z']) < .02
            assert result['localization']['path_offset'] == pytest.approx(-math.sin(pitch), abs=.005)
            assert result['localization']['rtk']['status'] == 'RTK_FIXED'
        else:
            assert pose['z'] == pytest.approx(0 if planar else -distance*math.sin(pitch), abs=.001)
        assert result['twist'][0] > 0
    finally:
        controller.close()
