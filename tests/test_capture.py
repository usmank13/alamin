import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np
from PIL import Image


def test_headless_capture_ids_and_timestamps(tmp_path):
    scene = Path(__file__).resolve().parents[1] / "examples/lab.xml"
    subprocess.run([sys.executable, "-m", "sim_harness.cli", str(scene),
                    "--seconds", "0.02", "--backend", "osmesa", "--output", str(tmp_path)],
                   check=True, capture_output=True, text=True)
    report = json.loads((tmp_path / "report.json").read_text())
    truth = json.loads((tmp_path / "ground_truth.json").read_text())
    with np.load(tmp_path / "state.npz", allow_pickle=False) as state:
        assert float(state['time']) == truth['time'] == report['capture_time']
        np.testing.assert_allclose(state['body_position'], [b['position'] for b in truth['bodies']])
    segmentation = np.load(tmp_path / "segmentation.npy")
    assert segmentation.shape == (480, 640, 2)
    geom_pixels = segmentation[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    assert geom_pixels.any()
    assert set(np.unique(segmentation[:, :, 0][geom_pixels])) <= {g['id'] for g in truth['geoms']}
    assert np.load(tmp_path / "depth.npy").shape == (480, 640)
    assert Image.open(tmp_path / "rgb.png").size == (640, 480)
    assert report['timestep'] == .002
    assert report['integrator'] == 'mjINT_IMPLICITFAST'
