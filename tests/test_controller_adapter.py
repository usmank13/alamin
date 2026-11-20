from pathlib import Path
from types import SimpleNamespace

import math

import mujoco
import pytest

from sim_harness.adapters import load_factory, stepping


def test_adapter_step_and_cleanup_on_failure():
    model = mujoco.MjModel.from_xml_string("<mujoco/>")
    data = mujoco.MjData(model)
    observed = []

    def bind(m, d):
        assert (m, d) == (model, data)

        def before_step():
            observed.append(d.time)
            if len(observed) == 2:
                raise RuntimeError("controller failed")

        return SimpleNamespace(before_step=before_step, close=lambda: observed.append("closed"))

    with pytest.raises(RuntimeError, match="controller failed"):
        with stepping(model, data, bind(model, data)) as step:
            step()
            step()
    assert observed == [0, model.opt.timestep, "closed"]
    assert data.time == model.opt.timestep


def test_import_external_factory():
    assert load_factory("types:SimpleNamespace") is SimpleNamespace
    with pytest.raises(ValueError, match="module:factory"):
        load_factory("missing_separator")


def test_soccer_example_stops_after_measured_travel(monkeypatch):
    from sim_harness.scene import load_scene

    examples = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(examples))
    model, data = load_scene(examples / "lab.xml", examples / "robots.json")
    position = data.joint("base/base_freejoint").qpos
    start = position[:2].copy()
    with stepping(model, data, load_factory("soccer_adapter:Drive")(model, data)) as step:
        for _ in range(round(3 / model.opt.timestep)):
            step()
        assert math.dist(position[:2], start) == pytest.approx(0.25, abs=0.01)
        assert all(data.actuator(f"base/wheel{i}_speed").ctrl[0] == 0 for i in (1, 2, 3))
