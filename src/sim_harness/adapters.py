"""Run a robot controller against a composed MuJoCo world."""
from contextlib import contextmanager
from importlib import import_module

import mujoco


def load_factory(spec):
    module, separator, name = spec.partition(":")
    if not separator or not module or not name.isidentifier():
        raise ValueError("Expected an adapter of the form package.module:factory")
    return getattr(import_module(module), name)


@contextmanager
def stepping(model, data, controller=None):
    if controller is None:
        yield lambda: mujoco.mj_step(model, data)
        return

    def step():
        controller.before_step()
        mujoco.mj_step(model, data)

    close = controller.close
    try:
        yield step
    finally:
        close()
