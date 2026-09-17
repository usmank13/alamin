"""Portable native scene export; independent-format adapters belong beside this."""
from pathlib import Path


def export_mujoco(spec, data, destination: str | Path):
    """Save composed MJCF/assets plus current state as the 'harness_initial' key.

    This is MuJoCo portability, not the brief's independent-format proof.
    """
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite export: {destination}")
    snapshot = spec.copy()
    for key in list(snapshot.keys):
        snapshot.delete(key)
    snapshot.add_key(name="harness_initial", time=data.time, qpos=data.qpos,
                     qvel=data.qvel, act=data.act, ctrl=data.ctrl,
                     mpos=data.mocap_pos.flatten(), mquat=data.mocap_quat.flatten())
    destination.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_zip(str(destination))
