"""Compose MJCF with namespaced robot instances without editing source assets."""
import json
import hashlib
from pathlib import Path
import re

import mujoco
import numpy as np


def _read_spec(path):
    """Embed file assets under content-addressed names for portable composition."""
    path = Path(path).resolve()
    if path.suffix == ".mjz":
        return mujoco.MjSpec.from_zip(str(path))
    spec = mujoco.MjSpec.from_file(str(path))
    for items, directory in ((spec.meshes, "meshdir"), (spec.textures, "texturedir"),
                             (spec.hfields, None)):
        for asset in items:
            if hasattr(asset, "cubefiles") and any(asset.cubefiles):
                raise ValueError("Six-file cubemap textures are not yet supported; use a single texture file")
            if not asset.file:
                continue
            folder = getattr(asset.compiler, directory) if directory else ""
            source = path.parent / folder / asset.file
            if not asset.name:
                asset.name = Path(asset.file).stem
            content = source.read_bytes()
            filename = hashlib.sha256(content).hexdigest() + source.suffix
            spec.assets[filename] = content
            asset.file = filename
    return spec


def compose_scene(scene: str | Path, robots: str | Path | None = None):
    """Return (spec, model, data) for exporters and runtime consumers.

    Placements are in metres; quaternions are wxyz.

    Scene owns global physics settings. Robot paths are relative to the manifest.
    Optional robot keyframes initialize scalar joints and actuator controls.
    Floating base placement comes from the attachment frame, not a keyframe.
    """
    spec = _read_spec(scene)
    entries = [] if robots is None else json.loads(Path(robots).read_text())
    if not isinstance(entries, list):
        raise ValueError("Robot manifest must be a JSON array")
    initial = []
    names = set()
    for entry in entries:
        name = entry["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) or name in names:
            raise ValueError(f"Invalid or duplicate robot name: {name}")
        names.add(name)
        pos = np.asarray(entry.get("pos", [0, 0, 0]), dtype=float)
        quat = np.asarray(entry.get("quat", [1, 0, 0, 0]), dtype=float)
        if pos.shape != (3,) or not np.isfinite(pos).all():
            raise ValueError("pos must contain three finite numbers")
        if quat.shape != (4,) or not np.isfinite(quat).all() or not np.isclose(np.linalg.norm(quat), 1):
            raise ValueError("quat must be a unit quaternion in wxyz order")
        path = (Path(robots).resolve().parent / entry["model"]).resolve()
        child = _read_spec(path)
        prefix = name + "/"
        if "keyframe" in entry:
            source = child.compile()
            key = source.key(entry["keyframe"])
            for j in range(source.njnt):
                if int(source.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
                    raise ValueError("Keyframe initialization supports only scalar robot joints; omit keyframe for floating robots")
                initial.append(("joint", prefix + source.joint(j).name, key.qpos[source.jnt_qposadr[j]].copy()))
            for a in range(source.nu):
                initial.append(("actuator", prefix + source.actuator(a).name, key.ctrl[a].copy()))
        for key in list(child.keys):
            child.delete(key)
        frame = spec.worldbody.add_frame(pos=pos, quat=quat)
        spec.attach(child, prefix=prefix, frame=frame)
    model = spec.compile()
    data = mujoco.MjData(model)
    if Path(scene).suffix == ".mjz":
        mujoco.mj_resetDataKeyframe(model, data, model.key("harness_initial").id)
    for kind, name, value in initial:
        if kind == "joint":
            data.joint(name).qpos[:] = value
        else:
            data.actuator(name).ctrl[:] = value
    mujoco.mj_forward(model, data)
    return spec, model, data


def load_scene(scene: str | Path, robots: str | Path | None = None):
    """Convenience runtime API returning (model, data)."""
    _, model, data = compose_scene(scene, robots)
    return model, data
