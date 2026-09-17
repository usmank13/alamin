"""Simulator ground truth, not noisy sensor observations or semantic labels."""
import mujoco
import numpy as np


def snapshot(model, data):
    """Return JSON-compatible poses, per-geom conservative bounds and joint state.

    Call mj_forward first after stepping or changing state. IDs are model-local.
    Geometry bounds enclose transformed local AABBs; they are not object bounds.
    """
    bodies = [dict(id=i, name=model.body(i).name,
                   parent_id=int(model.body_parentid[i]),
                   position=data.xpos[i].tolist(), quaternion_wxyz=data.xquat[i].tolist(),
                   mass=float(model.body_mass[i]),
                   principal_inertia=model.body_inertia[i].tolist())
              for i in range(model.nbody)]
    geoms = []
    for i in range(model.ngeom):
        bounds = None
        if int(model.geom_type[i]) != int(mujoco.mjtGeom.mjGEOM_PLANE):
            rotation = data.geom_xmat[i].reshape(3, 3)
            center = data.geom_xpos[i] + rotation @ model.geom_aabb[i, :3]
            half = np.abs(rotation) @ model.geom_aabb[i, 3:]
            bounds = dict(min=(center-half).tolist(), max=(center+half).tolist())
        geoms.append(dict(id=i, name=model.geom(i).name, body_id=int(model.geom_bodyid[i]),
                          type=mujoco.mjtGeom(int(model.geom_type[i])).name,
                          world_aabb=bounds, contype=int(model.geom_contype[i]),
                          conaffinity=int(model.geom_conaffinity[i])))
    joints = []
    for i in range(model.njnt):
        qstart = int(model.jnt_qposadr[i])
        vstart = int(model.jnt_dofadr[i])
        qend = int(model.jnt_qposadr[i+1]) if i+1 < model.njnt else model.nq
        vend = int(model.jnt_dofadr[i+1]) if i+1 < model.njnt else model.nv
        joints.append(dict(id=i, name=model.joint(i).name, body_id=int(model.jnt_bodyid[i]),
                           type=mujoco.mjtJoint(int(model.jnt_type[i])).name,
                           limited=bool(model.jnt_limited[i]), range=model.jnt_range[i].tolist(),
                           qpos=data.qpos[qstart:qend].tolist(), qvel=data.qvel[vstart:vend].tolist()))
    return dict(schema_version=1, time=float(data.time), units="metres, kilograms, seconds, radians",
                bodies=bodies, geoms=geoms, joints=joints,
                actuators=[dict(id=i, name=model.actuator(i).name, control=float(data.ctrl[i]))
                           for i in range(model.nu)])


def contacts(data):
    """Current contacts and signed separation; negative distance means penetration."""
    return [dict(geom1=int(c.geom1), geom2=int(c.geom2), distance=float(c.dist))
            for c in data.contact]
