"""Dress the Gen 3 sim robot in rhizome's production meshes and colours.

Only visual geoms change: bodies, joints, collision and mass stay as built, so
recorded qpos replays on a reskinned model.
"""
from pathlib import Path
import struct
import sys

import mujoco
import numpy as np

# rhizome src/robot_description/viz/robot.js MATERIALS with the powder-coated
# frame: colour, Phong specular colour, Phong shininess.
MATERIALS = {
    'frame': (0x2a2b2f, 0x333333, 25), 'solar': (0x141a26, 0x111111, 80), 'starlink': (0xf2f2f0, 0x111111, 30),
    'metal': (0xb4b9bf, 0xdddddd, 70), 'motor': (0x101010, 0x333333, 40), 'tyre': (0x161616, 0x111111, 4),
    'hub': (0x2e3034, 0x111111, 20), 'enclosure': (0xe6e7e9, 0x111111, 30), 'battery': (0xdfe0e2, 0x111111, 30),
    'camera': (0x121212, 0x333333, 40), 'tool': (0xb9bec5, 0xeeeeee, 90),
}
ARM_FRAME = np.diag([-1., -1., 1.])  # rhizome's arm frame in the robot frame: x rearward along the stowed tube


def _rgb(colour):
    return [(colour >> 16 & 255) / 255, (colour >> 8 & 255) / 255, (colour & 255) / 255]


def reskin(spec, rhizome_root, prefix=''):
    """Replace the robot's `*_visual` geoms in `spec`. `prefix` is the robot's
    name prefix in a composed scene ('element/')."""
    import yaml
    root = Path(rhizome_root)
    sys.path.insert(0, str(root / 'src'))
    try:
        from robot_description import robot_geometry as geo
        from robot_description.resolve import resolve
    finally:
        sys.path.pop(0)
    resolved = resolve(yaml.safe_load((root / 'src/robot_description/robot_cad.yml').read_text()),
                       yaml.safe_load((root / 'src/base_control/tools.yml').read_text()), 'gen-3.0', 'raptor_4x_30in_gen3')
    meshes = root / 'src/robot_description/viz/meshes'

    def read(name):
        """read_mesh, with each material group's triangle count (it keeps only the names)."""
        mesh, raw = geo.read_mesh(meshes / name), (meshes / name).read_bytes()
        return mesh, [struct.unpack_from('<I', raw, 12 + 16 * i)[0] for i in range(len(mesh.materials))]

    def posed(part, R, t, scale=(1., 1., 1.)):
        mesh, counts = part
        return geo._posed(mesh, R, t, scale), counts

    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    chassis = model.body(prefix + 'chassis').id
    Rc, pc = data.xmat[chassis].reshape(3, 3), data.xpos[chassis]

    def frame(name):
        """Body pose in the chassis frame, which is rhizome's robot frame."""
        b = model.body(prefix + name).id
        return Rc.T @ data.xmat[b].reshape(3, 3), Rc.T @ (data.xpos[b] - pc)

    def visual_box(name):
        g = model.geom(prefix + name + '_visual').id
        m = model.geom_dataid[g]
        v = model.mesh_vert[model.mesh_vertadr[m]:model.mesh_vertadr[m] + model.mesh_vertnum[m]]
        v = (v @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - pc) @ Rc
        return v.min(0), v.max(0)

    placed = []  # (body, (Mesh with vertices in the robot frame, triangles per material group))
    parts = geo.robot_parts(resolved)
    placed.append(('chassis', posed(read(parts[0].mesh), parts[0].R, parts[0].t)))
    for side, part in (('right', parts[1]), ('left', parts[2])):
        # The rockers' zero is the CAD export's pose, not level: turn the side to where its wheels are.
        pivot = frame('rocker_' + side)[1]
        d = frame('front_' + side)[1] - frame('rear_' + side)[1]
        a = np.arctan2(-d[2], d[0])
        tilt = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        mesh, counts = posed(read(part.mesh), part.R, part.t)
        placed.append(('rocker_' + side, (mesh._replace(vertices=(mesh.vertices - pivot) @ tilt.T + pivot), counts)))
        for wheel in ('front_' + side, 'rear_' + side):
            lo, hi = visual_box(wheel)
            radius = max(hi[0] - lo[0], hi[2] - lo[2]) / 2
            placed.append((wheel, posed(read('gen-3.0/wheel.bin'), part.R, (lo + hi) / 2, (radius, hi[1] - lo[1], radius))))
    arm = resolved.description.arms[0] if not callable(resolved.description.arms) else resolved.description.arms()[0]
    tube = resolved.description.arm_config(arm)['tube']
    outline = resolved.tool_catalog.entry(resolved.description.tool_type(arm)).tool
    blade = np.array([[p['x'], p['y'], p['z']] for p in outline])
    lo, hi = blade.min(0) - .0015, blade.max(0) + .0015
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    box = geo.Mesh(corners, np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]), ['tool']), [12]
    tubes = []
    for i in range(1, 5):
        yaw, pitch = f'arm_{i}_yaw', f'arm_{i}_pitch'
        origin = np.array([*frame(yaw)[1][:2], frame(pitch)[1][2]])  # where the yaw axis meets the pitch axis
        pose = geo.arm_pose(ARM_FRAME, origin, tube['length'], 0., 0.)
        placed += [('chassis', posed(read(geo.ARM_MESHES['bracket']), ARM_FRAME, origin)),
                   (yaw, posed(read(geo.ARM_MESHES['yaw']), pose.yaw_axes, origin)),
                   (pitch, posed(read(geo.ARM_MESHES['pitch']), pose.axes, origin)),
                   (pitch, posed(read(geo.ARM_MESHES['adapter']), pose.axes, pose.tube_end)),
                   (pitch, posed(box, pose.axes, pose.tube_end))]
        tubes.append((pitch, origin, pose.tube_end))

    old = [g for g in spec.geoms if g.name.startswith(prefix) and g.name.endswith('_visual')]
    old_meshes = {g.meshname for g in old}
    for g in old:
        spec.delete(g)
    for name in old_meshes:
        spec.delete(spec.mesh(name))
    for name, (colour, specular, shininess) in MATERIALS.items():
        spec.add_material(name=f'{prefix}robot_{name}', rgba=[*_rgb(colour), 1], specular=float(np.mean(_rgb(specular))),
                          shininess=shininess / 128)
    look = dict(group=2, contype=0, conaffinity=0, density=0)
    for n, (body, (mesh, counts)) in enumerate(placed):
        R, p = frame(body)
        vertices = (mesh.vertices - p) @ R  # into the body frame
        start = 0
        for material, count in zip(mesh.materials, counts):
            used, faces = np.unique(mesh.triangles[start:start + count], return_inverse=True)
            start += count
            key = f'{prefix}skin_{n}_{material}'
            spec.add_mesh(name=key, uservert=vertices[used].astype(np.float64).ravel(), userface=faces.ravel().astype(np.int32),
                          inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL, maxhullvert=16)
            spec.body(prefix + body).add_geom(name=key, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=key,
                                              material=f'{prefix}robot_{material}', **look)
    for n, (body, a, b) in enumerate(tubes):
        R, p = frame(body)
        spec.body(prefix + body).add_geom(name=f'{prefix}skin_tube_{n}', type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[tube['radius'], 0, 0],
                                          fromto=[*((a - p) @ R), *((b - p) @ R)], material=f'{prefix}robot_frame', **look)
    return spec

