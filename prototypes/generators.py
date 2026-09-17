"""P4 pilot: a fixed box-family route and two agent-authored category generators.

Not a 12-category benchmark. Proportions/joint limits are declared experimental
design choices; input envelopes are test specifications, not claimed product CAD.
"""
import math
import trimesh

from .physics import box, part


def joint(name, child, pos, axis, limits, kind='hinge'):
    return dict(name=name, parent='root', child=child, pos=pos, rpy=[0, 0, 0],
                axis=axis, range=limits, type=kind)


def fixed_box(category, size, thickness):
    if category not in ('base_cabinet', 'drawer_unit'):
        raise ValueError('G1 domain rejection: not a supported box category')
    w, d, h = size
    t = thickness
    if min(size) <= 6*t:
        raise ValueError('Envelope too small for the specified wall thickness')
    g = t/5
    parts = []
    for side in (-1, 1):
        parts.append(part('root', f'side{side}', box([t, d, h], [side*(w-t)/2, 0, h/2])))
    parts.extend([
        part('root', 'bottom', box([w-2*t, d, t], [0, 0, t/2])),
        part('root', 'top', box([w-2*t, d, t], [0, 0, h-t/2])),
        part('root', 'back', box([w-2*t, t, h-2*t], [0, (d-t)/2, h/2])),
    ])
    if category == 'base_cabinet':
        dw = w-2*t-2*g
        parts.append(part('moving', 'door', box([dw, t, h-2*t-2*g], [dw/2, t/2, h/2])))
        joints = [joint('door', 'moving', [-w/2+t+g, -d/2, 0], [0, 0, -1], [0, math.pi/2])]
    else:
        # A single open drawer tray with independent panel collision primitives.
        dw, dd, dh = w-2*t-2*g, d-t-g, h-2*t-2*g
        parts.extend([
            part('moving', 'drawer_bottom', box([dw, dd, t], [0, -t/2, t+g+t/2])),
            part('moving', 'drawer_front', box([dw, t, dh-t], [0, -(d-t)/2, h/2+t/2])),
        ])
        joints = [joint('drawer', 'moving', [0, 0, 0], [0, -1, 0], [0, d/2], 'slide')]
    return parts, joints


def authored(category, size, revision=0):
    """Two minimal G2 candidates authored by Codex in this session.

    Variant 0's lid is a controlled oversized-envelope negative example;
    variant 1 puts its pivot on the closed-state support face.
    """
    w, d, h = size
    t = min(size)/20
    if category == 'hinged_lid_bin':
        parts = [part('root', 'bottom', box([w, d, t], [0, 0, t/2]))]
        for sign in (-1, 1):
            parts.append(part('root', f'side{sign}', box([t, d, h-2*t], [sign*(w-t)/2, 0, h/2])))
            parts.append(part('root', f'end{sign}', box([w-2*t, t, h-2*t], [0, sign*(d-t)/2, h/2])))
        # Local lid extends toward -Y. Positive rotation opens upward.
        parts.append(part('moving', 'lid', box([w, d, t], [0, -d/2, t/2])))
        z = h if revision == 0 else h-t
        return parts, [joint('lid', 'moving', [0, d/2, z], [-1, 0, 0], [0, math.pi/2])]
    if category == 'swing_arm_faucet':
        # A simple swivel-arm proxy with a vertical hinge, not detailed faucet CAD.
        stem = trimesh.creation.cylinder(radius=min(w, d)/8, height=h-t, sections=24)
        stem.apply_translation([0, 0, (h-t)/2])
        parts = [part('root', 'stem', stem),
                 part('moving', 'arm', box([w, t, t], [0, 0, 0]))]
        # Small base reaches the requested depth: simplified fixture, not faucet CAD.
        parts.append(part('root', 'mount', box([t, d, t], [0, 0, t/2])))
        return parts, [joint('swing', 'moving', [0, 0, h-t/2], [0, 0, 1], [-math.pi/2, math.pi/2])]
    raise ValueError(f'No authored pilot generator for {category}')
