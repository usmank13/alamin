from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('trimesh')
pytest.importorskip('coacd')

from prototypes.generators import authored, fixed_box
from prototypes.microchecks import extract_triplet, run
from prototypes.p5 import scaled, read_urdf, decompose
from prototypes.physics import box, part, emit, validate


def test_fixed_route_rejects_unsupported_category():
    with pytest.raises(ValueError, match='domain rejection'):
        fixed_box('valve', [1, 1, 1], .02)


def test_uniform_and_protected_scaling_and_slide_limits():
    parts = [part('root', 'handle', box([.2, .06, .04], [.5, 0, 0]))]
    joints = [dict(type='slide', pos=[1, 0, 0], range=[0, .2])]
    uniform, uj, metrics = scaled(parts, joints, 1.4, 'G3')
    protected, pj, _ = scaled(parts, joints, 1.4, 'G4')
    np.testing.assert_allclose(uniform[0]['mesh'].extents, [.28, .084, .056])
    np.testing.assert_allclose(protected[0]['mesh'].extents, [.2, .06, .04])
    assert uj[0]['range'][1] == pytest.approx(.28)
    assert pj[0]['pos'][0] == pytest.approx(1.4)
    assert not metrics[0]['under_80mm']
    np.testing.assert_allclose(parts[0]['mesh'].extents, [.2, .06, .04])


def test_validator_rejects_collision_not_just_compile(tmp_path):
    parts, joints = fixed_box('base_cabinet', [1, 1, 1], .02)
    # Add an obstacle inside the moving door, not attached to it.
    parts.append(part('root', 'obstacle', box([.3, .3, .3], [0, -.5, .5])))
    result = validate(emit(parts, joints, tmp_path))
    assert result['compiles']
    assert not result['passed']
    assert result['settle_max_penetration'] > .002 or result['sweep'][0]['max_penetration'] > .002


def test_quote_membership_is_insufficient():
    assert run()['verbatim_only']['accepted']
    np.testing.assert_allclose(extract_triplet('30x24x30 in', '30x24x30 in', 'in'), [.762, .6096, .762])
    with pytest.raises(ValueError):
        extract_triplet('30x24x30 in', '30x24x30 in', 'cm')


def test_rotated_urdf_geometry_and_unsupported_joint(tmp_path):
    path = tmp_path/'test.urdf'
    path.write_text('''<robot name="test"><link name="root"><collision>
      <origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>
      <geometry><box size=".1 .2 .3"/></geometry></collision></link></robot>''')
    parts, _ = read_urdf(path)
    np.testing.assert_allclose(parts[0]['mesh'].extents, [.2, .1, .3])
    np.testing.assert_allclose(parts[0]['mesh'].bounds.mean(axis=0), [1, 2, 3])
    hulls = decompose(parts[0]['mesh'], tmp_path/'cache')
    assert len(hulls) == 1
    assert len(decompose(parts[0]['mesh'], tmp_path/'cache')) == 1
