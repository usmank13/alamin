from types import SimpleNamespace

import pytest

from scene_pipeline.rhizome import validate_config
from scene_pipeline.rhizome_worker import Runtime


@pytest.mark.parametrize("approach, targets", [(0, [3, 0, 3, 0]), (4, [4, 7, 4, 7, 4])])
def test_shuttle_route(approach, targets):
    route = Runtime.__new__(Runtime)
    route.config = {'shuttle': [3., 2]}
    route.course_distance, route.course_endpoints = approach, None
    route.course_leg, route.leg_index, route.has_reversed = 'out', 0, False
    route.node = SimpleNamespace(config_={'arrival_distance': .25})
    route.p = SimpleNamespace(WaypointsMsg=dict)
    pose = dict(x=0, y=0, qw=1, qx=0, qy=0, qz=0)
    for index, target in enumerate(targets):
        route.update_course(1 + index, pose)
        expected = 'REVERSE' if target < pose['x'] else 'FORWARD'
        goal = route.goals['waypoints'][-1]
        assert goal['point']['x'] == target
        assert goal['dir'] == expected
        assert route.leg_index == index
        pose['x'] = target
    route.update_course(10, pose)
    assert route.course_leg == 'done'
    assert route.goals == {}


@pytest.mark.parametrize("option", [[0, 2], [3., 0], [3., 2.5], [float('nan'), 1]])
def test_invalid_shuttle(option):
    with pytest.raises(ValueError, match='shuttle'):
        validate_config({'shuttle': option})
