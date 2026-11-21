from scene_pipeline.rhizome_mission_check import Arrival


def test_mission_requires_movement_then_continuous_fresh_odometry_near_goal():
    arrival = Arrival((0, 0))

    def sample(stamp, x, speed):
        return arrival.observe(stamp, {'pose': {'x': x, 'y': 0}, 'twist': {'v': speed, 'w': 0}})

    assert not sample(0, 0, 0)
    arrival.target = (1.2, 0)
    assert not sample(1, 0.2, 0.15)
    for n in range(15):
        assert not sample(2 + n/10, 0.3, 0)
    assert not sample(4, 1.1, 0)
    assert not sample(6, 1.1, 0)
    for n in range(10):
        assert not sample(6 + n/10, 1.1, 0)
    assert sample(7, 1.1, 0)
