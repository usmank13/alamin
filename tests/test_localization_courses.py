"""The collision and visible surfaces share the same course geometry."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from build_localization_courses import balance_rockers, build
from run_localization import score
sys.path.pop(0)


@pytest.mark.parametrize('course,expected', [
    ('flat', [0, 0, 0, 0, 0]),
    ('rolling', [0, .12, .12, .24, .18]),
    ('hazards', [0, -.4, .4, 0, 0]),
])
def test_course_surface(tmp_path, course, expected):
    robot = mujoco.MjSpec.from_string('<mujoco><worldbody><body name="chassis"><freejoint/>'
                                      '<geom type="sphere" size=".1"/></body></worldbody></mujoco>')
    robot.to_zip(str(tmp_path / 'robot.mjz'))
    profile = dict(wheels=[dict(center_m=[0., 0., 0.], radius_m=.1, half_width_m=.1)])
    (tmp_path / 'robot_profile.json').write_text(json.dumps(profile))
    output = tmp_path / course
    build(tmp_path, output, course)
    model = mujoco.MjSpec.from_zip(str(output / 'environment.mjz')).compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for (x, y), height in zip([(0, 0), (6, -.5), (6, .5), (12, 0), (40, 0)], expected):
        distance = mujoco.mj_ray(model, data, np.array([x, y, 3.]), np.array([0., 0., -1.]),
                                 None, 1, -1, np.empty(1, np.int32))
        assert 3. - distance == pytest.approx(height, abs=1e-6)

def test_relative_score_and_completion(tmp_path):
    rows = []
    for t, x, z, odom_z, leg in [(0, 0, 2., 0., 'out'), (1, 0, 1., 0., 'out'),
                                  (5, 4, 1.1, .3, 'back'), (10, .1, 1., .01, 'done')]:
        rows.append(dict(op='step', simulation_time=t, input=dict(position=[x, 0, z]),
                         output=dict(localization=dict(pose=dict(z=odom_z), leg=leg))))
    verdict = dict(blocked=True, grid_stop=True, violations=7)
    rows.append(dict(op='nav_infer', simulation_time=4.1, output=dict(
        prediction=dict(obstacle_fwd=verdict), violation_cells=dict(hole=[2, 0], obstacle=[5, 0]))))
    rows.sort(key=lambda row: row['simulation_time'])
    path = tmp_path / 'messages.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = score(path, 4, .25)
    assert result['out_and_back_completed']
    assert result['peak_abs_z_error_m'] == pytest.approx(.2)
    assert result['peak_z_error_time_s'] == result['turnaround_time_s'] == 5
    assert result['turnaround_z_error_m'] == pytest.approx(.2)
    assert result['final_z_error_m'] == pytest.approx(.01)
    assert result['obstacles']['fwd'] == dict(stop_ticks=1, grid_stop_ticks=1,
        first_stop_s=4.1, first_violation_s=4.1, first_violation_distance_m=0.,
        max_violation_cells=7, violation_cell_ticks=7, hole_cell_ticks=2, obstacle_cell_ticks=5)
    assert result['obstacles']['rev']['stop_ticks'] == 0
    assert not score(path, 40, .25)['out_and_back_completed']

def test_shuttle_leg_errors_and_violation_distance(tmp_path):
    rows = []
    for t, x, z_error, leg, index in [(1, 0, 0, 'out', 0), (2, 4, .1, 'out', 1),
                                     (3, 7, .2, 'back', 2), (4, 5, .25, 'back', 2),
                                     (5, 4, .3, 'done', 2)]:
        rows.append(dict(op='step', simulation_time=t, input=dict(position=[x, 0, 2]),
                         output=dict(localization=dict(pose=dict(z=z_error), leg=leg, leg_index=index))))
        if t == 4:
            rows.append(dict(op='nav_infer', simulation_time=t, output=dict(
                prediction=dict(obstacle_rev=dict(blocked=False, grid_stop=False, violations=1)))))
    path = tmp_path / 'messages.jsonl'
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    result = score(path, 4, .25, [3, 1])
    assert result['out_and_back_completed']
    assert not score(path, 4, .25, [3, 2])['out_and_back_completed']
    assert result['travel_distance_m'] == 10
    assert result['first_violation_s'] == 4
    assert result['first_violation_distance_m'] == result['obstacles']['rev']['first_violation_distance_m'] == 9
    assert result['obstacles']['rev']['first_stop_s'] is None
    assert result['obstacles']['fwd']['first_violation_s'] is None
    assert [leg['index'] for leg in result['legs']] == [0, 1, 2]
    assert [leg['end_z_error_m'] for leg in result['legs']] == pytest.approx([0, .1, .25])
    assert result['legs'][-1]['peak_abs_z_error_m'] == pytest.approx(.25)


def test_spring_preload_balances_only_sprung_mass():
    robot = mujoco.MjSpec.from_string("""<mujoco>
      <compiler angle="radian"/><worldbody><body name="chassis"><freejoint/>
        <geom type="box" size=".25 .25 .1" mass="10" pos="-.1 0 0"/>
        <body name="left" pos="0 .4 0"><joint name="left" axis="0 -1 0" stiffness="100" springref=".01"/>
          <geom type="sphere" size=".1" pos=".25 0 0" mass="2"/>
        </body>
        <body name="right" pos="0 -.4 0"><joint name="right" axis="0 1 0" stiffness="100"/>
          <geom type="sphere" size=".1" pos=".25 0 0" mass="2"/>
        </body>
      </body></worldbody></mujoco>""")
    profile = dict(suspension=[dict(joint='left'), dict(joint='right')])
    balance_rockers(robot, profile)
    trim = 10 * .1 * 9.81 / 200
    assert robot.joint('left').springref == pytest.approx(.01 + trim)
    assert robot.joint('right').springref == pytest.approx(-trim)


def test_loaded_course_pitch_steps_and_contact():
    from scene_pipeline.agriculture import compose_robot
    from scene_pipeline.agriculture_drive import wheel_targets
    from scene_pipeline.localization_sensors import ChassisPitch, pitch_at
    from sim_harness.scene import _read_spec

    scene = Path(__file__).resolve().parents[1]/'outputs/localization_courses_balanced/flat'
    if not scene.exists():
        pytest.skip('requires generated balanced Gen3 course')
    profile = json.loads((scene/'robot_profile.json').read_text())
    bounds = json.loads((scene/'manifest.json').read_text())['terrain']['bounds_m']
    _, model, data = compose_robot(_read_spec(scene/'environment.mjz'), _read_spec(scene/'robot.mjz'),
                                   profile, dict(position_m=[0., 0.]), bounds)
    chassis = model.body('element/chassis').id
    control = ChassisPitch(model, data, profile, chassis)
    wheel_names = ['element/' + wheel['id'] for wheel in profile['wheels']]
    servos = [model.actuator(name).id for name in wheel_names]
    wheels = {model.geom(name + '_collision').id for name in wheel_names}
    tools = set()
    for arm in profile['arms']:
        data.ctrl[model.actuator('element/' + arm['pitch_joint']).id] = 0.
        tools.add(model.geom('element/' + arm['pitch_joint'] + '_tool_collision').id)
    terrain = model.geom('terrain').id
    for tick in range(6000):
        t = tick * model.opt.timestep
        command = pitch_at([[0, 1.5], [4, -1.5], [8, 1.5]], t)
        if tick % 5 == 0:
            control.apply(t, command)
        # Smooth velocity changes let suspension contact be tested through both pitch steps.
        speed = np.interp(t, [0., 1., 2., 5., 6.5], [0., 0., .45, .45, -.45])
        data.ctrl[servos] = wheel_targets(profile, speed, 0.)[0]
        mujoco.mj_step(model, data)
        touching = {c.geom2 if c.geom1 == terrain else c.geom1 for c in data.contact
                    if terrain in (c.geom1, c.geom2)}
        if t > .5:
            assert wheels <= touching
            assert not tools & touching
        if 2.5 < t < 4 or 4.99 < t < 5 or 6.99 < t < 8 or 8.99 < t:
            assert np.degrees(np.arcsin(-data.xmat[chassis, 6])) == pytest.approx(command, abs=.1)
