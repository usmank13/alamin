"""The replay's reading of events.jsonl: trace entries, who drives when, playback pacing."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import replay_video as rv
sys.path.pop(0)

EVENTS = [
    dict(t=0.0, kind='operator', text='weed the plot'),
    dict(t=1.0, kind='thinking', text='plan  first'),
    dict(t=2.0, kind='tool', text='robot_state({})'),
    dict(t=3.0, kind='tool', text='drive_toward({"goal": "the road"})'),
    dict(t=5.0, kind='nav_source', text='NAV_MODEL'),
    dict(t=9.0, kind='nav_model', steer_source='ROWS'),
    dict(t=30.0, kind='nav_model', steer_source='TURN'),
    dict(t=60.0, kind='nav_source', text='NONE'),
]


def test_trace_drivers_schedule():
    assert rv.trace(EVENTS) == [(0.0, 'operator', 'weed the plot'), (1.0, 'think', 'plan first'),
                                (3.0, 'tool', 'drive_toward\tgoal="the road"'), (5.0, 'handoff', 'NAV MODEL'),
                                (30.0, 'event', 'row-end turn'), (60.0, 'handoff', 'STOPPED')]
    assert rv.drivers(EVENTS, 70.0) == [(0.0, 5.0, 'AGENT'), (5.0, 60.0, 'NAV_MODEL'), (60.0, 70.0, 'NONE')]
    times = rv.schedule(0, 70, [1.0, 5.0, 60.0], fast=8, slow=2)
    step = lambda t: next(b - a for a, b in zip(times, times[1:]) if a >= t)
    assert abs(step(3.0) - 2 / rv.FPS) < 1e-9 and abs(step(20.0) - 8 / rv.FPS) < 1e-9 and times[-1] < 70
