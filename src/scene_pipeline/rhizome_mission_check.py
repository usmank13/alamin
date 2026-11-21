"""Exercise the native agent's point-to-point navigation through operator services."""
import json
import math
from pathlib import Path
from queue import Empty, Queue
import threading
import time


class Arrival:
    """Require fresh odometry across the stop interval, near the last commanded goal."""
    def __init__(self, start):
        self.start = start
        self.target = None
        self.moved = False
        self.still_since = None
        self.last_stamp = None
        self.distance = self.remaining = self.speed = None
        self.stop_seconds = 0.0

    def observe(self, stamp, msg):
        if self.target is None:
            return False
        pose = msg['pose']['x'], msg['pose']['y']
        self.distance = math.dist(self.start, pose)
        self.remaining = math.dist(self.target, pose)
        self.speed = abs(msg['twist']['v'])
        yaw_rate = abs(msg['twist']['w'])
        if self.last_stamp is not None and stamp <= self.last_stamp:
            return False
        if self.last_stamp is None or stamp - self.last_stamp > 0.25:
            self.still_since = None
        self.last_stamp = stamp
        self.moved |= self.distance > 0.1 and self.speed > 0.05
        if self.moved and self.remaining <= 0.35 and self.speed < 0.04 and yaw_rate < 0.05:
            if self.still_since is None:
                self.still_since = stamp
            self.stop_seconds = stamp - self.still_since
            return self.stop_seconds >= 1
        self.still_since = None
        self.stop_seconds = 0.0
        return False


def run(config):
    from pyzome import core
    from .rhizome_agent import save

    output = Path(config['output'])
    result = {'passed': False, 'mission_submitted': False, 'agent_goal_observed': False}
    messages = Queue()
    class Observer(core.BaseNode):
        def Tick(self):
            pass

    node = Observer(core.NodeConfig('alamin_mission_check'))
    topics = ('/agent/enable', '/agent/transcript', '/nav/source', '/drive_mux/state',
              '/nav/goals', '/odom', 'local_control/health', 'drive_control/health')
    for topic in topics:
        node.Subscribe(topic, None, lambda msg, key=topic: messages.put((key, json.loads(str(msg)))))
    thread = threading.Thread(target=node.Run)
    thread.start()
    latest = {}
    arrival = None

    def receive(until):
        try:
            topic, msg = messages.get(timeout=max(0, until - time.monotonic()))
        except Empty:
            raise TimeoutError('Timed out waiting for native mission evidence') from None
        if time.monotonic() > until:
            raise TimeoutError('Native mission deadline exceeded')
        latest[topic] = msg
        return topic, msg

    def request(service, message, expected):
        replies = [json.loads(str(body)) for _, body in core.sendRequest(service, message, 3)
                   if body is not None]
        if not any(all(reply.get(k) == v for k, v in expected.items()) for reply in replies):
            raise RuntimeError(f'{service} did not confirm {expected}: {replies}')

    try:
        (output/'clock-start').touch()
        ready = time.monotonic() + 40
        health = ('local_control/health', 'drive_control/health')
        while ('/odom' not in latest or '/nav/source' not in latest or '/agent/enable' not in latest
               or any(t not in latest or latest[t]['stale_topics'] or latest[t]['invariant_failures']
                      for t in health)):
            receive(ready)
        request('agent-enable', core.AgentEnableMsg(dict(enabled=True)), {'str': 'enabled = True'})
        request('set-nav-source', core.NavSourceMsg(dict(requested='AGENT')), {'requested': 'AGENT'})
        request('set-drive-mux', core.MuxSelectMsg(dict(requested='AUTONOMY')), {'requested': 'AUTONOMY'})
        while (not latest['/agent/enable']['enabled'] or latest['/nav/source']['requested'] != 'AGENT'
               or latest.get('/drive_mux/state', {}).get('requested') != 'AUTONOMY'):
            receive(ready)
        start = latest['/odom']['pose']
        arrival = Arrival((start['x'], start['y']))
        mission = config['mission']
        submitted = core.now().seconds()
        request('agent-send', core.AgentChatMsg(dict(entry=dict(role='operator', source='portal', text=mission))),
                {'str': 'ok'})
        result['mission_submitted'] = True
        until = time.monotonic() + config['seconds']
        operator_seen = False
        while True:
            topic, msg = receive(until)
            timestamp = msg['header']['timestamp']
            stamp = timestamp['secs'] + timestamp['nsecs'] * 1e-9
            if stamp < submitted:
                continue
            if topic == '/agent/transcript':
                operator_seen |= any(e['role'] == 'operator' and e['text'] == mission and e['ts'] >= submitted
                                     for e in msg['entries'])
            elif topic == '/nav/goals' and msg.get('waypoints') and latest['/nav/source']['requested'] == 'AGENT':
                point = msg['waypoints'][-1]['point']
                if point['frame'] != 'odom':
                    raise ValueError('Point-to-point mission check expects an odom goal')
                target = point['x'], point['y']
                if arrival.target != target:
                    arrival.still_since = None
                arrival.target = target
                result.update(agent_goal_observed=True, goal_xy_m=target)
            elif topic == '/odom' and arrival.observe(stamp, msg) and operator_seen:
                result.update(passed=True, mission_transcript_verified=True)
                break
    except BaseException as error:
        result['error'] = str(error)
        raise
    finally:
        if arrival:
            result.update(travel_m=arrival.distance, distance_to_goal_m=arrival.remaining,
                          final_speed_mps=arrival.speed, stopped_seconds=arrival.stop_seconds)
        node.shutdown()
        thread.join()
        save(output/'control_check.json', result)
