#!/usr/bin/env python3
"""Score agent-nav runs for A/B comparison: adds a `score` block to each
run's report.json and prints one table row per run.

    scripts/score_run.py outputs/plots10 outputs/map5 ...

Model calls and tokens come from the Agent SDK's own session file for the
run (the SDK runs with cwd = the run dir); cost from the SDK's ResultMessage.
"""
import glob
import json
import os
import math

import numpy as np
import re
import sys
from pathlib import Path

# Tools that change nothing: a wake followed only by these was not needed.
PASSIVE = {'wait_for_event', 'remember', 'send_message', 'robot_state', 'memory_read', 'memory_write'}


def usage(run: Path) -> dict:
    """Model calls and tokens, one count per API message id."""
    # The SDK names its project dir after the cwd it saw: symlinked or resolved.
    slugs = {re.sub(r'[^A-Za-z0-9]', '-', str(p)) for p in (run.absolute(), run.resolve())}
    by_id = {}
    paths = [f for s in slugs for f in glob.glob(str(Path.home() / '.claude/projects' / s / '*.jsonl'))]
    # A rerun under the same name leaves the earlier attempt's session beside it.
    for path in sorted(paths, key=os.path.getmtime)[-1:]:
        for line in open(path):
            msg = json.loads(line).get('message') or {}
            if msg.get('id') and msg.get('usage'):
                by_id[msg['id']] = msg['usage']  # the last line of a message has its final counts
    total = lambda key: sum(u.get(key) or 0 for u in by_id.values())
    return {'llm_calls': len(by_id), 'input_tokens': total('input_tokens'), 'cache_read_tokens': total('cache_read_input_tokens'),
            'cache_write_tokens': total('cache_creation_input_tokens'), 'output_tokens': total('output_tokens')}


def wakes(events: list) -> dict:
    """What the model did after each wait_for_event return, and what the wake
    gate waited past without it."""
    counts = {'wakes': 0, 'wakes_noop': 0, 'gate_waited': 0, 'model_wall_s': []}
    after = None  # tools called since the last wake
    last_wall = None

    def close():
        if after is not None:
            counts['wakes'] += 1
            counts['wakes_noop'] += all(t in PASSIVE for t in after)

    for e in events:
        if e['kind'] == 'result' and e['text'].startswith('wait_for_event'):
            close()
            after, last_wall = [], e['wall']
        elif e['kind'] == 'trace' and re.match(r'gate \S+ → wait', e['text']):
            counts['gate_waited'] += 1
        elif e['kind'] == 'tool' and after is not None:
            if not after:
                counts['model_wall_s'].append(e['wall'] - last_wall)
            name = e['text'].split('(')[0]
            if name == 'wait_for_event':
                continue
            after.append(name)
    close()
    lat = sorted(counts.pop('model_wall_s'))
    counts['model_wall_s_median'] = round(lat[len(lat) // 2], 1) if lat else None
    return counts


def map_error(run: Path, scene: Path) -> dict | None:
    """Along-row distance from each saved lane end to the end of the nearest
    true row (scene manifest)."""
    saved = run / 'field_map.json'
    if not saved.exists():
        return None
    rows = [r['endpoints_m'] for r in json.loads((scene / 'manifest.json').read_text())['rows']]
    errors = []
    for lane in json.loads(saved.read_text())['lanes']:
        for x, y in lane:
            def along(row):
                (ax, ay), (bx, by) = row
                ux, uy = (bx - ax) / math.dist(row[0], row[1]), (by - ay) / math.dist(row[0], row[1])
                s, across = (x - ax) * ux + (y - ay) * uy, abs((x - ax) * uy - (y - ay) * ux)
                return across, min(abs(s), abs(s - math.dist(row[0], row[1])))
            errors.append(min(along(r) for r in rows)[1])
    return {'lane_ends': len(errors), 'end_error_m_mean': round(sum(errors) / len(errors), 2),
            'end_error_m_max': round(max(errors), 2)}


def score(run: Path) -> dict:
    report = json.loads((run / 'report.json').read_text())
    events = [json.loads(line) for line in open(run / 'events.jsonl')]
    finals = [e for e in events if e['kind'] == 'agent_final']
    costs = re.findall(r'total_cost_usd=([0-9.]+)', (run / 'conversation.json').read_text())
    out = {
        # The agent reported done; a later Ctrl-C does not undo that.
        'completed': bool(finals) and report.get('failure') in (None, 'interrupted'),
        'completed_sim_s': round(finals[0]['t']) if finals else None,
        'cost_usd': round(float(costs[-1]), 2) if costs else None,
        **usage(run), **wakes(events),
    }
    if finals and (run / 'positions.npy').exists():
        # Share of the time up to the agent's "done" that the rover was moving;
        # the report's moving_pct also counts whatever idle time followed.
        xy = np.load(run / 'positions.npy')[:int(finals[0]['t'] * 100) + 1, :2]  # 100 Hz
        speed = np.linalg.norm(np.diff(xy, axis=0), axis=1) * 100
        out['moving_pct_to_done'] = round(100 * float((np.convolve(speed, np.ones(50) / 50, 'same') > 0.05).mean()), 1)
    scene = Path(json.loads((run / 'run.json').read_text())['scene'])
    if (err := map_error(run, scene)):
        out['map'] = err
    report['score'] = out
    (run / 'report.json').write_text(json.dumps(report, indent=1))
    return {**out, 'moving_pct': report.get('moving_pct'), 'failure': report.get('failure'),
            'weeds_hit': [p['weeds_hit'] for p in report.get('weeding', {}).get('plots', {}).values()]}


if __name__ == '__main__':
    for arg in sys.argv[1:]:
        print(Path(arg).name, json.dumps(score(Path(arg))))
