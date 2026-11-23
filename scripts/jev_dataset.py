#!/usr/bin/env python3
"""Labeled wakes from past agent-nav runs, for tuning Jev prompts offline.

    scripts/jev_dataset.py outputs/jev_eval/wakes.jsonl outputs/plots* outputs/map* ...

One line per wait_for_event return: the event, the [world] digest, the
mission, the front frame taken at the wake (while the agent drove) and what
the reasoning model did next, as a label:
  noop     nothing but passive tools, then waited again
  drive    pushed a forward leg (drive_toward / push_waypoints)
  look     scanned, asked Gemini about the view or turned before any leg
  back_up  pushed a reverse leg
  hand_off changed who drives, stopped, or ended its turn
The label is what the model did, not ground truth of what was right.
"""
import ast
import json
import re
import sys
from pathlib import Path

PASSIVE = {'wait_for_event', 'remember', 'send_message', 'robot_state', 'memory_read', 'memory_write', 'grab_frame'}
LOOK = {'scan', 'gemini_describe', 'gemini_points', 'elevation_view', 'turn', 'pixel_to_ground'}
HAND_OFF = {'set_nav_source', 'cancel_nav', 'stop', 'disable_agent', 'record_field_map', 'capture_object', 'capture_area'}


def label(tools: list[str], ended: bool) -> str:
    for text in tools:
        name = text.split('(')[0]
        if name in LOOK:
            return 'look'
        if name in HAND_OFF:
            return 'hand_off'
        if name == 'drive_toward':
            return 'drive'
        if name == 'push_waypoints':
            return 'back_up' if '"reverse": true' in text else 'drive'
    return 'hand_off' if ended else 'noop'


def wakes(run: Path):
    frames = sorted((p.stat().st_mtime, p) for p in (run / 'frames').glob('*_front.jpg')) if (run / 'frames').is_dir() else []
    mission, out, cur, leg = '', [], None, ''
    for line in open(run / 'events.jsonl'):
        e = json.loads(line)
        if e['kind'] == 'tool' and e['text'].split('(')[0] in ('drive_toward', 'push_waypoints'):
            leg = e['text']  # the leg under way at the next wake
        if e['kind'] in ('thinking', 'says') and cur is not None:
            cur['thinking'] = (cur.get('thinking', '') + ' ' + e['text'])[:1500]
        if e['kind'] == 'operator' and not mission:
            mission = e['text']
        elif e['kind'] == 'result' and e['text'].startswith('wait_for_event -> '):
            body = e['text'][len('wait_for_event -> '):]
            driving = body.startswith('[{')
            if driving:  # [text, image, snapshot] printed as a python list, cut short
                quoted = re.match(r"\[\{'type': 'text', 'text': ('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\")", body)
                if not quoted:
                    cur = None
                    continue
                body = ast.literal_eval(quoted.group(1))
            head, _, digest = body.partition('\n')
            try:
                event = json.loads(head)
            except ValueError:
                cur = None
                continue
            frame = None
            if driving and frames:  # the frame grabbed for this wake: the last one saved before it was logged
                before = [p for m, p in frames if m <= e['wall'] + 0.5]
                frame = str(before[-1]) if before else None
            cur = dict(run=run.name, t=round(e['t'], 1), event=event, digest=digest, mission=mission,
                       driving=driving, frame=frame, leg=leg, tools=[], ended=False)
            out.append(cur)
        elif e['kind'] == 'tool' and cur is not None:
            if e['text'].startswith('wait_for_event'):
                cur = None
            else:
                cur['tools'].append(e['text'])
        elif e['kind'] == 'agent_final' and cur is not None:
            cur['ended'] = True
    for w in out:
        w['label'] = label(w['tools'], w.pop('ended'))
    return out


if __name__ == '__main__':
    rows = [w for run in sys.argv[2:] if (Path(run) / 'events.jsonl').exists() for w in wakes(Path(run))]
    Path(sys.argv[1]).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    counts = {}
    for r in rows:
        key = (r['event'].get('event'), r['label'])
        counts[key] = counts.get(key, 0) + 1
    for key in sorted(counts):
        print(f'{key[0]:14} {key[1]:9} {counts[key]}')
    print(len(rows), 'wakes,', sum(bool(r['frame']) for r in rows), 'with a frame, from', len({r['run'] for r in rows}), 'runs')
