#!/usr/bin/env python3
"""Jev on one graph transition: a driving leg is ending; keep going toward
the same goal, treat it as reached, or hand the step to the reasoning model?

    OPENROUTER_API_KEY=... RHIZOME_ROOT=... scripts/jev_legs.py [-v]

Wakes: approaching/arrived with a frame, from outputs/jev_eval/wakes.jsonl.
Label from what the reasoning model did: a forward leg = continue; handed
driving over or ended = reached; looked around, backed up, or did nothing
after thinking = supervisor.
"""
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(os.environ['RHIZOME_ROOT']).expanduser() / 'src'))
from ai_agent import decide  # noqa: E402

CFG = {'url': 'https://openrouter.ai/api/alpha/decisions', 'model': 'typesafe/jev-1.14',
       'key': os.environ.get('OPENROUTER_API_KEY'), 'timeout_s': 20}
TRUTH = {'drive': 'continue', 'hand_off': 'reached', 'look': 'supervisor', 'back_up': 'supervisor', 'noop': 'supervisor'}


def goal(w):
    g = re.search(r'"goal": "([^"]*)"', w['leg'])
    return g.group(1) if g else 'drive to a point a few metres ahead'


def lines(w, *prefixes):
    return '\n'.join(line for line in w['digest'].split('\n') if line.startswith(prefixes))


CHOICE = {'type': 'choice', 'instructions': (
    'A farm rover is driving one leg of a route and that leg is about to end. The state gives the leg it was '
    'driving and the view from its front camera now. Which fits?'),
    'criteria': {
        'supervisor': ('Neither of the others clearly fits: what the leg was heading for is not in view, the view '
                       'does not match the leg, something is in the way, or it is unclear what comes next.'),
        'continue': ('Routine: what the leg was heading for is still ahead and the ground toward it is open, so '
                     'the next leg simply carries on the same way.'),
        'reached': 'The rover is now at what the leg was heading for; that leg\'s purpose is complete.'}}
NOULS = {
    'ahead': {'type': 'noul', 'instructions': (
        'The image is a farm rover\'s front camera. The state names what the rover is driving toward. Is that '
        'thing, or the path it names, visible ahead with open ground leading toward it?')},
    'at': {'type': 'noul', 'instructions': (
        'The image is a farm rover\'s front camera. The state names what the rover is driving toward. Is the rover '
        'already there: is that thing right in front of it or under it?')},
    'blocked': {'type': 'noul', 'instructions': (
        'The image is a farm rover\'s front camera. Is a large object (a tree trunk, a bush, a vehicle) close in '
        'front of the rover, filling a good part of the view?')},
}
STATES = {
    'leg': lambda w: f'The leg it was driving: {goal(w)}\nFront camera now:',
    'leg+mission': lambda w: (f'Mission: {w["mission"]}\n{lines(w, "plan", "recent")}\n'
                              f'The leg it was driving: {goal(w)}\nEvent: {w["event"]["event"]}\nFront camera now:'),
}

rows = [json.loads(line) for line in open('outputs/jev_eval/wakes.jsonl')]
legs = [w for w in rows if w['frame'] and w['event'].get('event') in ('approaching', 'arrived')]
verbose = '-v' in sys.argv


def run(state, questions):
    def one(w):
        return decide.ask(CFG, STATES[state](w), questions, Path(w['frame']).read_bytes())
    with ThreadPoolExecutor(8) as pool:
        return list(pool.map(one, legs))


for state in STATES:
    got = run(state, {'q': CHOICE})
    picks = [(g['q']['choice'], max(g['q']['probabilities'].values()), TRUTH[w['label']]) for g, w in zip(got, legs)]
    print(f"choice/{state:12} 3-way {sum(c == t for c, _, t in picks)}/{len(picks)}", end='  ')
    for th in (0.0, 0.8, 0.9):
        # "handled" = Jev confidently took a routine edge; right if the model also did that
        handled = [(c, t) for c, p, t in picks if c != 'supervisor' and p >= th]
        print(f"p>={th}: handles {len(handled)}, right {sum(c == t for c, t in handled)}", end='  ')
    print()
    if verbose:
        for (c, p, t), w in zip(picks, legs):
            if c != t:
                print(f"   {w['run']} {w['t']} jev={c} {p:.2f} model={t} ({w['label']}) leg: {goal(w)[:70]} | {w.get('thinking', '')[:150]}")

got = run('leg', NOULS)
for th in (0.8, 0.9):
    handled = right = 0
    for g, w in zip(got, legs):
        ahead, at, blocked = (g[k]['noul'] for k in ('ahead', 'at', 'blocked'))
        pick = 'continue' if ahead >= th and blocked <= 1 - th and at <= 0.5 else 'reached' if at >= th else 'supervisor'
        if pick != 'supervisor':
            handled += 1
            right += pick == TRUTH[w['label']]
    print(f"nouls/leg  p>={th}: handles {handled}, right {right}")
print('truth:', {t: sum(TRUTH[w['label']] == t for w in legs) for t in ('continue', 'reached', 'supervisor')})
