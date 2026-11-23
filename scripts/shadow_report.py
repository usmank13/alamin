#!/usr/bin/env python3
"""What the wake gate would have done in a shadow run (decision.shadow: true),
against what the reasoning model did at each wake.

    scripts/shadow_report.py outputs/shadow_map1 [-v]

In a shadow run every event wakes the model, and the rule's verdict and Jev's
probability for the expectation are logged first. "needed" = the model then
called something other than passive tools.
"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from score_run import PASSIVE  # noqa: E402

GATE = re.compile(r"gate (\w+) → \w+ \(rule (\w+); jev (.*): p=([\d.]+|None) \(shadow\)\)")

for run in [Path(a) for a in sys.argv[1:] if not a.startswith('-')]:
    rows, pending, cur = [], None, None
    for line in open(run / 'events.jsonl'):
        e = json.loads(line)
        if e['kind'] == 'trace' and (m := GATE.match(e['text'])):
            pending = dict(t=round(e['t']), event=m.group(1), rule=m.group(2), expect=m.group(3),
                           p=None if m.group(4) == 'None' else float(m.group(4)))
        elif e['kind'] == 'result' and e['text'].startswith('wait_for_event'):
            cur = None
            if pending:
                cur = {**pending, 'tools': []}
                rows.append(cur)
                pending = None
        elif e['kind'] == 'tool' and cur is not None:
            name = e['text'].split('(')[0]
            if name == 'wait_for_event':
                cur = None
            else:
                cur['tools'].append(name)
    for r in rows:
        r['needed'] = any(t not in PASSIVE for t in r['tools'])
    print(f"{run.name}: {len(rows)} wakes with a gate record, {sum(not r['needed'] for r in rows)} the model did not need")
    for th in (0.8, 0.9):
        tally = {}
        for r in rows:
            jev = r['p'] is not None and r['p'] >= th
            for gate, sleeps in (('rule', r['rule'] == 'wait'), ('rule+jev', r['rule'] == 'wait' and jev)):
                key = (gate, 'slept' if sleeps else 'woke', 'needed' if r['needed'] else 'unneeded')
                tally[key] = tally.get(key, 0) + 1
        for gate in ('rule', 'rule+jev'):
            get = lambda a, b: tally.get((gate, a, b), 0)
            print(f"  {gate:8} p>={th}: skips {get('slept', 'unneeded')} unneeded wakes, wrongly skips {get('slept', 'needed')}, "
                  f"still wakes for {get('woke', 'unneeded')} unneeded")
    if '-v' in sys.argv:
        for r in rows:
            print(f"   {r['t']:5} {r['event']:11} rule={r['rule']:8} p={r['p']} {'NEEDED' if r['needed'] else 'unneeded'} {r['tools'][:4]}  {r['expect'][:60]}")
