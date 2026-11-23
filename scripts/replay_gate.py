#!/usr/bin/env python3
"""Replay the wakes of past agent-nav runs through the wake gate
(rhizome ai_agent/decide.py) and compare with what the model did.

    RHIZOME_ROOT=~/Desktop/code/rhizome-ai-agent scripts/replay_gate.py --backend rules outputs/plots10 ...
    OPENROUTER_API_KEY=... scripts/replay_gate.py --backend jev --threshold 0.9 outputs/plots10 ...

A wake "needed the model" when it called anything but passive tools before
waiting again. A missed wake (gate waits, model acted) is the costly error.
Text only: past runs logged no frame at nav_model wakes.
"""
import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(os.environ['RHIZOME_ROOT']).expanduser() / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from ai_agent import decide  # noqa: E402
from score_run import PASSIVE  # noqa: E402


def wakes(run: Path):
    """(event dict, digest text, model acted) per wait_for_event return."""
    out, cur = [], None
    for line in open(run / 'events.jsonl'):
        e = json.loads(line)
        if e['kind'] == 'result' and e['text'].startswith('wait_for_event -> '):
            head, _, digest = e['text'][len('wait_for_event -> '):].partition('\n')
            try:
                cur = [json.loads(head), digest, False]
            except ValueError:  # with a frame (the agent was driving): only the name is recoverable
                name = re.search(r'"event": "(\w+)"', head)
                cur = [{'event': name.group(1) if name else '?'}, '', False]
            out.append(cur)
        elif e['kind'] == 'tool' and cur is not None and e['text'].split('(')[0] not in PASSIVE:
            cur[2] = True
    return out


async def main(args):
    cfg = {'backend': args.backend, 'threshold': args.threshold, 'images': 'off', 'timeout_s': 10,
           'url': args.url, 'model': args.model, 'key': os.environ.get('OPENROUTER_API_KEY')}
    tally = {'saved': 0, 'missed': 0, 'woke_needed': 0, 'woke_unneeded': 0}
    seconds = []
    for run in args.runs:
        for ev, digest, acted in wakes(run):
            ctx = SimpleNamespace(config={'decision': cfg}, agent=None,
                                  world=SimpleNamespace(digest=lambda d=digest: d))
            started = time.perf_counter()
            verdict, why = await decide.gate(ev, ctx)
            if why.startswith('jev'):
                seconds.append(time.perf_counter() - started)
            waited = verdict == decide.WAIT
            tally[('missed' if acted else 'saved') if waited else ('woke_needed' if acted else 'woke_unneeded')] += 1
            if args.verbose or (waited and acted):
                d = ev.get('details') or {}
                print(f"{run.name:10} {ev.get('event'):12} gate={d.get('gate')} driving={d.get('driving')} "
                      f"-> {verdict:8} ({why})  model {'acted' if acted else 'only waited'}")
    seconds.sort()
    print(json.dumps({**tally, 'jev_calls': len(seconds),
                      'jev_s_median': round(seconds[len(seconds) // 2], 2) if seconds else None}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+', type=Path)
    p.add_argument('--backend', default='rules', choices=['rules', 'jev'])
    p.add_argument('--threshold', type=float, default=0.9)
    p.add_argument('--url', default='https://openrouter.ai/api/alpha/decisions')
    p.add_argument('--model', default='typesafe/jev-1.14')
    p.add_argument('-v', '--verbose', action='store_true')
    asyncio.run(main(p.parse_args()))
