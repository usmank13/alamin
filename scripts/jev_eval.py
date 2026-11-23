#!/usr/bin/env python3
"""Try Jev prompt/state setups against labeled past wakes (jev_dataset.py).

    OPENROUTER_API_KEY=... RHIZOME_ROOT=~/Desktop/code/rhizome-ai-agent \\
        scripts/jev_eval.py outputs/jev_eval/wakes.jsonl [--only NAME ...] [--split dev|test|all]

Two tasks:
  wake    does the reasoning model need to act on this wake? (label != noop)
  action  for wakes it acted on while driving: drive / look / back_up / hand_off
Answers are cached next to the dataset, so re-runs only pay for new setups.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(os.environ['RHIZOME_ROOT']).expanduser() / 'src'))
from ai_agent import decide  # noqa: E402

CFG = {'url': 'https://openrouter.ai/api/alpha/decisions', 'model': 'typesafe/jev-1.14',
       'key': os.environ.get('OPENROUTER_API_KEY'), 'timeout_s': 20}


# ---- state builders ---------------------------------------------------------
def compact_event(w):
    ev = json.loads(json.dumps(w['event']))
    field = (ev.get('details') or {}).get('field')
    if isinstance(field, dict):  # corner lists are noise for a yes/no model
        ev['details']['field'] = {k: field[k] for k in ('inside', 'outside_m', 'edge_ahead_m') if k in field}
    return json.dumps(ev)


def digest_lines(w, *prefixes):
    return '\n'.join(line for line in w['digest'].split('\n') if line.startswith(prefixes))


STATES = {
    'event': lambda w: 'Event: ' + compact_event(w),
    'lean': lambda w: '\n'.join(filter(None, [
        'Mission: ' + w['mission'], digest_lines(w, 'plan', 'recent', 'here'), 'Event: ' + compact_event(w)])),
    'full': lambda w: '\n'.join(['Mission: ' + w['mission'], w['digest'], 'Event: ' + json.dumps(w['event'])]),
}

# ---- questions --------------------------------------------------------------
INTRO = ('A field rover is carrying out the mission in the state. Its supervisor, a slow and expensive AI, '
         'set things in motion and is asleep. An event has just arrived. ')
WAKE_PLAIN = {'type': 'choice', 'instructions': INTRO + 'Does the supervisor need to decide or do something now?',
              'criteria': {'wake': 'Yes: a new decision or action is needed.',
                           'sleep': 'No: things are progressing as intended and nothing needs deciding yet.'}}
WAKE_RUBRIC = {'type': 'choice', 'instructions': INTRO + 'Does the supervisor need to decide or do something now?',
               'criteria': {
                   'wake': ('A leg of driving is ending or has ended and a next one must be chosen; something blocks '
                            'the way or the rover is stuck; the row-following controller stopped, finished the field '
                            'or lost its rows; the operator said something; or anything unexpected.'),
                   'sleep': ('The row-following controller (nav_model) reports it is still driving its rows or turning '
                             'to the next lane; or an obstacle report says the way is clear again; or a wait merely '
                             'timed out while the rover is still making progress.')}}
WAKE_NOUL = {'type': 'noul', 'instructions': INTRO + 'Is everything progressing as intended, so that the supervisor '
             'can stay asleep and nothing needs deciding yet?'}
ACTION = {'type': 'choice', 'instructions': INTRO + 'The supervisor must act. What kind of step fits best?',
          'criteria': {
              'drive': 'Drive a next leg forward: the way toward the goal is visible and open.',
              'look': 'Look first: scan around, turn, or study the view, because the goal or the way to it is not clear from here.',
              'back_up': 'Reverse away: the rover is too close to something, blocked or stuck facing it.',
              'hand_off': 'This phase is finished: hand driving to the row-following controller, stop, or report to the operator.'}}


def reorder(q):
    return {**q, 'criteria': dict(reversed(list(q['criteria'].items())))}


# name: (task, state, question, with_image)
SETUPS = {
    'wake/event/plain': ('wake', 'event', WAKE_PLAIN, False),
    'wake/lean/plain': ('wake', 'lean', WAKE_PLAIN, False),
    'wake/full/plain': ('wake', 'full', WAKE_PLAIN, False),
    'wake/event/rubric': ('wake', 'event', WAKE_RUBRIC, False),
    'wake/lean/rubric': ('wake', 'lean', WAKE_RUBRIC, False),
    'wake/lean/rubric-rev': ('wake', 'lean', reorder(WAKE_RUBRIC), False),
    'wake/lean/noul': ('wake', 'lean', WAKE_NOUL, False),
    'wake/lean/rubric+img': ('wake', 'lean', WAKE_RUBRIC, True),
    'action/lean': ('action', 'lean', ACTION, False),
    'action/full': ('action', 'full', ACTION, False),
    'action/lean+img': ('action', 'lean', ACTION, True),
    'action/event+img': ('action', 'event', ACTION, True),
}


def split_of(w):
    return 'test' if zlib.crc32(w['run'].encode()) % 3 == 0 else 'dev'


def main(args):
    rows = [json.loads(line) for line in open(args.dataset)]
    rows = [w for w in rows if args.split == 'all' or split_of(w) == args.split]
    if args.digest_only:  # runs of the current harness: every wake carries the [world] digest
        rows = [w for w in rows if w['digest']]
    cache_path = Path(args.dataset).with_suffix('.cache.json')
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    def answer(w, state, question, image):
        text = STATES[state](w)
        jpeg = Path(w['frame']).read_bytes() if image and w['frame'] else None
        key = hashlib.sha256(json.dumps([CFG['model'], text, question, w['frame'] if jpeg else None]).encode()).hexdigest()
        if key not in cache:
            for attempt in range(3):
                try:
                    started = time.perf_counter()
                    cache[key] = {'a': decide.ask(CFG, text, {'q': question}, jpeg)['q'],
                                  's': round(time.perf_counter() - started, 2)}
                    break
                except Exception as e:
                    cache[key] = {'error': f'{type(e).__name__}: {e}'}
                    time.sleep(2 * (attempt + 1))
        return cache[key]

    for name, (task, state, question, image) in SETUPS.items():
        if args.only and not any(o in name for o in args.only):
            continue
        todo = [w for w in rows if (task == 'wake' or (w['label'] != 'noop' and w['driving']))
                and (not image or w['frame'])]
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda w: answer(w, state, question, image), todo))
        cache_path.write_text(json.dumps(cache))
        ok = [(w, g) for w, g in zip(todo, got) if 'a' in g]
        secs = sorted(g['s'] for _, g in ok)
        line = f"{name:24} n={len(ok):3} err={len(todo) - len(ok)} {secs[len(secs) // 2] if secs else 0:.2f}s"
        if task == 'wake':
            # p(sleep) per wake; "saved" = slept through a noop, "missed" = slept through one the model acted on
            ps = [(g['a']['noul'] if g['a']['type'] == 'noul' else g['a']['probabilities']['sleep'], w['label'] == 'noop')
                  for w, g in ok]
            noops = sum(n for _, n in ps)
            line += f"  noops={noops}"
            for th in (0.5, 0.8, 0.9, 0.95):
                saved = sum(1 for p, n in ps if p >= th and n)
                missed = sum(1 for p, n in ps if p >= th and not n)
                line += f"  @{th}: saved {saved} missed {missed}"
            if args.by_event:
                for name_ in sorted({w['event'].get('event') for w, _ in ok}):
                    sub = [(p, n) for (p, n), (w, _) in zip(ps, ok) if w['event'].get('event') == name_]
                    line += (f"\n    {name_:12} noop {sum(n for _, n in sub):3} acted {sum(not n for _, n in sub):3}"
                             f"  @0.9: saved {sum(1 for p, n in sub if p >= 0.9 and n):3} missed {sum(1 for p, n in sub if p >= 0.9 and not n):3}"
                             f"  @0.5: saved {sum(1 for p, n in sub if p >= 0.5 and n):3} missed {sum(1 for p, n in sub if p >= 0.5 and not n):3}")
        else:
            right = sum(g['a']['choice'] == w['label'] for w, g in ok)
            sure = [(w, g) for w, g in ok if max(g['a']['probabilities'].values()) >= 0.8]
            line += (f"  acc {right}/{len(ok)}  when p>=0.8: {sum(g['a']['choice'] == w['label'] for w, g in sure)}/{len(sure)}")
            if args.confusion:
                labels = list(question['criteria'])
                for truth in labels:
                    line += f"\n    {truth:9}" + ''.join(
                        f" {sum(1 for w, g in ok if w['label'] == truth and g['a']['choice'] == pred):3}" for pred in labels)
        print(line, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('dataset')
    p.add_argument('--only', nargs='*')
    p.add_argument('--split', default='dev', choices=['dev', 'test', 'all'])
    p.add_argument('--confusion', action='store_true')
    p.add_argument('--by-event', action='store_true')
    p.add_argument('--digest-only', action='store_true')
    main(p.parse_args())
