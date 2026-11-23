#!/usr/bin/env python3
"""Jev yes/no questions about camera frames, against hand labels.

    OPENROUTER_API_KEY=... RHIZOME_ROOT=... scripts/jev_frames.py

Frames: outputs/jev_eval/frames36.json (36 wake frames sampled from past
runs). Labels below were made by eye from contact sheets; '?' = ambiguous,
left out.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(os.environ['RHIZOME_ROOT']).expanduser() / 'src'))
from ai_agent import decide  # noqa: E402

CFG = {'url': 'https://openrouter.ai/api/alpha/decisions', 'model': 'typesafe/jev-1.14',
       'key': os.environ.get('OPENROUTER_API_KEY'), 'timeout_s': 20}
CAM = "The image is the front camera of a farm rover. "
#            0         1         2         3
#            012345678901234567890123456789012345
LABELS = {
    'rows':  'NNYYYYY?NNNNYNNNNNN??NYYNNNNNNNN?NNN',
    'close': 'NNNNNNNNNYNNNYYY?YYY?NNNY?Y?YYYNNYYN',
    'road':  'NNNNNNNY?YYYY???N?NNNN?NNNYYYYYNYNNN',
    'tractor': 'NNNNNNNNYYNNNNNNNNNNNNNNNNYYY?YNNNNN',
}
QUESTIONS = {
    'rows': {
        'bare': 'Are rows of crop plants visible?',
        'cam': CAM + 'Are rows of crop plants (lines of green plants on tilled soil) visible anywhere in view?',
        'criteria': (CAM + 'Are crop rows in view?', {'true': 'Lines of green crop plants on bare tilled soil can be seen, near or far.',
                                                       'false': 'Only grass, road, trees, bushes, vehicles or sky: no planted rows.'}),
    },
    'close': {
        'bare': 'Is something blocking the way directly ahead?',
        'cam': CAM + 'Is a large object (a tree trunk, a bush, a vehicle) close in front of the rover, filling a good part of the view?',
    },
    'road': {
        'bare': 'Is there a dirt road ahead?',
        'cam': CAM + 'Is the rover on or facing a dirt road or track (a bare brown path through grass, not the tilled soil between crop rows)?',
    },
    'tractor': {'bare': 'Is a tractor visible?', 'cam': CAM + 'Is a tractor or part of one visible?'},
}

frames = json.loads(Path('outputs/jev_eval/frames36.json').read_text())


def ask(frame, q):
    question = {'type': 'noul', 'instructions': q} if isinstance(q, str) else \
        {'type': 'noul', 'instructions': q[0], 'criteria': q[1]}
    started = time.perf_counter()
    return decide.ask(CFG, 'Front camera frame:', {'q': question}, Path(frame).read_bytes())['q']['noul'], \
        time.perf_counter() - started


for topic, variants in QUESTIONS.items():
    truth = LABELS[topic]
    keep = [i for i, c in enumerate(truth) if c != '?']
    for name, q in variants.items():
        with ThreadPoolExecutor(8) as pool:
            got = list(pool.map(lambda i: ask(frames[i], q), keep))
        ps = [p for p, _ in got]
        right = sum((p >= 0.5) == (truth[i] == 'Y') for i, p in zip(keep, ps))
        sure = [(i, p) for i, p in zip(keep, ps) if p >= 0.9 or p <= 0.1]
        sure_right = sum((p >= 0.5) == (truth[i] == 'Y') for i, p in sure)
        wrong = [f"{i}:{p:.2f}" for i, p in zip(keep, ps) if (p >= 0.5) != (truth[i] == 'Y')]
        secs = sorted(s for _, s in got)
        print(f"{topic:8} {name:9} acc {right}/{len(keep)}  sure(p>=.9|<=.1) {sure_right}/{len(sure)}  "
              f"{secs[len(secs) // 2]:.2f}s  wrong: {' '.join(wrong)}", flush=True)

# Same request five times: does the answer move?
print('repeat:', [round(ask(frames[9], QUESTIONS['close']['cam'])[0], 3) for _ in range(5)])
