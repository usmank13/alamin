"""Print a run_agent_nav run's milestones from events.jsonl as they land; exit when the runner exits.
usage: watch_run.py <run_dir> [--from-start]   (--from-start: print what is logged so far and exit)"""
import json, subprocess, sys, time
from pathlib import Path

run = Path(sys.argv[1]); f = run / 'events.jsonl'
replay = '--from-start' in sys.argv
pos = 0 if replay else (f.stat().st_size if f.exists() else 0)


def alive():
    ps = subprocess.run(['ps', '-eo', 'args'], capture_output=True, text=True).stdout
    return any('python scripts/run_agent_nav' in l for l in ps.splitlines())


def show(e):
    k, t = e.get('kind'), e.get('t', 0)
    if k == 'nav_source':
        return f"{t:.0f}s driver -> {e['text']}"
    if k == 'tool' and e['text'].startswith('send_message'):
        return f"{t:.0f}s to operator: {e['text'][13:220]}"
    if k == 'tool' and 'skills/' in e['text']:
        return f"{t:.0f}s read {e['text'][:80]}"
    if k == 'nav_model' and e.get('gate') not in ('NONE', 'OTHER_SOURCE'):
        return f"{t:.0f}s nav_model gate {e.get('gate')}"


while True:
    if f.exists():
        with open(f) as h:
            h.seek(pos); chunk = h.read()
        done, _, rest = chunk.rpartition('\n')
        pos += len(chunk) - len(rest)
        for line in done.splitlines():
            try:
                msg = show(json.loads(line))
            except Exception as ex:
                msg = f"[watch: bad line: {ex}]"
            if msg:
                print(msg, flush=True)
    if replay:
        break
    if not alive():
        log = run.parent / (run.name + '.log')
        print(f"RUN EXITED: {log.read_text()[-300:] if log.exists() else ''}", flush=True)
        break
    time.sleep(5)
