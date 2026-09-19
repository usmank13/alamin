"""Local demo dashboard. Run with python -m scene_pipeline.web from the repo.

The HTTP process never imports MuJoCo. Jobs execute the existing CLI in isolated
process groups, one at a time; credentials remain in the server environment.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import quote, unquote, urlsplit


STATIC = Path(__file__).with_name('web_ui')
TERMINAL = {'succeeded', 'partial', 'failed', 'cancelled', 'interrupted'}


def read_json(path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {} if default is None else default


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def credentials(root):
    env = dict(os.environ)
    path = root / '.env'
    if path.exists():
        for line in path.read_text().splitlines():
            name, sep, value = line.strip().removeprefix('export ').partition('=')
            if sep and name.strip() in ('OPENROUTER_API_KEY', 'FAL_KEY', 'FAL_API_KEY'):
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    env.setdefault(name.strip(), parts[0])
    env.setdefault('SCENE_PIPELINE_RESOURCE_ROOT', str(root))
    env.setdefault('SCENE_PIPELINE_CACHE', str(root / 'vendor/fal_cache'))
    env.setdefault('SCENE_PIPELINE_ASSET_STORE', str(root / 'vendor/asset_library'))
    env.setdefault('MUJOCO_GL', 'osmesa')
    env['PYTHONUNBUFFERED'] = '1'
    return env


def number(data, key, default, low, high, integer=False):
    value = data.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{key} must be a number')
    if not low <= value <= high or (integer and int(value) != value):
        raise ValueError(f'{key} must be between {low} and {high}')
    return int(value) if integer else value


def choice(data, key, default, options):
    value = data.get(key, default)
    if value not in options:
        raise ValueError(f'Invalid {key}')
    return value


class Dashboard:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.outputs = self.root / 'outputs'
        self.store = self.outputs / '.ui'
        self.store.mkdir(parents=True, exist_ok=True)
        self.env = credentials(self.root)
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.jobs = read_json(self.store / 'jobs.json', [])
        for job in self.jobs:
            if job['status'] not in TERMINAL:
                job.update(status='interrupted', stage='Server restarted; output retained')
        self.queue = deque()
        self.process = None
        self.active = None
        self.stopping = False
        self._save()
        self.worker = threading.Thread(target=self._work, daemon=True)
        self.worker.start()

    def _save(self):
        write_json(self.store / 'jobs.json', self.jobs)

    def redact(self, text):
        for name, value in self.env.items():
            if len(value) >= 8 and any(k in name.upper() for k in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
                text = text.replace(value, '[REDACTED]')
        return text

    def path(self, relative):
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError('Invalid artifact path')
        path = (self.outputs / relative).resolve()
        if not path.is_relative_to(self.outputs.resolve()) or any(p.startswith('.') for p in Path(relative).parts):
            raise ValueError('Artifact path is outside the output library')
        return path

    def url(self, path):
        if path.is_file() and path.resolve().is_relative_to(self.outputs.resolve()):
            return '/artifacts/' + quote(path.resolve().relative_to(self.outputs.resolve()).as_posix(), safe='/') + '?v=' + str(path.stat().st_mtime_ns)
        return None

    def scenes(self):
        result = []
        # Prune work trees and asset bundles before walking: large outputs stay cheap.
        for current, dirs, files in os.walk(self.outputs):
            relative = Path(current).relative_to(self.outputs)
            dirs[:] = [d for d in dirs if not d.startswith('.') and not d.endswith('_work')
                       and d not in ('attempts', 'assets', 'preflight', 'render', 'runs', 'variants', 'resolution')]
            if len(relative.parts) >= 3:
                dirs[:] = []
            if 'scene.mjz' not in files or 'ir.json' not in files:
                continue
            folder = Path(current)
            if (folder / 'best_layout/scene.mjz').exists():
                continue
            ir = read_json(folder / 'ir.json')
            objects = ir.get('objects', [])
            validation = read_json(folder / 'validation.json')
            name = relative.as_posix()
            label = relative.parts[-2] if relative.name == 'best_layout' else relative.name
            result.append(dict(id=name, name=label.replace('_', ' ').title(),
                status='validated' if validation.get('passed') is True else 'partial' if validation else 'unchecked',
                object_count=len(objects), seed=ir.get('meta', {}).get('seed'),
                area=validation.get('area_m2'),
                image=self.url(folder / 'render/cycles.png') or self.url(folder / 'preview.png'),
                modified=(folder / 'scene.mjz').stat().st_mtime,
                drawers=[o['id'] for o in objects if o.get('category') == 'drawer_unit']))
        return sorted(result, key=lambda s: (s['id'] != 'small_office', -s['modified']))

    def scene(self, scene_id):
        item = next((s for s in self.scenes() if s['id'] == scene_id), None)
        if not item:
            raise ValueError('Scene is not in the library')
        folder = self.path(scene_id)
        ir = read_json(folder / 'ir.json')
        item.update(validation=read_json(folder / 'validation.json'),
                    cost=read_json(folder / 'cost.json'),
                    objects=[dict(id=o['id'], category=o['category']) for o in ir.get('objects', [])],
                    artifacts={name: self.url(folder / file) for name, file in {
                        'overview': 'render/cycles.png', 'preview': 'preview.png', 'topdown': 'topdown.png',
                        'articulation': 'articulation.gif', 'gallery': 'index.html', 'scene': 'scene.mjz',
                        'json': 'program.json', 'calls': 'agent_calls.json', 'validation': 'validation.json'}.items()})
        authorship = read_json(folder / 'authorship.json')
        for artifact, key in [('calls', 'agent_calls'), ('json', 'model_authored_input')]:
            if isinstance(authorship.get(key), str):
                item['artifacts'][artifact] = self.url(folder / authorship[key]) or item['artifacts'][artifact]
        folders = [folder / flow for flow in ('mapping', 'interaction', 'navigate')]
        folders += list((folder / 'runs').glob('*')) if (folder / 'runs').exists() else []
        runs = []
        for run in folders:
            if not (run / 'report.json').exists():
                continue
            report = read_json(run / 'report.json')
            runs.append(dict(id=run.relative_to(self.outputs).as_posix(), name=run.name,
                flow=report.get('flow', run.name), report=report,
                modified=(run / 'report.json').stat().st_mtime,
                artifacts={n: self.url(run / f) for n, f in {
                    'video': 'replay.mp4', 'image': 'replay.png', 'map': 'map.png',
                    'report': 'report.json', 'data': 'data.h5', 'gallery': 'index.html'}.items()}))
        item['runs'] = sorted(runs, key=lambda r: -r['modified'])
        return item

    def state(self):
        with self.lock:
            jobs = [{k: v for k, v in j.items() if k not in ('request',)} for j in self.jobs]
        for job in jobs:
            if job['kind']=='generate':
                progress=read_json(self.path(job['output'])/'progress.json')
                if progress:
                    job['generation']=progress
                    if job['status']=='running' and progress.get('status')=='running':
                        job['stage']=f'Attempt {progress["attempt"]}/{progress["max_attempts"]}: {progress["phase"]}'
        return dict(token=self.token, scenes=self.scenes(), jobs=jobs,
            settings=dict(openrouter=bool(self.env.get('OPENROUTER_API_KEY')),
                          fal=bool(self.env.get('FAL_KEY') or self.env.get('FAL_API_KEY')),
                          renderer=self.env['MUJOCO_GL'], default_model='anthropic/claude-sonnet-4.6'))

    def submit(self, data):
        if not isinstance(data, dict):
            raise ValueError('Expected a JSON object')
        kind = choice(data, 'kind', 'flow', ('generate', 'flow', 'render'))
        request = dict(kind=kind, seed=number(data, 'seed', 0, 0, 2147483647, True))
        scene_id = None
        if kind == 'generate':
            prompt = data.get('prompt', '').strip() if isinstance(data.get('prompt', ''), str) else ''
            if not 1 <= len(prompt) <= 16000:
                raise ValueError('Describe the scene (1–16,000 characters)')
            mode = choice(data, 'mode', 'openrouter', ('openrouter', 'codex', 'json'))
            model = data.get('model', 'anthropic/claude-sonnet-4.6' if mode == 'openrouter' else '')
            if not isinstance(model, str) or len(model) > 200 or model.startswith('-') or (mode == 'openrouter' and not model):
                raise ValueError('Enter a model ID')
            if mode == 'openrouter' and not self.env.get('OPENROUTER_API_KEY'):
                raise ValueError('Set OPENROUTER_API_KEY in the repository .env and restart the UI')
            appearance = choice(data, 'appearance', 'flat', ('flat', 'fal'))
            if appearance == 'fal' and not (self.env.get('FAL_KEY') or self.env.get('FAL_API_KEY')):
                raise ValueError('Set FAL_KEY or FAL_API_KEY in .env and restart the UI')
            request.update(prompt=prompt, mode=mode, model=model, appearance=appearance,
                max_cost=number(data, 'max_cost', 2, .01, 25) if mode=='openrouter' else None,
                max_fal=number(data, 'max_fal', 3, 0, 25),
                iterations=number(data, 'iterations', 5, 1, 10, True))
            if mode == 'json':
                program = data.get('program')
                if isinstance(program, str):
                    program = json.loads(program)
                from .contracts import validate_program
                try:
                    validate_program(program)
                except Exception as exc:
                    raise ValueError('Scene JSON does not match the SceneProgram schema') from exc
                if program['prompt'] != prompt:
                    raise ValueError('The prompt must match the prompt field in the supplied JSON')
                request['program'] = program
        else:
            scene_id = data.get('scene')
            scene = self.scene(scene_id)
            request['scene'] = scene_id
            if kind == 'flow':
                flow = choice(data, 'flow', 'mapping', ('mapping', 'interaction', 'navigate'))
                seconds = number(data, 'seconds', 60, 1, 300)
                if flow == 'interaction' and seconds < 60:
                    raise ValueError('The drawer sequence needs at least 60 simulated seconds')
                target = data.get('target')
                if flow == 'interaction' and target not in scene['drawers']:
                    raise ValueError('Choose an available drawer target')
                goal = data.get('goal', '')
                if flow == 'navigate' and (not isinstance(goal, str) or not 1 <= len(goal.strip()) <= 500):
                    raise ValueError('Enter a navigation goal')
                request.update(flow=flow, seconds=seconds, target=target, goal=goal,
                    tier=choice(data, 'tier', 'full', ('full', 'state')))
        job_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_') + secrets.token_hex(3)
        if kind == 'generate':
            output = 'ui/scene_' + job_id
        elif kind == 'flow':
            output = f'{scene_id}/runs/{request["flow"]}_{job_id}'
        else:
            output = scene_id
        job = dict(id=job_id, kind=kind, status='queued', stage='Waiting for worker',
                   created=time.time(), scene=scene_id, output=output, request=request,
                   title='Generate scene' if kind == 'generate' else 'Render scene' if kind == 'render' else request['flow'].title())
        with self.lock:
            if self.stopping:
                raise ValueError('Server is stopping')
            if len(self.queue) >= 10:
                raise ValueError('Queue is full; wait for an existing job to finish')
            self.jobs.insert(0, job)
            self.queue.append(job)
            self._save()
        return dict(id=job_id, status='queued')

    def cancel(self, job_id):
        with self.lock:
            job = next((j for j in self.jobs if j['id'] == job_id), None)
            if not job:
                raise ValueError('Unknown job')
            if job['status'] not in TERMINAL:
                job.update(status='cancelled', stage='Cancelled; existing artifacts retained', finished=time.time())
                if self.active == job_id and self.process and self.process.poll() is None:
                    os.killpg(self.process.pid, signal.SIGTERM)
                    process = self.process
                    threading.Thread(target=self._kill_later, args=(process,), daemon=True).start()
                self._save()
        return {'status': job['status']}

    @staticmethod
    def _kill_later(process):
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def log(self, job_id):
        with self.lock:
            if not any(j['id'] == job_id for j in self.jobs):
                raise ValueError('Unknown job')
        path = self.store / job_id / 'job.log'
        if not path.exists():
            return ''
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size - 32000))
            return self.redact(f.read().decode(errors='replace'))

    def commands(self, job):
        request = job['request']
        output = self.path(job['output'])
        work = self.store / job['id']
        work.mkdir(exist_ok=True)
        if job['kind'] == 'generate':
            args = ['generate', '--prompt', request['prompt'], '--output', str(output),
                    '--seed', str(request['seed']), '--max-iterations', str(request['iterations']), '--timeout', '1200',
                    '--materials', request['appearance'], '--clutter', 'fal' if request['appearance'] == 'fal' else 'off',
                    '--max-fal-usd', str(request['max_fal'])]
            if request['mode'] == 'json':
                write_json(work / 'program.json', request['program'])
                args += ['--program', str(work / 'program.json')]
            else:
                args += ['--agent-backend', request['mode']]
                if request['mode']=='codex':
                    args += ['--use-skill']
                if request['mode']=='openrouter':
                    args += ['--max-cost-usd', str(request['max_cost'])]
                if request['model']:
                    args += ['--model', request['model']]
            yield 'Following pipeline skill' if request['mode']=='codex' else 'Generating and validating scene', args
            if (output / 'scene.mjz').exists():
                yield 'Rendering scene', ['render', str(output), '--view', 'overview', '--samples', '32']
        elif job['kind'] == 'render':
            yield 'Rendering scene', ['render', str(output), '--view', 'overview', '--samples', '32']
        else:
            source = self.path(request['scene'])
            if request['flow'] == 'interaction':
                # The existing controller chooses the first drawer. Reorder only
                # its lookup table; never alter the physical scene or manifest.
                view = work / 'target_scene'
                view.mkdir(exist_ok=True)
                for name in ('scene.mjz', 'manifest.json'):
                    (view / name).symlink_to(source / name)
                ir = read_json(source / 'ir.json')
                ir['objects'].sort(key=lambda o: o['id'] != request['target'])
                write_json(view / 'ir.json', ir)
                write_json(work / 'target_selection.json', dict(scene=request['scene'], target=request['target'],
                    method='Reordered IR lookup only; scene.mjz and manifest.json unchanged'))
                source = view
            args = ['run', str(source), '--flow', request['flow'], '--seconds', str(request['seconds']),
                    '--tier', request['tier'], '--seed', str(request['seed']), '--output', str(output)]
            if request['flow'] == 'navigate':
                args += ['--goal', request['goal'], '--policy', 'planner']
            yield 'Simulating and recording robot flow', args
            if output.exists() and request['flow'] == 'interaction':
                write_json(output / 'target_selection.json', read_json(work / 'target_selection.json'))
        if output.exists():
            yield 'Preparing gallery', ['inspect', str(output), '--no-open']

    def _work(self):
        while not self.stopping:
            with self.lock:
                job = self.queue.popleft() if self.queue else None
                if job and job['status'] == 'cancelled':
                    job = None
                if job:
                    self.active = job['id']
                    job.update(status='running', started=time.time())
                    self._save()
            if job is None:
                time.sleep(.2)
                continue
            codes = []
            work = self.store / job['id']
            work.mkdir(exist_ok=True)
            try:
                with (work / 'job.log').open('w') as log:
                    for stage, args in self.commands(job):
                        with self.lock:
                            if job['status'] == 'cancelled':
                                break
                            job['stage'] = stage
                            self._save()
                            log.write(f'\n--- {stage} ---\n'); log.flush()
                            self.process = subprocess.Popen([sys.executable, '-m', 'scene_pipeline.cli', *args],
                                cwd=self.root, env=self.env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        codes.append(self.process.wait())
                    with self.lock:
                        if job['status'] != 'cancelled':
                            output = self.path(job['output'])
                            status = 'succeeded' if codes and all(c == 0 for c in codes) else 'partial' if (output / 'scene.mjz').exists() or (output / 'report.json').exists() else 'failed'
                            job.update(status=status, stage='Complete' if status == 'succeeded' else 'Finished with failed checks; review output', return_codes=codes)
                            if job['kind']=='generate':
                                result=read_json(output/'cost.json')
                                attempts=result.get('attempts_used',len(result.get('attempts',[])))
                                limit=result.get('max_attempts',job['request']['iterations'])
                                # A loadable scene or successful renderer is not validation.
                                if result.get('passed') is not True or read_json(output/'validation.json').get('passed') is not True:
                                    job['status']='partial' if (output/'scene.mjz').exists() else 'failed'
                                    reason=result.get('stop_reason') or (result.get('last_failure') or {}).get('code','generation failed')
                                    job['stage']=f'No validated scene after {attempts}/{limit} attempts ({reason}); review logs'
                                else:
                                    job['stage']=f'Validated on attempt {attempts}/{limit}' if status=='succeeded' else 'Scene validated; a later render/gallery step failed'
                        job['finished'] = time.time()
            except Exception as exc:
                with self.lock:
                    if job['status'] != 'cancelled':
                        job.update(status='failed', stage=self.redact(str(exc)), finished=time.time())
            finally:
                with self.lock:
                    self.process = None
                    self.active = None
                    self._save()

    def close(self):
        with self.lock:
            self.stopping = True
            for job in self.jobs:
                if job['status'] not in TERMINAL:
                    self.cancel(job['id'])
        self.worker.join(timeout=5)


class Handler(BaseHTTPRequestHandler):
    server_version = 'SceneStudio/1.0'

    @property
    def app(self):
        return self.server.app

    def log_message(self, format, *args):
        pass

    def headers_ok(self):
        return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

    def json(self, value, status=200):
        data = self.app.redact(json.dumps(value, allow_nan=False)).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(data)))
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if not self.headers_ok():
            return self.json({'error': 'Invalid host'}, 403)
        route = unquote(urlsplit(self.path).path)
        try:
            if route == '/api/state':
                return self.json(self.app.state())
            if route.startswith('/api/scenes/'):
                return self.json(self.app.scene(route[len('/api/scenes/'):]))
            if route.startswith('/api/logs/'):
                return self.json({'log': self.app.log(route[len('/api/logs/'):])})
            if route.startswith('/artifacts/'):
                path = self.app.path(route[len('/artifacts/'):])
            elif route in ('/', '/app.js', '/style.css'):
                path = STATIC / ('index.html' if route == '/' else route[1:])
            else:
                return self.json({'error': 'Not found'}, 404)
            return self.file(path)
        except (ValueError, OSError):
            self.json({'error': 'Artifact not found'}, 404)

    def file(self, path):
        if not path.is_file():
            return self.json({'error': 'Artifact not found'}, 404)
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        requested = self.headers.get('Range')
        if requested:
            try:
                unit, value = requested.split('=')
                left, right = value.split('-')
                if unit != 'bytes' or not (left or right):
                    raise ValueError()
                start = int(left) if left else max(0, size - int(right))
                end = min(int(right), size - 1) if left and right else size - 1
                if not 0 <= start <= end < size:
                    raise ValueError()
                status = 206
            except ValueError:
                self.send_response(416)
                self.send_header('Content-Range', f'bytes */{size}')
                self.end_headers()
                return
        self.send_response(status)
        self.send_header('Content-Type', mimetypes.guess_type(path.name)[0] or 'application/octet-stream')
        self.send_header('Content-Length', str(max(0, end - start + 1)))
        self.send_header('Accept-Ranges', 'bytes')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; media-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'; object-src 'none'")
        if status == 206:
            self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = f.read(min(1024 * 256, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def do_POST(self):
        if not self.headers_ok() or not secrets.compare_digest(self.headers.get('X-Studio-Token', ''), self.app.token):
            return self.json({'error': 'Refresh the dashboard before submitting a job'}, 403)
        origin = self.headers.get('Origin')
        if origin and origin != 'http://' + self.headers['Host']:
            return self.json({'error': 'Cross-origin requests are not allowed'}, 403)
        try:
            length = int(self.headers.get('Content-Length', 0))
            if not 0 < length <= 128000:
                raise ValueError('Request must be at most 128 KB')
            data = json.loads(self.rfile.read(length))
            route = urlsplit(self.path).path
            if route == '/api/jobs':
                return self.json(self.app.submit(data), 202)
            if route.startswith('/api/cancel/'):
                return self.json(self.app.cancel(route[len('/api/cancel/'):]))
            return self.json({'error': 'Not found'}, 404)
        except (ValueError, TypeError) as exc:
            return self.json({'error': str(exc)}, 400)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd(), help='Repository root (contains outputs/ and .env)')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    app = Dashboard(args.root)
    server.app = app
    print(f'Scene Studio: http://127.0.0.1:{server.server_port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.close()
        server.server_close()


if __name__ == '__main__':
    main()
