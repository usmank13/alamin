"""Dashboard contracts: no live providers, robot downloads, or paid generation."""
import json
import signal
import sys
import time
from pathlib import Path
from io import BytesIO
from types import SimpleNamespace

import pytest

from scene_pipeline.web import Dashboard, Handler, credentials, write_json


def scene_fixture(root, name='office'):
    scene = root / 'outputs' / name
    scene.mkdir(parents=True)
    (scene / 'scene.mjz').write_bytes(b'portable scene')
    write_json(scene / 'ir.json', {'meta': {'seed': 7}, 'objects': [
        {'id': 'drawer_a', 'category': 'drawer_unit'}, {'id': 'drawer_b', 'category': 'drawer_unit'}]})
    write_json(scene / 'manifest.json', {'instances': {}})
    write_json(scene / 'validation.json', {'passed': True, 'area_m2': 30, 'checks': {'stability': True}})
    return scene


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setattr(Dashboard, '_work', lambda self: None)
    scene_fixture(tmp_path)
    dashboard = Dashboard(tmp_path)
    yield dashboard
    dashboard.close()


def test_library_prunes_work_and_prefers_best_layout(app):
    scene_fixture(app.root, 'debug_work/seed_1')
    scene_fixture(app.root, 'dorm')
    scene_fixture(app.root, 'dorm/best_layout')
    scene_fixture(app.root, 'office/attempts/0')
    assert {s['id'] for s in app.scenes()} == {'office', 'dorm/best_layout'}
    write_json(app.outputs / 'office/mapping/report.json', {'flow': 'mapping', 'passed': False})
    detail = app.scene('office')
    assert detail['drawers'] == ['drawer_a', 'drawer_b']
    assert detail['runs'][0]['report']['passed'] is False


@pytest.mark.parametrize('path', ['../.env', '/etc/passwd', '.ui/jobs.json', 'office/../../.env'])
def test_path_confinement(app, path):
    with pytest.raises(ValueError):
        app.path(path)


def test_symlink_escape_is_rejected(app, tmp_path):
    (app.outputs / 'outside').symlink_to(tmp_path.parent)
    with pytest.raises(ValueError):
        app.path('outside/secret.txt')


@pytest.mark.parametrize('body', [
    {'kind': 'shell'},
    {'kind': 'flow', 'scene': '../private'},
    {'kind': 'flow', 'scene': 'office', 'seconds': float('nan')},
    {'kind': 'flow', 'scene': 'office', 'seconds': True},
    {'kind': 'flow', 'scene': 'office', 'flow': 'navigate', 'goal': ''},
    {'kind': 'flow', 'scene': 'office', 'flow': 'interaction', 'target': 'missing'},
    {'kind': 'flow', 'scene': 'office', 'flow': 'interaction', 'target': 'drawer_a', 'seconds': 10},
])
def test_job_validation(app, body):
    with pytest.raises(ValueError):
        app.submit(body)
    assert not app.jobs


def test_drawer_target_view_does_not_mutate_scene(app):
    source = app.outputs / 'office'
    before = {name: (source / name).read_bytes() for name in ('ir.json', 'scene.mjz', 'manifest.json')}
    app.submit({'kind': 'flow', 'scene': 'office', 'flow': 'interaction', 'target': 'drawer_b'})
    stage, args = next(app.commands(app.jobs[0]))
    target = Path(args[1])
    assert json.loads((target / 'ir.json').read_text())['objects'][0]['id'] == 'drawer_b'
    assert (target / 'scene.mjz').resolve() == source / 'scene.mjz'
    assert {name: (source / name).read_bytes() for name in before} == before
    assert args[args.index('--tier') + 1] == 'full'
    assert '--policy' not in args


def test_generation_uses_explicit_budget_and_argument_vector(app):
    app.env['OPENROUTER_API_KEY'] = 'test-secret-not-real'
    app.submit({'kind': 'generate', 'prompt': 'A room; $(echo harmless)', 'max_cost': 1.25, 'model': 'provider/model'})
    _, args = next(app.commands(app.jobs[0]))
    assert args[args.index('--prompt') + 1] == 'A room; $(echo harmless)'
    assert args[args.index('--max-cost-usd') + 1] == '1.25'
    assert args[args.index('--agent-backend') + 1] == 'openrouter'
    assert '--program' not in args
    assert '--use-skill' not in args
    assert 'test-secret-not-real' not in json.dumps(app.state())


def test_json_generation_avoids_model_calls(app):
    from scene_pipeline.dsl import load
    root = Path(__file__).resolve().parents[1]
    program = load(root / 'examples/cafe_program.py')
    app.submit({'kind': 'generate', 'mode': 'json', 'prompt': program['prompt'], 'program': program})
    _, args = next(app.commands(app.jobs[0]))
    assert '--program' in args and '--agent-backend' not in args
    assert '--use-skill' not in args
    assert json.loads(Path(args[args.index('--program') + 1]).read_text()) == program


def test_codex_uses_its_default_model_when_unspecified(app):
    app.submit({'kind': 'generate', 'mode': 'codex', 'prompt': 'A small office'})
    _, args = next(app.commands(app.jobs[0]))
    assert args[args.index('--agent-backend') + 1] == 'codex'
    assert '--use-skill' in args
    assert '--model' not in args
    assert '--max-cost-usd' not in args  # Codex provides tokens, not billed USD.
    assert args[args.index('--max-iterations')+1]=='5'


def test_dashboard_reports_agent_attempt_progress(app):
    app.submit({'kind':'generate','mode':'codex','prompt':'A room','iterations':4})
    job=app.jobs[0];job['status']='running'
    write_json(app.path(job['output'])/'progress.json',dict(status='running',attempt=2,max_attempts=4,phase='Validating scene'))
    shown=app.state()['jobs'][0]
    assert shown['stage']=='Attempt 2/4: Validating scene'
    assert shown['generation']['attempt']==2


def test_derived_scene_links_to_original_model_calls(app):
    root = app.outputs / 'office'
    (root / 'agent_calls.json').write_text('[{"backend":"openrouter"}]')
    child = scene_fixture(app.root, 'office/best_layout')
    write_json(child / 'authorship.json', {'agent_calls': '../agent_calls.json'})
    assert app.scene('office/best_layout')['artifacts']['calls'].startswith('/artifacts/office/agent_calls.json?')


def test_credentials_are_parsed_as_data_and_logs_are_redacted(app, monkeypatch):
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    (app.root / '.env').write_text('OPENROUTER_API_KEY="literal-$(do-not-execute)"\nPATH=/not/allowed\n')
    env = credentials(app.root)
    assert env['OPENROUTER_API_KEY'] == 'literal-$(do-not-execute)'
    assert env['PATH'] != '/not/allowed'
    app.env = env
    job = app.submit({'kind': 'render', 'scene': 'office'})
    folder = app.store / job['id']; folder.mkdir()
    (folder / 'job.log').write_text('failure ' + env['OPENROUTER_API_KEY'])
    assert app.log(job['id']) == 'failure [REDACTED]'


def test_cancellation_and_restart_do_not_resume_paid_jobs(app):
    job = app.submit({'kind': 'render', 'scene': 'office'})
    assert app.cancel(job['id'])['status'] == 'cancelled'
    app.submit({'kind': 'render', 'scene': 'office'})
    restarted = Dashboard(app.root)
    assert [j['status'] for j in restarted.jobs] == ['interrupted', 'cancelled']
    assert not restarted.queue
    restarted.close()


def await_job(app, job_id, timeout=8):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        job = next(j for j in app.jobs if j['id'] == job_id)
        if job['status'] in ('succeeded', 'partial', 'failed', 'cancelled'):
            return job
        time.sleep(.03)
    raise AssertionError('Worker did not finish')


def test_worker_retains_nonzero_status_and_continues_queue(tmp_path, monkeypatch):
    import subprocess
    real_popen = subprocess.Popen
    return_codes = iter([1, 0])
    def fake_pipeline(command, **kwargs):
        return real_popen([sys.executable, '-c', f'raise SystemExit({next(return_codes)})'], **kwargs)
    monkeypatch.setattr('scene_pipeline.web.subprocess.Popen', fake_pipeline)
    monkeypatch.setattr(Dashboard, 'commands', lambda self, job: iter([('Render', ['render'])]))
    scene_fixture(tmp_path)
    app = Dashboard(tmp_path)
    try:
        first = app.submit({'kind': 'render', 'scene': 'office'})
        second = app.submit({'kind': 'render', 'scene': 'office'})
        assert await_job(app, first['id'])['status'] == 'partial'
        assert await_job(app, second['id'])['status'] == 'succeeded'
    finally:
        app.close()


def test_running_cancel_terminates_process_group(tmp_path, monkeypatch):
    import subprocess
    real_popen = subprocess.Popen
    def fake_pipeline(command, **kwargs):
        return real_popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
    monkeypatch.setattr('scene_pipeline.web.subprocess.Popen', fake_pipeline)
    monkeypatch.setattr(Dashboard, 'commands', lambda self, job: iter([('Long run', ['run'])]))
    scene_fixture(tmp_path)
    app = Dashboard(tmp_path)
    try:
        job = app.submit({'kind': 'render', 'scene': 'office'})
        end = time.monotonic() + 5
        while app.process is None and time.monotonic() < end:
            time.sleep(.03)
        process = app.process
        assert process is not None
        app.cancel(job['id'])
        assert process.wait(timeout=4) == -signal.SIGTERM
        assert app.jobs[0]['status'] == 'cancelled'
    finally:
        app.close()


def handler(app, path, headers=None, body=None):
    request = Handler.__new__(Handler)
    request.server = SimpleNamespace(app=app, server_port=8765)
    request.path = path
    payload = json.dumps(body).encode() if body is not None else b''
    request.headers = {'Host': '127.0.0.1:8765', 'Content-Length': str(len(payload)), **(headers or {})}
    request.rfile = BytesIO(payload)
    request.wfile = BytesIO()
    request.response_headers = {}
    request.send_response = lambda status: setattr(request, 'status', status)
    request.send_header = lambda name, value: request.response_headers.update({name: value})
    request.end_headers = lambda: None
    return request


def test_http_submission_requires_local_host_and_token(app):
    body = {'kind': 'render', 'scene': 'office'}
    for headers in [{}, {'Host': 'evil.example:8765', 'X-Studio-Token': app.token},
                    {'X-Studio-Token': app.token, 'Origin': 'https://evil.example'}]:
        request = handler(app, '/api/jobs', headers, body)
        request.do_POST()
        assert request.status == 403
    assert not app.jobs
    request = handler(app, '/api/jobs', {'X-Studio-Token': app.token, 'Origin': 'http://127.0.0.1:8765'}, body)
    request.do_POST()
    assert request.status == 202
    assert len(app.jobs) == 1


def test_http_file_range_and_private_paths(app):
    video = app.outputs / 'office/replay.mp4'
    video.write_bytes(b'0123456789')
    request = handler(app, '/artifacts/office/replay.mp4', {'Range': 'bytes=2-5'})
    request.do_GET()
    assert request.status == 206
    assert request.response_headers['Content-Range'] == 'bytes 2-5/10'
    assert request.wfile.getvalue() == b'2345'
    request = handler(app, '/artifacts/office/replay.mp4', {'Range': 'bytes=20-25'})
    request.do_GET()
    assert request.status == 416
    for path in ['/artifacts/%2e%2e/.env', '/artifacts/.ui/jobs.json', '/.env']:
        request = handler(app, path)
        request.do_GET()
        assert request.status == 404
