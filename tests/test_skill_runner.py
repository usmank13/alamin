"""Skill execution uses real tool artifacts, shared limits, and one agent session."""
import json
import os
from pathlib import Path
import signal
from types import SimpleNamespace
import sys
import time

import pytest

from scene_pipeline import skill_runner as runner
from scene_pipeline.contracts import PipelineError, read_json, write_json
from scene_pipeline.dsl import load


def job(tmp_path, monkeypatch, attempts=2):
    root = tmp_path / 'job'
    root.mkdir()
    config = dict(output=str(root), prompt='A kitchen', seed=7, max_attempts=attempts,
                  deadline=time.time()+60, timeout=60, materials='fal', clutter='fal', max_fal_usd=1.)
    write_json(root / 'generation.json', config)
    write_json(root / 'builds.json', [])
    monkeypatch.setenv(runner.JOB_ENV, str(root / 'generation.json'))
    program = load(Path(__file__).resolve().parents[1] / 'examples/cafe_program.py')
    program['prompt'] = config['prompt']
    program['space']['inferred_fields'] = ['area_m2', 'shape', 'annexes']
    write_json(root / 'input.json', program)
    def args(index):
        return SimpleNamespace(command='generate', program=root/'input.json', use_skill=False,
                               output=root/'attempts'/str(index), prompt=config['prompt'], timeout=900,
                               seed=999, materials='flat', clutter='off', max_fal_usd=50.)
    return root, config, program, args


def test_builds_share_attempt_budget_fal_budget_and_intent(tmp_path, monkeypatch):
    root, config, program, args = job(tmp_path, monkeypatch)
    first = args(0)
    runner.prepare_build(first)
    assert first.seed == 7 and first.max_fal_usd == 1.
    assert first.materials == first.clutter == 'fal' and 0 < first.timeout <= 60
    write_json(root/'attempts/0/cost.json', dict(fal_spend_usd=.4))
    second = args(1)
    runner.prepare_build(second)
    assert second.seed == 10 and second.max_fal_usd == pytest.approx(.6)
    with pytest.raises(PipelineError, match='exhausted'):
        runner.prepare_build(args(2))
    assert len(read_json(root/'builds.json')) == 2


def test_skill_repairs_cannot_drop_required_inventory(tmp_path, monkeypatch):
    root, _, program, args = job(tmp_path, monkeypatch)
    runner.prepare_build(args(0))
    program['objects'][0]['count'] += 1
    write_json(root/'input.json', program)
    with pytest.raises(PipelineError, match='frozen required'):
        runner.prepare_build(args(1))
    assert len(read_json(root/'builds.json')) == 2
    assert read_json(root/'attempts/1/failure.json')['code'] == 'INTENT_DRIFT'


def test_job_requires_supplied_json_and_confines_outputs(tmp_path, monkeypatch):
    root, _, _, args = job(tmp_path, monkeypatch)
    request = args(0); request.program = None
    with pytest.raises(PipelineError, match='nested agents'):
        runner.prepare_build(request)
    request = args(0); request.output = tmp_path/'other'
    with pytest.raises(PipelineError, match='next scene output'):
        runner.prepare_build(request)
    with pytest.raises(PipelineError, match='shared fal budget'):
        runner.prepare_build(SimpleNamespace(command='asset-generate'))
    assert read_json(root/'builds.json') == []


def test_missing_skill_fails_before_launching_agent(tmp_path):
    with pytest.raises(PipelineError, match='Required pipeline skill missing'):
        runner.generate('a room', 0, tmp_path/'output', root=tmp_path)
    assert not (tmp_path/'output').exists()


def test_launch_loads_current_skill_in_one_tool_capable_session(tmp_path, monkeypatch):
    skill = tmp_path/runner.SKILL
    skill.parent.mkdir(parents=True)
    skill.write_text('Unique skill instruction: inspect existing kitchen assets first.')
    seen = []
    def fake_agent(command, prompt, output, env, timeout, progress):
        seen.append((command, prompt, env))
        progress()
        assert skill.read_text() in prompt
        assert env[runner.JOB_ENV] == str(output/'generation.json')
        return dict(passed=True, seconds=0.)
    monkeypatch.setattr(runner, 'run_agent', fake_agent)
    result = runner.generate('A kitchen', 7, tmp_path/'output', root=tmp_path, model='my-model')
    assert len(seen) == 1
    command = seen[0][0]
    assert command[command.index('--sandbox')+1] == 'workspace-write'
    assert command[command.index('--model')+1] == 'my-model'
    assert '--output-schema' not in command
    assert '--dangerously-bypass-approvals-and-sandbox' not in command
    assert result['passed'] is False  # An agent saying done is not a scene.
    assert (tmp_path/'output/skill_snapshot.md').read_text() == skill.read_text()


def test_stream_records_usage_and_redacts_logs(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_API_KEY', 'sensitive-secret-value')
    code = '''import json,sys
sys.stdin.read()
print(json.dumps({'type':'thread.started','thread_id':'one-session'}))
print(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'sensitive-secret-value'}}))
print(json.dumps({'type':'turn.completed','usage':{'input_tokens':30,'output_tokens':5}}))
'''
    record = runner.run_agent([sys.executable, '-c', code], 'skill prompt', tmp_path, dict(os.environ), 5, lambda:None)
    assert record['passed'] and record['thread_id'] == 'one-session'
    assert record['input_tokens'] == 30
    log = (tmp_path/'agent_trajectory.jsonl').read_text()
    assert 'sensitive-secret-value' not in log and '[REDACTED]' in log
    assert read_json(tmp_path/'agent_calls.json') == [record]


def test_timeout_keeps_diagnostics_and_restores_signal_handlers(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    code = "import sys,time; sys.stdin.read(); print('starting tool', flush=True); time.sleep(10)"
    started = time.monotonic()
    record = runner.run_agent([sys.executable, '-c', code], 'skill', tmp_path, dict(os.environ), .3, lambda:None)
    assert time.monotonic()-started < 4
    assert record['error'] == 'AGENT_TIMEOUT'
    assert 'starting tool' in (tmp_path/'agent_trajectory.jsonl').read_text()
    assert signal.getsignal(signal.SIGTERM) == before


def test_final_result_is_revalidated_and_agent_usage_is_preserved(tmp_path, monkeypatch):
    root, config, _, args = job(tmp_path, monkeypatch)
    runner.prepare_build(args(0))
    scene = root/'attempts/0'; scene.mkdir(parents=True)
    (scene/'scene.mjz').write_bytes(b'scene')
    write_json(scene/'validation.json', {'passed':True,'checks':{'robot_access':True}})
    write_json(scene/'scene_checks.json', {'passed':True})
    write_json(scene/'agent_calls.json', [])
    write_json(root/'agent_calls.json', [{'kind':'skill_session'}])
    def failed_verification(path):
        assert path == scene
        report = {'passed':False, 'checks':{'robot_access':False}}
        write_json(path/'validation.json', report)
        return report
    monkeypatch.setattr('scene_pipeline.validation.validate_scene', failed_verification)
    result = runner.finish(root, config, {'passed':True})
    assert result['status'] == 'partial' and result['passed'] is False
    assert read_json(root/'validation.json')['passed'] is False
    assert read_json(root/'agent_calls.json') == [{'kind':'skill_session'}]


def test_cancellation_terminates_agent_and_its_tools(tmp_path):
    import subprocess
    code = """import subprocess,sys,time
from pathlib import Path
sys.stdin.read()
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(60)
"""
    outer = f"""import os,sys
from pathlib import Path
from scene_pipeline.skill_runner import run_agent
root=Path(sys.argv[1])
run_agent([sys.executable,'-c',{code!r},str(root/'tool.pid')],'skill',root,dict(os.environ),60,lambda:None)
"""
    process = subprocess.Popen([sys.executable, '-c', outer, str(tmp_path)], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        deadline = time.monotonic()+5
        while not (tmp_path/'tool.pid').exists() and time.monotonic()<deadline:
            time.sleep(.03)
        assert (tmp_path/'tool.pid').exists()
        tool_pid = int((tmp_path/'tool.pid').read_text())
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=5) != 0
        stat = Path(f'/proc/{tool_pid}/stat')
        assert not stat.exists() or stat.read_text().split()[2] == 'Z'
        assert (tmp_path/'agent_calls.json').exists()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
