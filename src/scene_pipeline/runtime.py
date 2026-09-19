"""Optional model transports. Deterministic scene code depends only on this boundary."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import jsonschema

from .contracts import PipelineError, read_json, write_json

CURRENT = ContextVar('scene_agent_runtime', default=None)


def redact(text):
    for name, value in os.environ.items():
        if value and len(value) >= 8 and any(word in name.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            text = text.replace(value, '[REDACTED]')
    return text


def strict(schema):
    """Provider schema; the original contract is still validated locally.

    Codex structured output rejects uniqueItems. Keep uniqueness enforcement in
    jsonschema.validate below, rather than weakening the scene contract itself.
    https://developers.openai.com/api/docs/guides/structured-outputs
    """
    if isinstance(schema, list): return [strict(s) for s in schema]
    if not isinstance(schema, dict): return schema
    out = {('anyOf' if k == 'oneOf' else k): strict(v) for k, v in schema.items() if k != 'uniqueItems'}
    if out.get('type') == 'object' and 'properties' in out:
        required = set(out.get('required', []))
        for key, prop in out['properties'].items():
            if key not in required:
                out['properties'][key] = {'anyOf': [prop, {'type': 'null'}]}
        out['required'] = list(out['properties'])
    return out


def strip_null(value):
    if isinstance(value, dict): return {k: strip_null(v) for k, v in value.items() if v is not None}
    if isinstance(value, list): return [strip_null(v) for v in value]
    return value


@dataclass
class Runtime:
    backend: str = 'codex'
    model: str | None = None
    deadline: float | None = None
    max_cost_usd: float | None = None
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)
    max_calls: int = 60

    def remaining(self, limit=180.):
        remaining = limit if self.deadline is None else min(limit, self.deadline-time.monotonic())
        if not math.isfinite(remaining) or remaining <= 0:
            raise PipelineError('BUDGET_EXHAUSTED', 'Wall-time deadline exhausted')
        return remaining

    def summary(self):
        costs = [c.get('cost_usd') for c in self.calls]
        return dict(calls=len(self.calls), input_tokens=sum(c.get('input_tokens', 0) or 0 for c in self.calls),
                    output_tokens=sum(c.get('output_tokens', 0) or 0 for c in self.calls),
                    cost_usd=sum(costs) if all(c is not None for c in costs) else None,
                    known_cost_usd=sum(c for c in costs if c is not None),
                    cost_note='Provider-reported cost; cap checked between calls, not a hard billing ceiling')

    def request(self, instruction, schema, work, *, name='program', timeout=180., search=False, image=None):
        if self.backend not in ('codex', 'openrouter', 'recorded'):
            raise PipelineError('AGENT_BACKEND', self.backend)
        if self.max_cost_usd is not None:
            if not math.isfinite(self.max_cost_usd) or self.max_cost_usd <= 0:
                raise PipelineError('COST_BUDGET', 'Spend limit must be positive and finite')
            totals = self.summary()
            if self.calls and totals['cost_usd'] is None:
                raise PipelineError('COST_UNKNOWN', 'Cannot continue a cost-limited run without reported costs')
            if totals['known_cost_usd'] >= self.max_cost_usd:
                raise PipelineError('BUDGET_EXHAUSTED', 'Reported model spend limit reached')
        if len(self.calls) >= self.max_calls:
            raise PipelineError('BUDGET_EXHAUSTED', 'Model request limit reached')
        timeout = self.remaining(timeout); work = Path(work); work.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        record = dict(backend=self.backend, model=self.model, kind=name, search=search,
                      image=bool(image), cost_usd=None, passed=False)
        try:
            if self.backend == 'recorded':
                if not self.responses: raise PipelineError('RECORDING_EXHAUSTED', 'No recorded response remains')
                reply = self.responses.pop(0)
                if reply['kind'] != name: raise PipelineError('RECORDING_ORDER', f'Expected {name}')
                output = reply['output']; record.update(cost_usd=0., input_tokens=0, output_tokens=0)
            elif self.backend == 'codex':
                write_json(work/f'{name}.schema.json', strict(schema))
                command = ['codex'] + (['--search'] if search else []) + ['exec']
                if image: command += ['-i', str(Path(image).resolve())]
                if self.model: command += ['--model', self.model]
                command += ['--sandbox', 'read-only', '--skip-git-repo-check', '--ignore-user-config', '--ephemeral', '--json',
                            '--output-schema', str((work/f'{name}.schema.json').resolve()),
                            '--output-last-message', str((work/f'{name}.json').resolve()), '--cd', str(work.resolve()), instruction]
                result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
                (work/'trajectory.jsonl').write_text(redact(result.stdout))
                (work/'agent.stderr.log').write_text(redact(result.stderr))
                for line in result.stdout.splitlines():
                    try: event = json.loads(line)
                    except ValueError: continue
                    if event.get('type') == 'turn.completed': record.update(event.get('usage', {}))
                if result.returncode or not (work/f'{name}.json').exists():
                    failures=[]
                    for line in result.stdout.splitlines():
                        try:event=json.loads(line)
                        except ValueError:continue
                        if event.get('type') in ('error','turn.failed'):
                            failures.append(event.get('message') or event.get('error'))
                    message=redact(json.dumps(failures,ensure_ascii=False))[:2000] if failures else 'See redacted agent logs'
                    code='AGENT_REQUEST_INVALID' if 'invalid_json_schema' in message else 'AGENT_CREDENTIALS' if any(
                        marker in message for marker in ('invalid_api_key','authentication_error')) else 'AGENT_FAILED'
                    raise PipelineError(code, f'Codex failed: {message}', dict(returncode=result.returncode))
                output = read_json(work/f'{name}.json')
            else:
                if not self.model: raise PipelineError('AGENT_MODEL', 'OpenRouter requires an explicit model ID')
                if not os.getenv('OPENROUTER_API_KEY'): raise PipelineError('AGENT_CREDENTIALS', 'Set OPENROUTER_API_KEY')
                request = dict(instruction=instruction, schema=schema, model=self.model, search=search,
                               image=str(Path(image).resolve()) if image else None)
                # Process isolation provides a real wall-clock cancellation boundary, including SDK retries.
                result = subprocess.run([sys.executable, '-m', 'scene_pipeline.openrouter_worker'],
                                        input=json.dumps(request), text=True, capture_output=True, timeout=timeout)
                (work/'agent.stderr.log').write_text(redact(result.stderr))
                if result.returncode: raise PipelineError('AGENT_FAILED', 'OpenRouter worker failed; see redacted log')
                reply = json.loads(result.stdout)
                if 'error' in reply: raise PipelineError(reply['error']['code'], reply['error']['message'])
                output = reply['output']; record.update(reply['usage'])
                write_json(work/'provider_messages.json', json.loads(redact(json.dumps(reply.get('messages', [])))))
            output = strip_null(output)
            jsonschema.validate(output, schema)
            self.remaining(timeout)
            write_json(work/f'{name}.json', output); record['passed'] = True
            return output
        except subprocess.TimeoutExpired as exc:
            raise PipelineError('AGENT_TIMEOUT', 'Model call exceeded remaining wall-time budget') from exc
        except FileNotFoundError as exc:
            raise PipelineError('AGENT_UNAVAILABLE', 'Agent executable is not installed') from exc
        except (jsonschema.ValidationError, json.JSONDecodeError) as exc:
            raise PipelineError('AGENT_SCHEMA', 'Model output did not satisfy the local schema') from exc
        finally:
            record['seconds'] = time.monotonic()-started
            self.calls.append(record)
            write_json(work/f'{name}.usage.json', record)


@contextmanager
def using(runtime):
    token = CURRENT.set(runtime)
    try: yield runtime
    finally: CURRENT.reset(token)


def request(instruction, schema, work, model=None, timeout=180, **kwargs):
    return (CURRENT.get() or Runtime(model=model)).request(instruction, schema, work, timeout=timeout, **kwargs)
