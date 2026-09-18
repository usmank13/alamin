"""fal.ai jobs cached by request digest. Only the orchestrator submits; layout, compiler and render read the cache."""
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .contracts import PipelineError, digest, read_json, write_json
from .registry import CATALOG, MATERIALS

ROOT = Path(os.environ.get('SCENE_PIPELINE_CACHE', Path(os.environ.get('XDG_CACHE_HOME', Path.home()/'.cache'))/'scene-pipeline'))/'fal'
PATINA = 'fal-ai/patina/material'
HUNYUAN = 'fal-ai/hunyuan3d-v3/text-to-3d'
PATINA_ROLES = {'basecolor': 'rgb', 'normal': 'normal', 'roughness': 'roughness', 'metalness': 'metallic'}  # PATINA map -> MuJoCo texture role
# fal list prices read on 2026-09-17, not provider-reported billing. PATINA: $0.01 + $0.02/MP + $0.01/MP per map at
# 1024^2 with four maps. Hunyuan3D v3 text-to-3d 'Normal' (textured, no PBR pass) is a flat fee.
PRICE_USD = {PATINA: .07, HUNYUAN: .375}
PRICE_BASIS = 'fal list price 2026-09; cache hits cost 0; not provider-reported billing'
KEEP = ('.png', '.jpg', '.jpeg', '.webp', '.glb')


def patina_request(finish, seed=0, *, prompt=None):
    return PATINA, dict(prompt=prompt if prompt is not None else MATERIALS[finish]['prompt'], image_size='square_hd', seed=int(seed), enable_prompt_expansion=False,
                        maps=list(PATINA_ROLES), tiling_mode='both', output_format='png')


def decor_request(category, config=None):
    config = config or CATALOG[category]
    return HUNYUAN, dict(prompt=config['prompt']+', a single object centred on a plain background', face_count=40000,
                         generate_type='Normal', enable_pbr=False)


def material_jobs(seed=0):
    jobs = []
    for finish, spec in MATERIALS.items():
        if 'prompt' in spec:
            model, payload = patina_request(finish, seed); jobs.append(dict(kind='material', finish=finish, seed=int(seed), model=model, payload=payload))
    return jobs


def decor_jobs(program):
    jobs = []
    seen=set()
    for item in program['objects']:
        if item.get('asset_ref') or item.get('asset_request'):continue
        category=item['category'];config=None
        if item.get('generated_request'):
            from .generated import config_from_request
            config=config_from_request(item['generated_request'])
        elif CATALOG.get(category,{}).get('route')!='G6':continue
        model,payload=decor_request(category,config);identity=key(model,payload)
        if identity not in seen:
            jobs.append(dict(kind='decor',category=category,model=model,payload=payload));seen.add(identity)
    return jobs


def key(model, payload):
    return digest(dict(model=model, payload=payload))


def cached(model, payload):
    folder = ROOT/key(model, payload)
    return folder if (folder/'meta.json').exists() else None


def _http(method, url, body=None, timeout=60.):
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, method=method,
                                     headers={'Authorization': f"Key {os.environ.get('FAL_KEY') or os.environ.get('FAL_API_KEY', '')}", 'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _download(url, timeout=300.):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def _files(response):
    """Every {'url': ...} object in a fal response, named by map_type or by its parent key."""
    found = []
    def walk(value, name):
        if isinstance(value, dict):
            if 'url' in value:
                suffix = Path(value.get('file_name') or value['url'].split('?')[0]).suffix.lower() or '.bin'
                found.append((value.get('map_type') or name, suffix, value['url']))
            for k, v in value.items(): walk(v, k)
        elif isinstance(value, list):
            for v in value: walk(v, name)
    walk(response, 'file')
    return found


def _store(folder, job, reply, response, started, download):
    folder.mkdir(parents=True, exist_ok=True)
    hashes = {}; seen = set()
    for name, suffix, url in _files(response):
        if suffix not in KEEP or url in seen: continue
        seen.add(url); data = download(url)
        if suffix != '.glb':
            # MuJoCo reads 8-bit RGB PNG; normalise depth, alpha and format once at download time.
            buffer = io.BytesIO(); Image.open(io.BytesIO(data)).convert('RGB').save(buffer, 'PNG'); data = buffer.getvalue(); suffix = '.png'
        (folder/f'{name}{suffix}').write_bytes(data); hashes[f'{name}{suffix}'] = hashlib.sha256(data).hexdigest()
    write_json(folder/'request.json', dict(model=job['model'], payload=job['payload']))
    write_json(folder/'response.json', response)
    meta = dict(model=job['model'], request_id=reply.get('request_id'), submitted_at=started, completed_at=datetime.now(timezone.utc).isoformat(),
                list_price_usd=PRICE_USD[job['model']], price_basis=PRICE_BASIS, files=hashes)
    write_json(folder/'meta.json', meta)  # written last: cached() only ever sees complete entries
    return meta


def prefetch(jobs, *, budget_usd=None, spent_usd=0., deadline_s=600., interval=2., http=None, download=None):
    """Submit uncached jobs to the fal queue, poll them together, store results. One job's failure never raises.

    jobs: dicts with kind, model, payload and any labels. Returns one record per job carrying cached, cost_usd and
    error, so callers degrade per item and write the ledger. FAL_KEY is required only when something is uncached.
    """
    http = http or _http; download = download or _download
    records = []; pending = []; spent = float(spent_usd)
    for job in jobs:
        folder = ROOT/key(job['model'], job['payload']); labels = {k: v for k, v in job.items() if k != 'payload'}
        if (folder/'meta.json').exists():
            records.append(dict(**labels, cached=True, cost_usd=0., request_id=read_json(folder/'meta.json').get('request_id'))); continue
        if not (os.environ.get('FAL_KEY') or os.environ.get('FAL_API_KEY')):
            raise PipelineError('FAL_CREDENTIALS', 'Set FAL_KEY or FAL_API_KEY to generate materials or clutter with fal')
        price = PRICE_USD[job['model']]
        if budget_usd is not None and spent+price > budget_usd:
            records.append(dict(**labels, cached=False, cost_usd=0., error=dict(code='BUDGET_EXHAUSTED', message=f'fal list-price budget {budget_usd} USD would be exceeded'))); continue
        started = datetime.now(timezone.utc).isoformat()
        try:
            reply = http('POST', f"https://queue.fal.run/{job['model']}", job['payload'])
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            records.append(dict(**labels, cached=False, cost_usd=0., error=dict(code='FAL_SUBMIT', message=str(exc)[:300]))); continue
        spent += price  # billed on submission whether or not the result is collected
        pending.append(dict(job=job, labels=labels, folder=folder, reply=reply, started=started, clock=time.monotonic()))
    t0 = time.monotonic()
    while pending:
        if time.monotonic()-t0 > deadline_s:
            for item in pending:
                records.append(dict(**item['labels'], cached=False, cost_usd=PRICE_USD[item['job']['model']], request_id=item['reply'].get('request_id'),
                                    error=dict(code='FAL_TIMEOUT', message='Job did not complete before the deadline')))
            break
        for item in list(pending):
            try:
                status = http('GET', item['reply']['status_url']+'?logs=0'); state = status.get('status')
                if state in ('IN_QUEUE', 'IN_PROGRESS'): continue
                if state != 'COMPLETED': raise ValueError(f"fal status {state}: {status.get('error') or status.get('error_type') or ''}")
                response = http('GET', item['reply']['response_url'])
                meta = _store(item['folder'], item['job'], item['reply'], response, item['started'], download)
                records.append(dict(**item['labels'], cached=False, cost_usd=meta['list_price_usd'], request_id=meta['request_id'], seconds=time.monotonic()-item['clock']))
            except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
                records.append(dict(**item['labels'], cached=False, cost_usd=PRICE_USD[item['job']['model']], request_id=item['reply'].get('request_id'),
                                    error=dict(code='FAL_JOB', message=str(exc)[:300])))
            pending.remove(item)
        if pending: time.sleep(interval)
    return records
