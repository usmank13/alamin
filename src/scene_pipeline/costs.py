"""Wall-clock and token accounting read from existing artifacts; nothing is estimated."""
import json
from pathlib import Path

from .contracts import PipelineError,read_json,write_json


def tokens(root):
    """Sum Codex `turn.completed` usage over attempt trajectories; None when no agent ran."""
    total=None
    for path in sorted(Path(root).glob('attempts/*/trajectory.jsonl')):
        for line in path.read_text().splitlines():
            try:event=json.loads(line)
            except ValueError:continue
            if event.get('type')=='turn.completed' and 'usage' in event:
                total=total or {'input':0,'output':0}
                total['input']+=int(event['usage'].get('input_tokens',0));total['output']+=int(event['usage'].get('output_tokens',0))
    return total


def rows(artifacts):
    out=[]
    for path in map(Path,artifacts):
        if (path/'cost.json').exists():
            cost=read_json(path/'cost.json');stages=[]
            for key,label in [('layout_seconds','layout'),('compile_seconds','compile'),('validation_seconds','validation')]:
                stages.append(dict(artifact=path.name,stage=f'generate/{label}',wall_s=sum(a.get(key,0.) for a in cost['attempts'])))
            # Whatever generate() spent outside the deterministic stages: agent calls, previews, file copies.
            stages.append(dict(artifact=path.name,stage='generate/agent+io',wall_s=cost['seconds']-sum(s['wall_s'] for s in stages),
                               attempts=len(cost['attempts']),status=cost.get('status'),tokens=tokens(path),api_spend_usd=cost.get('api_spend_usd')))
            if cost.get('fal_calls'):
                # fal wall time is inside generate/agent+io; spend is list price, with cache hits at 0.
                stages.append(dict(artifact=path.name,stage='generate/fal',wall_s=None,attempts=cost['fal_calls'],api_spend_usd=cost.get('fal_spend_usd'),status=cost.get('fal_spend_basis')))
            out+=stages
            if (path/'render'/'report.json').exists():out.append(dict(artifact=path.name,stage='render/cycles',wall_s=read_json(path/'render'/'report.json')['seconds']))
            if (path/'urdf'/'verification.json').exists():out.append(dict(artifact=path.name,stage='export/urdf+pybullet',wall_s=read_json(path/'urdf'/'verification.json').get('seconds')))
        elif (path/'report.json').exists() and 'flow' in read_json(path/'report.json'):
            r=read_json(path/'report.json');out.append(dict(artifact=path.name,stage=f"flow/{r['flow']}",wall_s=r['wall_seconds'],sim_s=r['simulated_seconds'],passed=r['passed']))
        elif (path/'dataset.json').exists():
            d=read_json(path/'dataset.json');runs=[v['report'] for v in d['variants'] if 'report' in v]
            generation=[read_json(p) for p in sorted(path.glob('variant_*/cost.json'))]
            out.append(dict(artifact=path.name,stage=f"dataset/generate x{len(generation)}",wall_s=sum(c['seconds'] for c in generation)))
            out.append(dict(artifact=path.name,stage=f"dataset/flows x{len(runs)} of {len(d['variants'])}",wall_s=sum(r['wall_seconds'] for r in runs),sim_s=sum(r['simulated_seconds'] for r in runs),passed=d['passed']))
            if (path/'fal_calls.json').exists():
                calls=read_json(path/'fal_calls.json');out.append(dict(artifact=path.name,stage='dataset/fal',wall_s=None,attempts=len(calls),api_spend_usd=sum(c.get('cost_usd') or 0. for c in calls)))
        else:raise PipelineError('UNKNOWN_ARTIFACT',f'{path} has no cost.json, flow report.json or dataset.json')
    return out


def table(artifacts,output,usd_in=None,usd_out=None):
    """Markdown + JSON cost table. Spend is only computed when both token prices are supplied."""
    out=rows(artifacts);output=Path(output)
    for r in out:
        if r.get('tokens') and usd_in is not None and usd_out is not None:
            r['api_spend_usd']=r['tokens']['input']/1e6*usd_in+r['tokens']['output']/1e6*usd_out
    total=sum(r['wall_s'] for r in out if r.get('wall_s') is not None)
    fmt=lambda v:'n/a' if v is None else (f'{v:.1f}' if isinstance(v,float) else str(v))
    lines=['| artifact | stage | wall s | sim s | attempts | tokens in/out | API spend USD | passed |','| --- | --- | --- | --- | --- | --- | --- | --- |']
    for r in out:
        t=r.get('tokens');lines.append(f"| {r['artifact']} | {r['stage']} | {fmt(r.get('wall_s'))} | {fmt(r.get('sim_s'))} | {fmt(r.get('attempts'))} | "
                                    f"{f'{t['input']}/{t['output']}' if t else 'n/a'} | {fmt(r.get('api_spend_usd'))} | {fmt(r.get('passed'))} |")
    lines.append(f'| **total** | | **{total:.1f}** | | | | | |')
    note=('\n\nWall seconds are measured on this CPU-only machine. `n/a` spend means the Codex CLI reports tokens but no price; '
          'pass `--usd-per-mtok-in/--usd-per-mtok-out` to price them. Deterministic stages have no API spend. '
          'fal rows are list price at request time with cache hits at 0, not provider billing.\n')
    output.write_text('# Cost and timing table\n\n'+'\n'.join(lines)+note)
    result=dict(rows=out,total_wall_s=total,table=str(output),usd_per_mtok=dict(input=usd_in,output=usd_out))
    write_json(output.with_suffix('.json'),result)
    return result
