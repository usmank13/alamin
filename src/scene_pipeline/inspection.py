"""Static inspection pages: no server, generated code or external web dependencies."""
import html
import json
from pathlib import Path
import subprocess
import sys
import webbrowser

from .contracts import PipelineError,read_json,write_json

STYLE='body{font:16px system-ui;max-width:1300px;margin:2em auto;padding:0 1em;background:#fafafa}img,video{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}td,th{padding:.6em;text-align:left;border-bottom:1px solid #ddd}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:1em}article,figure{margin:0;padding:1em;background:white;border:1px solid #ddd}a{color:#135ca4}'


def details(root):
    return {p.name:read_json(p) for p in (root/'cost.json',root/'validation.json',root/'architecture_validation.json',
            root/'scene_checks.json',root/'report.json',root/'generation.json',root/'restyle_report.json',root/'failure.json') if p.exists()}


def scene_page(root,summary=None):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);e=html.escape
    if summary is not None:write_json(root/'inspection_summary.json',summary)
    elif (root/'inspection_summary.json').exists():summary=read_json(root/'inspection_summary.json')
    reports=details(root)
    if summary is not None:reports={'evaluation':summary,**reports}
    media=[]
    for name,label in [('preview.png','MuJoCo perspective'),('topdown.png','Plan: object footprint bounds'),
                       ('provenance.png','Object provenance; origin is not functional verification'),
                       ('articulation.gif','Kinematic sweep — not robot interaction'),('render/cycles.png','Cycles — same compiled scene'),
                       ('render_overview/cycles.png','Cycles overview — same compiled scene'),
                       ('map.png','Range reconstruction'),('replay.mp4','Recorded physics replay')]:
        if (root/name).exists():
            tag=f'<video controls preload="metadata" src="{name}"></video>' if name.endswith('.mp4') else f'<img loading="lazy" src="{name}">'
            media.append(f'<figure><figcaption>{e(label)}</figcaption>{tag}</figure>')
    links=[]
    for name in ('ir.json','program.json','manifest.json','provenance.json','dimension_sources.json','asset_resolutions.json','cost.json','validation.json','unmet.json','agent_calls.json','fal_calls.json','restyle_request.json','restyle_report.json','floor_material/basecolor.png','render/report.json','urdf/verification.json','mapping/index.html','interaction/index.html'):
        if (root/name).exists():links.append(f'<a href="{name}">{name}</a>')
    command=f'pipeline inspect {__import__("shlex").quote(str(root.resolve()))} --viewer'
    manifest=read_json(root/'manifest.json') if (root/'manifest.json').exists() else {}
    rows=[]
    for identity,obj in manifest.get('instances',{}).items():
        source=obj.get('source_classification',{})
        rows.append('<tr>'+''.join(f'<td>{e(str(v))}</td>' for v in (identity,obj['category'],source.get('origin','unknown'),len(obj.get('affordances',[])),
                    'template proxy; function unverified' if obj.get('provenance',{}).get('kind')=='sourced_template' else 'functional suitability not certified'))+'</tr>')
    page=f'<!doctype html><meta charset="utf-8"><title>Scene inspection</title><style>{STYLE}</style><h1>Scene inspection</h1>'
    page+='<p>Physics, evidence coverage and task success are separate. A loadable proxy is not a verified functional object.</p>'
    if (root/'scene.mjz').exists() or (root/'rollout_scene.mjz').exists():page+=f'<pre>{e(command)}</pre>'
    page+='<div class="grid">'+''.join(media)+'</div><p>'+' | '.join(links)+'</p>'
    page+='<table><tr><th>Instance</th><th>Category</th><th>Origin</th><th>Affordances</th><th>Scope</th></tr>'+''.join(rows)+'</table>'
    page+='<h2>Reports and failures</h2><pre>'+e(json.dumps(reports,indent=2))+'</pre>'
    (root/'index.html').write_text(page)
    return root/'index.html'


def batch_page(root,report):
    root=Path(root);e=html.escape;cards=[]
    for run in report['runs']:
        path=run['path'];scene=root/path
        thumbnail=f'<img loading="lazy" src="{e(path)}/preview.png">' if (scene/'preview.png').exists() else '<p>No compiled preview available</p>'
        diagnostic=run.get('diagnostic_scene')
        diagnostic_link=f'<p><a href="{e(diagnostic)}/index.html">Inspect diagnostic attempt (not accepted)</a></p>' if diagnostic and (root/diagnostic/'index.html').exists() else ''
        cards.append(f'<article><h2>{e(run["domain"])} / {e(str(run.get("model") or run["backend"]))}</h2>'
                     f'<p>{e(run["status"])} · seed {run["seed"]}</p><a href="{e(path)}/index.html">{thumbnail}Inspect run</a>'
                     f'{diagnostic_link}<p>{e(run["prompt"])}</p><pre>{e(json.dumps(run.get("error") or run.get("scores",{}),indent=2))}</pre></article>')
    (root/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Pipeline evaluation</title><style>{STYLE}</style>'
        '<h1>Cross-domain pipeline evaluation</h1><p>Fixed-program runs are regression fixtures, not prompt-generation evidence. '
        'Generic architecture backoff is not domain-specific empirical coverage. No visual or LLM approval gate.</p>'
        '<a href="evaluation.json">Machine-readable report</a><div class="grid">'+''.join(cards)+'</div>')
    return root/'index.html'


def inspect_path(path,*,viewer=False,open_browser=True):
    path=Path(path).resolve()
    if not path.exists():raise PipelineError('INSPECTION_PATH',str(path))
    root=path if path.is_dir() else path.parent
    if viewer:
        scene=path if path.suffix in ('.mjz','.xml') else next((root/n for n in ('scene.mjz','rollout_scene.mjz','asset.mjz') if (root/n).exists()),None)
        if scene is None:raise PipelineError('INSPECTION_SCENE','No compiled scene in this folder')
        result=subprocess.run([sys.executable,'-m','sim_harness.cli',str(scene),'--viewer','--backend','glfw'])
        return dict(passed=result.returncode==0,scene=str(scene))
    if (root/'dataset.json').exists():page=dataset_page(root)
    elif (root/'evaluation.json').exists():page=batch_page(root,read_json(root/'evaluation.json'))
    elif (root/'report.json').exists() and read_json(root/'report.json').get('kind')=='asset_retrieval_smoke' and (root/'index.html').exists():
        page=root/'index.html'
    elif (root/'asset.json').exists():
        from .asset_library import asset_preview,asset_page
        page=asset_page(root) if (root/'preview.png').exists() else asset_preview(root)
    else:
        if (root/'scene.mjz').exists() and not (root/'preview.png').exists():
            from .render import preview
            preview(root)
        page=scene_page(root)
    opened=webbrowser.open(page.as_uri()) if open_browser else False
    return dict(passed=True,page=str(page),browser_opened=opened)


def dataset_page(root):
    """Browse existing captures and per-variant failures; does not rerun physics."""
    root=Path(root);report=read_json(root/'dataset.json');e=html.escape;cards=[]
    for run in report['variants']:
        relative=f'variant_{run["variant"]:03d}';folder=root/relative
        scene_page(folder)
        if (folder/'mapping/report.json').exists():scene_page(folder/'mapping')
        links=[f'<a href="{relative}/index.html">Scene and checks</a>']
        for name,label in [('mapping/index.html','Mapping / video'),('mapping/data.h5','HDF5 streams')]:
            if (folder/name).exists():links.append(f'<a href="{relative}/{name}">{label}</a>')
        thumbnail=f'<img loading="lazy" src="{relative}/preview.png">' if (folder/'preview.png').exists() else ''
        passed=run.get('report',{}).get('passed',False)
        cards.append(f'<article><h2>{relative}: {"passed" if passed else "failed"}</h2>{thumbnail}<p>'+ ' | '.join(links)+
                     f'</p><pre>{e(json.dumps(run,indent=2))}</pre></article>')
    (root/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><title>Dataset inspection</title><style>{STYLE}</style>'
        '<h1>Dataset inspection</h1><p><a href="dataset.json">Dataset report</a> | <a href="DATA_CARD.md">Data card</a></p>'
        '<p>Recorded rollouts and deterministic checks; visual plausibility is a separate review.</p><div class="grid">'+''.join(cards)+'</div>')
    return root/'index.html'
