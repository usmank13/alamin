"""Apply one fal floor material to an existing scene; preserve its geometry."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shlex
import shutil
import time

from scene_pipeline import fal
from scene_pipeline.compiler import compile_scene
from scene_pipeline.contracts import PipelineError,read_json,write_json,digest
from scene_pipeline.inspection import scene_page
from scene_pipeline.render import preview,cycles
from scene_pipeline.validation import validate_scene,load


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--floor-prompt',required=True);parser.add_argument('--tile-m',type=float,default=2.)
    parser.add_argument('--max-fal-usd',type=float,default=.10);parser.add_argument('--env-file',type=Path,default=Path('.env'))
    args=parser.parse_args()
    if args.output.exists():raise PipelineError('OUTPUT_EXISTS','Use a fresh output; the source is never modified')
    if not 0<args.tile_m<=20:raise PipelineError('MATERIAL_SCALE','Expected a positive tile size in metres, at most 20')
    # Read only the two credential names as data, never execute a dotenv file.
    if args.env_file.exists():
        for line in args.env_file.read_text().splitlines():
            name,separator,value=line.removeprefix('export ').partition('=')
            if separator and name.strip() in ('FAL_KEY','FAL_API_KEY'):
                parts=shlex.split(value,comments=True)
                if len(parts)==1:os.environ.setdefault(name.strip(),parts[0])
    args.output.mkdir(parents=True)
    started=time.perf_counter();original=read_json(args.scene/'ir.json');ir=deepcopy(original)
    options=dict(source='fal',seed=0,overrides={'floor_tile':dict(prompt=args.floor_prompt,tile_m=args.tile_m)})
    ir['meta']['appearance']={**ir['meta'].get('appearance',{}),'materials':options}
    write_json(args.output/'restyle_request.json',dict(source=str(args.scene.resolve()),source_ir_sha256=digest(original),
        materials=options,max_fal_usd=args.max_fal_usd,api_documentation='https://fal.ai/models/fal-ai/patina/material/api'))
    model,payload=fal.patina_request('floor_tile',0,prompt=args.floor_prompt)
    # Keep the cache in the explicit writable workspace, separate from source scenes.
    fal.ROOT=Path(os.environ.get('SCENE_PIPELINE_CACHE','vendor/fal_cache'))/'fal'
    jobs=[dict(kind='material',finish='floor_tile',model=model,payload=payload)]
    calls=fal.prefetch(jobs,budget_usd=args.max_fal_usd,deadline_s=600)
    write_json(args.output/'fal_calls.json',calls)
    if any(c.get('error') for c in calls):raise PipelineError('FAL_JOB','Material generation failed; see fal_calls.json')
    entry=fal.cached(model,payload)
    if entry is None or not all((entry/f'{role}.png').exists() for role in fal.PATINA_ROLES):
        raise PipelineError('FAL_MAPS','The requested four-map material set is incomplete')
    shutil.copytree(entry,args.output/'floor_material')
    shutil.copytree(args.scene/'assets',args.output/'assets')
    for name in ('program.json','input_program.json','intent.json','dimension_sources.json','asset_resolutions.json'):
        if (args.scene/name).exists():shutil.copy2(args.scene/name,args.output/name)
    write_json(args.output/'ir.json',ir)
    compile_scene(ir,args.output)
    # Appearance-only changes must preserve the compiled collision and joint arrays.
    import numpy as np
    # Compare both saved packages at the same serialization boundary.
    before=load(args.scene);compiled=load(args.output)
    fields=('geom_type','geom_size','geom_pos','geom_quat','geom_contype','geom_conaffinity',
            'body_pos','body_quat','body_mass','body_inertia','jnt_type','jnt_axis','jnt_range')
    unchanged=all(np.array_equal(getattr(before,k),getattr(compiled,k)) for k in fields)
    if not unchanged:raise PipelineError('RESTYLE_GEOMETRY','Appearance pass changed geometry/physics')
    report=validate_scene(args.output)
    preview(args.output)
    cycles(args.output,samples=64,resolution=1024,view='overview')
    write_json(args.output/'restyle_report.json',dict(passed=report['passed'],geometry_physics_unchanged=unchanged,
        checked_arrays=list(fields),fal_estimated_usd=sum(c['cost_usd'] for c in calls),cost_basis=fal.PRICE_BASIS,
        seconds=time.perf_counter()-started,materials=read_json(args.output/'manifest.json')['materials']))
    scene_page(args.output)
    print(args.output/'index.html')
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
