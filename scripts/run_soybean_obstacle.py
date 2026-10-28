#!/usr/bin/env python3
"""Run Rhizome vision driving and obstacle stopping in the soybean cube scene."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    repo=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,default=repo/'outputs/soybean30_obstacle_gen3')
    parser.add_argument('--config',type=Path,default=repo/'examples/agriculture/soybean30_obstacle_drive.json')
    parser.add_argument('--output',type=Path,help='Fresh rollout directory; default is timestamped')
    parser.add_argument('--seconds',type=float,default=25)
    parser.add_argument('--disable-stopping',action='store_true',help='Comparison run: keep perception but turn off obstacle gating')
    parser.add_argument('--renderer',choices=['egl','osmesa'],default='egl')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--viewer',action='store_true',help='Open recorded MuJoCo replay, initially paused')
    args=parser.parse_args()
    scene=args.scene.expanduser().resolve();config_path=args.config.expanduser().resolve()
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output=args.output.expanduser().resolve() if args.output else repo/'outputs'/('soybean_obstacle_'+stamp)
    if output.exists():parser.error(f'Choose a fresh output directory: {output}')
    if not (scene/'scene.mjz').is_file():parser.error(f'Missing scene: {scene}; see docs/agriculture.md for generation')
    # Config paths use the same repository-relative convention as pipeline run.
    os.chdir(repo)
    os.environ['MUJOCO_GL']=args.renderer
    os.environ.setdefault('LP_NUM_THREADS','2')
    config=json.loads(config_path.read_text())
    controller=config['controller'];nav=controller['nav_model']
    nav.update(depth=True,obstacle_detection=not args.disable_stopping)
    package=Path(nav['package']).expanduser().resolve()
    if not (package/'model.ts').is_file():parser.error(f'Missing obstacle-grid package: {package}')
    from scene_pipeline.rhizome_build import verify
    host=Path(controller['host_build']).expanduser().resolve()
    source=Path(controller['root']).expanduser().resolve()
    try:
        manifest=verify(host)
        if Path(manifest['source_root']).resolve()!=source:raise ValueError('Build belongs to another checkout')
    except (OSError,ValueError):
        build=repo/'outputs'/('rhizome_host_obstacle_'+stamp)
        print(f'Building matching Rhizome host extensions in {build}',flush=True)
        subprocess.run([sys.executable,'-m','scene_pipeline.rhizome_build','--root',str(source),
                        '--output',str(build),'--jobs','4'],check=True)
        host=build/'host'
    controller['host_build']=str(host)
    from scene_pipeline.agriculture_drive import run
    report=run(scene,output,config=config,seconds=args.seconds,video=not args.no_video,timeout=1800)
    nav_report=report.get('rhizome',{}).get('navigation',{})
    print(json.dumps(dict(output=str(output),simulation_passed=report['passed'],failure=report['failure'],
        travel_m=report['displacement_m'],navigation=nav_report),indent=2))
    print(f'Inspect: {output / "index.html"}')
    print(f'Replay: .venv/bin/pipeline replay {output} --viewer --paused')
    if args.viewer:
        subprocess.run([str(repo/'.venv/bin/pipeline'),'replay',str(output),'--viewer','--paused'],check=True)
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
