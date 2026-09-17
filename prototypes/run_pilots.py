"""Run the minimal pilots against the prepared subset; writes only outputs/."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    folder = root/'vendor/partnet_pilot/manipgen_subset/partnet/meshdata'
    assets = [folder / name / 'coacd.urdf' for name in
              ('door-33507-18-0', 'door-40402-16-0', 'door-30857-13-0')]
    if not all(p.exists() for p in assets):
        raise SystemExit('Run python prototypes/fetch_manipgen.py --download first')
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def run(*args, timeout=180):
        subprocess.run(args, check=True, cwd=root, env=env, timeout=timeout)
    run('bash', 'prototypes/run_p4.sh')
    run(sys.executable, '-m', 'prototypes.microchecks')
    for source, output in [('collision', 'p5'), ('visual', 'p5_visual')]:
        run(sys.executable, '-m', 'prototypes.p5', *map(str, assets),
            '--collision-source', source, '--output', f'outputs/prototypes/{output}')
    run(sys.executable, '-m', 'prototypes.report_pilots')


if __name__ == '__main__':
    main()
