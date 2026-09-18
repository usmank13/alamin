"""Prompt -> autonomous scene generation -> Cycles render -> inspection gallery."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shlex


ROOT = Path(__file__).resolve().parents[1]


def load_credentials(path):
    """Read supported credentials as data, never execute dotenv contents."""
    if path.exists():
        for line in path.read_text().splitlines():
            name, separator, value = line.strip().removeprefix('export ').partition('=')
            if separator and name.strip() in ('FAL_KEY', 'FAL_API_KEY', 'OPENROUTER_API_KEY'):
                parts = shlex.split(value, comments=True)
                if len(parts) == 1:
                    os.environ.setdefault(name.strip(), parts[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--env-file', type=Path, default=ROOT / '.env')
    parser.add_argument('--materials', choices=['flat', 'fal'], default='flat')
    parser.add_argument('--clutter', choices=['off', 'fal'], default='off')
    parser.add_argument('--max-fal-usd', type=float, default=5.)
    parser.add_argument('--agent-backend', choices=['codex', 'openrouter'], default='codex')
    parser.add_argument('--model')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=float, default=1200.)
    parser.add_argument('--max-iterations', type=int, default=5)
    parser.add_argument('--layout-backend', choices=['heuristic', 'empirical', 'architecture'], default='heuristic')
    parser.add_argument('--priors', type=Path)
    args = parser.parse_args(argv)
    output = args.output or ROOT / 'outputs' / datetime.now(timezone.utc).strftime('scene_%Y%m%d_%H%M%S_%f')
    if output.exists():
        parser.error('Output already exists; choose a fresh directory.')
    load_credentials(args.env_file)
    os.environ.setdefault('SCENE_PIPELINE_RESOURCE_ROOT', str(ROOT))
    os.environ.setdefault('SCENE_PIPELINE_CACHE', str(ROOT / 'vendor' / 'fal_cache'))
    os.environ.setdefault('SCENE_PIPELINE_ASSET_STORE', str(ROOT / 'vendor' / 'asset_library'))
    # Import after environment setup: cache/resource roots are resolved on import.
    from scene_pipeline.cli import main as pipeline
    command = ['generate', '--prompt', args.prompt, '--output', str(output),
               '--agent-backend', args.agent_backend, '--seed', str(args.seed),
               '--materials', args.materials, '--clutter', args.clutter,
               '--max-fal-usd', str(args.max_fal_usd), '--timeout', str(args.timeout),
               '--max-iterations', str(args.max_iterations), '--layout-backend', args.layout_backend]
    if args.model:
        command += ['--model', args.model]
    if args.priors:
        command += ['--priors', str(args.priors)]
    print(f'Generating from prompt; artifacts: {output.resolve()}', flush=True)
    status = pipeline(command)
    if (output / 'scene.mjz').exists():
        print('Rendering the compiled scene with Cycles...', flush=True)
        render_status = pipeline(['render', str(output), '--view', 'overview', '--samples', '64'])
        status = status or render_status
    if output.exists():
        gallery_status = pipeline(['inspect', str(output), '--no-open'])
        status = status or gallery_status
    return status


if __name__ == '__main__':
    raise SystemExit(main())
