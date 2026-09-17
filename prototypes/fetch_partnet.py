"""Fetch a small public ManiSkill PartNet subset, with checksums and safe extraction."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

META = "https://raw.githubusercontent.com/mani-skill/ManiSkill/main/mani_skill/assets/partnet_mobility/meta"
BASE = "https://storage1.ucsd.edu/datasets/ManiSkill2022-assets/partnet_mobility/dataset"


def get(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('vendor/partnet_pilot'))
    parser.add_argument('--per-category', type=int, default=1)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for category in ('cabinet_door', 'cabinet_drawer', 'faucet'):
        metadata = get(f'{META}/info_{category}_train.json')
        models = json.loads(metadata)
        for model_id in sorted(models, key=int)[:args.per_category]:
            url = f'{BASE}/{model_id}.zip'
            archive = args.output / f'{model_id}.zip'
            if not archive.exists():
                archive.write_bytes(get(url))
            raw = archive.read_bytes()
            target = args.output / model_id
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(raw)) as z:
                for member in z.infolist():
                    dest = (target / member.filename).resolve()
                    if not dest.is_relative_to(target.resolve()):
                        raise ValueError('Unsafe archive path')
                    if member.file_size > 256 * 1024 * 1024:
                        raise ValueError('Unexpectedly large member')
                z.extractall(target)
            records.append(dict(id=model_id, category=category, url=url,
                                sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw),
                                metadata=models[model_id]))
            print(category, model_id, len(raw), flush=True)
    (args.output / 'sources.json').write_text(json.dumps(records, indent=2)+'\n')


if __name__ == '__main__':
    main()
