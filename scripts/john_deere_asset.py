"""Fetch the freely redistributed CC BY John Deere 9RX from AllenAI Objaverse.

Run `.venv/bin/python scripts/john_deere_asset.py` to cache it without an account.
"""
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1] / 'outputs/outdoor_assets/john_deere_9rx'
TRACTOR = ROOT / 'tractor.glb'
URL = 'https://huggingface.co/datasets/allenai/objaverse/resolve/main/glbs/000-093/d1d7db9b4f424b6583b315eb5edc9deb.glb'
SHA256 = '308c87fd1b05fa598629aaa32fea53734974f66420e00233e679f4a74856db17'
METADATA = dict(
    name='John Deere Trekker 9RX', author='Daan van Leeuwen',
    source_url='https://sketchfab.com/3d-models/john-deere-trekker-9rx-d1d7db9b4f424b6583b315eb5edc9deb',
    download_url=URL, source_sha256=SHA256, license='CC BY 4.0',
    license_url='https://creativecommons.org/licenses/by/4.0/',
    length_m=8., scale_basis='Approximate scene scale, not calibrated manufacturer geometry',
    changes='Converted to MuJoCo, rescaled, materials packed into an RGB atlas, static convex collision proxy.')
CREDIT = (f"{METADATA['name']} by {METADATA['author']}\n{METADATA['source_url']}\n"
          f"Licensed under Creative Commons Attribution 4.0: {METADATA['license_url']}\n"
          f"Downloaded from AllenAI Objaverse: {URL}\n"
          "Objaverse database: ODC Attribution 1.0, https://opendatacommons.org/licenses/by/1-0/\n"
          f"Changes: {METADATA['changes']}\n")


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not TRACTOR.exists():
        request = urllib.request.Request(URL, headers={'User-Agent': 'alamin-scene-builder'})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        if hashlib.sha256(payload).hexdigest() != SHA256:
            raise ValueError('John Deere source checksum mismatch')
        TRACTOR.write_bytes(payload)
    elif hashlib.sha256(TRACTOR.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f'John Deere source checksum mismatch: {TRACTOR}')
    (ROOT / 'tractor_LICENSE.txt').write_text(CREDIT)
    (ROOT / 'tractor.json').write_text(json.dumps(METADATA, indent=2)+'\n')
    return TRACTOR


if __name__ == '__main__':
    print(prepare())
