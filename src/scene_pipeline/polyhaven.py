"""CC0 assets from polyhaven.com, cached under outputs/outdoor_assets/polyhaven."""
import json
from pathlib import Path
import urllib.request

import numpy as np
from PIL import Image

CACHE = Path(__file__).resolve().parents[2] / 'outputs/outdoor_assets/polyhaven'
Image.MAX_IMAGE_PIXELS = None  # the tonemapped skies are 16k


def _get(url, path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(url, headers={'User-Agent': 'scene-builder'})
        with urllib.request.urlopen(request, timeout=300) as reply:
            path.write_bytes(reply.read())
        (CACHE / 'LICENSE.txt').write_text('Assets in this folder are from https://polyhaven.com, licensed CC0 1.0.\n')
    return path


def _files(asset):
    return json.loads(_get(f'https://api.polyhaven.com/files/{asset}', CACHE / asset / 'files.json').read_text())


def texture(asset, resolution='2k'):
    """Diffuse map as float32 RGB 0-255, and its side in metres."""
    info = json.loads(_get(f'https://api.polyhaven.com/info/{asset}', CACHE / asset / 'info.json').read_text())
    path = _get(_files(asset)['Diffuse'][resolution]['jpg']['url'], CACHE / asset / f'diffuse_{resolution}.jpg')
    return np.asarray(Image.open(path).convert('RGB'), np.float32), info['dimensions'][0] / 1000


def sky(asset):
    """Tonemapped equirectangular panorama, uint8 RGB."""
    return np.asarray(Image.open(_get(_files(asset)['tonemapped']['url'], CACHE / asset / 'tonemapped.jpg')).convert('RGB'))


def tiled(asset, x, y, cm_per_px=1.):
    """The texture laid over world coordinates x, y (metres, arrays of one shape)."""
    rgb, side = texture(asset)
    n = max(8, round(side * 100 / cm_per_px))
    tile = np.asarray(Image.fromarray(rgb.astype(np.uint8)).resize((n, n), Image.LANCZOS), np.float32)
    return tile[(y / side * n).astype(np.int64) % n, (x / side * n).astype(np.int64) % n]


def sun(panorama):
    """(azimuth, elevation) in radians of the panorama's brightest patch; azimuth 0 at its centre column, positive left."""
    small = np.asarray(Image.fromarray(panorama).resize((720, 360)).convert('L'), np.float32)
    small = small[:180]  # above the horizon
    from scipy.ndimage import gaussian_filter
    row, col = np.unravel_index(np.argmax(gaussian_filter(small, 3)), small.shape)
    return (.5 - (col + .5) / 720) * 2 * np.pi, (.5 - (row + .5) / 360) * np.pi


def skybox(panorama, yaw=0., size=1024):
    """MuJoCo skybox grid image (gridsize 4x3, layout '.U..LFRB.D..') with the
    panorama's centre column at world azimuth `yaw` (radians from +x)."""
    c = (np.arange(size) + .5) / size * 2 - 1
    s, t = np.meshgrid(c, c)
    one = np.ones_like(s)
    faces = dict(R=(one, -t, -s), L=(-one, -t, s), U=(s, one, t), D=(s, -one, -t), F=(s, -t, one), B=(-s, -t, -one))
    grid = np.zeros((4 * size, 3 * size, 3), np.uint8)
    height, width = panorama.shape[:2]
    for k, face in enumerate('.U..LFRB.D..'):
        if face != '.':
            cx, cy, cz = faces[face]
            x, y, z = cx, -cz, cy  # MuJoCo samples the cube at (x, z, -y) of a z-up world direction
            azimuth = np.arctan2(y, x) - yaw
            elevation = np.arctan2(z, np.hypot(x, y))
            col = ((.5 - azimuth / (2 * np.pi)) % 1 * width).astype(np.int64) % width
            row = np.clip(((.5 - elevation / np.pi) * height).astype(np.int64), 0, height - 1)
            grid[(k // 3) * size:(k // 3 + 1) * size, (k % 3) * size:(k % 3 + 1) * size] = panorama[row, col]
    return grid


def hdri(asset, resolution='4k'):
    """Radiance .hdr panorama, for image-based lighting."""
    return _get(_files(asset)['hdri'][resolution]['hdr']['url'], CACHE / asset / f'{resolution}.hdr')


def model(asset, resolution='1k'):
    """glTF model with its textures; returns the .gltf path."""
    entry = _files(asset)['gltf'][resolution]['gltf']
    for relative, item in entry.get('include', {}).items():
        _get(item['url'], CACHE / asset / relative)
    return _get(entry['url'], CACHE / asset / f'{asset}.gltf')
