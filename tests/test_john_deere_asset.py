"""The public tractor download is cached, checked, and carries attribution."""
import hashlib
import importlib.util
from io import BytesIO
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('john_deere_asset', Path(__file__).resolve().parents[1]/'scripts/john_deere_asset.py')
asset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(asset)


def test_verified_download_and_cached_corruption(tmp_path, monkeypatch):
    payload = b'test model bytes'
    monkeypatch.setattr(asset, 'ROOT', tmp_path)
    monkeypatch.setattr(asset, 'TRACTOR', tmp_path/'tractor.glb')
    monkeypatch.setattr(asset, 'SHA256', hashlib.sha256(payload).hexdigest())
    calls = []

    def fetch(request, timeout):
        calls.append(request.full_url)
        return BytesIO(payload)

    monkeypatch.setattr(asset.urllib.request, 'urlopen', fetch)
    assert asset.prepare().read_bytes() == payload
    assert asset.prepare().read_bytes() == payload
    assert calls == [asset.URL]
    assert 'Daan van Leeuwen' in (tmp_path/'tractor_LICENSE.txt').read_text()
    asset.TRACTOR.write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='checksum mismatch'):
        asset.prepare()
    assert asset.TRACTOR.read_bytes() == b'corrupted'
