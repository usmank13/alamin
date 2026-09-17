"""Fetch only SVG annotations from the official archive using bounded HTTP ranges.

No images/models, no full 5 GB download. CubiCasa5K is CC BY-NC 4.0; this is a
local research sample, not a redistributable commercial training corpus.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

URL = 'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1'


class RemoteZip(io.RawIOBase):
    def __init__(self, url, budget=32 * 1024**2):
        self.url, self.budget, self.used, self.position = url, budget, 0, 0
        with urllib.request.urlopen(urllib.request.Request(url, method='HEAD'), timeout=30) as r:
            self.size = int(r.headers['Content-Length'])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        if self.position < 0: raise ValueError('Negative seek')
        return self.position

    def read(self, size=-1):
        size = min(self.size-self.position, size if size >= 0 else self.size)
        if not size: return b''
        if size < 0 or self.used+size > self.budget: raise ValueError('Download budget exceeded')
        start, end = self.position, self.position+size-1
        # A distinct query avoids intermediaries reusing a different cached range.
        request = urllib.request.Request(self.url+f'&range_start={start}', headers={'Range': f'bytes={start}-{end}'})
        with urllib.request.urlopen(request, timeout=45) as r:
            if r.status != 206 or r.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                raise ValueError('Server did not honor the requested byte range; refusing full download')
            data = r.read(size+1)
        if len(data) != size: raise ValueError('Truncated/oversized range')
        self.position += size; self.used += size
        return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=24)
    p.add_argument('--exclude-manifest',type=Path)
    p.add_argument('--freeze-splits',action='store_true',help='Assign train/calibration/test by source before downloading')
    a = p.parse_args()
    if not 1 <= a.count <= 100: p.error('count must be 1..100')
    if a.output.exists(): p.error('Refusing to overwrite an existing sample')
    remote = RemoteZip(URL)
    with zipfile.ZipFile(remote) as archive:
        names = sorted(n for n in archive.namelist() if n.endswith('/model.svg') and '__MACOSX' not in n)
        excluded=set()
        if a.exclude_manifest:
            excluded={e['archive_member'].split('/')[-2] for e in json.loads(a.exclude_manifest.read_text())['annotations']}
            names=[n for n in names if n.split('/')[-2] not in excluded]
        if len(names) < a.count: raise ValueError('Insufficient SVG entries')
        selected = [names[i*(len(names)-1)//max(a.count-1,1)] for i in range(a.count)]
        a.output.mkdir(parents=True)
        assignments={name:('test' if i%5==0 else 'calibration' if i%5==1 else 'train') for i,name in enumerate(selected)}
        if a.freeze_splits:
            (a.output/'split_assignment.json').write_text(json.dumps(assignments,indent=2)+'\n')
        entries = []
        for index, name in enumerate(selected):
            info = archive.getinfo(name)
            if info.file_size > 8*1024**2: raise ValueError('Oversized annotation')
            payload = archive.read(name)  # ZIP CRC checked by stdlib.
            filename = f'{index:03d}.svg'
            (a.output/filename).write_bytes(payload)
            entry=dict(file=filename,archive_member=name,sha256=hashlib.sha256(payload).hexdigest())
            if a.freeze_splits:entry['split']=assignments[name]
            entries.append(entry)
            print(f'{index+1}/{a.count}: {name}', flush=True)
    manifest = dict(schema_version=1, dataset='CubiCasa5K', source_url=URL,
                    license='CC-BY-NC-4.0', license_url='https://github.com/CubiCasa/CubiCasa5k/blob/master/LICENSE',
                    selection='evenly spaced in sorted SVG archive members; feasibility sample, not representative',
                    archive_bytes=remote.size,downloaded_bytes=remote.used,annotations=entries,
                    excluded_building_ids=sorted(excluded),splits_frozen_before_download=a.freeze_splits)
    (a.output/'sources.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(downloaded_bytes=remote.used, annotations=len(entries))))


if __name__ == '__main__': main()
