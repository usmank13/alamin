#!/usr/bin/env python3
"""Bounded RoboCasa fixture fetcher; reads ZIP entries over HTTP ranges only."""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import time
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "vendor/robocasa_native"
COMMIT = "4f8a2980def75a55dff96b990745b83540425f09"
OLD_COMMIT = "756598a5be52e052339bb2d957426e39015c2afb"
ROBOSUITE_COMMIT = "5ce6643f3092639d08f7b0f90ed1c6a84f50552c"
ARCHIVES = {
    "v0.2": "https://utexas.box.com/shared/static/pobhbsjyacahg2mx8x4rm5fkz3wlmyzp.zip",
    "lightwheel": "https://utexas.box.com/shared/static/idbncsadpnaz1jfl4i6m8qejawk7p9pi.zip",
}
PREFIXES = ['fixtures/microwaves/Microwave075', 'fixtures/dishwashers/Dishwasher051',
            'fixtures/fridges/Refrigerator055']
MANIFEST = ROOT / 'prototypes/robocasa_sources.json'


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.source = url
        request = urllib.request.Request(url + "?download=1&nonce=" + str(time.time_ns()),
                                         headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            if response.status != 206:
                raise ValueError("Server lacks byte ranges; refusing full download")
            self.url = response.geturl()
            self.size = int(response.headers["Content-Range"].split("/")[-1])
            response.read(1)
        self.position = 0
        self.downloaded = 1

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if not 0 <= self.position <= self.size:
            raise ValueError("Out-of-range seek")
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = self.size-self.position if count < 0 else min(count, self.size-self.position)
        if not count:
            return b""
        if count > 64*1024**2 or self.downloaded + count > 160*1024**2:
            raise ValueError("Range fetch exceeds 64 MiB/request or 160 MiB/run")
        end = self.position + count - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
        with urllib.request.urlopen(request, timeout=30) as response:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if response.status != 206 or response.headers.get("Content-Range") != expected:
                raise ValueError("Incorrect range response; refusing full download")
            data = response.read(count+1)
        if len(data) != count:
            raise ValueError("Truncated range")
        self.position += count
        self.downloaded += count
        return data


def sha(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe path: {name}")
    return path


def save(path, data):
    target = DEST.joinpath(*safe_path(path).parts)
    if target.is_symlink() or any(p.is_symlink() for p in target.parents):
        raise ValueError("Symlink target")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != data:
            raise ValueError(f"Refusing overwrite: {target}")
    else:
        with target.open("xb") as out:
            out.write(data)
    return {"path": str(target.relative_to(ROOT)), "bytes": len(data), "sha256": sha(data)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", choices=ARCHIVES, default="lightwheel")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--prefix", action="append", default=[])
    parser.add_argument("--verify", action="store_true", help="Verify manifest hashes offline, without downloads")
    args = parser.parse_args()
    locked = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if args.verify:
        if locked is None:
            parser.error('No manifest to verify')
        for record in locked['files']:
            data = (ROOT / record['path']).read_bytes()
            if len(data) != record['bytes'] or sha(data) != record['sha256']:
                raise ValueError(f"Hash mismatch: {record['path']}")
        print(f"Verified {len(locked['files'])} file hashes offline")
        return
    args.prefix = args.prefix or PREFIXES
    expected = {r['path']: r['sha256'] for r in locked['files']} if locked else {}

    def save_locked(path, data):
        relative = str((DEST / path).relative_to(ROOT))
        if locked and ((relative not in expected and not path.startswith('provenance/')) or
                       (relative in expected and sha(data) != expected[relative])):
            raise ValueError(f'Source differs from locked manifest: {relative}')
        return save(path, data)

    remote = RemoteZip(ARCHIVES[args.archive])
    with zipfile.ZipFile(remote) as archive:
        members = archive.infolist()
        if args.list:
            for member in members:
                if member.filename.endswith('.xml') and any(s in member.filename.lower() for s in ('microwave','dishwasher','fridge')):
                    prefix = member.filename.rsplit('/',1)[0]+'/'
                    group = [x for x in members if x.filename.startswith(prefix)]
                    print(member.filename, 'files',len(group),'bytes',sum(x.file_size for x in group),flush=True)
            print('archive bytes',remote.size,'transferred',remote.downloaded)
            return
        if not args.prefix:
            parser.error("Specify fixture prefixes with --prefix")
        selected = [m for m in members if not m.is_dir() and any(m.filename.startswith(p.rstrip('/')+'/') for p in args.prefix)]
        if len(selected) > 2000 or sum(m.file_size for m in selected) > 128*1024**2:
            raise ValueError("Selection exceeds 2000 files / 128 MiB")
        records = []
        for member in selected:
            if member.file_size > 32*1024**2 or (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Oversized member or symlink")
            local = DEST.joinpath(*safe_path(member.filename).parts)
            key = str(local.relative_to(ROOT))
            if key in expected and local.is_file():
                data = local.read_bytes()
            else:
                data = archive.read(member)  # ZIP CRC verified by Python
            record = save_locked(member.filename, data)
            record.update(archive_member=member.filename, crc32=f'{member.CRC:08x}')
            records.append(record)
            print('Ready',record['path'],flush=True)
        commit = OLD_COMMIT if args.archive == 'v0.2' else COMMIT
        for path in ['LICENSE','README.md','robocasa/models/assets/box_links/box_links_assets.json',
                     'robocasa/models/fixtures/fixture.py', 'robocasa/models/fixtures/microwave.py',
                     'robocasa/models/fixtures/dishwasher.py',
                     'robocasa/models/objects/objects.py',
                     'robocasa/environments/kitchen/kitchen.py',
                     'robocasa/models/assets/fixtures/fixture_registry/microwave.yaml',
                     'robocasa/models/assets/fixtures/fixture_registry/dishwasher.yaml',
                     'robocasa/models/assets/fixtures/fixture_registry/fridge_bottom_freezer.yaml',
                     'robocasa/models/assets/fixtures/fixture_registry/fridge_side_by_side.yaml']:
            if args.archive == 'v0.2' and 'box_links' in path:
                continue
            url = f'https://raw.githubusercontent.com/robocasa/robocasa/{commit}/{path}'
            with urllib.request.urlopen(url, timeout=30) as response:
                data=response.read(1024*1024)
            record=save_locked('provenance/'+path,data)
            record['url']=url
            records.append(record)
        for path in ['LICENSE', 'robosuite/models/objects/objects.py',
                     'robosuite/models/base.py', 'robosuite/models/world.py',
                     'robosuite/models/tasks/task.py', 'robosuite/models/assets/base.xml']:
            url=f'https://raw.githubusercontent.com/ARISE-Initiative/robosuite/{ROBOSUITE_COMMIT}/{path}'
            with urllib.request.urlopen(url, timeout=30) as response:
                data=response.read(1024*1024)
            record=save_locked('provenance/robosuite_upstream/'+path,data)
            record['url']=url
            records.append(record)
        models=[]
        for record in records:
            if record['path'].endswith('.xml'):
                path=ROOT/record['path']
                xml=ET.parse(path).getroot()
                missing=[]
                for node in xml.findall('.//*[@file]'):
                    if not (path.parent/node.attrib['file']).is_file():
                        missing.append(node.attrib['file'])
                handles=[]
                for body in xml.findall('.//body'):
                    for geom in body.findall('geom'):
                        if 'handle' in geom.get('name',''):
                            handles.append({'body':body.get('name'), **geom.attrib})
                models.append({'xml':record['path'],'compiler':[n.attrib for n in xml.findall('compiler')],
                               'joints':[n.attrib for n in xml.findall('.//joint')],
                               'handle_collision_geoms':handles,
                               'main_region':[n.attrib for n in xml.findall('.//geom') if n.get('name') == 'reg_main'],
                               'closed_reference':'All joint positions zero; source joint refs default to zero.',
                               'missing_dependencies':missing})
        manifest={'repository':'https://github.com/robocasa/robocasa','commit':commit,
                  'archive_url':remote.source,'archive_bytes':remote.size,
                  'archive_hash':None,'archive_hash_note':'Partial range download: individual files SHA-256 and ZIP CRC verified; full archive not downloaded.',
                  'transferred_bytes':remote.downloaded,'prefixes':args.prefix,'files':records,'models':models}
        manifest['license']={'assets':'CC BY 4.0, per pinned upstream README', 'code':'MIT',
                             'asset_license_url':'https://creativecommons.org/licenses/by/4.0/',
                             'attribution':'RoboCasa Team; Lightwheel fixture collection distributed by RoboCasa. See preserved README and licenses.',
                             'raw_assets_modified':False}
        manifest['runtime_import_review']={
            'robosuite_repository':'https://github.com/ARISE-Initiative/robosuite',
            'robosuite_reviewed_commit':ROBOSUITE_COMMIT,
            'version_scope':'RoboCasa README requests robosuite master; this is the separately pinned inspected revision, not a robocasa dependency lock.',
            'scene_compiler_inertiagrouprange':'0 0',
            'effect':'Group-1 visual and region geoms excluded from inferred inertia; group-0 collision geoms remain.',
            'fixture_density_override_found':False,
            'joint_dynamics_override_found_in_reviewed_fixture_import_path':False,
            'g4_visual_handle_segmentation':'Not supplied. Named collision handle boxes do not establish visual mesh segmentation.',
            'diagnostic_mass_sums_kg':{
                'Microwave075':{'raw':383.04991626067687,'only_group0_inertia':67.47729127518737},
                'Dishwasher051':{'raw':1335.527190313961,'only_group0_inertia':228.18857792734633},
                'Refrigerator055':{'raw':4134.119522158324,'only_group0_inertia':515.2673981562925}},
            'diagnostic_scope':'MuJoCo 3.13 native scale, only inertiagrouprange changed in memory. Sum includes anchored bodies, not physical measured appliance mass or full runtime scene.'}
        if any(m['missing_dependencies'] for m in models):
            raise ValueError('Missing MJCF dependencies')
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
        print('Transferred',remote.downloaded,'bytes; manifest ready',flush=True)


if __name__ == '__main__':
    main()
