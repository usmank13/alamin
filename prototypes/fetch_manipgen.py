#!/usr/bin/env python3
"""Fetch a pinned archive (opt-in), safely extract six assets, and inventory them.

Standard library only; never imports or executes anything from the archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.request
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REVISION = "5459c7323a1c04e0207ad30fc28f78e62ff168fc"
REPO = "https://huggingface.co/datasets/minliu01/ManipGen-PartNet"
URL = f"{REPO}/resolve/{REVISION}/manipgen-partnet.tar.gz"
SHA256 = "a4120e4eb663f2976f7067547e19c55a09dc839d2fc4e9d06a98731c8448d555"
SIZE = 552155633
SELECTED = {
    "door-33507-18-0": "short horizontal handle",
    "door-30857-13-0": "compact round handle",
    "door-40402-16-0": "slender vertical handle",
    "drawer-46254-5-0": "round handle",
    "drawer-47167-8-0": "wide horizontal handle",
    "drawer-47710-9-0": "broad horizontal handle (bounding-box observation)",
}
# Exact pinned README, retrieved as text on 2026-09-16. Keep spelling intact.
README = """---
license: mit
task_categories:
- robotics
language:
- en
size_categories:
- 100M<n<1B
---
# ManipGen-PartNet
ManipGen-PartNet is used to train grasp handle, open, and close policies in [ManipGen](https://mihdalal.github.io/manipgen/). 

The dataset contains 2K+ cabinet assets (drawers and doors) for training robotic manipulation policies in simulation. We sample 1K+ handles from [PartNet](https://partnet.cs.stanford.edu/) and assemble them with procedually generated cabinet bodies.


### Structure
* `meshdata`: mesh and urdf
* `graspdata/`: pre-sampled grasp poses for Franka arm with UMI gripper
* `trainset3419.txt`: the list of 2655 objects used to train grasp handle, open, and close policies in ManipGen

"""


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="manipgen-download-", dir=path.parent)
    start = last = time.monotonic()
    size = 0
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(URL, timeout=30) as src:
            while chunk := src.read(1024 * 1024):
                size += len(chunk)
                if size > SIZE or time.monotonic() - start > 1800:
                    raise ValueError("Download exceeded pinned size or 30-minute budget")
                out.write(chunk)
                if time.monotonic() - last >= 5:
                    print(f"Downloaded {size:,}/{SIZE:,} bytes", flush=True)
                    last = time.monotonic()
        if size != SIZE or digest(Path(temporary)) != SHA256:
            raise ValueError("Downloaded archive failed size/SHA-256 verification")
        if path.exists():
            raise FileExistsError(path)
        os.rename(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract(archive, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = total = 0
    found = set()
    with tempfile.TemporaryDirectory(prefix="manipgen-subset-", dir=destination.parent) as temp:
        stage = Path(temp) / "subset"
        stage.mkdir()
        with tarfile.open(archive, "r|gz") as src:
            for member in src:
                count += 1
                if count > 25000:
                    raise ValueError("Archive member count exceeded bound")
                path = PurePosixPath(member.name)
                if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                    raise ValueError(f"Unsafe archive path: {member.name}")
                if not (member.isfile() or member.isdir()):
                    raise ValueError(f"Non-regular archive member: {member.name}")
                parts = path.parts
                if len(parts) != 4 or parts[:2] != ("partnet", "meshdata") or parts[2] not in SELECTED:
                    continue
                allowed = {"coacd.urdf", "meta.yaml", "decomposed.obj", "cabinet_body.obj"}
                if parts[2].startswith("drawer-"):
                    allowed.add("drawer_body.obj")
                if parts[3] not in allowed or not member.isfile():
                    raise ValueError(f"Unexpected selected member: {member.name}")
                if member.name in found or not 0 < member.size <= 16 * 1024 * 1024:
                    raise ValueError(f"Duplicate/oversized/empty member: {member.name}")
                found.add(member.name)
                total += member.size
                if total > 64 * 1024 * 1024:
                    raise ValueError("Subset exceeds 64 MiB")
                target = stage.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with src.extractfile(member) as incoming, target.open("xb") as out:
                    shutil.copyfileobj(incoming, out)
                print(f"Extracted {member.name} ({member.size:,} bytes)", flush=True)
        for asset in SELECTED:
            folder = stage / "partnet/meshdata" / asset
            expected = {"coacd.urdf", "meta.yaml", "decomposed.obj", "cabinet_body.obj"}
            if asset.startswith("drawer-"):
                expected.add("drawer_body.obj")
            if not folder.is_dir() or {p.name for p in folder.iterdir()} != expected:
                raise ValueError(f"Incomplete asset: {asset}")
        (stage / "SOURCE_README.md").write_text(README)
        if destination.is_symlink():
            raise ValueError("Destination must not be a symlink")
        if destination.exists():
            wanted = {p.relative_to(stage) for p in stage.rglob("*") if p.is_file()}
            existing = {p.relative_to(destination) for p in destination.rglob("*") if p.is_file()}
            if wanted != existing or any(p.is_symlink() for p in destination.rglob("*")):
                raise ValueError("Existing subset differs; refusing to overwrite")
            for rel in wanted:
                if digest(stage / rel) != digest(destination / rel):
                    raise ValueError(f"Existing file differs: {rel}; refusing to overwrite")
        else:
            os.rename(stage, destination)
    return count, total


def inventory(destination):
    assets = []
    for name, form in SELECTED.items():
        folder = destination / "partnet/meshdata" / name
        robot = ET.parse(folder / "coacd.urdf").getroot()
        links = [x.attrib["name"] for x in robot.findall("link")]
        joints, issues, seen = [], [], set()
        for joint in robot.findall("joint"):
            data = dict(joint.attrib)
            for field in ("parent", "child", "axis", "origin", "limit", "dynamics"):
                node = joint.find(field)
                if node is not None:
                    data[field] = dict(node.attrib)
            if data["name"] in seen:
                issues.append(f"Duplicate joint name: {data['name']}")
            seen.add(data["name"])
            for field in ("parent", "child"):
                if data[field]["link"] not in links:
                    issues.append(f"Joint {data['name']} has missing {field} link: {data[field]['link']}")
            joints.append(data)
        meshes = []
        for mesh in robot.findall(".//mesh"):
            filename = mesh.attrib["filename"]
            if PurePosixPath(filename).name != filename or not (folder / filename).is_file():
                raise ValueError(f"Unresolved/nonlocal mesh: {filename}")
            if filename not in meshes:
                meshes.append(filename)
        vertices = []
        with (folder / "decomposed.obj").open() as src:
            for line in src:
                if line.startswith("v "):
                    vertices.append(list(map(float, line.split()[1:4])))
        lower = [min(v[i] for v in vertices) for i in range(3)]
        upper = [max(v[i] for v in vertices) for i in range(3)]
        files = [{"path": str(p.relative_to(ROOT)), "bytes": p.stat().st_size,
                  "sha256": digest(p)} for p in sorted(folder.iterdir())]
        assets.append({"id": name, "category": name.split("-")[0], "handle_form_observation": form,
                       "urdf": str((folder / "coacd.urdf").relative_to(ROOT)),
                       "links": links, "joints": joints, "mesh_filenames": meshes,
                       "handle_bounds_xyz": {"min": lower, "max": upper,
                                             "extent": [b-a for a, b in zip(lower, upper)]},
                       "inertial_elements": len(robot.findall(".//inertial")),
                       "source_issues": issues, "files": files})
    return assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download only if the pinned archive is absent")
    args = parser.parse_args()
    archive = ROOT / "vendor/partnet_pilot/manipgen-partnet.tar.gz"
    destination = ROOT / "vendor/partnet_pilot/manipgen_subset"
    if not archive.exists():
        if not args.download:
            parser.error("Archive missing. Supply --download to fetch the pinned 552 MB archive.")
        fetch(archive)
    if archive.stat().st_size != SIZE or digest(archive) != SHA256:
        raise ValueError("Existing archive failed pinned size/SHA-256 verification")
    print("Archive size/SHA-256 verified", flush=True)
    members, size = extract(archive, destination)
    readme = destination / "SOURCE_README.md"
    manifest = {
        "schema_version": 1,
        "source": {"dataset": "minliu01/ManipGen-PartNet", "revision": REVISION,
                   "url": URL, "archive": str(archive.relative_to(ROOT)), "bytes": SIZE,
                   "sha256": SHA256, "reviewed_date": "2026-09-16",
                   "readme_url": f"{REPO}/raw/{REVISION}/README.md",
                   "readme_snapshot": str(readme.relative_to(ROOT)), "readme_sha256": digest(readme),
                   "declared_license": "MIT (dataset-card metadata)",
                   "license_caveat": "Archive contains no README/LICENSE; upstream PartNet handle terms not independently resolved. Dataset-card MIT is a declaration, not proof of relicensing upstream geometry.",
                   "upstream_handles": "https://partnet.cs.stanford.edu/",
                   "project": "https://mihdalal.github.io/manipgen/",
                   "composition": "Procedurally generated cabinet bodies assembled with PartNet handles; not original PartNet-Mobility scenes or the 15-asset/five-category P5 set."},
        "extraction": {"archive_members_scanned": members, "asset_bytes": size,
                       "max_asset_bytes": 64 * 1024 * 1024, "asset_count": len(SELECTED),
                       "raw_files_unmodified": True, "graspdata_excluded": True},
        "assets": inventory(destination),
    }
    output = ROOT / "prototypes/dataset_sources.json"
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {output}; {len(SELECTED)} assets, {size:,} raw bytes", flush=True)


if __name__ == "__main__":
    main()
