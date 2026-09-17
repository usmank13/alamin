"""Install an official checksum-verified Blender binary locally, without sudo."""
import hashlib
from pathlib import Path
import tarfile
import urllib.request

VERSION='4.2.3'
BASE='https://download.blender.org/release/Blender4.2/'


def main():
    root=Path('vendor/blender'); root.mkdir(parents=True,exist_ok=True)
    name=f'blender-{VERSION}-linux-x64.tar.xz'
    def fetch(url,timeout):
        return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=timeout)
    checks=fetch(BASE+f'blender-{VERSION}.sha256',60).read().decode()
    expected=next(line.split()[0] for line in checks.splitlines() if line.strip().endswith(name))
    archive=root/name
    if not archive.exists():
        with fetch(BASE+name,120) as src, archive.open('wb') as dest:
            while chunk:=src.read(1024*1024): dest.write(chunk)
    with archive.open('rb') as source:
        actual=hashlib.file_digest(source,'sha256').hexdigest()
    if actual!=expected: raise RuntimeError('Blender checksum mismatch; archive not extracted')
    folder=root/name.removesuffix('.tar.xz')
    if not folder.exists():
        with tarfile.open(archive) as tar:
            tar.extractall(root,filter='data')
    print(folder/'blender')


if __name__=='__main__': main()
