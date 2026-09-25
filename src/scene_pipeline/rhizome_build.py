"""Build and identify a coherent, isolated set of Rhizome host extensions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sysconfig

MODULES=('core','physics','nav','tracker','node_test')


def source_identity(root):
    root=Path(root)
    paths=subprocess.check_output(['git','-C',str(root),'ls-files','-z'],text=True).split('\0')
    digest=hashlib.sha256()
    for name in sorted(paths):
        path=root/name
        if path.is_file() and (path.suffix in ('.hpp','.h','.cpp','.c','.mk') or name=='Makefile'):
            digest.update(name.encode()+b'\0');digest.update(path.read_bytes())
    return dict(revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                compiled_sources_sha256=digest.hexdigest())


def stamp(root, host, identity):
    root=Path(root).resolve();host=Path(host).resolve()
    if source_identity(root)!=identity:raise ValueError('Rhizome sources changed during build; rebuild in a fresh directory')
    modules={}
    for name in MODULES:
        matches=list((host/'py/pyzome').glob(name+'.*.so'))
        if len(matches)!=1:raise ValueError(f'Expected exactly one compiled {name} extension')
        modules[matches[0].name]=hashlib.sha256(matches[0].read_bytes()).hexdigest()
    manifest=dict(source_root=str(root),**identity,modules_sha256=modules)
    (host/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify(host):
    host=Path(host)
    manifest=json.loads((host/'build_manifest.json').read_text())
    if source_identity(manifest['source_root'])['compiled_sources_sha256']!=manifest['compiled_sources_sha256']:
        raise ValueError('Rhizome source/build mismatch; rebuild host extensions in a fresh directory')
    for name,digest in manifest['modules_sha256'].items():
        if hashlib.sha256((host/'py/pyzome'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Rhizome extension changed after verified build: {name}')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh build directory; host modules go under output/host')
    parser.add_argument('--jobs',type=int,default=4)
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    if output.exists():parser.error('Choose a fresh output directory to avoid stale dependency files and archive members')
    if args.jobs<1:parser.error('--jobs must be positive')
    identity=source_identity(root);suffix=sysconfig.get_config_var('EXT_SUFFIX')
    output.mkdir(parents=True)
    host=output/'host'
    targets=[str(host/'py/pyzome'/(name+suffix)) for name in MODULES]
    with (output/'build.log').open('w') as log:
        subprocess.run(['make','-C',str(root),f'BUILD_DIR={output}',f'JOBS={args.jobs}','CCACHE=',*targets],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(stamp(root,host,identity),indent=2))


if __name__=='__main__':main()
