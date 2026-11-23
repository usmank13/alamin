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
    paths=subprocess.check_output(['git','-C',str(root),'ls-files','--cached','--others','--exclude-standard','-z'],text=True).split('\0')
    digest=hashlib.sha256()
    for name in sorted(paths):
        path=root/name
        if path.is_file() and (path.suffix in ('.hpp','.h','.cpp','.c','.mk') or name=='Makefile'):
            digest.update(name.encode()+b'\0');digest.update(path.read_bytes())
    return dict(revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
                compiled_sources_sha256=digest.hexdigest())


def stamp(root, host, identity, *, binaries=(), sim_odometry=False):
    root=Path(root).resolve();host=Path(host).resolve()
    if source_identity(root)!=identity:raise ValueError('Rhizome sources changed during build; rebuild in a fresh directory')
    modules={}
    for name in MODULES:
        matches=list((host/'py/pyzome').glob(name+'.*.so'))
        if len(matches)!=1:raise ValueError(f'Expected exactly one compiled {name} extension')
        modules[matches[0].name]=hashlib.sha256(matches[0].read_bytes()).hexdigest()
    executables={}
    for name in binaries:
        if Path(name).name!=name:
            raise ValueError('Invalid executable name')
        executables[name]=hashlib.sha256((host/'bin'/name).read_bytes()).hexdigest()
    manifest=dict(source_root=str(root),**identity,modules_sha256=modules,binaries_sha256=executables)
    if sim_odometry:
        manifest['sim_odometry_hooks_sha256'] = {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((host.parent/'sim-odometry').glob('*.cpp'))}
    (host/'build_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify(host, root=None):
    """Verify the selected checkout, not just the checkout named by the build."""
    host=Path(host).resolve()
    manifest=json.loads((host/'build_manifest.json').read_text())
    root=Path(root or manifest['source_root']).resolve()
    if source_identity(root)['compiled_sources_sha256']!=manifest['compiled_sources_sha256']:
        raise ValueError('Rhizome source/build mismatch; rebuild host extensions in a fresh directory')
    for module in MODULES:
        matches=list((host/'py/pyzome').glob(module+'.*.so'))
        if len(matches)!=1 or matches[0].name not in manifest['modules_sha256']:
            raise ValueError(f'Expected exactly one verified {module} extension')
    for name,digest in manifest['modules_sha256'].items():
        if Path(name).name!=name:
            raise ValueError('Invalid extension path in Rhizome build manifest')
        if hashlib.sha256((host/'py/pyzome'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Rhizome extension changed after verified build: {name}')
    for name,digest in manifest.get('binaries_sha256',{}).items():
        if Path(name).name!=name:
            raise ValueError('Invalid executable path in Rhizome build manifest')
        if hashlib.sha256((host/'bin'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Rhizome executable changed after verified build: {name}')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='Fresh build directory; host modules go under output/host')
    parser.add_argument('--sim-odometry', action='store_true',
                        help='Build native simulation clock and Odometry description hooks')
    parser.add_argument('--jobs',type=int,default=4)
    selection=parser.add_mutually_exclusive_group()
    selection.add_argument('--binaries', nargs='+', metavar='NAME',
                           help='Build and verify specific host executables from this checkout')
    selection.add_argument('--task',help='Also build native executables selected by this Rhizome task')
    args=parser.parse_args();root=args.root.resolve();output=args.output.resolve()
    if output.exists():parser.error('Choose a fresh output directory to avoid stale dependency files and archive members')
    if args.jobs<1:parser.error('--jobs must be positive')
    identity=source_identity(root);suffix=sysconfig.get_config_var('EXT_SUFFIX')
    host=output/'host'
    targets=[str(host/'py/pyzome'/(name+suffix)) for name in MODULES]
    names=set(args.binaries or [])
    if args.task:
        from .rhizome_task import Task
        task=Task(root,args.task)
        names={name for nodes in task.services().values() for name in nodes}
    native={path.name.removesuffix('.main.cpp') for path in (root/'src').glob('*/*.main.cpp')}
    scripts={path.name.removesuffix('.main.py') for path in (root/'src').glob('*/*.main.py')} if args.task else set()
    unknown=names-native-scripts
    if unknown:
        parser.error(f'No buildable entry point for {", ".join(sorted(unknown))}')
    binaries=sorted(native & names)
    targets.extend(str(host/'bin'/name) for name in binaries)
    output.mkdir(parents=True)
    makefiles = []
    if args.sim_odometry:
        from .rhizome_sim_build import host_overlay
        makefiles = ['-f', 'Makefile', '-f', str(host_overlay(root, output))]
    with (output/'build.log').open('w') as log:
        subprocess.run(['make','-C',str(root),*makefiles,f'BUILD_DIR={output}',f'JOBS={args.jobs}','CCACHE=',*targets],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(stamp(root,host,identity,binaries=binaries,sim_odometry=args.sim_odometry),indent=2))


if __name__=='__main__':main()
