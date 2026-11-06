"""Select a checkout and its verified host build without importing native code."""
import os
from pathlib import Path
import subprocess

from .contracts import PipelineError
from .rhizome_build import source_identity, verify


def install_paths(root=None, host_build=None):
    root=root or os.environ.get('RHIZOME_ROOT')
    if not root:
        raise PipelineError('RHIZOME_CONFIG', 'Set RHIZOME_ROOT or specify the Rhizome checkout explicitly')
    root=Path(root).expanduser().resolve()
    host=Path(host_build or os.environ.get('RHIZOME_HOST_BUILD') or root/'build/host').expanduser().resolve()
    return root,host


def inspect_install(root=None, host_build=None):
    root,host=install_paths(root,host_build)
    if not (root/'src').is_dir():
        raise PipelineError('RHIZOME_INSTALL', f'Not a Rhizome checkout: {root}')
    try:
        manifest=verify(host,root)
        identity=source_identity(root)
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as exc:
        raise PipelineError('RHIZOME_INSTALL', f'{exc}; build with scene_pipeline.rhizome_build') from exc
    return dict(root=str(root),host_build=str(host),revision=identity['revision'],
                compiled_sources_sha256=identity['compiled_sources_sha256'],build_manifest=manifest,
                python_source=str(root/'src'))
