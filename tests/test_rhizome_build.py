import subprocess

import pytest

from scene_pipeline.rhizome_build import MODULES,source_identity,stamp,verify


@pytest.fixture
def install(tmp_path):
    root=tmp_path/'source';(root/'src').mkdir(parents=True)
    subprocess.run(['git','init','-q',str(root)],check=True)
    (root/'src/node.cpp').write_text('int value = 1;\n')
    (root/'src/agent.py').write_text('version = 1\n')
    subprocess.run(['git','-C',str(root),'add','.'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','fixture'],check=True)
    host=tmp_path/'host';(host/'py/pyzome').mkdir(parents=True)
    for name in MODULES:(host/'py/pyzome'/f'{name}.test.so').write_bytes(name.encode())
    stamp(root,host,source_identity(root))
    return root,host


def test_build_checks_selected_checkout_but_allows_python_changes(install,tmp_path):
    root,host=install
    other=tmp_path/'other'
    subprocess.run(['git','clone','-q',str(root),str(other)],check=True)
    (other/'src/agent.py').write_text('version = 2\n')
    subprocess.run(['git','-C',str(other),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qam','Python-only update'],check=True)
    assert verify(host,other)['source_root']==str(root)
    assert source_identity(root)['revision']!=source_identity(other)['revision']
    (other/'src/node.cpp').write_text('int value = 2;\n')
    with pytest.raises(ValueError,match='source/build mismatch'):verify(host,other)


def test_manifest_must_cover_all_loaded_extensions(install):
    root,host=install
    (host/'py/pyzome/core.stale.so').write_bytes(b'stale')
    with pytest.raises(ValueError,match='exactly one verified core'):verify(host,root)


def test_manifest_checks_built_node_executables(install):
    root,host=install
    (host/'bin').mkdir()
    binary=host/'bin/drive_control'
    binary.write_bytes(b'original native node')
    stamp(root,host,source_identity(root),binaries=['drive_control'])
    assert 'drive_control' in verify(host,root)['binaries_sha256']
    binary.write_bytes(b'replaced native node')
    with pytest.raises(ValueError,match='executable changed'):verify(host,root)


def test_new_untracked_native_sources_require_rebuild(install):
    root,host=install
    (root/'src/new_dependency.hpp').write_text('constexpr int version = 2;\n')
    with pytest.raises(ValueError,match='source/build mismatch'):verify(host,root)


def test_install_selection_uses_explicit_paths_before_environment(install,monkeypatch,tmp_path):
    from scene_pipeline.rhizome_install import inspect_install,install_paths
    root,host=install
    monkeypatch.setenv('RHIZOME_ROOT',str(root))
    monkeypatch.setenv('RHIZOME_HOST_BUILD',str(host))
    assert install_paths()==(root,host)
    assert inspect_install()['root']==str(root)
    assert install_paths(tmp_path,root)==(tmp_path,root)
    monkeypatch.delenv('RHIZOME_ROOT')
    with pytest.raises(ValueError,match='RHIZOME_ROOT'):install_paths()


def test_install_rejects_unstamped_build(install):
    from scene_pipeline.rhizome_install import inspect_install
    root,host=install
    (host/'build_manifest.json').unlink()
    with pytest.raises(ValueError,match='rhizome_build'):inspect_install(root,host)


def test_build_stamp_detects_changed_source_and_binary(tmp_path):
    root=tmp_path/'source';root.mkdir()
    subprocess.run(['git','init','-q',str(root)],check=True)
    source=root/'arm.cpp';source.write_text('int radius = 1;\n')
    subprocess.run(['git','-C',str(root),'add','arm.cpp'],check=True)
    subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                    'commit','-qm','fixture'],check=True)
    host=tmp_path/'host';modules=host/'py/pyzome';modules.mkdir(parents=True)
    for name in MODULES:(modules/(name+'.test.so')).write_bytes(name.encode())
    identity=source_identity(root)
    manifest=stamp(root,host,identity)
    assert verify(host)==manifest
    source.write_text('int radius = 2;\n')
    with pytest.raises(ValueError,match='source/build mismatch'):verify(host)
    with pytest.raises(ValueError,match='changed during build'):stamp(root,host,identity)
    source.write_text('int radius = 1;\n')
    (modules/'physics.test.so').write_bytes(b'stale binary')
    with pytest.raises(ValueError,match='extension changed'):verify(host)
