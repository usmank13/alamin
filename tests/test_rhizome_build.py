import subprocess

import pytest

from scene_pipeline.rhizome_build import MODULES,source_identity,stamp,verify


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
