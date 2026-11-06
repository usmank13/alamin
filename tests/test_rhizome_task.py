import json
import os
from pathlib import Path

import pytest
import yaml

from scene_pipeline.rhizome_task import Task
from test_rhizome_build import install


def write(path, text):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(text)


@pytest.fixture
def source(tmp_path):
    write(tmp_path/'src/nav/nav.node.yml', '''
trigger: {type: ON_FUNCTION}
target_devices: [rover]
includes: [camera/nav.yml, payloads.yml]
---
drive: false
invariants: {old/health: 1, retained/health: 2}
camera: {nav: {existing: 10}}
''')
    write(tmp_path/'src/arm/arm.node.yml', 'target_devices: [weeder]\n---\n{}\n')
    write(tmp_path/'src/agent/agent.node.yml', 'target_devices: []\n---\n{}\n')
    write(tmp_path/'src/camera/nav.yml', 'on: true\nresolution: {width: 640, height: 480}\n')
    write(tmp_path/'src/tasks/example.yml', '''
disabled_services: [arm]
extra_services: {rover: [agent]}
camera:
  nav: {resolution: {width: 848}}
nodes:
  nav:
    drive: true
    invariants: {old/health: null}
''')
    return tmp_path


def test_task_resolves_device_services_includes_and_merge_patch(source):
    task=Task(source,'example')
    snapshot=task.snapshot()
    assert snapshot['services']=={'rover':['agent','nav'],'weeder':[]}
    node=snapshot['nodes']['nav']
    assert node['config']['drive'] is True
    assert node['config']['invariants']=={'retained/health':2}
    assert node['config']['camera']['nav']==dict(on=True,existing=10,resolution=dict(width=848,height=480))
    assert node['missing_includes']==['payloads.yml']
    assert snapshot['configuration_complete'] is False
    # The dedicated loader must not change PyYAML's global resolver table.
    assert yaml.safe_load('on: true')=={True:True}


def test_hardware_includes_require_explicit_configuration(source,tmp_path):
    config=tmp_path/'hardware'
    write(config/'payloads.yml','payloads: [1, 2]\n')
    snapshot=Task(source,'example',config_dir=config).snapshot()
    assert snapshot['configuration_complete']
    assert snapshot['nodes']['nav']['config']['payloads']=={'payloads':[1,2]}
    assert str(config/'payloads.yml') in snapshot['source_sha256']


def test_exclusive_services_and_unknown_nodes(source):
    path=source/'src/tasks/example.yml'
    path.write_text('services: {rover: [agent]}\nextra_services: {rover: [nav]}\n')
    assert Task(source,'example').services()=={'rover':['agent'],'weeder':[]}
    path.write_text('services: {rover: [typo]}\n')
    with pytest.raises(ValueError,match='without node configs'):Task(source,'example').snapshot()
    path.write_text('services: {typo: [agent]}\n')
    with pytest.raises(ValueError,match='without node targets'):Task(source,'example').snapshot()


def test_task_changes_are_read_live_and_cannot_escape_checkout(source):
    first=Task(source,'example').snapshot()
    path=source/'src/tasks/example.yml'
    path.write_text(path.read_text().replace('drive: true','drive: false'))
    second=Task(source,'example').snapshot()
    assert first['source_sha256']!=second['source_sha256']
    assert second['nodes']['nav']['config']['drive'] is False
    with pytest.raises(ValueError,match='Task must name'):Task(source,'../example')


def test_inspection_cli_verifies_build_and_records_task(install,capsys,tmp_path):
    from scene_pipeline.cli import main
    root,host=install
    write(root/'src/tasks/example.yml','services: {rover: [agent]}\n')
    write(root/'src/agent/agent.node.yml','target_devices: [rover]\n---\n{}\n')
    output=tmp_path/'task.json'
    assert main(['rhizome-inspect','--root',str(root),'--host-build',str(host),
                 '--task','example','--output',str(output)])==0
    report=json.loads(capsys.readouterr().out)
    assert report==json.loads(output.read_text())
    assert report['services']=={'rover':['agent']}
    assert report['install']['python_source']==str(root/'src')


def test_selected_checkout_tasks():
    root=os.environ.get('RHIZOME_ROOT')
    if not root:pytest.skip('Set RHIZOME_ROOT to check real task contracts')
    agent=Task(root,'dual-agent').snapshot()
    drive=Task(root,'nav-model-drive').snapshot()
    assert 'ai_agent' in agent['services']['rover']
    assert agent['nodes']['nav_model']['config']['drive'] is False
    assert agent['nodes']['local_control']['config']['goals_topic']=='/nav/goals'
    assert drive['nodes']['nav_model']['config']['drive'] is True
    assert drive['nodes']['local_control']['config']['goals_topic']=='/nav_model/goals'
