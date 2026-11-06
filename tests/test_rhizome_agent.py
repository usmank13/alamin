import json
import struct

import numpy as np
import pytest

from scene_pipeline.rhizome_agent import stage
from scene_pipeline.rhizome_camera import combo_frame


def test_combo_wire_format():
    rgb = np.arange(24,dtype=np.uint8).reshape(2,4,3)
    depth = np.array([[1,2,3,4],[.1,np.nan,7,2.5]])
    payload = combo_frame(rgb,depth,123456789012345)
    color = np.frombuffer(payload[:32],np.uint8).reshape(2,4,4)
    np.testing.assert_array_equal(color[:,:,:3],rgb)
    assert np.all(color[:,:,3] == 255)
    assert struct.unpack_from('<Q',payload,32)[0] == 123456789012345
    assert np.frombuffer(payload[40:],'<u2').tolist() == [1000,0,0,25000]
    with pytest.raises(ValueError,match='dimensions'):
        combo_frame(rgb,depth[:1],1)


def test_stage_preserves_task_and_native_node_config(tmp_path):
    root = tmp_path/'rhizome'
    nodes = root/'src/ai_agent'
    nodes.mkdir(parents=True)
    node = nodes/'ai_agent.node.yml'
    node.write_text('target_devices: [rover]\ntrigger: {type: ON_TIMER, period: 1.0}\n---\nmodels: {claude: source-model}\n')
    (root/'src/tasks').mkdir()
    (root/'src/tasks/test.yml').write_text('nodes:\n  ai_agent:\n    odom_max_age_s: 0.7\n    models: {claude: task-model}\n')
    output = tmp_path/'run'
    runtime = stage(root,output,'test',{'keys':{'anthropic':'/credential/file'},'voice':{'tts_voice':'custom'}})
    task = json.loads((runtime/'opt/task/task.json').read_text())
    cfg = task['nodes']['ai_agent']
    assert cfg['models']['claude'] == 'task-model'
    assert cfg['odom_max_age_s'] == .7
    assert cfg['keys']['anthropic'] == '/credential/file'
    assert cfg['voice'] == {'dir':'/data/agent/voice','tts_voice':'custom'}
    assert (runtime/'conf/nodes/ai_agent.yml').resolve() == node
    assert 'source-model' in node.read_text()


def test_stage_rejects_task_without_agent(tmp_path):
    root = tmp_path/'rhizome'
    (root/'src/tasks').mkdir(parents=True)
    (root/'src/tasks/test.yml').write_text('{}')
    with pytest.raises(ValueError,match='does not select ai_agent'):
        stage(root,tmp_path/'run','test')
