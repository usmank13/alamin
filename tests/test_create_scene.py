"""The one-prompt wrapper delegates to existing tools; no live services."""
from pathlib import Path
import runpy

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'create_scene.py'


def test_prompt_runner(tmp_path, monkeypatch):
    import scene_pipeline.cli
    calls = []
    output = tmp_path / 'scene'
    def pipeline(argv):
        calls.append(argv)
        if argv[0] == 'generate':
            output.mkdir()
            (output / 'scene.mjz').touch()
        return 0
    monkeypatch.setattr(scene_pipeline.cli, 'main', pipeline)
    main = runpy.run_path(str(SCRIPT))['main']
    assert main(['A tiled home kitchen', '--output', str(output), '--env-file', str(tmp_path / 'absent'),
                 '--materials', 'fal', '--clutter', 'fal']) == 0
    assert [c[0] for c in calls] == ['generate', 'render', 'inspect']
    assert '--program' not in calls[0]
    assert calls[0][calls[0].index('--prompt') + 1] == 'A tiled home kitchen'
    with pytest.raises(SystemExit):
        main(['new prompt', '--output', str(output)])


def test_credentials_are_data_and_environment_wins(tmp_path, monkeypatch):
    monkeypatch.setenv('FAL_API_KEY', 'existing')
    monkeypatch.delenv('FAL_KEY', raising=False)
    path = tmp_path / '.env'
    path.write_text('FAL_API_KEY=ignored\nexport FAL_KEY="literal-$(echo no)" # comment\nPATH=ignored\n')
    load = runpy.run_path(str(SCRIPT))['load_credentials']
    load(path)
    import os
    assert os.environ['FAL_API_KEY'] == 'existing'
    assert os.environ['FAL_KEY'] == 'literal-$(echo no)'
    assert os.environ['PATH'] != 'ignored'


def test_failed_generation_keeps_failure_status(tmp_path, monkeypatch):
    import scene_pipeline.cli
    output = tmp_path / 'failed'
    calls = []
    def pipeline(argv):
        calls.append(argv[0])
        if argv[0] == 'generate':
            output.mkdir()
            return 2
        return 0
    monkeypatch.setattr(scene_pipeline.cli, 'main', pipeline)
    main = runpy.run_path(str(SCRIPT))['main']
    assert main(['prompt', '--output', str(output), '--env-file', str(tmp_path / 'absent')]) == 2
    assert calls == ['generate', 'inspect']
