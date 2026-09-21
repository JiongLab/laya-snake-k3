"""Portability and model integrity are part of the installation contract."""
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys
import pytest


@pytest.fixture(autouse=True)
def restore_policy_environment(monkeypatch):
    yield
    monkeypatch.undo()
    from snake_k3 import policy
    importlib.reload(policy)


def test_default_model_follows_user_data_directory(monkeypatch, tmp_path):
    from snake_k3 import policy
    monkeypatch.setenv('XDG_DATA_HOME', str(tmp_path / 'user data'))
    monkeypatch.delenv('LAYA_K3_HOME', raising=False)
    importlib.reload(policy)
    expected = tmp_path / 'user data/laya-snake-k3/models/multilingual'
    expected.mkdir(parents=True)
    assert policy.local_checkpoint() == expected


def test_explicit_install_home_is_independent_of_working_directory(monkeypatch, tmp_path):
    from snake_k3 import policy
    monkeypatch.setenv('LAYA_K3_HOME', str(tmp_path / 'another user'))
    monkeypatch.chdir(tmp_path)
    importlib.reload(policy)
    expected = tmp_path / 'another user/models/multilingual'
    expected.mkdir(parents=True)
    assert policy.local_checkpoint() == expected


def test_model_verification_accepts_exact_files_and_rejects_damage(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts/models.py'
    model = tmp_path / 'models'
    model.mkdir()
    (model / 'sample.bin').write_bytes(b'known model')
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': [{'path': 'sample.bin',
        'url': 'https://example.invalid/not-used',
        'sha256': hashlib.sha256(b'known model').hexdigest()}]}))
    command = [sys.executable, str(script), '--manifest', str(manifest),
               '--directory', str(model), '--verify-only']
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    (model / 'sample.bin').write_bytes(b'damaged')
    run = subprocess.run(command, capture_output=True, text=True)
    assert run.returncode != 0
    assert 'SHA256' in run.stderr


def test_launcher_missing_environment_explains_setup_without_loading_model(tmp_path):
    import os
    launcher = Path(__file__).resolve().parents[1] / 'start-snake.sh'
    run = subprocess.run(['sh', str(launcher)], capture_output=True, text=True,
                         env={**os.environ, 'LAYA_K3_HOME': str(tmp_path)})
    assert run.returncode != 0
    assert 'setup' in run.stderr.lower()


def test_setup_help_does_not_install_anything(tmp_path):
    import os
    script = Path(__file__).resolve().parents[1] / 'setup.sh'
    run = subprocess.run(['sh', str(script), '--help'], capture_output=True, text=True,
                         env={**os.environ, 'LAYA_K3_HOME': str(tmp_path/'empty')})
    assert run.returncode == 0, run.stderr
    assert '--model-source' in run.stdout
    assert not (tmp_path/'empty').exists()


def test_model_source_copy_is_verified_before_installation(tmp_path):
    script = Path(__file__).resolve().parents[1]/'scripts/models.py'
    source = tmp_path/'source'
    source.mkdir()
    (source/'weights').write_bytes(b'right weights')
    manifest = tmp_path/'lock.json'
    manifest.write_text(json.dumps({'files':[{'path':'weights','url':'unused',
        'sha256':hashlib.sha256(b'right weights').hexdigest()}]}))
    target = tmp_path/'new user models'
    run = subprocess.run([sys.executable,str(script),'--manifest',str(manifest),
        '--source',str(source),'--directory',str(target)],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    assert (target/'weights').read_bytes() == b'right weights'


def test_k3_help_does_not_advertise_unsupported_mlx_optimization():
    run = subprocess.run([sys.executable,'-m','snake_k3','--help'],capture_output=True,text=True)
    assert run.returncode == 0
    assert 'Unsupported on K3' in run.stdout
    assert 'Also: laya-snake benchmark' not in run.stdout
