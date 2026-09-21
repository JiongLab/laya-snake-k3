"""Build/extract the actual release package when the wheel bundle is present."""
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not shutil.which('dpkg-deb') or len(list((ROOT/'wheelhouse').glob('*.whl'))) != 16,
                    reason='package build needs dpkg-deb and the release wheel bundle')
def test_package_permissions_payload_and_extracted_help(tmp_path):
    subprocess.run([sys.executable,str(ROOT/'scripts/build-deb.py')],check=True)
    package = ROOT/'dist/laya-snake-k3_0.1.0_riscv64.deb'
    info = subprocess.check_output(['dpkg-deb','--field',str(package),'Architecture'],text=True)
    assert info.strip() == 'riscv64'
    data = subprocess.check_output(['dpkg-deb','--fsys-tarfile',str(package)])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        members = {m.name:m for m in archive.getmembers()}
        assert members['.'].mode == 0o755, 'Archive root must not inherit private temporary-directory permissions'
        assert members['./usr/bin/laya-snake-k3'].mode & 0o111
        assert './usr/share/laya-snake-k3/docs/licenses/tokenizers-LICENSE' in members
        assert not any(name.endswith(('.onnx','.safetensors','.pyc')) or '/.git/' in name for name in members)
    subprocess.run(['dpkg-deb','--extract',str(package),str(tmp_path)],check=True)
    app = tmp_path/'usr/share/laya-snake-k3'
    result = subprocess.run(['sh',str(app/'setup.sh'),'--help'],capture_output=True,text=True)
    assert result.returncode == 0
    assert '--model-source' in result.stdout
