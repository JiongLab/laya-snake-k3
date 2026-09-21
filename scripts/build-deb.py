#!/usr/bin/env python3
"""Build the riscv64 application package on any Debian system with dpkg-deb."""
from pathlib import Path
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[1]
out = root/'dist'
out.mkdir(exist_ok=True)
if len(list((root/'wheelhouse').glob('*.whl'))) != 16:
    raise SystemExit('Expected 16 pinned runtime wheels; see docs/BUILD.md')
with tempfile.TemporaryDirectory(prefix='laya-deb-') as temporary:
    stage = Path(temporary)
    stage.chmod(0o755)
    app = stage/'usr/share/laya-snake-k3'
    app.mkdir(parents=True)
    for name in ('snake_k3','scripts','docs','wheelhouse'):
        shutil.copytree(root/name, app/name, ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for name in ('LICENSE','NOTICE','README.md','models.lock.json','requirements-runtime.txt','setup.sh','start-snake.sh'):
        shutil.copy2(root/name, app/name)
    for name in ('setup.sh','start-snake.sh'):
        (app/name).chmod(0o755)
    (stage/'DEBIAN').mkdir()
    shutil.copy2(root/'packaging/control', stage/'DEBIAN/control')
    binary = stage/'usr/bin'
    binary.mkdir(parents=True)
    for command, script in (('laya-snake-k3','start-snake.sh'),('laya-snake-k3-setup','setup.sh')):
        entry = binary/command
        entry.write_text(f'#!/bin/sh\nexec /usr/share/laya-snake-k3/{script} "$@"\n')
        entry.chmod(0o755)
    applications = stage/'usr/share/applications'
    applications.mkdir()
    shutil.copy2(root/'packaging/laya-snake-k3.desktop', applications/'laya-snake-k3.desktop')
    subprocess.run(['dpkg-deb','--root-owner-group','-Zgzip','--build',str(stage),
                    str(out/'laya-snake-k3_0.1.0_riscv64.deb')], check=True)
