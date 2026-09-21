#!/usr/bin/env python3
"""Download pinned public model files, or verify/copy a pre-downloaded set."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'models.lock.json')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--source', type=Path, help='Copy verified files from this local models directory')
    args = parser.parse_args()
    for item in json.loads(args.manifest.read_text())['files']:
        target = args.directory / item['path']
        if target.is_file() and digest(target) == item['sha256']:
            print('SHA256 OK', item['path'], flush=True)
            continue
        if args.verify_only:
            raise ValueError(f"SHA256 mismatch or missing file: {item['path']}")
        if target.exists():
            raise ValueError(f"SHA256 mismatch: {target}. Move the damaged file aside before retrying.")
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + '.part')
        if args.source:
            source = args.source / item['path']
            if not source.is_file() or digest(source) != item['sha256']:
                raise ValueError(f"SHA256 mismatch or missing source: {source}")
            shutil.copyfile(source, partial)
        else:
            print('Downloading', item['path'], flush=True)
            subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                            '--connect-timeout', '30', '--continue-at', '-',
                            '--output', str(partial), item['url']], check=True)
        if digest(partial) != item['sha256']:
            raise ValueError(f'SHA256 mismatch: {partial}; file not installed')
        partial.replace(target)
        print('SHA256 OK', item['path'], flush=True)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
