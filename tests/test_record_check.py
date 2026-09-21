from pathlib import Path
import subprocess
import sys


def test_checker_validates_a_real_record_and_rejects_wrong_step_count():
    root = Path(__file__).resolve().parents[1]
    record = root/'benchmarks/results/snake-fast.jsonl'
    command = [sys.executable,str(root/'scripts/check-record.py'),str(record)]
    run = subprocess.run(command,capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    assert 'steady_mean_ms' in run.stdout
    run = subprocess.run(command+['--steps','1'],capture_output=True,text=True)
    assert run.returncode != 0
