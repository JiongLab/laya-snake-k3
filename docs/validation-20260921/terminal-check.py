"""Exercise the installed public entry point in a real PTY, without old app paths."""
import fcntl
import json
import os
from pathlib import Path
import pty
import re
import select
import struct
import subprocess
import termios
import time

root = Path(__file__).resolve().parent
master, slave = pty.openpty()
fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 40, 120, 0, 0))
initial = termios.tcgetattr(slave)
env = {**os.environ, 'LAYA_K3_HOME': str(root/'data'), 'TERM': 'xterm-256color', 'COLORTERM': 'truecolor'}
started = time.monotonic()
proc = subprocess.Popen(['/usr/bin/laya-snake-k3'], stdin=slave, stdout=slave, stderr=slave, env=env)
raw = bytearray()
pending = ''
stage = 'running'
result = {}
stage_at = started
try:
    while time.monotonic()-started < 180:
        if select.select([master], [], [], .1)[0]:
            chunk = os.read(master, 65536)
            raw.extend(chunk)
            pending += chunk.decode(errors='replace')
        clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', pending)
        now = time.monotonic()
        if stage == 'running' and 'NEXT MOVE' in clean and 'AI CORE / FP16' in clean:
            result['first_frame_seconds'] = now-started
            os.write(master, b' ')
            stage, pending = 'pause', ''
        elif stage == 'pause' and 'PAUSED' in clean:
            result['pause_displayed'] = True
            stage, stage_at, pending = 'hold', now, ''
        elif stage == 'hold' and now-stage_at > 1:
            assert 'PAUSED' in clean, 'Pause was not maintained'
            os.write(master, b' ')
            stage, pending = 'resume', ''
        elif stage == 'resume' and 'LIVE' in clean:
            result['resume_displayed'] = True
            os.write(master, b'r')
            stage, pending = 'reset', ''
        elif stage == 'reset' and 'ROUND 02' in clean:
            result['reset_displayed'] = True
            os.write(master, b'q')
            stage = 'quit'
        if proc.poll() is not None:
            break
    assert proc.poll() == 0, f'Exit={proc.poll()}, stage={stage}, tail={pending[-1000:]}'
    assert stage == 'quit', stage
    assert termios.tcgetattr(slave) == initial, 'TTY settings not restored'
    result.update(exit_code=proc.returncode, terminal_restored=True, total_seconds=time.monotonic()-started)
    (root/'terminal-result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
finally:
    (root/'terminal-capture.ansi').write_bytes(raw)
    if proc.poll() is None:
        proc.terminate()
        proc.wait(timeout=20)
    os.close(master)
    os.close(slave)
