import sys

import pytest


# https://www.youtube.com/watch?v=DhUpxWjOhME
@pytest.fixture
def capture_stdout(monkeypatch):
    buffer = {'stdout': '', 'stderr': '', 'write_calls': 0}

    def fake_write(s):
        buffer['stdout'] += s
        buffer['write_calls'] += 1
    def fake_write_err(s):
        buffer['stderr'] += s
        buffer['write_calls'] += 1

    monkeypatch.setattr(sys.stdout, 'write', fake_write)
    monkeypatch.setattr(sys.stderr, 'write', fake_write_err)
    return buffer
