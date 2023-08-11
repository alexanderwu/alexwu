import sys

# import pytest

import alexwu as aw


def test_reload(capture_stdout):
    aw.reload()
    assert capture_stdout['stdout'] == '%load_ext autoreload\n%autoreload 2\n'

def test_reload_copy(capture_stdout):
    sys.modules['pyperclip'] = None
    # with pytest.raises(Exception) as _:
    aw.reload(True)
    assert capture_stdout['stderr'] == 'Cannot copy. Try `pip install pyperclip`\n'
    assert capture_stdout['stdout'] == '%load_ext autoreload\n%autoreload 2\n'
