"""Run the renderer's own test suite as part of `pytest -q`.

chat_render.js is frontend code with no bundler and no npm, so its tests run on
node's built-in runner. Bridging them into pytest means one command still covers
the whole project — a green backend and a silently broken renderer is exactly the
state this avoids.

Skipped, not failed, where node is absent: the backend does not need it.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SUITE = Path(__file__).resolve().parent / "chat_render.test.js"


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_the_message_renderer_suite_passes():
    result = subprocess.run(
        [shutil.which("node"), "--test", str(SUITE)],
        cwd=ROOT, capture_output=True, text=True, timeout=180)
    combined = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, combined
    assert "# fail 0" in combined or "fail 0" in combined, combined
