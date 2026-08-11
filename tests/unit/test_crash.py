"""The Resumability Law, demonstrated for real: hard-kill a writer process
mid-write and prove the on-disk contract — the destination is always either
absent or a complete previous version, and the only debris is ``.partial``
files that the startup sweep removes.

This is the strongest offline approximation of "killing the process at any
instant" (§3.3): TerminateProcess on Windows is not catchable, not
graceful, and lands at an arbitrary instant inside the write loop.
"""

import json
import subprocess
import sys
import time
from pathlib import Path  # noqa: F401 - used in type hints below

from clipforge import paths

# The child rewrites the same destination as fast as it can, forever, with
# payloads big enough (~1 MiB) that a kill has a real chance of landing
# mid-write. It prints READY once the first write completed.
_CHILD = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[2])
from clipforge import paths

dest = Path(sys.argv[1]) / "artifact.json"
payload = {"n": 0, "pad": "x" * (1 << 20)}
i = 0
while True:
    payload["n"] = i
    paths.atomic_write_json(dest, payload)
    if i == 0:
        print("READY", flush=True)
    i += 1
"""


def test_hard_kill_mid_write_never_torn(tmp_path: Path) -> None:
    repo_root = str(Path(__file__).resolve().parents[2])
    proc = subprocess.Popen(
        [sys.executable, "-c", _CHILD, str(tmp_path), repo_root],
        stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        # Wait until at least one complete artifact exists.
        line = proc.stdout.readline()
        assert line.strip() == "READY", f"child failed to start: {line!r}"
        # Let it churn, then kill at an arbitrary instant (TerminateProcess).
        time.sleep(0.25)
        proc.kill()
        proc.wait(timeout=10)
    finally:
        if proc.poll() is None:  # pragma: no cover - safety net
            proc.kill()
        proc.stdout.close()

    dest = tmp_path / "artifact.json"
    # Contract 1: destination exists (first write completed) and is a
    # COMPLETE JSON document — never torn, whatever instant the kill hit.
    #
    # Windows releases a killed process's handles asynchronously, so the
    # read itself can transiently fail with PermissionError even though the
    # file is intact. Retry briefly: the contract under test is the file's
    # CONTENT, not the OS's handle-release latency.
    deadline = time.monotonic() + 10
    text = None
    while time.monotonic() < deadline:
        try:
            text = dest.read_text(encoding="utf-8")
            break
        except PermissionError:
            time.sleep(0.1)
    assert text is not None, "destination stayed locked after the kill"
    data = json.loads(text)
    assert data["pad"] == "x" * (1 << 20)
    assert isinstance(data["n"], int)

    # Contract 2: the only possible debris is .partial, and the startup
    # sweep removes it without touching the artifact.
    #
    # Windows releases a killed process's file handles asynchronously, so a
    # sweep run microseconds after the kill can legitimately find the temp
    # still locked (discard_partials tolerates that by design). Retry
    # briefly: the contract is that the sweep CONVERGES, not that a single
    # pass succeeds against a handle the OS has not closed yet.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        paths.discard_partials(tmp_path)
        if not list(paths.iter_partials(tmp_path)):
            break
        time.sleep(0.1)
    assert list(paths.iter_partials(tmp_path)) == [], "sweep did not converge"
    assert json.loads(dest.read_text(encoding="utf-8"))["n"] == data["n"]
