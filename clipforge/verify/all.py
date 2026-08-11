"""Run EVERY deterministic gate module in one command.

Round 8's meta domain measured a false-green hazard: two neutralizations
(``MAX_CONNECTS_PER_HOUR`` raised 20 -> 500, ``MIN_SEGMENT_BYTES`` dropped to
0) were caught ONLY by ``clipforge.verify.ingestion`` while all pytest tests
stayed green. Anyone reaching for the pytest half alone — a CI shorthand, a
pre-commit hook, a tired operator — would read that as a pass.

The §8 gate is therefore TWO commands, and this module is the second::

    python -m pytest tests/unit tests/integration
    python -m clipforge.verify.all

Exit code is nonzero if ANY module fails, so a shell `&&` chain is safe.
"""

from __future__ import annotations

import sys

from clipforge.verify import ai, ingestion, skeleton

#: Order is cheapest-first so an obvious breakage reports fast.
MODULES = (("skeleton", skeleton), ("ingestion", ingestion), ("ai", ai))


def main() -> int:
    failed: list[str] = []
    for name, module in MODULES:
        print(f"\n=== verify: {name} ===", flush=True)
        try:
            rc = module.main()
        except Exception as exc:  # a crashing gate is a FAILING gate
            print(f"[ERROR] {name} raised {type(exc).__name__}: {exc}")
            rc = 1
        if rc != 0:
            failed.append(name)
    print("\n" + "=" * 62)
    if failed:
        print(f"GATE FAILED: {', '.join(failed)}")
        return 1
    print(f"GATE PASSED: {', '.join(n for n, _ in MODULES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
