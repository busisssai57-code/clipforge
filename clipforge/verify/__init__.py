"""Per-module deterministic self-checks (spec §8 gate #1).

Each submodule exposes ``main() -> int`` (0 = pass) and is runnable both via
``python -m clipforge.verify.<module>`` and ``clipforge verify <module>``.
Checks here are REAL executions against temp dirs — not mocks — but never
require GPU, network, or external binaries unless the module is explicitly
about them (those degrade to SKIP with a message).
"""
