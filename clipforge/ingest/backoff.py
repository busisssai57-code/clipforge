"""Deterministic exponential backoff with jitter.

The spec demands BOTH jitter (§S0: reconnect storms against a platform are
antisocial and get you rate-limited) AND determinism (§3.2: no `random` in
decision paths). Both at once: the jitter is drawn from an RNG seeded by
``(seed_key, attempt)``, so the delay sequence for a given channel is fully
reproducible — same channel, same attempt number, same delay — while still
decorrelating retry timing ACROSS channels (each channel's seed differs).
"""

from __future__ import annotations

import random


def backoff_delay(attempt: int, *, base_s: float, cap_s: float,
                  seed_key: str) -> float:
    """Delay before reconnect ``attempt`` (0-based): min(cap, base·2^n),
    jittered into [0.5·d, 1.0·d] deterministically per (seed_key, attempt).
    """
    if attempt < 0:
        raise ValueError("attempt must be >= 0")
    # 2**attempt overflows fast; cap the exponent before the multiply.
    raw = base_s * (2 ** min(attempt, 16))
    capped = min(cap_s, raw)
    rng = random.Random(f"{seed_key}:{attempt}")  # deterministic jitter
    return capped * (0.5 + 0.5 * rng.random())
