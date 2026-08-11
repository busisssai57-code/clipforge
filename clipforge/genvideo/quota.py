"""Quota ledger for metered generation providers.

The routing requirement is: use the premium provider until its quota is
gone, fall back to a local one, and go BACK when the quota resets. That is
only trustworthy if "quota is gone" and "the window reset" are both
recorded on disk — an in-memory flag forgets across runs and would hammer
a exhausted API on every invocation, which is how a daily quota turns into
a rate-limit ban.

Design notes that matter:

* **The clock is injected.** Every decision here is a function of (state,
  now). Tests pass a fake clock and step it; nothing sleeps.
* **Exhaustion is recorded with its reset time**, taken from the provider's
  own ``retry-after``/quota metadata when it gives one, and from a
  configured window length when it does not. Guessing "midnight UTC" for
  an API that actually meters per-minute would strand the primary provider
  for a day.
* **State is written atomically.** A half-written ledger that fails to
  parse must not be able to lock a provider out permanently, so a corrupt
  file is discarded with a warning rather than raising.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from clipforge.log import get_logger

log = get_logger(__name__)

Clock = Callable[[], float]

#: Used when a provider says "exhausted" without saying until when.
DEFAULT_WINDOW_S = 24 * 3600.0

#: Never trust an absurd retry-after; a provider bug should not sideline
#: the primary for a week.
MAX_BACKOFF_S = 7 * 24 * 3600.0


@dataclass
class ProviderState:
    """What we know about one provider's quota."""

    calls: int = 0
    seconds_generated: float = 0.0
    #: Epoch seconds when the current exhaustion lifts. 0 = not exhausted.
    exhausted_until: float = 0.0
    #: Why it was marked exhausted, for the dashboard and the log.
    last_reason: str = ""
    last_used_at: float = 0.0
    consecutive_errors: int = 0

    def available(self, now: float) -> bool:
        return now >= self.exhausted_until


@dataclass
class QuotaLedger:
    """Persistent per-provider quota state."""

    path: Path
    clock: Clock = time.time
    providers: dict[str, ProviderState] = field(default_factory=dict)

    # ------------------------------------------------------------ io

    @classmethod
    def load(cls, path: Path, clock: Clock = time.time) -> "QuotaLedger":
        ledger = cls(path=Path(path), clock=clock)
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            for name, blob in (raw.get("providers") or {}).items():
                known = {k: v for k, v in blob.items()
                         if k in ProviderState.__dataclass_fields__}
                ledger.providers[str(name)] = ProviderState(**known)
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            # A corrupt ledger must not permanently strand a provider.
            log.warning("genvideo.quota_ledger_unreadable",
                        path=str(path), error=str(exc)[:200],
                        note="starting from empty state")
        return ledger

    def save(self) -> None:
        """Atomic write — a torn ledger could otherwise pin a provider off."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"providers": {n: asdict(s) for n, s in self.providers.items()}}
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, sort_keys=True)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # --------------------------------------------------------- state

    def state(self, provider: str) -> ProviderState:
        return self.providers.setdefault(provider, ProviderState())

    def available(self, provider: str) -> bool:
        return self.state(provider).available(self.clock())

    def seconds_until_available(self, provider: str) -> float:
        return max(0.0, self.state(provider).exhausted_until - self.clock())

    # -------------------------------------------------------- record

    def record_success(self, provider: str, *, seconds: float) -> None:
        st = self.state(provider)
        now = self.clock()
        st.calls += 1
        st.seconds_generated += max(0.0, float(seconds))
        st.last_used_at = now
        st.consecutive_errors = 0
        # A success proves the window reopened even if we had it marked
        # exhausted — trust the observation over the prediction.
        if st.exhausted_until and now < st.exhausted_until:
            log.info("genvideo.quota_reopened_early", provider=provider)
        st.exhausted_until = 0.0
        st.last_reason = ""
        self.save()

    def record_exhausted(self, provider: str, *, reason: str,
                         retry_after_s: float | None = None,
                         window_s: float = DEFAULT_WINDOW_S) -> float:
        """Mark a provider out of quota. Returns when it becomes available."""
        st = self.state(provider)
        now = self.clock()
        wait = retry_after_s if retry_after_s is not None else window_s
        wait = max(0.0, min(float(wait), MAX_BACKOFF_S))
        st.exhausted_until = now + wait
        st.last_reason = reason[:300]
        st.last_used_at = now
        self.save()
        log.warning("genvideo.quota_exhausted", provider=provider,
                    reason=reason[:200], available_in_s=round(wait, 1))
        return st.exhausted_until

    def record_error(self, provider: str, *, reason: str,
                     cooldown_s: float = 60.0,
                     max_errors: int = 3) -> None:
        """A non-quota failure. Repeated failures sideline the provider.

        Distinct from exhaustion on purpose: a 500 or a network blip
        should cost seconds, not a day, but a provider failing repeatedly
        is broken and should stop being tried first.
        """
        st = self.state(provider)
        now = self.clock()
        st.consecutive_errors += 1
        st.last_reason = reason[:300]
        st.last_used_at = now
        if st.consecutive_errors >= max_errors:
            st.exhausted_until = now + cooldown_s
            log.warning("genvideo.provider_sidelined", provider=provider,
                        consecutive_errors=st.consecutive_errors,
                        cooldown_s=cooldown_s, reason=reason[:200])
        self.save()

    # ------------------------------------------------------- reports

    def snapshot(self) -> dict[str, dict[str, float | int | str | bool]]:
        """Dashboard-shaped view of every provider."""
        now = self.clock()
        return {
            name: {
                "calls": st.calls,
                "seconds_generated": round(st.seconds_generated, 2),
                "available": st.available(now),
                "available_in_s": round(max(0.0, st.exhausted_until - now), 1),
                "last_reason": st.last_reason,
                "last_used_at": st.last_used_at,
                "consecutive_errors": st.consecutive_errors,
            }
            for name, st in sorted(self.providers.items())
        }
