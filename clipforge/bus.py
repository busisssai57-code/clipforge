"""Bounded asyncio queues with backpressure accounting.

Every hand-off between worker pools (spec §6) goes through a
:class:`Bus` channel. Queues are BOUNDED — a slow consumer applies
backpressure to its producer instead of growing memory unboundedly during a
week-long run. ``put()`` awaits; there is deliberately no ``put_nowait``
escape hatch in the public API.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

from clipforge.log import get_logger

log = get_logger(__name__)

T = TypeVar("T")


@dataclass
class ChannelStats:
    put_count: int = 0
    get_count: int = 0
    # How often a producer had to wait — the backpressure signal to watch.
    blocked_puts: int = 0


class Channel(Generic[T]):
    """A named, bounded queue with stats. Thin by design."""

    def __init__(self, name: str, maxsize: int) -> None:
        if maxsize < 1:
            raise ValueError("Channel maxsize must be >= 1 (unbounded is forbidden)")
        self.name = name
        self._q: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self.stats = ChannelStats()

    async def put(self, item: T) -> None:
        if self._q.full():
            self.stats.blocked_puts += 1
            log.debug("bus.backpressure", channel=self.name,
                      blocked_puts=self.stats.blocked_puts)
        await self._q.put(item)
        self.stats.put_count += 1

    async def get(self) -> T:
        item = await self._q.get()
        self.stats.get_count += 1
        return item

    def task_done(self) -> None:
        self._q.task_done()

    async def join(self) -> None:
        await self._q.join()

    def qsize(self) -> int:
        return self._q.qsize()


@dataclass
class Bus:
    """The pipeline's channel registry. Channels are created once at startup
    with sizes from OrchestrationConfig; workers receive references, never
    create their own (keeps the topology auditable in one place)."""

    channels: dict[str, Channel[Any]] = field(default_factory=dict)

    def create(self, name: str, maxsize: int) -> Channel[Any]:
        if name in self.channels:
            raise ValueError(f"Channel {name!r} already exists")
        ch: Channel[Any] = Channel(name, maxsize)
        self.channels[name] = ch
        return ch

    def __getitem__(self, name: str) -> Channel[Any]:
        return self.channels[name]

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Stats for logging/monitoring."""
        return {
            name: {"qsize": ch.qsize(), "puts": ch.stats.put_count,
                   "gets": ch.stats.get_count, "blocked": ch.stats.blocked_puts}
            for name, ch in sorted(self.channels.items())
        }
