"""Bounded channels: backpressure accounting, no unbounded queues."""

import asyncio

import pytest

from clipforge.bus import Bus, Channel


def test_unbounded_channel_is_forbidden():
    with pytest.raises(ValueError):
        Channel("bad", 0)


async def test_put_get_roundtrip():
    ch: Channel[int] = Channel("t", 4)
    await ch.put(1)
    await ch.put(2)
    assert await ch.get() == 1
    assert await ch.get() == 2
    assert ch.stats.put_count == 2 and ch.stats.get_count == 2


async def test_backpressure_blocks_and_counts():
    ch: Channel[int] = Channel("t", 1)
    await ch.put(1)

    async def slow_consumer():
        await asyncio.sleep(0.01)
        return await ch.get()

    consumer = asyncio.create_task(slow_consumer())
    await ch.put(2)  # must block until the consumer drains one
    assert ch.stats.blocked_puts == 1
    await consumer


def test_bus_registry_rejects_duplicates():
    bus = Bus()
    bus.create("chunks", 4)
    with pytest.raises(ValueError):
        bus.create("chunks", 4)
    snap = bus.snapshot()
    assert snap == {"chunks": {"qsize": 0, "puts": 0, "gets": 0, "blocked": 0}}
