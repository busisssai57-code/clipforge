"""Trend discovery: parsing, filtering, and selection order — all offline.

The yt-dlp runner is injected, so these tests never touch the network. They
pin the two properties that make ``bta trending`` trustworthy: junk rows never
become candidates, and the survivor order is view-count order (so ``--pick 1``
is the most-viewed clip-ready video).
"""

from __future__ import annotations

import subprocess

from clipforge import trending


def fake_run(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
    def _run(name, args, **kw):
        return subprocess.CompletedProcess([name, *args], returncode,
                                           stdout=stdout, stderr=stderr)
    return _run


# yt-dlp --flat-playlist output for the search: id \t duration \t views \t
# is_live \t title. Row 1 is a search shelf (unusable id, NA duration), which
# is exactly the kind of row that must be dropped.
SAMPLE = "\n".join([
    "https://www.youtube.com/podcasts\tNA\tNA\tNA\tPodcasts",  # shelf, no real id
    "dFGVGrc5xHU\t1631\t3503983\tFalse\tSumitra Reincarnation Case",   # 27m keep
    "Ary1gIbaOTc\t4666\t1757217\tFalse\tEx-CIA Spy John Kiriakou",     # 77m keep
    "Z1rRH5CM0uM\t482\t1192317\tFalse\tJeff Teague goes off",          # 8m keep
    "liveAAAAAAA\t0\t42000\tTrue\tSomeone is live right now",   # valid id, live: skip
    "shortBBBBBB\t45\t880000\tFalse\tA 45-second short",        # valid id, short: skip
])


def test_parse_drops_non_video_rows():
    cands = trending.parse_candidates(SAMPLE)
    ids = [c.video_id for c in cands]
    assert "https://www.youtube.com/podcasts" not in ids
    # Every survivor is a real 11-char id.
    assert all(len(c.video_id) == 11 for c in cands)
    # The shelf row is gone; the five id-bearing rows remain.
    assert len(cands) == 5


def test_filter_skips_live_short_and_overlong():
    cands = trending.parse_candidates(SAMPLE)
    kept = trending.filter_candidates(cands, min_minutes=3.0, max_minutes=90.0)
    ids = [c.video_id for c in kept]
    assert "liveAAAAAAA" not in ids, "live source has no fixed VOD"
    assert "shortBBBBBB" not in ids, "45s cannot yield a distinct 9:16 moment"
    assert ids == ["dFGVGrc5xHU", "Ary1gIbaOTc", "Z1rRH5CM0uM"]


def test_discover_preserves_view_count_order():
    # The search already returns view-sorted; discovery must not reshuffle it,
    # so cands[0] (== --pick 1) is the most-viewed usable video.
    cands = trending.discover("podcast", run=fake_run(SAMPLE))
    assert cands[0].video_id == "dFGVGrc5xHU"
    assert cands[0].view_count == 3503983
    assert cands[0].url == "https://www.youtube.com/watch?v=dFGVGrc5xHU"


def test_max_minutes_ceiling_is_enforced():
    # The Ex-CIA interview is 4666s (~77m); a 30-minute ceiling must exclude it
    # while the 27m and 8m sources survive.
    cands = trending.discover("podcast", max_minutes=30.0, run=fake_run(SAMPLE))
    assert [c.video_id for c in cands] == ["dFGVGrc5xHU", "Z1rRH5CM0uM"]


def test_discovery_fetches_wide_enough_to_survive_duration_filtering():
    # Regression: DEFAULT_LIMIT was 15, and a real 3-12 minute window left
    # exactly ONE usable candidate out of 15 — which then made --lang fail with
    # nothing to match. Discovery is one listing call regardless of the number,
    # so it must fetch wide.
    assert trending.DEFAULT_LIMIT >= 40

    captured = {}

    def spy_run(name, args, **kw):
        captured["playlist_end"] = int(args[args.index("--playlist-end") + 1])
        return fake_run(SAMPLE)(name, args, **kw)

    trending.discover("podcast", run=spy_run)
    assert captured["playlist_end"] == trending.DEFAULT_LIMIT


def test_limit_is_passed_through_to_yt_dlp():
    captured = {}

    def spy_run(name, args, **kw):
        captured["playlist_end"] = args[args.index("--playlist-end") + 1]
        return fake_run(SAMPLE)(name, args, **kw)

    trending.discover("podcast", limit=7, run=spy_run)
    assert captured["playlist_end"] == "7"


def test_discover_reports_why_candidates_were_dropped():
    # Without these counts the CLI blamed --lang for an empty pool that the
    # DURATION window had actually emptied (measured: "podcast" with a 4-10
    # minute window dropped 58 of 59 as too long, and the error said the
    # language gate found nothing).
    stats: dict = {}
    kept = trending.discover("podcast", max_minutes=30.0, stats=stats,
                             run=fake_run(SAMPLE))
    assert stats["raw"] == 5
    assert stats["usable"] == len(kept) == 2
    assert stats["too_long"] == 1      # the 77m interview
    assert stats["too_short"] == 1     # the 45s short
    assert stats["live"] == 1


def test_stats_is_optional_and_absence_changes_nothing():
    assert trending.discover("podcast", run=fake_run(SAMPLE)) == \
        trending.discover("podcast", stats={}, run=fake_run(SAMPLE))


def test_build_search_url_encodes_query_and_filter():
    url = trending.build_search_url("joe rogan", region="gb")
    assert "search_query=joe+rogan" in url
    assert f"sp={trending.SEARCH_SP}" in url
    assert "gl=GB" in url


def test_duration_hms_formats_hours_and_minutes():
    c = trending.Candidate("aaaaaaaaaaa", "t", 7904, 1, False)
    assert c.duration_hms == "2:11:44"
    short = trending.Candidate("bbbbbbbbbbb", "t", 95, 1, False)
    assert short.duration_hms == "1:35"
