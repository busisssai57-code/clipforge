"""Live chat as an engagement signal (clipforge.ingest.chat).

The three properties that make this trustworthy rather than a message
counter, each pinned here: one account cannot manufacture a peak, a paid
message weighs more than a plain one, and the same log scores identically
every time. The parsers are pinned on the shapes real exporters emit.
"""

from __future__ import annotations

import json

import pytest

from clipforge.ingest.chat import (ChatEvent, PAID_WEIGHT, build_curve,
                                    parse_chat_log, peaks, score_window)


# ----------------------------------------------------------- anti-spam

def test_one_user_flooding_a_second_counts_at_most_the_cap():
    """Fifty emotes from one account is one opinion, not fifty votes."""
    flood = [ChatEvent(t_s=10.0, user="spammer") for _ in range(50)]
    curve = build_curve(flood, per_user_cap=3)
    assert curve.bins[10] == 3.0, "the per-user cap did not bound the flood"


def test_a_real_crowd_outweighs_a_flood():
    """Ten different people beat one person typing ten times."""
    flood = [ChatEvent(t_s=5.0, user="spammer") for _ in range(10)]
    crowd = [ChatEvent(t_s=8.0, user=f"u{i}") for i in range(10)]
    curve = build_curve(flood + crowd, per_user_cap=3)
    assert curve.bins[8] > curve.bins[5]


def test_a_paid_message_weighs_more_than_a_plain_one():
    plain = build_curve([ChatEvent(t_s=1.0, user="a")])
    paid = build_curve([ChatEvent(t_s=1.0, user="a", weight=PAID_WEIGHT)])
    assert paid.bins[1] > plain.bins[1]
    assert paid.bins[1] == pytest.approx(PAID_WEIGHT)


def test_a_gift_cannot_swamp_the_whole_crowd_scale_silently():
    """A single gift is strong but bounded; the busiest crowd second still
    anchors the top of the scale when it is larger."""
    events = [ChatEvent(t_s=2.0, user="whale", weight=PAID_WEIGHT)]
    events += [ChatEvent(t_s=30.0, user=f"u{i}") for i in range(20)]
    curve = build_curve(events, per_user_cap=3)
    assert curve.peak == curve.bins[30]


# --------------------------------------------------------- determinism

def test_the_curve_is_order_independent():
    a = [ChatEvent(t_s=t, user=u) for t, u in
         [(1.0, "x"), (1.0, "y"), (2.0, "x"), (5.5, "z")]]
    assert build_curve(a).bins == build_curve(list(reversed(a))).bins


def test_binning_floors_to_the_second():
    curve = build_curve([ChatEvent(t_s=3.0, user="a"),
                         ChatEvent(t_s=3.9, user="b")])
    assert set(curve.bins) == {3}
    assert curve.bins[3] == 2.0


# ------------------------------------------------------------- scoring

def test_a_window_with_no_chat_scores_zero():
    curve = build_curve([ChatEvent(t_s=100.0, user="a")])
    assert score_window(0.0, 30.0, curve) == 0.0


def test_the_busiest_window_approaches_one():
    events = [ChatEvent(t_s=10.0, user=f"u{i}") for i in range(8)]
    curve = build_curve(events)
    hot = score_window(9.0, 11.0, curve)
    cold = score_window(40.0, 70.0, curve)
    assert hot > cold
    assert 0.0 <= hot <= 1.0


def test_an_empty_curve_scores_zero_and_is_falsey():
    curve = build_curve([])
    assert not curve
    assert score_window(0.0, 30.0, curve) == 0.0


def test_score_is_average_not_sum_so_length_is_not_rewarded():
    """Two windows over the same reaction, one longer: the longer must not
    win on chat alone — length is a different heuristic's concern."""
    events = [ChatEvent(t_s=10.0, user=f"u{i}") for i in range(6)]
    curve = build_curve(events)
    tight = score_window(9.0, 12.0, curve)
    loose = score_window(9.0, 40.0, curve)
    assert tight > loose


# ------------------------------------------------------------- parsing

def test_parses_youtube_style_jsonl(tmp_path):
    p = tmp_path / "v.live_chat.json"
    lines = [
        {"videoOffsetTimeMsec": "5000", "authorName": "alice", "message": "lol"},
        {"videoOffsetTimeMsec": "5200", "author": "bob"},
        {"videoOffsetTimeMsec": "6000", "authorName": "carol",
         "_type": "liveChatPaidMessageRenderer"},
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    events = parse_chat_log(p)
    assert len(events) == 3
    assert events[0].t_s == 5.0
    paid = [e for e in events if e.weight > 1.0]
    assert len(paid) == 1 and paid[0].user == "carol"


def test_parses_twitch_style_comments_array(tmp_path):
    p = tmp_path / "chat.json"
    doc = {"comments": [
        {"content_offset_seconds": 12.5, "commenter": {"display_name": "dan"}},
        {"content_offset_seconds": 13.0, "commenter": {"display_name": "eve"}},
    ]}
    p.write_text(json.dumps(doc), encoding="utf-8")
    events = parse_chat_log(p)
    assert [e.t_s for e in events] == [12.5, 13.0]


def test_parses_irc_style_log(tmp_path):
    p = tmp_path / "chat.log"
    p.write_text("[00:00:05] alice: first\n[00:01:00] bob: later\n",
                 encoding="utf-8")
    events = parse_chat_log(p)
    assert [e.t_s for e in events] == [5.0, 60.0]
    assert events[0].user == "alice"


def test_parses_csv_with_header(tmp_path):
    p = tmp_path / "chat.csv"
    p.write_text("offset_seconds,user\n3,alice\n3,bob\n9,carol\n",
                 encoding="utf-8")
    curve = build_curve(parse_chat_log(p))
    assert curve.bins[3] == 2.0 and curve.bins[9] == 1.0


def test_a_missing_or_junk_log_is_not_an_error(tmp_path):
    assert parse_chat_log(tmp_path / "nope.json") == []
    junk = tmp_path / "j.txt"
    junk.write_text("this is not chat in any format\n", encoding="utf-8")
    assert parse_chat_log(junk) == []


# --------------------------------------------------------------- peaks

def test_peaks_are_the_busy_seconds_sorted_by_time():
    events = ([ChatEvent(t_s=10.0, user=f"u{i}") for i in range(10)]
              + [ChatEvent(t_s=50.0, user=f"v{i}") for i in range(9)]
              + [ChatEvent(t_s=90.0, user="lonely")])
    curve = build_curve(events)
    hot = peaks(curve, min_fraction=0.5)
    times = [t for t, _ in hot]
    assert 10 in times and 50 in times and 90 not in times
    assert times == sorted(times)
