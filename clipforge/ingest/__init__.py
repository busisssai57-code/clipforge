"""Ingestion subsystem (CP1) — monitors, VOD download, live chunker.

Data flow::

    monitor.ChannelMonitor          (async poll loops, disk guard, backoff)
      ├─ youtube.new_vod_ids/download_vod   (yt-dlp, DB-deduped)
      ├─ twitch.is_live ─┐
      ├─ kick.is_live ───┴─ chunker.ChunkerSession  (streamlink→ffmpeg pipe)
      │                        └─ SegmentEvent per READY .ts segment
      └─ on_media(path, abs_start_s)        (DAG seam — orchestrator at CP5)

    overlap.build_virtual_window   (T1: prev-tail + chunk, concat demuxer)
    watch.watcher.DirectoryWatcher (stable-file debounce for foreign files)
"""
