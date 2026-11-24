"""Advisory recipient state and bounded, non-destructive mailbox housekeeping."""
from __future__ import annotations

import logging
import math
import threading
import time
from bisect import bisect_right
from collections import OrderedDict

from persistence.conversation.sessions import END_REASON_COMPRESSION
from persistence_assembly import open_persistence_reader

from . import mailbox, teams

LOG = logging.getLogger(__name__)
ARCHIVE_AGE_SECONDS = 24 * 3600
SWEEP_INTERVAL_SECONDS = 600
SWEEP_BATCH = 16
TERMINAL_WORKER_STATES = {"completed", "stopped", "superseded", "expired"}
_sweep_lock = threading.Lock()
_sweeps = OrderedDict()


def worker_record(store, participant):
    if not participant.is_child or not participant.subagent_id:
        return None
    record = store.coordination.get_subagent(subagent_id=participant.subagent_id)
    # Short IDs can be reused; verify the exact logical child.
    if record is None or record.document.get("session_id") not in teams.compression_ids(store, participant):
        return None
    return record


def recipient_state(store, participant, live_owners):
    """Neither a lease nor a durable running record proves a current LLM turn."""
    row = participant.row
    state = {"owner_present": None if live_owners is None else row.session_id in live_owners,
             "subagent_state": None, "last_activity_at": row.last_activity_at}
    if participant.is_child:
        try:
            record = worker_record(store, participant)
            if record is not None:
                state["subagent_state"] = record.state.value
        except Exception:
            LOG.debug("Peer recipient state unavailable", exc_info=True)
    return state


def eligible(store, participant, live_owners, now):
    row = participant.row
    if (row.ended_at is None or not isinstance(row.ended_at, (int, float))
            or not math.isfinite(row.ended_at)
            or row.end_reason == END_REASON_COMPRESSION
            or now - row.ended_at < ARCHIVE_AGE_SECONDS
            or live_owners is None
            or any(sid in live_owners for sid in teams.compression_ids(store, participant))):
        return False
    if participant.is_child:
        record = worker_record(store, participant)
        return (record is not None and record.state.value in TERMINAL_WORKER_STATES
                and not record.pinned and not record.owed
                and math.isfinite(record.updated_at)
                and now - record.updated_at >= ARCHIVE_AGE_SECONDS)
    return True


def sweep(home, live_owners):
    """At most one small batch per profile/interval on explicit root discovery.

    Unknown metadata, live/claimed/resumable workers, and compression-only
    segments are never retired. Reopened participants can read archived mail.
    No worker, timer, model request, or deletion of unacknowledged mail.
    """
    from .service import root

    if live_owners is None:
        return
    key = str(home)
    now, tick = time.time(), time.monotonic()
    with _sweep_lock:
        last, cursor = _sweeps.get(key, (None, ""))
        if last is not None and tick - last < SWEEP_INTERVAL_SECONDS:
            return
        _sweeps[key] = (tick, cursor)
        _sweeps.move_to_end(key)
        while len(_sweeps) > 16:
            _sweeps.popitem(last=False)
    try:
        folder = root(home)
        if not folder.is_dir():
            return
        names = sorted(p.name for p in folder.iterdir() if not p.name.startswith(".") and p.is_dir()
                       and not p.is_symlink())
        start = bisect_right(names, cursor)
        batch = (names[start:] + names[:start])[:SWEEP_BATCH]
        if not batch:
            return
        with _sweep_lock:
            if key in _sweeps and _sweeps[key][0] == tick:
                _sweeps[key] = (tick, batch[-1])
        for sid in batch:
            try:
                mailbox.session_component(sid)
                inbox = folder / sid
                if not mailbox._active_paths(inbox):
                    continue
                with open_persistence_reader(home, patience_s=0.05) as store:
                    participant = teams.resolve(store, sid, allow_ended=True)
                    if participant.thread_id != sid or not eligible(store, participant, live_owners, now):
                        continue
                    mailbox.archive_pending(inbox)
            except Exception:
                LOG.debug("Peer archive skipped one inbox", exc_info=True)
    except Exception:
        LOG.debug("Peer mailbox housekeeping unavailable", exc_info=True)
