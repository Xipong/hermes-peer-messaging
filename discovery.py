"""Short-lived profile candidate snapshots; never cache routing permission."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow


@dataclass(frozen=True)
class CandidateSnapshot:
    candidates: tuple[tuple[str, float | None], ...]
    limited_sources: tuple[str, ...]
    live_owners: frozenset[str] | None


def scan(store, live_owners=None, *, roots_only=False):
    from . import service

    candidates, limited = [], []
    groups = [("local", tuple(sorted(service.STORED)))]
    if not roots_only:
        groups.append(("subagent", ("subagent",)))
    for label, sources in groups:
        rows = store.sessions.list_sessions(
            SessionQuery(sources=sources, local_created_sources=tuple(sorted(service.LOCAL)) if label == "local" else None,
                         include_children=True, include_subagents=not roots_only,
                         order=SessionOrder.LAST_ACTIVE, project_compression_tips=False),
            SessionWindow(limit=service.SCAN_LIMIT), compact_rows=True,
        )
        if len(rows) >= service.SCAN_LIMIT:
            limited.append(label)
        candidates.extend((entry.session.session_id, entry.last_active) for entry in rows)
    return CandidateSnapshot(tuple(candidates), tuple(limited),
                             None if live_owners is None else frozenset(live_owners))


class CandidateCache:
    """Bounded per-process cache shared by local agents in the same profile.

    Only discovery candidates and informational owner presence can be stale.
    Callers still resolve every identity and check the current link graph and
    project mapping. No store handles or session records outlive their reader.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.snapshots = OrderedDict()
        self.generation = 0

    def get(self, home, store, *, now, max_age, owner_loader, refresh=False, roots_only=False):
        key = (str(home), roots_only)
        with self.lock:
            previous = self.snapshots.get(key)
            if not refresh and previous is not None and 0 <= now - previous[0] < max_age:
                self.snapshots.move_to_end(key)
                return previous[1]
            self.generation += 1
            generation = self.generation
        # Do not serialize another agent behind storage or owner I/O. A first
        # concurrent miss may scan twice; completed snapshots are then shared.
        snapshot = scan(store, owner_loader(home), roots_only=roots_only)
        with self.lock:
            previous = self.snapshots.get(key)
            if previous is not None and previous[2] > generation:
                return snapshot
            self.snapshots.pop(key, None)
            self.snapshots[key] = (now, snapshot, generation)
            while len(self.snapshots) > 16:
                self.snapshots.popitem(last=False)
        return snapshot
