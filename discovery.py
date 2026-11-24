"""Root discovery cache and native immediate-family candidate queries."""
from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass

from persistence.conversation.sessions import SessionOrder, SessionQuery, SessionWindow

from . import teams

ROOT_SCAN_LIMIT = 100


@dataclass(frozen=True)
class CandidateSnapshot:
    candidates: tuple[tuple[str, float | None], ...]
    limited_sources: tuple[str, ...]
    live_owners: frozenset[str] | None


def scan(store, live_owners=None):
    rows = store.sessions.list_sessions(
        SessionQuery(sources=tuple(sorted(teams.STORED)), local_created_sources=tuple(sorted(teams.LOCAL)),
                     include_children=True, include_subagents=False,
                     order=SessionOrder.LAST_ACTIVE, project_compression_tips=False),
        SessionWindow(limit=ROOT_SCAN_LIMIT), compact_rows=True,
    )
    return CandidateSnapshot(tuple((entry.session.session_id, entry.last_active) for entry in rows),
                             ("local",) if len(rows) >= ROOT_SCAN_LIMIT else (),
                             None if live_owners is None else frozenset(live_owners))


def family_candidates(store, own):
    """Find children and siblings by native delegation markers, not a global list.

    The marker lookup is only a candidate index. The caller still resolves
    every returned identity and verifies its delegation/compression edge.
    """
    parents = [own]
    if own.is_child:
        try:
            parents.append(teams.resolve(store, own.parent_session_id, allow_ended=True))
        except ValueError:
            pass
    candidates = set()
    for parent in parents:
        for segment in teams.compression_ids(store, parent):
            candidates.update(store.sessions.subagent_session_ids(segment, "_delegate_from"))
    return tuple(sorted(candidates))


class CandidateCache:
    """Short-lived independent-root candidates, never access permissions."""

    def __init__(self):
        self.lock = threading.Lock()
        self.snapshots = OrderedDict()
        self.generation = 0

    def get(self, home, store, *, now, max_age, owner_loader, refresh=False):
        key = str(home)
        with self.lock:
            previous = self.snapshots.get(key)
            if not refresh and previous is not None and 0 <= now - previous[0] < max_age:
                self.snapshots.move_to_end(key)
                return previous[1]
            self.generation += 1
            generation = self.generation
        snapshot = scan(store, owner_loader(home))
        with self.lock:
            previous = self.snapshots.get(key)
            if previous is not None and previous[2] > generation:
                return snapshot
            self.snapshots.pop(key, None)
            self.snapshots[key] = (now, snapshot, generation)
            while len(self.snapshots) > 16:
                self.snapshots.popitem(last=False)
        return snapshot
