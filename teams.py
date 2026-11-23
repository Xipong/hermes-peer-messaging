"""Durable local-team identities, derived from verified stored parent edges.

Subagent markers locate candidates; they never establish ownership on their own.
Only a direct link between the two local roots grants communication across teams.
"""
from __future__ import annotations

from dataclasses import dataclass

from persistence.conversation.sessions import SessionRecord

from . import mailbox


@dataclass(frozen=True)
class Participant:
    row: SessionRecord
    thread_id: str
    root_id: str
    root: SessionRecord
    parent_session_id: str | None
    subagent_id: str | None
    inbox_ids: tuple[str, ...]

    @property
    def is_child(self) -> bool:
        return self.parent_session_id is not None


def _child(row):
    return row.source == "subagent" or bool(row.model_config.get("_delegate_from"))


def _delegated(row):
    parent = row.parent_session_id
    return bool(parent and row.model_config.get("_delegate_from") == parent
                and parent not in (row.branched_from, row.reset_from))


def _subagent_id(row):
    value = row.model_config.get("_subagent_id")
    return mailbox.session_component(value) if value is not None else None


class _Resolver:
    def __init__(self, store, allow_ended):
        self.store = store
        self.allow_ended = allow_ended
        self.rows = {}

    def get(self, sid):
        sid = mailbox.session_component(sid)
        if sid not in self.rows:
            row = self.store.sessions.get(sid)
            if row is None or not self.store.sessions.owns_session(sid):
                raise ValueError("Unknown session or session belongs to another profile")
            if row.hidden or row.archived or row.model_config_corrupt:
                raise ValueError("Session is hidden, archived, or has invalid routing metadata")
            self.rows[sid] = row
        return self.rows[sid]

    def segments(self, row, *, stop_at=None):
        """Compression ancestors, optionally bounded by a validated local root."""
        segments, seen = [row], {row.session_id}
        while row.session_id != stop_at and row.parent_session_id and self.store.lineage.is_compression_child(row):
            parent = row.parent_session_id
            if parent in seen:
                raise ValueError("Cyclic session lineage")
            row = self.get(parent)
            segments.append(row)
            seen.add(parent)
        if stop_at is not None and row.session_id != stop_at:
            raise ValueError("Local conversation lineage changed; retry routing")
        return tuple(segments)

    def live(self, row):
        if row.ended_at is not None and not self.allow_ended:
            raise ValueError("Session has ended; refresh peer_sessions")

    @staticmethod
    def worker(row):
        from . import service

        if row.source != "subagent" and (row.source not in service.STORED or service.surface(row) not in service.LOCAL):
            raise ValueError("Peer teams require local delegated sessions")

    def local(self, row, *, follow=True):
        # Import lazily: service dispatch uses this resolver, while the existing
        # local route remains the authority for supported conversation surfaces.
        from . import service, threads

        current = service.route(self.store, row.session_id, follow=follow, allow_ended=self.allow_ended)
        root_id, current = threads.locate(self.store, current.session_id, allow_ended=self.allow_ended)
        current = self.get(current.session_id)
        # locate already stopped at an archived, foreign, or ineligible ancestor.
        # The inbox walk must preserve that boundary and keep own mail usable.
        segments = self.segments(current, stop_at=root_id)
        return Participant(current, root_id, root_id, current, None, None,
                           tuple(item.session_id for item in segments))

    def team_root(self, row):
        seen = set()
        while _child(row):
            if row.session_id in seen:
                raise ValueError("Cyclic delegation lineage")
            seen.add(row.session_id)
            self.worker(row)
            if not row.parent_session_id or not (_delegated(row) or self.store.lineage.is_compression_child(row)):
                raise ValueError("Subagent has no verified delegation route to a local conversation")
            row = self.get(row.parent_session_id)
        return self.local(row)

    def child(self, row, *, follow):
        segments = self.segments(row)
        first = segments[-1]
        if not _delegated(first):
            raise ValueError("Subagent has no verified delegation parent")
        subagent_id = _subagent_id(row) or _subagent_id(first)
        candidates = {row.session_id: segments}
        if subagent_id:
            for sid in self.store.sessions.subagent_session_ids(subagent_id, "_subagent_id"):
                try:
                    candidate = self.get(sid)
                    lineage = self.segments(candidate)
                except ValueError:
                    continue
                # Reused short subagent IDs and branch/reset descendants do not
                # join this child's mailbox or change its canonical address.
                if lineage[-1].session_id == first.session_id and any(item.session_id == row.session_id for item in lineage):
                    candidates[sid] = lineage
        continued = {lineage[0].parent_session_id for lineage in candidates.values()
                     if len(lineage) > 1}
        tips = [lineage for sid, lineage in candidates.items() if sid not in continued]
        if len(tips) != 1:
            raise ValueError("Ambiguous subagent compression route")
        segments = tips[0]
        current = segments[0]
        for segment in segments:
            self.worker(segment)
        if not follow and current.session_id != row.session_id:
            raise ValueError("This chat rotated; use its current session")
        self.live(current)
        root = self.team_root(first)
        return Participant(current, first.session_id, root.root_id, root.root,
                           first.parent_session_id, _subagent_id(current) or subagent_id,
                           tuple(item.session_id for item in segments))


def resolve(store, sid, *, follow=True, allow_ended=False) -> Participant:
    resolver = _Resolver(store, allow_ended)
    row = resolver.get(sid)
    return resolver.child(row, follow=follow) if _child(row) else resolver.local(row, follow=follow)


def allowed(a, b, links, *, require_link=False, project_index=None) -> bool:
    from . import service

    if not service.same_profile(a.root.profile_name, b.root.profile_name):
        return False
    if a.root_id == b.root_id:
        return True
    if not (require_link or a.is_child or b.is_child):
        return True
    project = service.project(a.root, index=project_index)
    if not project or project != service.project(b.root, index=project_index):
        return False
    key = ":".join(sorted((a.root_id, b.root_id)))
    edge = links.get(key)
    return bool(isinstance(edge, dict) and edge.get("since") is not None
                and set(edge.get("offers", ())) == {a.root_id, b.root_id})


def describe(participant) -> dict:
    from . import service

    return {"session_id": participant.row.session_id, "thread_id": participant.thread_id,
            "root_thread_id": participant.root_id, "parent_session_id": participant.parent_session_id,
            "subagent_id": participant.subagent_id, "is_child": participant.is_child,
            "title": participant.row.title, "source": service.surface(participant.row)}
