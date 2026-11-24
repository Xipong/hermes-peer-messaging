"""Durable participant identities and immediate-family communication boundaries.

Subagent markers locate candidates; they never establish ownership on their own.
Coordinators carry information between branches, never opening access to foreign workers.
"""
from __future__ import annotations

from dataclasses import dataclass

from persistence.conversation.sessions import END_REASON_COMPRESSION, SessionLineageKind, SessionRecord

from . import mailbox

LOCAL = frozenset({"cli", "tui", "desktop", "gui"})
STORED = LOCAL | {"local"}


def surface(row):
    label = row.created_source if row.source == "local" else row.source
    return "desktop" if label == "gui" else label


def same_profile(a, b):
    return not a or not b or a == b


def _route(store, sid, *, allow_ended=False) -> SessionRecord:
    sid = mailbox.session_component(sid)
    chain = store.lineage.chain(sid) or (sid,)
    if chain[0] != sid:
        raise ValueError("This chat rotated; use its current session")
    previous, profile = None, None
    for index, segment in enumerate(chain):
        row = store.sessions.get(mailbox.session_component(segment))
        if row is None:
            raise ValueError("Unknown session; use an exact ID from peer_sessions")
        if not store.sessions.owns_session(row.session_id):
            raise ValueError("Peer routing cannot cross profiles")
        continues = index < len(chain) - 1
        valid_end = (continues and row.end_reason == END_REASON_COMPRESSION) or (not continues and allow_ended)
        if row.hidden or row.archived or (row.ended_at is not None and not valid_end):
            raise ValueError("Session is hidden, ended, or archived; refresh peer_sessions")
        if (row.source not in STORED or surface(row) not in LOCAL or row.model_config_corrupt
                or row.lineage_kind is SessionLineageKind.DELEGATE or row.model_config.get("_delegate_from")):
            raise ValueError("Peer messaging is only for independent local CLI, TUI, and Desktop chats")
        if previous and (row.parent_session_id != previous.session_id
                         or previous.end_reason != END_REASON_COMPRESSION
                         or previous.session_id in (row.branched_from, row.reset_from)):
            raise ValueError("Session no longer follows the selected compression route")
        if not same_profile(profile, row.profile_name):
            raise ValueError("Peer routing cannot cross profiles")
        profile = profile or row.profile_name
        previous = row
    assert previous is not None  # The chain includes at least the requested session.
    return previous


def _locate(store, sid, *, allow_ended=False):
    tip = _route(store, sid, allow_ended=allow_ended)
    first = tip
    seen = {tip.session_id}
    while first.parent_session_id and first.parent_session_id not in seen:
        parent = store.sessions.get(mailbox.session_component(first.parent_session_id))
        if not parent or parent.end_reason != "compression":
            break
        try:
            if _route(store, parent.session_id, allow_ended=allow_ended).session_id != tip.session_id:
                break
        except ValueError:
            break
        seen.add(parent.session_id)
        first = parent
    return first.session_id, tip




@dataclass(frozen=True)
class Participant:
    row: SessionRecord
    thread_id: str
    root_id: str
    root: SessionRecord
    parent_session_id: str | None
    parent_thread_id: str | None
    subagent_id: str | None

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

    def segments(self, row):
        """Verified compression ancestry for delegated identities."""
        segments, seen = [row], {row.session_id}
        while row.parent_session_id and self.store.lineage.is_compression_child(row):
            parent = row.parent_session_id
            if parent in seen:
                raise ValueError("Cyclic session lineage")
            row = self.get(parent)
            segments.append(row)
            seen.add(parent)
        return tuple(segments)

    def live(self, row):
        if row.ended_at is not None and not self.allow_ended:
            raise ValueError("Session has ended; refresh peer_sessions")

    @staticmethod
    def worker(row):
        if row.source != "subagent" and (row.source not in STORED or surface(row) not in LOCAL):
            raise ValueError("Peer teams require local delegated sessions")

    def local(self, row, *, follow=True):
        root_id, current = _locate(self.store, row.session_id, allow_ended=self.allow_ended)
        if not follow and current.session_id != row.session_id:
            raise ValueError("This chat rotated; use its current session")
        current = self.get(current.session_id)
        return Participant(current, root_id, root_id, current, None, None, None)

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
        parent = self.get(first.parent_session_id)
        # The stored delegation edge can point to any verified compression
        # segment. Siblings born before and after that rotation share a parent.
        parent_id = self.segments(parent)[-1].session_id if _child(parent) else root.thread_id
        return Participant(current, first.session_id, root.root_id, root.root,
                           first.parent_session_id, parent_id, _subagent_id(current) or subagent_id)


def resolve(store, sid, *, follow=True, allow_ended=False) -> Participant:
    resolver = _Resolver(store, allow_ended)
    row = resolver.get(sid)
    return resolver.child(row, follow=follow) if _child(row) else resolver.local(row, follow=follow)


def relationship(a, b) -> str | None:
    """Describe b from a's permitted neighborhood, using stable parent edges."""
    if not same_profile(a.root.profile_name, b.root.profile_name):
        return None
    if a.thread_id == b.thread_id:
        return "self"
    if not (a.is_child or b.is_child):
        return "peer"
    if a.root_id != b.root_id:
        return None
    if a.parent_thread_id == b.thread_id:
        return "parent"
    if b.parent_thread_id == a.thread_id:
        return "child"
    if a.parent_thread_id is not None and a.parent_thread_id == b.parent_thread_id:
        return "sibling"
    return None


def allowed(a, b) -> bool:
    return relationship(a, b) not in {None, "self"}


def compression_ids(store, participant) -> tuple[str, ...]:
    """Only the already verified participant's own compression segments."""
    if participant.is_child:
        resolver = _Resolver(store, allow_ended=True)
        return tuple(segment.session_id for segment in reversed(resolver.segments(participant.row)))
    return tuple(store.lineage.chain(participant.thread_id) or (participant.thread_id,))


def describe(participant, *, viewer) -> dict:
    relation = relationship(viewer, participant)
    if relation is None:
        raise ValueError("Participant is outside the immediate communication neighborhood")
    result = {"session_id": participant.row.session_id, "thread_id": participant.thread_id,
              "root_thread_id": participant.root_id, "parent_session_id": participant.parent_session_id,
              "parent_thread_id": participant.parent_thread_id,
              "subagent_id": participant.subagent_id, "is_child": participant.is_child,
              "title": participant.row.title, "source": surface(participant.row),
              "relation": relation}
    if viewer.is_child and viewer.parent_thread_id != participant.root_id:
        result.pop("root_thread_id")
    # Returning a visible parent's own parent would leak a grandparent address.
    if participant.parent_thread_id not in {None, viewer.thread_id, viewer.parent_thread_id}:
        result["parent_session_id"] = None
        result["parent_thread_id"] = None
    return result
