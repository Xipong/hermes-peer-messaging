"""Stable coordinator contacts between independent local roots, without access grants."""
from __future__ import annotations

import json
from contextlib import contextmanager

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from persistence_assembly import open_persistence_reader
from utils import atomic_json_write

from . import mailbox, service

_MAX_LINKS = 1024
_MAX_PEERS = 32


@contextmanager
def graph(home, *, write=False, locked=False):
    folder = service.root(home)
    path = folder / ".threads.json"
    lock = folder / ".threads.lock"
    if path.is_symlink() or lock.is_symlink():
        raise ValueError("Thread link storage must not be a symlink")
    if write or locked:
        folder.mkdir(mode=0o700, parents=True, exist_ok=True)
    def load():
        try:
            with path.open("rb") as stream:
                data = stream.read(512_001)
        except FileNotFoundError:
            return {}
        if len(data) > 512_000:
            raise ValueError("Thread link store too large")
        value = json.loads(data)
        if not isinstance(value, dict) or value.get("version") != 2 or not isinstance(value.get("links"), dict):
            raise ValueError("Invalid thread link store")
        return value["links"]
    if not (write or locked):
        yield load()  # Atomic publication gives readers a complete old or new snapshot.
        return
    try:
        with _FileLock(lock, timeout=2.0):
            links = load()
            yield links
            if write:
                atomic_json_write(path, {"version": 2, "links": links}, indent=None, mode=0o600)
    except RuntimeError as exc:
        if isinstance(exc.__cause__, BlockingIOError):
            raise TimeoutError("Peer links are busy; retry the operation") from exc
        raise


def locate(store, sid, *, allow_ended=False):
    tip = service.route(store, sid, follow=True, allow_ended=allow_ended)
    first = tip
    seen = {tip.session_id}
    while first.parent_session_id and first.parent_session_id not in seen:
        parent = store.sessions.get(mailbox.session_component(first.parent_session_id))
        if not parent or parent.end_reason != "compression":
            break
        try:
            if service.route(store, parent.session_id, follow=True, allow_ended=allow_ended).session_id != tip.session_id:
                break
        except ValueError:
            break
        seen.add(parent.session_id)
        first = parent
    return first.session_id, tip


def pair(own, other):
    if own == other:
        raise ValueError("Choose another conversation")
    return ":".join(sorted((own, other)))


def accepted(edge, a, b):
    return isinstance(edge, list) and set(edge) == {a, b}


def neighbors(links, own):
    for key, offers in links.items():
        left, right = key.split(":")
        if own in (left, right):
            yield (right if left == own else left), offers


def link(args, sid):
    home = get_hermes_home()
    requested = mailbox.session_component(args.get("thread_id"))
    action = args.get("action", "offer")
    if action not in {"offer", "unlink"}:
        raise ValueError("action must be offer or unlink")
    with graph(home, write=True) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        if action == "unlink":
            other_id = requested if pair(own_id, requested) in links else locate(store, requested, allow_ended=True)[0]
            links.pop(pair(own_id, other_id), None)
            return {"success": True, "status": "unlinked", "thread_id": other_id}
        other_id, other = locate(store, requested)
        if not service.same_profile(own.profile_name, other.profile_name):
            raise ValueError("Thread links cannot cross profiles")
        key = pair(own_id, other_id)
        if key not in links and (len(links) >= _MAX_LINKS or any(
                sum(1 for _ in neighbors(links, root)) >= _MAX_PEERS for root in (own_id, other_id))):
            raise ValueError("Thread link capacity reached; unlink unused conversations")
        offers = links[key] = sorted({*links.get(key, ()), own_id})
        return {"success": True, "status": "linked" if accepted(offers, own_id, other_id) else "awaiting_other_chat",
                "thread_id": other_id}


def directory(store, own, links, *, incoming_only=False):
    """Root contacts ignore project/recency filters but stay private to roots."""
    if own.is_child:
        return []
    result = []
    for other_id, offers in neighbors(links, own.root_id):
        state = "linked" if accepted(offers, own.root_id, other_id) else (
            "offered" if own.root_id in offers else "incoming")
        if incoming_only and state != "incoming":
            continue
        other = None
        try:
            _, candidate = locate(store, other_id, allow_ended=True)
            if service.same_profile(own.root.profile_name, candidate.profile_name):
                other = candidate
        except ValueError:
            pass
        # Keep the stable link address revocable even if the peer disappeared.
        result.append({"thread_id": other_id, "session_id": other.session_id if other else None,
                       "ended_at": other.ended_at if other else None, "link": state})
    return result
