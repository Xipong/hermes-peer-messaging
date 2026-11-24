"""Explicitly related, passive local threads. No scheduling or inference here."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from persistence.conversation.messages import TranscriptRead
from persistence_assembly import open_persistence_reader
from utils import atomic_json_write

from . import mailbox, service
from .project_identity import ProjectIndex

_MAX_LINKS = 1024
_MAX_PEERS = 32
_SCAN = 80
_MAX_TEXT = 3000


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
        if not isinstance(value, dict) or value.get("version") != 1 or not isinstance(value.get("links"), dict):
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
                atomic_json_write(path, {"version": 1, "links": links}, indent=None, mode=0o600)
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


def same_project(a, b, *, index=None):
    index = index if index is not None else ProjectIndex(get_hermes_home())
    a_project = service.project(a, index=index)
    return a_project is not None and a_project == service.project(b, index=index)


def accepted(edge, a, b):
    return bool(isinstance(edge, dict) and edge.get("since") is not None
                and set(edge.get("offers", ())) == {a, b})


def shares_transcript(edge, own, other, *, index=None):
    # Legacy links could only be offered in the same project. A mail-only link
    # must not turn into transcript consent when its participants move.
    return edge.get("mail_only", False) is False and same_project(own, other, index=index)


def neighbors(links, own):
    for key, edge in links.items():
        left, right = key.split(":")
        if own in (left, right):
            yield (right if left == own else left), edge


def new_edge(links, a, b):
    key = pair(a, b)
    if key not in links:
        if len(links) >= _MAX_LINKS or any(sum(1 for _ in neighbors(links, sid)) >= _MAX_PEERS for sid in (a, b)):
            raise ValueError("Thread link capacity reached; unlink unused conversations")
        links[key] = {"offers": [], "since": None}
    return links[key]


def link(args, sid):
    home = get_hermes_home()
    requested = mailbox.session_component(args.get("thread_id"))
    action = args.get("action", "offer")
    if action not in {"offer", "unlink"}:
        raise ValueError("action must be offer or unlink")
    with graph(home, write=True) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        if action == "unlink":
            # The stable ID can revoke a link even if the other chat is gone.
            other_id = requested if pair(own_id, requested) in links else locate(store, requested, allow_ended=True)[0]
            links.pop(pair(own_id, other_id), None)
            return {"success": True, "status": "unlinked", "thread_id": other_id}
        other_id, other = locate(store, requested)
        if not service.same_profile(own.profile_name, other.profile_name):
            raise ValueError("Thread links cannot cross profiles")
        edge = new_edge(links, own_id, other_id)
        index = ProjectIndex(home)
        if edge["since"] is None:
            if not same_project(own, other, index=index):
                edge["mail_only"] = True
            edge["offers"] = sorted({*edge["offers"], own_id})
            if set(edge["offers"]) == {own_id, other_id}:
                edge["since"] = time.time()
        is_linked = accepted(edge, own_id, other_id)
        status = "linked" if is_linked else "awaiting_other_chat"
        return {"success": True, "status": status, "thread_id": other_id,
                "transcript_shared": is_linked and shares_transcript(edge, own, other, index=index),
                "note": "Both parents must offer the exact link. Once linked, use peer_sessions(same_project=false) "
                        "to find teammates across projects. Cross-project links grant messaging only; no chat is started."}


def describe(key, row):
    return {"thread_id": key, "session_id": row.session_id, "title": row.title,
            "source": service.surface(row), "started_at": row.started_at,
            "last_activity_at": row.last_activity_at, "ended_at": row.ended_at, "message_count": row.message_count}


def list_threads(args, sid):
    from . import teams

    home = get_hermes_home()
    with graph(home) as links, open_persistence_reader(home) as store:
        participant = teams.resolve(store, sid)
        own_id, own = participant.root_id, participant.root
        result = []
        index = ProjectIndex(home)
        for other_id, edge in neighbors(links, own_id):
            try:
                _, other = locate(store, other_id, allow_ended=True)
            except ValueError:
                continue
            if not service.same_profile(own.profile_name, other.profile_name):
                continue
            is_linked = accepted(edge, own_id, other_id)
            state = "linked" if is_linked else ("offered" if own_id in edge["offers"] else "incoming")
            result.append({**describe(other_id, other), "link": state,
                           "transcript_shared": is_linked and shares_transcript(edge, own, other, index=index)})
        return {"success": True, "thread_id": own_id, "session_id": participant.row.session_id,
                "root_thread_id": own_id, "threads": result}


def linked(store, links, own_id, own, target, *, allow_ended=False):
    target = mailbox.session_component(target)
    other_id, row = locate(store, target, allow_ended=allow_ended)
    edge = links.get(pair(own_id, other_id))
    if not service.same_profile(own.profile_name, row.profile_name) or not accepted(edge, own_id, other_id):
        raise ValueError("This thread has not accepted the link. Use link_thread from both chats first.")
    if not shares_transcript(edge, own, row):
        raise ValueError("This link does not share transcripts. Use peer_send for coordination.")
    return other_id, row, edge["since"]


def text_only(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part["text"] for part in value if isinstance(part, dict)
                         and part.get("type") in {"text", "input_text", "output_text"}
                         and isinstance(part.get("text"), str))
    return ""


def public_message(msg, since):
    if msg.role not in {"user", "assistant"} or msg.timestamp < since or msg.compressed_summary or msg.display_kind or msg.tool_calls:
        return None
    meta = msg.display_metadata or {}
    if msg.role == "assistant":
        # Only an explicitly finished answer can satisfy a wait; commentary, errors and tool rows cannot.
        if msg.finish_reason not in {"stop", "end_turn", "completed", "final", "final_answer"}:
            return None
        value = msg.content
    else:
        if meta.get("withheld_prompt") or msg.api_content is not None and "typed_ask" not in meta:
            return None
        value = meta.get("typed_ask", msg.content)
    text = text_only(value)
    if not text.strip():
        return None
    return {"row_id": int(msg.id), "role": msg.role, "text": text[:_MAX_TEXT],
            "truncated": len(text) > _MAX_TEXT, "timestamp": msg.timestamp}


def recent(store, row, since, *, limit=5, after=0, roles=("user", "assistant")):
    out, limited = [], []
    with store.messages.read_snapshot() as snapshot:
        for role in roles:
            found = 0
            for offset in range(_SCAN):
                candidate = store.messages.latest_row_id(row.session_id, role=role, offset=offset,
                                                        require_text=role == "assistant")
                if candidate is None or candidate <= after:
                    break
                # Select the exact active row, retaining summary/visibility metadata.
                messages = store.messages.read_transcript(row.session_id, TranscriptRead(after_id=candidate - 1, limit=1))
                if messages and messages[0].id == candidate:
                    if message := public_message(messages[0], since):
                        out.append(message)
                        found += 1
                        if found >= limit:
                            break
            else:
                older = store.messages.latest_row_id(row.session_id, role=role, offset=_SCAN,
                                                    require_text=role == "assistant")
                if older is not None and older > after:
                    limited.append(role)
        if not snapshot():
            raise ValueError("Thread transcript changed during the read; retry the observation")
    out.sort(key=lambda message: message["row_id"])
    final = next((message["row_id"] for message in reversed(out) if message["role"] == "assistant"), None)
    return {"messages": out[-limit:], "latest_final_row_id": final,
            "scan_limit_per_role": _SCAN, "scan_limited_roles": limited}


def read_thread(args, sid):
    home = get_hermes_home()
    limit = args.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("limit must be 1–10")
    with graph(home) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        other_id, row, since = linked(store, links, own_id, own, args.get("thread_id"), allow_ended=True)
        data = recent(store, row, since, limit=limit)
        return {"success": True, "authority": "peer_data", **describe(other_id, row), **data, "shared_since": since,
                "note": "Public text from the current segment after link acceptance. scan_limited_roles names incomplete scans; not a live turn status."}


def wait_threads(args, sid):
    targets = args.get("threads")
    wait = args.get("wait_seconds", 15)
    if not isinstance(targets, list) or not 1 <= len(targets) <= 5 or type(wait) not in (float, int) or not 0 <= wait <= 30:
        raise ValueError("Supply 1–5 threads and wait_seconds between 0 and 30")
    cursors = {}
    for item in targets:
        if not isinstance(item, dict) or set(item) - {"thread_id", "after_row_id"}:
            raise ValueError("Expected thread_id and optional after_row_id per observation")
        target = mailbox.session_component(item.get("thread_id"))
        after = item.get("after_row_id", 0)
        if type(after) is not int or after < 0 or target in cursors:
            raise ValueError("Use unique thread IDs and non-negative after_row_id values")
        cursors[target] = after
    home = get_hermes_home()
    deadline = time.monotonic() + wait
    cached = {}
    while True:
        with graph(home) as links, open_persistence_reader(home) as store:
            own_id, own = locate(store, sid)
            ready, pending, closed, limited = [], [], [], []
            for target, cursor in cursors.items():
                other_id, row, since = linked(store, links, own_id, own, target, allow_ended=True)
                # The row's transcript_revision tracks rewrites, not appends.
                # The public fingerprint also changes when a new final is saved.
                cursor_key = (row.session_id, store.messages.conversation_revision(row.session_id), since)
                if target not in cached or cached[target][0] != cursor_key:
                    observation = recent(store, row, since, limit=1, after=cursor, roles=("assistant",))
                    cached[target] = (cursor_key, observation)
                else:
                    observation = cached[target][1]
                final = next(iter(observation["messages"]), None)
                if observation["scan_limited_roles"]:
                    limited.append(other_id)
                if final:
                    ready.append({"thread_id": other_id, "session_id": row.session_id, "result": final})
                elif row.ended_at is not None:
                    closed.append(other_id)
                else:
                    pending.append(other_id)
        remaining = deadline - time.monotonic()
        if ready or not pending or remaining <= 0:
            status = "ready" if ready else "scan_limit" if limited else "timeout" if pending else "closed"
            return {"success": True, "authority": "peer_data", "status": status, "ready": ready,
                    "pending": pending, "closed": closed, "scan_limited": limited, "scan_limit_per_role": _SCAN,
                    "note": "Saved finals in the current segment only. scan_limited names incomplete scans; no chat was started."}
        time.sleep(min(0.75, remaining))


def register(ctx):
    def add(name, operation, description, properties, required=()):
        ctx.register_tool(name=name, toolset=service.TOOLSET, check_fn=service.requirements,
                          handler=service.handler(operation), emoji="🧵", schema={"name": name,
                              "description": description, "parameters": {"type": "object", "properties": properties,
                              "required": list(required), "additionalProperties": False}})
    id_field = {"type": "string", "description": "Exact thread_id or current session_id"}
    add("list_threads", list_threads, "List your team's directly linked local parent chats and pending offers. Delegated agents inherit their root parent's directory. No wake.", {})
    add("link_thread", link, "For independent local parents in one profile: propose or revoke a link between teams. Both must offer; links are not transitive. Linked teams can message each other across projects. Only links formed in the same project can share new public parent text. No prompt is sent.",
        {"thread_id": id_field, "action": {"type": "string", "enum": ["offer", "unlink"]}}, ("thread_id",))
    add("read_thread", read_thread, "For independent parents: read bounded public text after a shared-transcript link in the same project. Mail-only links do not grant access. Delegated agents receive mail as peer_data with their tool results. scan_limited_roles reports incomplete scans. No raw tool results, reasoning, or wake.",
        {"thread_id": id_field, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("thread_id",))
    add("wait_threads", wait_threads, "For independent parents: wait up to 30 seconds for a same-project parent's saved final newer than the cursor on a shared-transcript link. Mail replies arrive as peer_data with ordinary tool results; do not use this to poll the inbox or track child completion. scan_limit reports an incomplete search. Does not launch or resume anyone.",
        {"threads": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "object", "properties": {
            "thread_id": id_field, "after_row_id": {"type": "integer", "minimum": 0}}, "required": ["thread_id"], "additionalProperties": False}},
         "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30}}, ("threads",))
