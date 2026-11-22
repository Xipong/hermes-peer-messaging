"""Explicitly related, passive local threads. No scheduling or inference here."""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from uuid import uuid4

from hermes_cli.active_sessions import _FileLock
from hermes_constants import get_hermes_home
from persistence.conversation.messages import TranscriptRead
from persistence.conversation.sessions import NewSession, sanitize_title
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope
from utils import atomic_json_write

from . import mailbox, service

_MAX_LINKS = 1024
_MAX_PEERS = 32
_SCAN = 80
_MAX_TEXT = 3000


@contextmanager
def graph(home, *, write=False):
    folder = service.root(home)
    path = folder / ".threads.json"
    lock = folder / ".threads.lock"
    if path.is_symlink() or lock.is_symlink():
        raise ValueError("Thread link storage must not be a symlink")
    if write:
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
    if not write:
        yield load()  # Atomic publication gives readers a complete old or new snapshot.
        return
    with _FileLock(lock, timeout=2.0):
        links = load()
        yield links
        atomic_json_write(path, {"version": 1, "links": links}, indent=None, mode=0o600)


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


def project_pair(a, b):
    if not service.same_profile(a.profile_name, b.profile_name) or not service.project(a) or service.project(a) != service.project(b):
        raise ValueError("Thread links require local conversations in the same profile and project")


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
        if action == "unlink" and pair(own_id, requested) in links:
            other_id = requested  # Revocation still works for an ended or deleted participant.
        else:
            other_id, other = locate(store, requested)
            if action == "offer":
                project_pair(own, other)
        key = pair(own_id, other_id)
        if action == "unlink":
            links.pop(key, None)
            return {"success": True, "status": "unlinked", "thread_id": other_id}
        edge = new_edge(links, own_id, other_id)
        edge["offers"] = sorted({*edge["offers"], own_id})
        if edge["since"] is None and len(edge["offers"]) == 2:
            edge["since"] = time.time()
        status = "linked" if edge["since"] is not None else "awaiting_other_chat"
        return {"success": True, "status": status, "thread_id": other_id,
                "note": "Both chats must offer the exact link. Linking never sends a prompt or wakes a chat."}


def describe(key, row):
    return {"thread_id": key, "session_id": row.session_id, "title": row.title,
            "source": service.surface(row), "started_at": row.started_at,
            "last_activity_at": row.last_activity_at, "ended_at": row.ended_at, "message_count": row.message_count}


def list_threads(args, sid):
    home = get_hermes_home()
    with graph(home) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        result = []
        for other_id, edge in neighbors(links, own_id):
            try:
                _, other = locate(store, other_id, allow_ended=True)
                project_pair(own, other)
            except ValueError:
                continue
            state = "linked" if edge["since"] is not None else ("offered" if own_id in edge["offers"] else "incoming")
            result.append({**describe(other_id, other), "link": state})
        return {"success": True, "thread_id": own_id, "threads": result}


def linked(store, links, own_id, own, target, *, allow_ended=False):
    target = mailbox.session_component(target)
    other_id, row = locate(store, target, allow_ended=allow_ended)
    project_pair(own, row)
    edge = links.get(pair(own_id, other_id))
    if not edge or edge["since"] is None:
        raise ValueError("This thread has not accepted the link. Use link_thread from both chats first.")
    return other_id, row, edge["since"]


def text_only(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part["text"] for part in value if isinstance(part, dict)
                         and part.get("type") in {"text", "input_text", "output_text"}
                         and isinstance(part.get("text"), str))
    return ""


def recent(store, row, since):
    out = []
    for msg in store.messages.read_transcript(row.session_id, TranscriptRead(limit=_SCAN, latest=True)):
        if msg.role not in {"user", "assistant"} or msg.timestamp < since or msg.compressed_summary or msg.display_kind or msg.tool_calls:
            continue
        meta = msg.display_metadata or {}
        if msg.role == "assistant":
            # Only an explicitly finished answer can satisfy a wait; commentary, errors and tool rows cannot.
            if msg.finish_reason not in {"stop", "end_turn", "completed", "final", "final_answer"}:
                continue
            value = msg.content
        else:
            if meta.get("withheld_prompt") or msg.api_content is not None and "typed_ask" not in meta:
                continue
            value = meta.get("typed_ask", msg.content)
        text = text_only(value)
        if not text.strip():
            continue
        out.append({"row_id": int(msg.id), "role": msg.role, "text": text[:_MAX_TEXT],
                    "truncated": len(text) > _MAX_TEXT, "timestamp": msg.timestamp})
    return out


def read_thread(args, sid):
    home = get_hermes_home()
    limit = args.get("limit", 5)
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("limit must be 1–10")
    with graph(home) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        other_id, row, since = linked(store, links, own_id, own, args.get("thread_id"), allow_ended=True)
        data = recent(store, row, since)
        last_final = next((m["row_id"] for m in reversed(data) if m["role"] == "assistant"), None)
        return {"success": True, "authority": "peer_data", **describe(other_id, row), "messages": data[-limit:],
                "latest_final_row_id": last_final, "shared_since": since,
                "note": "Bounded public text from the current session segment after link acceptance; not a live turn status."}


def send_message(args, sid):
    home = get_hermes_home()
    with graph(home) as links, open_persistence_reader(home) as store:
        own_id, own = locate(store, sid)
        _, row, _ = linked(store, links, own_id, own, args.get("thread_id"))
    return service.send({**args, "target_session_id": row.session_id}, sid)


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
            ready, pending, closed = [], [], []
            for target, cursor in cursors.items():
                other_id, row, since = linked(store, links, own_id, own, target, allow_ended=True)
                cursor_key = (row.session_id, row.transcript_revision, since)
                if target not in cached or cached[target][0] != cursor_key:
                    final = next((m for m in reversed(recent(store, row, since)) if m["role"] == "assistant"
                                  and m["row_id"] > cursor), None)
                    cached[target] = (cursor_key, final)
                else:
                    final = cached[target][1]
                if final:
                    ready.append({"thread_id": other_id, "session_id": row.session_id, "result": final})
                elif row.ended_at is not None:
                    closed.append(other_id)
                else:
                    pending.append(other_id)
        remaining = deadline - time.monotonic()
        if ready or not pending or remaining <= 0:
            status = "ready" if ready else "timeout" if pending else "closed"
            return {"success": True, "authority": "peer_data", "status": status, "ready": ready,
                    "pending": pending, "closed": closed, "note": "Observed saved finals only; no chat was started."}
        time.sleep(min(0.75, remaining))


def create_thread(args, sid):
    if args.get("confirm") is not True:
        raise ValueError("Use confirm=true only when the user explicitly asked to create a new chat")
    name = args.get("title")
    if not isinstance(name, str) or not (title := sanitize_title(name)):
        raise ValueError("A non-empty chat title is required")
    home = get_hermes_home()
    target = f"thread-{uuid4().hex}"
    with graph(home, write=True) as links, open_persistence(profile_scope(home=home)) as store:
        own_id, own = locate(store, sid)
        if not service.project(own) or not own.cwd:
            raise ValueError("Set the current chat's project before creating another thread")
        edge = new_edge(links, own_id, target)
        # Do not copy messages, system prompt, runtime policy, secrets, or parent DB lineage.
        # This creates a named local chat record only. A human may open it separately.
        store.sessions.create_exclusive(target, source=service.surface(own), title=title)
        store.sessions.create(NewSession(target, service.surface(own), cwd=own.cwd, git_repo_root=own.git_repo_root))
        row = store.sessions.get(target)
        project_pair(own, row)
        edge.update(offers=sorted((own_id, target)), since=time.time())
    return {"success": True, "status": "created_idle", **describe(target, row),
            "note": "Created a new self-contained chat without starting a turn. Open it by title or session ID."}


def register(ctx):
    def add(name, operation, description, properties, required=()):
        ctx.register_tool(name=name, toolset=service.TOOLSET, check_fn=service.requirements,
                          handler=service.handler(operation), emoji="🧵", schema={"name": name,
                              "description": description, "parameters": {"type": "object", "properties": properties,
                              "required": list(required), "additionalProperties": False}})
    id_field = {"type": "string", "description": "Exact thread_id or current session_id"}
    add("list_threads", list_threads, "List only explicitly linked local chats and pending link offers. No wake.", {})
    add("link_thread", link, "Propose a link to an exact same-project chat, or revoke it. Existing chats share only new public text after both independently offer. No prompt is sent.",
        {"thread_id": id_field, "action": {"type": "string", "enum": ["offer", "unlink"]}}, ("thread_id",))
    add("read_thread", read_thread, "Read bounded public text and state from an explicitly linked chat, only after link acceptance. No raw tool results, reasoning, or wake.",
        {"thread_id": id_field, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("thread_id",))
    add("send_message_to_thread", send_message, "Queue an attributed note to an explicitly linked chat without waking it. A queued note is not a new task or delivery receipt.",
        {"thread_id": id_field, "message": {"type": "string", "maxLength": mailbox.MAX_MESSAGE_CHARS}, "in_reply_to": {"type": "string"}}, ("thread_id", "message"))
    add("wait_threads", wait_threads, "Wait up to 30 seconds for a saved final answer from any linked thread newer than the supplied cursor. Does not launch or resume anyone.",
        {"threads": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "object", "properties": {
            "thread_id": id_field, "after_row_id": {"type": "integer", "minimum": 0}}, "required": ["thread_id"], "additionalProperties": False}},
         "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30}}, ("threads",))
    add("create_thread", create_thread, "Only on an explicit user request: create a fresh independent named chat in this project and link it. No messages copied, no model called, no chat awakened.",
        {"title": {"type": "string", "maxLength": 100}, "confirm": {"type": "boolean"}}, ("title", "confirm"))
