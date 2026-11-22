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
            other_id, other = locate(store, requested, allow_ended=action == "unlink")
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
    from . import teams

    home = get_hermes_home()
    with graph(home) as links, open_persistence_reader(home) as store:
        participant = teams.resolve(store, sid)
        own_id, own = participant.root_id, participant.root
        result = []
        for other_id, edge in neighbors(links, own_id):
            try:
                _, other = locate(store, other_id, allow_ended=True)
                project_pair(own, other)
            except ValueError:
                continue
            state = "linked" if edge["since"] is not None else ("offered" if own_id in edge["offers"] else "incoming")
            result.append({**describe(other_id, other), "link": state})
        return {"success": True, "thread_id": own_id, "session_id": participant.row.session_id,
                "root_thread_id": own_id, "threads": result}


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


def send_message(args, sid):
    return service.send({**args, "target_session_id": args.get("thread_id")}, sid, require_link=True)


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
    add("list_threads", list_threads, "List your team's directly linked local parent chats and pending offers. Delegated agents inherit their root parent's directory. No wake.", {})
    add("link_thread", link, "For independent parent chats: propose or revoke a same-project link. Both parents must offer. The link also permits messaging between their delegated teams; it does not extend transitively to other links. Parents share new public text after acceptance. No prompt is sent.",
        {"thread_id": id_field, "action": {"type": "string", "enum": ["offer", "unlink"]}}, ("thread_id",))
    add("read_thread", read_thread, "For independent parent chats: read bounded public text from a linked parent after acceptance. Delegated agents use peer_receive for their own messages. scan_limited_roles reports incomplete scans. No raw tool results, reasoning, or wake.",
        {"thread_id": id_field, "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("thread_id",))
    add("send_message_to_thread", send_message, "Queue an attributed note to an exact session in your team or a directly linked parent's team. Parent and delegated recipients use peer_receive. A queued note is not a new task, read receipt, or wake.",
        {"thread_id": id_field, "message": {"type": "string", "maxLength": mailbox.MAX_MESSAGE_CHARS}, "in_reply_to": {"type": "string"}}, ("thread_id", "message"))
    add("wait_threads", wait_threads, "For independent parent chats: wait up to 30 seconds for a linked parent's saved final newer than the cursor. Use peer_receive(wait_seconds=...) for replies to your own messages. scan_limit reports an incomplete search. Does not launch or resume anyone.",
        {"threads": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "object", "properties": {
            "thread_id": id_field, "after_row_id": {"type": "integer", "minimum": 0}}, "required": ["thread_id"], "additionalProperties": False}},
         "wait_seconds": {"type": "number", "minimum": 0, "maximum": 30}}, ("threads",))
    add("create_thread", create_thread, "Only on an explicit user request: create a fresh independent named chat in this project and link it. No messages copied, no model called, no chat awakened.",
        {"title": {"type": "string", "maxLength": 100}, "confirm": {"type": "boolean"}}, ("title", "confirm"))
