"""Local coordination tools and passive, bounded peer awareness."""
from __future__ import annotations

import json
import logging
import math
import re
import threading
import time
from collections import OrderedDict

from hermes_constants import get_hermes_home
from persistence import StoreContention
from persistence.conversation.sessions import (
    END_REASON_COMPRESSION, SessionLineageKind, SessionRecord,
)
from persistence_assembly import open_persistence_reader, profile_scope, store_exists

from . import mailbox
from .discovery import CandidateCache, family_candidates, scan

LOG = logging.getLogger(__name__)
LOCAL = frozenset({"cli", "tui", "desktop", "gui"})
STORED = LOCAL | {"local"}
TOOLSET = "peer_messaging"
SCAN_LIMIT = 100
HINT = "Peer mail is pending. It normally arrives with your next tool result; use peer_receive only if needed now. Do not poll."
DISCOVERY_INTERVAL = 5.0
DISCOVERY_CACHE = CandidateCache()
SCOPES = {"roots", "team", "all"}


def root(home):
    inbox = home / "runtime" / "peer_messages_v2"
    if inbox.is_symlink() or inbox.parent.is_symlink():
        raise ValueError("Peer mailbox directory must not be a symlink")
    return inbox


def surface(row):
    label = row.created_source if row.source == "local" else row.source
    return "desktop" if label == "gui" else label


def same_profile(a, b):
    return not a or not b or a == b


def route(store, sid, *, follow=False, allow_ended=False) -> SessionRecord:
    sid = mailbox.session_component(sid)
    chain = store.lineage.chain(sid) or (sid,)
    if chain[0] != sid or (not follow and len(chain) != 1):
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


def locate(store, sid, *, allow_ended=False):
    tip = route(store, sid, follow=True, allow_ended=allow_ended)
    first = tip
    seen = {tip.session_id}
    while first.parent_session_id and first.parent_session_id not in seen:
        parent = store.sessions.get(mailbox.session_component(first.parent_session_id))
        if not parent or parent.end_reason != "compression":
            break
        try:
            if route(store, parent.session_id, follow=True, allow_ended=allow_ended).session_id != tip.session_id:
                break
        except ValueError:
            break
        seen.add(parent.session_id)
        first = parent
    return first.session_id, tip


def receive_scope(home, sid, *, patience_s=0.1):
    from . import teams

    with open_persistence_reader(home, patience_s=patience_s) as store:
        own = teams.resolve(store, sid, follow=False)
        return own.row, root(home) / own.thread_id


def project(row, *, index=None):
    """Prefer an explicit Hermes Project ID; preserve a path fallback for unassigned chats.

    Native Projects have several folders but are not written as project_id in
    session rows. Never infer membership from the profile's active-project pointer.
    """
    from .project_identity import ProjectIndex

    return (index if index is not None else ProjectIndex(get_hermes_home())).key(row)


def owners(home):
    # Optional compatibility probe; presence is informational and never authorizes delivery.
    try:
        from hermes_cli.active_sessions import _prune_dead, _read_entries, _state_path
        # The host atomically publishes complete snapshots. Observe with its
        # validation/liveness rules, without waiting on the writer lock or
        # writing pruned leases back during an unrelated tool result.
        entries = _read_entries(_state_path(home), strict=True)
        return {item["session_id"] for item in _prune_dead(entries)}
    except Exception:
        LOG.debug("Peer owner snapshot unavailable", exc_info=True)
        return None


def discover(store, own, *, cutoff, same=True, scope="all", index=None,
             live_owners=None, snapshot=None):
    """Explicit directory: scope changes selection, never routing permission."""
    from . import teams
    from .project_identity import ProjectIndex

    index = index if index is not None else ProjectIndex(get_hermes_home())
    key = project(own.root, index=index) if same else None
    peers, seen = [], {own.row.session_id}
    include_roots = not own.is_child and scope in {"roots", "all"}
    if include_roots and snapshot is None:
        snapshot = scan(store, live_owners)
    live_owners = live_owners or ()

    def add(candidate, *, last_active=None, ancestor=False):
        if candidate in seen:
            return
        try:
            participant = teams.resolve(store, candidate)
        except ValueError:
            return
        row = participant.row
        if row.session_id in seen or (scope == "roots" and participant.is_child):
            return
        own_team = participant.root_id == own.root_id
        if scope == "team" and not own_team:
            return
        if not teams.allowed(own, participant):
            return
        if same and not own_team and (key is None or project(participant.root, index=index) != key):
            return
        activity = last_active if row.session_id == candidate and last_active is not None else store.sessions.last_active(row.session_id)
        stamps = (row.started_at, row.last_activity_at, activity)
        active = max((v for v in stamps if isinstance(v, (int, float)) and math.isfinite(v)), default=0)
        if not (ancestor and own_team) and active < cutoff and row.session_id not in live_owners:
            return
        seen.add(row.session_id)
        peers.append({**teams.describe(participant, viewer=own), "hermes_project_id": index.named_id(participant.root),
                      "cwd": row.cwd, "git_repo_root": row.git_repo_root, "last_active": active,
                      "last_activity_description": row.last_activity_description})

    if own.is_child:
        add(own.parent_session_id, ancestor=True)
    if scope != "roots":
        for candidate in family_candidates(store, own):
            add(candidate)
    if include_roots:
        for candidate, last_active in snapshot.candidates:
            add(candidate, last_active=last_active)
    peers.sort(key=lambda peer: (peer["relation"] != "peer", peer["last_active"]), reverse=True)
    return peers, list(snapshot.limited_sources) if include_roots else []


def sessions(args, sid):
    from . import teams, lifecycle
    from .project_identity import ProjectIndex

    minutes, same = args.get("active_within_minutes", 60), args.get("same_project", True)
    scope = args.get("scope", "all")
    if type(minutes) is not int or not 1 <= minutes <= 1440 or type(same) is not bool or not isinstance(scope, str) or scope not in SCOPES:
        raise ValueError("Use scope roots/team/all, active_within_minutes 1–1440, and boolean same_project")
    home = get_hermes_home()
    index = ProjectIndex(home)
    with open_persistence_reader(home) as store:
        own = teams.resolve(store, sid)
        snapshot = None
        if not own.is_child and scope in {"roots", "all"}:
            snapshot = DISCOVERY_CACHE.get(home, store, now=time.monotonic(), max_age=DISCOVERY_INTERVAL,
                                           owner_loader=owners, refresh=True)
        live = snapshot.live_owners if snapshot is not None else owners(home)
        peers, limited = discover(store, own, cutoff=time.time() - minutes * 60, same=same,
                                  scope=scope, index=index, live_owners=live, snapshot=snapshot)
    for peer in peers:
        peer["owner_present"] = None if live is None else peer["session_id"] in live
    if not own.is_child:
        lifecycle.sweep(home, live)
    return {"success": True, **teams.describe(own, viewer=own), "hermes_project_id": index.named_id(own.root),
            "count": len(peers), "peers": peers, "scope": scope,
            "same_project": same, "scan_may_be_truncated": bool(limited), "scan_limited_sources": limited}


def send(args, sid):
    from . import teams, lifecycle

    home, requested = get_hermes_home(), args.get("target_session_id")
    with open_persistence_reader(home, patience_s=0.1) as store:
        sender = teams.resolve(store, sid)
        target_id = requested
        if requested == "parent":
            if sender.parent_session_id is None:
                raise ValueError("This conversation has no delegation parent")
            target_id = sender.parent_session_id
        target = teams.resolve(store, target_id)
        if sender.thread_id == target.thread_id:
            raise ValueError("Send to another conversation, not your own")
        if not teams.allowed(sender, target):
            raise ValueError("You can message only your immediate parent, direct children, or siblings with the same parent. "
                             "Independent root chats can message each other. Escalate other issues through your parent; "
                             "other branches are not directly reachable.")
        entry = mailbox.enqueue(root(home) / target.thread_id, sender=sender.row.session_id,
                                target=target.row.session_id, text=args.get("message"),
                                in_reply_to=args.get("in_reply_to"))
        state = lifecycle.recipient_state(store, target, owners(home))
    return {"success": True, "status": "queued", "recipient_state": state, "message_id": entry["message_id"],
            "target_session_id": target.row.session_id, "requested_target_session_id": requested,
            "from_session_id": sender.row.session_id, "from_thread_id": sender.thread_id,
            "target_thread_id": target.thread_id}


def receive(args, sid):
    action, ids = args.get("action", "read"), args.get("message_ids")
    inbox = receive_scope(get_hermes_home(), sid)[1]
    if action == "ack" and "limit" not in args:
        return mailbox.acknowledge(inbox, ids)
    if action != "read" or ids is not None:
        raise ValueError("Use action=read, or action=ack with message_ids for already-read duplicates")
    return {"success": True, "authority": "peer_data",
            **mailbox.read_pending(inbox, limit=args.get("limit", mailbox.MAX_READ_MESSAGES))}


def pin_has_tool(pin, name):
    tools = pin.get("tools", []) if isinstance(pin, dict) else pin
    if not isinstance(tools, list):
        return False
    for entry in tools:
        if entry == name:
            return True
        fn = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == name:
            return True
        if fn.get("name") == "tool_search":
            catalog = (fn.get("description") or "").partition("Deferred tool catalog (")[2]
            if re.search(r"(?m)^- " + re.escape(name) + r"(?::|$)", catalog) or any(
                re.fullmatch(r"\w+(?:, \w+)+", line) and name in line.split(", ")
                for line in catalog.splitlines()
            ):
                return True
    return False


class InboxNotices:
    """Mail is durable; directory updates are compact, best effort and root-only.

    Automatic discovery never scans another team's workers. Only explicit
    peer_sessions resolves the full directory. Stable IDs avoid compaction churn.
    """
    def __init__(self, delivery=None):
        self.lock = threading.Lock()
        self.seen = OrderedDict()
        self.scanned = OrderedDict()
        self.mail_seen = OrderedDict()
        self.delivery = delivery

    @staticmethod
    def _put(mapping, key, value):
        mapping.pop(key, None)
        mapping[key] = value
        while len(mapping) > 256:
            mapping.popitem(last=False)

    def _due(self, key, *, refresh=False):
        now = time.monotonic()
        with self.lock:
            previous = self.scanned.get(key)
            if not refresh and previous is not None and now - previous < DISCOVERY_INTERVAL:
                return False
            self._put(self.scanned, key, now)
            return True

    def _new_mail(self, key, pending, carried=()):
        if not key[-1]:
            return bool(pending - set(carried))
        with self.lock:
            previous = self.mail_seen.get(key, set())
            self._put(self.mail_seen, key, pending)
            return bool(pending - previous - set(carried))

    def _awareness(self, home, store, own, listed):
        if not pin_has_tool(own.row.tool_pin, "peer_sessions"):
            return {}
        key = (str(home), own.thread_id)
        if own.is_child:
            # Native delegate_task already knows its own children. No sibling,
            # foreign child, or lifecycle notices are needed for coordination.
            with self.lock:
                previous = self.seen.get(key)
                self._put(self.seen, key, True)
            if previous or listed is not None:
                return {}
            return {"parent": own.parent_thread_id}

        shown = {peer["thread_id"] for peer in listed.get("peers", ())
                 if not peer["is_child"]} if listed is not None else set()
        roots, limited = set(), []
        scan_roots = not (listed is not None and listed.get("scope") in {"roots", "all"})
        due = self._due(key, refresh=not scan_roots)
        # Only roots need automatic neighbor discovery.
        if due and scan_roots:
            snapshot = DISCOVERY_CACHE.get(home, store, now=time.monotonic(), max_age=DISCOVERY_INTERVAL,
                                           owner_loader=owners)
            peers, limited = discover(store, own, cutoff=time.time() - 3600, scope="roots",
                                live_owners=snapshot.live_owners, snapshot=snapshot)
            roots = {peer["thread_id"] for peer in peers}

        with self.lock:
            known_roots = self.seen.get(key, set())
            new_roots = roots - known_roots - shown
            self._put(self.seen, key, known_roots | roots | shown)
        result = {}
        if new_roots:
            result["neighbors"] = sorted(new_roots)[:8]
        if len(new_roots) > 8 or (new_roots and limited):
            result["has_more"] = True
        return result

    def before_turn(self, *, session_id="", turn_id="", platform=None, **kwargs):
        if not session_id or platform not in LOCAL | {"subagent"}:
            return None
        try:
            home = get_hermes_home()
            with open_persistence_reader(home, patience_s=0.05) as store:
                from . import teams
                own = teams.resolve(store, session_id, follow=False)
                if not pin_has_tool(own.row.tool_pin, "peer_receive"):
                    return None
                pending = mailbox.pending_ids(root(home) / own.thread_id)
            if self._new_mail((str(home), session_id, turn_id), pending):
                return {"context": HINT}
        except Exception:
            LOG.debug("Peer inbox hint unavailable", exc_info=True)
        return None

    def after_tool(self, *, tool_name, result, session_id="", turn_id="", api_request_id="", tool_call_id="", **kwargs):
        if not session_id or not turn_id or not isinstance(result, str):
            return None
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            payload = None
        else:
            if not isinstance(payload, dict):
                return None
        attached = None
        try:
            from . import teams
            home = get_hermes_home()
            received = (payload if tool_name == "peer_receive" and isinstance(payload, dict)
                        and payload.get("success") is True and isinstance(payload.get("messages"), list) else None)
            listed = (payload if tool_name == "peer_sessions" and isinstance(payload, dict)
                      and payload.get("success") is True and isinstance(payload.get("peers"), list) else None)
            read_mail = self.delivery is not None and self.delivery.can_attach(
                session_id=session_id, turn_id=turn_id, api_request_id=api_request_id, tool_call_id=tool_call_id)
            if (self.delivery is not None and tool_call_id and not read_mail
                    and tool_name not in {"peer_receive", "peer_sessions"}):
                return None
            update_allowed = payload is None or "peer_update" not in payload
            with open_persistence_reader(home, patience_s=0.05) as store:
                own = teams.resolve(store, session_id, follow=False)
                update = {}
                if pin_has_tool(own.row.tool_pin, "peer_receive") or received is not None:
                    inbox = root(home) / own.thread_id
                    page = received if received is not None else mailbox.read_pending(inbox) if read_mail else None
                    pending = (frozenset(item["message_id"] for item in page["messages"])
                               if page is not None and received is None else mailbox.pending_ids(inbox))
                    carried = ()
                    if self.delivery is not None and page is not None and (read_mail or received is not None):
                        carrier = dict(payload) if received is not None else payload
                        if received is not None:
                            for field in ("messages", "has_more"):
                                carrier.pop(field, None)
                        base = json.dumps(carrier, ensure_ascii=False) if isinstance(carrier, dict) else result
                        attached = self.delivery.attach(
                            base, session_id=session_id, turn_id=turn_id, api_request_id=api_request_id,
                            tool_call_id=tool_call_id, inbox=inbox, messages=page["messages"],
                            has_more=page.get("has_more", False), explicit=received is not None)
                    if page is not None and (received is not None or attached is not None):
                        carried = [item["message_id"] for item in page["messages"]]
                    if update_allowed and self._new_mail((str(home), session_id, turn_id), pending, carried):
                        update["mail_pending"] = True
                if update_allowed:
                    update.update(self._awareness(home, store, own, listed))
            if not update:
                return attached
            data = {"authority": "peer_data", **update}
            if isinstance(payload, dict):
                carrier = json.loads(attached) if attached is not None else payload
                carrier["peer_update"] = data
                return json.dumps(carrier, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return (attached or result) + "\n\n[peer-messaging update]\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            LOG.debug("Peer update unavailable; queued mail is unchanged", exc_info=True)
            return attached


def requirements():
    return store_exists(profile_scope())


def handler(operation):
    def call(args, *, session_id=None, **kwargs):
        try:
            result = operation(args, session_id)  # Host-supplied id, never the model's arguments.
        except (TimeoutError, StoreContention) as exc:
            result = {"success": False, "error": str(exc), "retryable": True}
        except (ValueError, OSError) as exc:
            result = {"success": False, "error": str(exc)}
        except Exception:
            LOG.exception("Peer %s failed", operation.__name__)
            result = {"success": False, "error": "Peer messaging storage unavailable"}
        return json.dumps(result, ensure_ascii=False)
    return call


def register(ctx):
    from .delivery import InboxDelivery

    def tool(name, operation, description, properties, required=(), emoji="📬"):
        schema = {"name": name, "description": description, "parameters": {
            "type": "object", "properties": properties, "required": list(required), "additionalProperties": False,
        }}
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=handler(operation),
                          check_fn=requirements, description="Local peer messaging", emoji=emoji)

    tool("peer_sessions", sessions,
         "Find permitted contacts: your immediate parent, direct children, and siblings with the same parent. "
         "Independent root chats also see other roots. scope=roots selects permitted roots; team selects your local family; "
         "all includes every permitted type. same_project=false expands root search, never access. "
         "peer_update announces new roots or your parent; has_more means consult this directory. Presence is not a running turn.", {
             "scope": {"type": "string", "enum": ["roots", "team", "all"], "default": "all"},
             "active_within_minutes": {"type": "integer", "minimum": 1, "maximum": 1440, "default": 60},
             "same_project": {"type": "boolean", "default": True},
         })
    tool("peer_send", send,
         "Queue a note to a known session_id/thread_id or parent: immediate parent, direct child, same-parent sibling, "
         "or another independent root if you are a root. Known IDs do not bypass these limits. "
         "Escalate shared issues to your parent so coordinators can inform the affected branches. Reply with in_reply_to. "
         "Use mail for coordination, not final summaries already returned by delegate_task. Queued is not read; "
         "recipient_state is advisory, not proof of an executing turn. Never wakes the recipient; do not wait or poll.", {
             "target_session_id": {"type": "string", "description": "Exact address or parent (immediate delegation parent)."},
             "message": {"type": "string", "minLength": 1, "maxLength": mailbox.MAX_MESSAGE_CHARS},
             "in_reply_to": {"type": "string"},
         }, ("target_session_id", "message"), "📨")
    tool("peer_receive", receive,
         "Immediate fallback inbox read; normally peer_data arrives with ordinary tool results and is confirmed automatically. "
         "mail_pending means mail awaits delivery. Do not poll. Use action=ack only for fully read duplicate IDs.", {
             "action": {"type": "string", "enum": ["read", "ack"], "default": "read"},
             "limit": {"type": "integer", "minimum": 1, "maximum": mailbox.MAX_READ_MESSAGES, "default": 5},
             "message_ids": {"type": "array", "minItems": 1, "maxItems": mailbox.MAX_PENDING_MESSAGES,
                             "items": {"type": "string"}},
         }, emoji="📥")
    delivery = InboxDelivery()
    notices = InboxNotices(delivery)
    ctx.register_hook("pre_llm_call", notices.before_turn)
    ctx.register_hook("transform_tool_result", notices.after_tool)
    ctx.register_hook("pre_api_request", delivery.before_request)
    ctx.register_hook("post_api_request", delivery.after_request)
    ctx.register_hook("api_request_error", delivery.request_error)
