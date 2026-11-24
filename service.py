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
from persistence_assembly import open_persistence_reader, profile_scope, store_exists

from . import lifecycle, mailbox, teams
from .delivery import InboxDelivery
from .discovery import CandidateCache, family_candidates, scan
from .project_identity import ProjectIndex

LOG = logging.getLogger(__name__)
TOOLSET = "peer_messaging"
HINT = "Peer mail pending. Use peer_receive only if needed before your next tool result."
DISCOVERY_INTERVAL = 5.0
DISCOVERY_CACHE = CandidateCache()
SCOPES = {"roots", "team", "all"}


def owners(home):
    # Presence is advisory and never authorizes delivery.
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
    index = index if index is not None else ProjectIndex(get_hermes_home())
    key = index.key(own.root) if same else None
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
        if same and not own_team and (key is None or index.key(participant.root) != key):
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
            raise ValueError("Recipient is outside your communication neighborhood; coordinate through your parent.")
        entry = mailbox.enqueue(mailbox.root(home) / target.thread_id, sender=sender.row.session_id,
                                target=target.row.session_id, text=args.get("message"),
                                in_reply_to=args.get("in_reply_to"))
        state = lifecycle.recipient_state(store, target, owners(home))
    return {"success": True, "status": "queued", "recipient_state": state, "message_id": entry["message_id"],
            "target_session_id": target.row.session_id, "requested_target_session_id": requested,
            "from_session_id": sender.row.session_id, "from_thread_id": sender.thread_id,
            "target_thread_id": target.thread_id}


def receive(args, sid):
    action, ids = args.get("action", "read"), args.get("message_ids")
    home = get_hermes_home()
    with open_persistence_reader(home, patience_s=0.1) as store:
        own = teams.resolve(store, sid, follow=False)
    inbox = mailbox.root(home) / own.thread_id
    if action == "ack" and "limit" not in args:
        return mailbox.acknowledge(inbox, ids)
    if action != "read" or ids is not None:
        raise ValueError("Use limit with read, or message_ids with ack.")
    return {"success": True, "authority": "peer_data",
            **mailbox.read_pending(inbox, limit=args.get("limit", mailbox.MAX_READ_MESSAGES))}


def pin_has_tool(pin, name):
    tools = pin.get("tools") if isinstance(pin, dict) else None
    if not isinstance(tools, list):
        return False
    for entry in tools:
        fn = entry.get("function") if isinstance(entry, dict) else None
        if not isinstance(fn, dict):
            continue
        if fn.get("name") == name:
            return True
        if fn.get("name") == "tool_search":
            description = fn.get("description")
            if not isinstance(description, str):
                continue
            catalog = description.partition("Deferred tool catalog (")[2]
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
        self.delivery = delivery if delivery is not None else InboxDelivery()

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
        if not session_id or platform not in teams.LOCAL | {"subagent"}:
            return None
        try:
            home = get_hermes_home()
            with open_persistence_reader(home, patience_s=0.05) as store:
                own = teams.resolve(store, session_id, follow=False)
                if not pin_has_tool(own.row.tool_pin, "peer_receive"):
                    return None
                pending = mailbox.pending_ids(mailbox.root(home) / own.thread_id)
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
            home = get_hermes_home()
            received = (payload if tool_name == "peer_receive" and isinstance(payload, dict)
                        and payload.get("success") is True and isinstance(payload.get("messages"), list) else None)
            listed = (payload if tool_name == "peer_sessions" and isinstance(payload, dict)
                      and payload.get("success") is True and isinstance(payload.get("peers"), list) else None)
            read_mail = self.delivery.can_attach(
                session_id=session_id, turn_id=turn_id, api_request_id=api_request_id, tool_call_id=tool_call_id)
            if (tool_call_id and not read_mail
                    and tool_name not in {"peer_receive", "peer_sessions"}):
                return None
            update_allowed = payload is None or "peer_update" not in payload
            with open_persistence_reader(home, patience_s=0.05) as store:
                own = teams.resolve(store, session_id, follow=False)
                update = {}
                if pin_has_tool(own.row.tool_pin, "peer_receive") or received is not None:
                    inbox = mailbox.root(home) / own.thread_id
                    page = received if received is not None else mailbox.read_pending(inbox) if read_mail else None
                    pending = (frozenset(item["message_id"] for item in page["messages"])
                               if page is not None and received is None else mailbox.pending_ids(inbox))
                    carried = ()
                    if page is not None and (read_mail or received is not None):
                        carrier = dict(payload) if received is not None else payload
                        if received is not None:
                            for field in ("messages", "has_more", "authority"):
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
            if isinstance(payload, dict):
                carrier = json.loads(attached) if attached is not None else payload
                carrier["peer_update"] = update
                return json.dumps(carrier, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            return (attached or result) + "\n\n[peer-messaging update]\n" + json.dumps(update, ensure_ascii=False, separators=(",", ":"))
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
    def tool(name, operation, description, properties, required=(), emoji="📬"):
        schema = {"name": name, "description": description, "parameters": {
            "type": "object", "properties": properties, "required": list(required), "additionalProperties": False,
        }}
        ctx.register_tool(name=name, toolset=TOOLSET, schema=schema, handler=handler(operation),
                          check_fn=requirements, description="Local peer messaging", emoji=emoji)

    tool("peer_sessions", sessions,
         "Find your immediate parent, direct children, same-parent siblings, or other roots if you are a root. "
         "Use for unknown addresses or peer_update.has_more.", {
             "scope": {"type": "string", "enum": ["roots", "team", "all"], "default": "all",
                       "description": "roots: independent chats; team: immediate family; all: both."},
             "active_within_minutes": {"type": "integer", "minimum": 1, "maximum": 1440, "default": 60,
                                       "description": "Recency in minutes; immediate parent and live owners are exempt."},
             "same_project": {"type": "boolean", "default": True,
                              "description": "Filter other roots by project only."},
         })
    tool("peer_send", send,
         "Send a decision, blocker, or handoff; combine routine progress into one update per recipient at a checkpoint. "
         "If a delay would block someone, send now. Otherwise, include the send in the same batch as your next independent work tool when possible. "
         "Raise cross-branch issues via your parent; delegate_task returns final summaries. "
         "Queued means stored, not read; recipients are not woken. recipient_state is advisory. Continue work without polling.", {
             "target_session_id": {"type": "string", "description": "Known session_id/thread_id or parent (immediate parent)."},
             "message": {"type": "string", "minLength": 1, "maxLength": mailbox.MAX_MESSAGE_CHARS},
             "in_reply_to": {"type": "string", "description": "ID of the message being answered."},
         }, ("target_session_id", "message"), "📨")
    tool("peer_receive", receive,
         "Read mail if needed now, or ACK fully read duplicates. Normally delivery and ACK are automatic.", {
             "action": {"type": "string", "enum": ["read", "ack"], "default": "read"},
             "limit": {"type": "integer", "minimum": 1, "maximum": mailbox.MAX_READ_MESSAGES, "default": 5,
                       "description": "Read only."},
             "message_ids": {"type": "array", "minItems": 1, "maxItems": mailbox.MAX_PENDING_MESSAGES,
                             "items": {"type": "string"}, "description": "ACK only: message IDs."},
         }, emoji="📥")
    delivery = InboxDelivery()
    notices = InboxNotices(delivery)
    ctx.register_hook("pre_llm_call", notices.before_turn)
    ctx.register_hook("transform_tool_result", notices.after_tool)
    ctx.register_hook("pre_api_request", delivery.before_request)
    ctx.register_hook("post_api_request", delivery.after_request)
    ctx.register_hook("api_request_error", delivery.request_error)
