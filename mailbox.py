"""File-mailbox operations for local peer sessions, not a turn scheduler.

Reads are non-destructive. Only confirmed delivery or an explicit recipient
acknowledgement consumes an envelope, so a failed result can retry the same ID.
This is at-least-once reading, not exactly-once task execution.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hermes_cli.active_sessions import _FileLock
from utils import atomic_json_write

logger = logging.getLogger(__name__)
MAX_MESSAGE_CHARS = 8_000
MAX_PENDING_MESSAGES = 50
MAX_READ_MESSAGES = 5
_MESSAGE_ID = re.compile(r"[0-9]{20}_[0-9a-f]{8}")
_SESSION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
MAX_ENVELOPE_BYTES = 64_000


@contextmanager
def _locked(inbox: Path, *, timeout=2.0):
    # Automatic receipts try once; explicit mutations retain a bounded wait.
    try:
        with _FileLock(inbox / ".lock", timeout=timeout):
            yield
    except RuntimeError as exc:
        if isinstance(exc.__cause__, BlockingIOError):
            raise TimeoutError("Peer inbox is busy; retry the operation") from exc
        raise


def session_component(value: str) -> str:
    """Reject, rather than sanitize, IDs: no traversal or many-to-one aliases."""
    if not isinstance(value, str) or _SESSION_ID.fullmatch(value) is None:
        raise ValueError("Expected an exact local session ID")
    return value


def message_id(value: str) -> str:
    if not isinstance(value, str) or _MESSAGE_ID.fullmatch(value) is None:
        raise ValueError("Expected a message_id returned by peer_send/peer_receive")
    return value


def _paths(inbox: Path) -> list[Path]:
    if inbox.is_symlink():
        raise ValueError("Peer inbox must not be a symlink")
    return sorted(p for p in inbox.glob("*.json") if _MESSAGE_ID.fullmatch(p.stem))


def _load(path: Path) -> dict[str, Any] | None:
    """Invalid envelopes are distinguishable from temporary I/O failures."""
    if path.is_symlink():
        raise ValueError("Peer envelope must not be a symlink")
    with path.open("rb") as stream:
        raw = stream.read(MAX_ENVELOPE_BYTES + 1)
    if len(raw) > MAX_ENVELOPE_BYTES:
        return None
    try:
        entry = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError):
        return None
    if not isinstance(entry, dict):
        return None
    body = entry.get("message")
    if not isinstance(body, str) or not body.strip() or len(body) > MAX_MESSAGE_CHARS:
        return None
    if entry.get("message_id") != path.stem:
        return None
    try:
        for field in ("from_session_id", "target_session_id"):
            session_component(entry.get(field))
        if entry.get("in_reply_to") is not None:
            message_id(entry["in_reply_to"])
    except ValueError:
        return None
    sent_at = entry.get("sent_at")
    if isinstance(sent_at, bool) or not isinstance(sent_at, (int, float)) or not math.isfinite(sent_at):
        return None
    # Project only data fields; no envelope-provided role/authority/control flags.
    return {"message_id": path.stem, "from_session_id": entry.get("from_session_id"),
            "target_session_id": entry.get("target_session_id"), "message": body,
            "sent_at": entry.get("sent_at"), "in_reply_to": entry.get("in_reply_to")}


def enqueue(inbox: Path, *, sender: str, target: str, text: str,
            in_reply_to: str | None = None) -> dict[str, Any]:
    """Publish under the existing process-shared lock; concurrent sends share the cap."""
    session_component(sender)
    session_component(target)
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_MESSAGE_CHARS:
        raise ValueError(f"message must contain 1–{MAX_MESSAGE_CHARS} characters")
    # Lone surrogates trigger the host writer's whole-envelope ASCII fallback,
    # which can exceed the read cap despite the character limit. Reject before
    # publication instead of accepting a message the reader would discard.
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("message must contain valid UTF-8 text") from exc
    if in_reply_to is not None:
        message_id(in_reply_to)
    if inbox.is_symlink():
        raise ValueError("Peer inbox must not be a symlink")
    inbox.mkdir(mode=0o700, parents=True, exist_ok=True)
    with _locked(inbox):
        if len(_paths(inbox)) >= MAX_PENDING_MESSAGES:
            raise ValueError("Target inbox is full; its recipient must read and acknowledge messages")
        key = f"{time.time_ns():020d}_{uuid.uuid4().hex[:8]}"
        while (inbox / f"{key}.json").exists():
            key = f"{time.time_ns():020d}_{uuid.uuid4().hex[:8]}"
        entry = {"message_id": key, "from_session_id": sender, "target_session_id": target,
                 "message": text.strip(), "sent_at": time.time(), "in_reply_to": in_reply_to}
        atomic_json_write(inbox / f"{key}.json", entry, indent=None, mode=0o600)
        return entry


def pending_ids(inbox: Path) -> frozenset[str]:
    """Content-free hint snapshot for one logical participant."""
    return frozenset(path.stem for path in _paths(inbox))


def read_pending(inbox: Path, *, limit: int = MAX_READ_MESSAGES) -> dict[str, Any]:
    """Read an inbox snapshot without waiting for a sender or ACK's write lock.

    Envelopes are immutable and published by atomic rename, so readers see a
    complete file or no file. A concurrent ACK can remove a snapshot entry.
    Only the requested FIFO page is decoded. pending_count includes uninspected
    or unreadable candidate files until pending_count_is_exact becomes true;
    has_more therefore also includes candidates that still need validation.
    """
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_READ_MESSAGES:
        raise ValueError(f"limit must be between 1 and {MAX_READ_MESSAGES}")
    paths = _paths(inbox)
    entries, unreadable, inspected, pending = [], 0, 0, len(paths)
    for path in paths:
        if len(entries) == limit:
            break
        inspected += 1
        try:
            entry = _load(path)
            if entry is None:
                # Only demonstrated invalidity permits poison-file removal.
                path.unlink(missing_ok=True)
                pending -= 1
            else:
                entries.append(entry)
        except FileNotFoundError:
            # A concurrent explicit ACK is not a failed read to retry.
            pending -= 1
        except (OSError, ValueError):
            unreadable += 1
            logger.debug("peer inbox: cannot read %s", path.name, exc_info=True)
    return {"messages": entries, "has_more": pending > len(entries),
            "pending_count": pending, "pending_count_is_exact": inspected == len(paths) and not unreadable,
            "unreadable_count": unreadable}


def acknowledge(inbox: Path, ids: list[str], *, lock_timeout=2.0) -> dict[str, Any]:
    """Only the caller's stable inbox is supplied. ACKs cannot consume a neighbor."""
    if not isinstance(ids, list) or not ids or len(ids) > MAX_PENDING_MESSAGES:
        raise ValueError(f"message_ids must contain 1–{MAX_PENDING_MESSAGES} IDs")
    wanted = list(dict.fromkeys(message_id(key) for key in ids))  # validate before any mutation
    acknowledged, failed = set(), set()
    if inbox.is_symlink():
        raise ValueError("Peer inbox must not be a symlink")
    if inbox.is_dir():
        with _locked(inbox, timeout=lock_timeout):
            for key in wanted:
                path = inbox / f"{key}.json"
                try:
                    path.unlink()
                    acknowledged.add(key)
                except FileNotFoundError:
                    pass
                except OSError:
                    failed.add(key)
                    logger.debug("peer inbox: acknowledgement failed for %s", key, exc_info=True)
    return {"success": not failed, "acknowledged": sorted(acknowledged),
            "not_pending": sorted(set(wanted) - acknowledged - failed), "failed": sorted(failed)}
