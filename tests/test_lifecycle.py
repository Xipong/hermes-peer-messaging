"""Recipient-state provenance and non-destructive closed-inbox maintenance."""
from importlib import import_module
from types import SimpleNamespace
import time

from persistence.authority.subagents import SubagentState, SubagentWrite
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401
from test_team_delivery import add_child


def stamp(store, sid, state, *, document_sid=None):
    key = "task-" + sid
    old = time.time() - 2 * 86400
    expected = store.coordination.get_subagent(subagent_id=key)
    assert store.coordination.put_subagent(
        subagent_id=key, expected=expected, now=old,
        write=SubagentWrite(state=state, pinned=False, stop_cause=None,
                            owner_pid=None, owner_started_at=None,
                            document={"session_id": document_sid or sid}, expires_at=None))


def test_recipient_state_is_advisory_and_rejects_reused_short_id(install):
    lifecycle = import_module("peer-messaging.lifecycle")
    teams = import_module("peer-messaging.teams")
    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "child", "sender")
        stamp(store, "child", SubagentState.RUNNING, document_sid="elsewhere")
    sent = call("peer_send", "sender", target_session_id="recipient", message="note")
    assert sent["status"] == "queued"
    assert set(sent["recipient_state"]) == {"owner_present", "subagent_state", "last_activity_at"}
    with open_persistence_reader(install) as store:
        own = teams.resolve(store, "recipient")
        assert lifecycle.recipient_state(store, own, {"recipient"})["owner_present"] is True
        assert lifecycle.recipient_state(store, own, None)["owner_present"] is None
        child = teams.resolve(store, "child")
        assert lifecycle.recipient_state(store, child, set())["subagent_state"] is None
    with open_persistence(profile_scope(home=install)) as store:
        stamp(store, "child", SubagentState.RUNNING)
    with open_persistence_reader(install) as store:
        assert lifecycle.recipient_state(store, teams.resolve(store, "child"), set())["subagent_state"] == "running"


def test_closed_mail_is_archived_not_lost_and_reopened_mail_is_fifo(install, monkeypatch):
    lifecycle = import_module("peer-messaging.lifecycle")
    box = import_module("peer-messaging.mailbox")
    folder = install / "runtime" / "peer_messages_v2"
    with open_persistence(profile_scope(home=install)) as store:
        for sid in ("finished", "resumable", "unknown", "still-live", "compressed"):
            add_child(store, sid, "sender")
        stamp(store, "finished", SubagentState.COMPLETED)
        stamp(store, "resumable", SubagentState.RESUMABLE)
        stamp(store, "still-live", SubagentState.COMPLETED)
    queued = {sid: box.enqueue(folder / sid, sender="sender", target=sid, text=sid)
              for sid in ("finished", "resumable", "unknown", "still-live", "compressed", "recipient")}
    with open_persistence(profile_scope(home=install)) as store:
        for sid in ("finished", "resumable", "unknown", "recipient"):
            store.sessions.end(sid, "agent_close")
        store.sessions.end("compressed", "compression")
        store.sessions.create(NewSession("compressed-tip", "subagent", parent_session_id="compressed",
                                         model_config={"_delegate_from": "sender", "_subagent_id": "task-compressed"}))
    future = time.time() + lifecycle.ARCHIVE_AGE_SECONDS + 1
    monkeypatch.setattr(lifecycle, "time", SimpleNamespace(time=lambda: future, monotonic=lambda: 100.0))
    lifecycle.sweep(install, None)
    assert not (folder / ".archive").exists()
    lifecycle.sweep(install, set())
    for sid, entry in queued.items():
        name = entry["message_id"] + ".json"
        archived = sid in {"finished", "recipient"}
        assert (folder / ".archive" / sid / name).is_file() == archived
        assert (folder / sid / name).is_file() != archived
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.reopen("finished")
    second = call("peer_send", "sender", target_session_id="finished", message="new mail")
    assert second["status"] == "queued"
    received = call("peer_receive", "finished")
    assert [m["message_id"] for m in received["messages"]] == [
        queued["finished"]["message_id"], second["message_id"]]
    assert call("peer_receive", "finished", action="ack",
                message_ids=[queued["finished"]["message_id"]])["acknowledged"] == [
        queued["finished"]["message_id"]]
    assert not (folder / ".archive" / "finished" / (queued["finished"]["message_id"] + ".json")).exists()


def test_archive_checks_owner_grace_and_bounded_rotation(install, monkeypatch):
    lifecycle = import_module("peer-messaging.lifecycle")
    box = import_module("peer-messaging.mailbox")
    folder = install / "runtime" / "peer_messages_v2"
    ids = [f"closed-{i:02d}" for i in range(lifecycle.SWEEP_BATCH + 3)]
    with open_persistence(profile_scope(home=install)) as store:
        for sid in ids:
            store.sessions.create(NewSession(sid, "cli"))
            box.enqueue(folder / sid, sender="sender", target=sid, text="pending")
            store.sessions.end(sid, "agent_close")
    clock = [time.time(), 100.0]
    monkeypatch.setattr(lifecycle, "time", SimpleNamespace(time=lambda: clock[0], monotonic=lambda: clock[1]))
    lifecycle.sweep(install, set())
    assert not (folder / ".archive").exists()
    clock[0] += lifecycle.ARCHIVE_AGE_SECONDS + 1
    clock[1] += lifecycle.SWEEP_INTERVAL_SECONDS
    lifecycle.sweep(install, {ids[0]})
    first = sum(len(box._active_paths(folder / sid)) == 0 for sid in ids)
    assert 0 < first <= lifecycle.SWEEP_BATCH
    lifecycle.sweep(install, {ids[0]})
    assert sum(len(box._active_paths(folder / sid)) == 0 for sid in ids) == first
    clock[1] += lifecycle.SWEEP_INTERVAL_SECONDS
    lifecycle.sweep(install, {ids[0]})
    assert len(box._active_paths(folder / ids[0])) == 1
    assert sum(len(box._active_paths(folder / sid)) == 0 for sid in ids) == len(ids) - 1
