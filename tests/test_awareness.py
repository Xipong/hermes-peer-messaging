"""Root-only automatic discovery, explicit scopes and bounded notifications."""
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from importlib import import_module
from types import SimpleNamespace

import pytest

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401 -- native temporary profile


def pin(home, sid, names=("peer_sessions", "peer_receive")):
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.update_tool_pin(sid, {"tools": [{"function": {"name": name}} for name in names]})


def start(sid, turn, *, platform="desktop", parent=""):
    from hermes_cli.lifecycle import invoke_hook

    values = invoke_hook("pre_llm_call", session_id=sid, turn_id=turn, platform=platform,
                         task_id=sid, parent_session_id=parent, is_first_turn=True)
    return "\n\n".join(value["context"] for value in values
                        if isinstance(value, dict) and value.get("context"))


def tool(sid, turn, name="peer_receive", **args):
    from model_tools import handle_function_call

    return json.loads(handle_function_call(name, args, session_id=sid, turn_id=turn,
                                          tool_call_id=f"observe-{name}", enabled_toolsets=["peer_messaging"]))


def assert_no_model_work(home, *ids):
    with open_persistence_reader(home) as store:
        for sid in ids:
            row = store.sessions.get(sid)
            assert row.message_count == row.api_call_count == 0


def work(notices, sid="recipient", turn="active-turn", result='{"ok":true}'):
    changed = notices.after_tool(tool_name="work", result=result, session_id=sid, turn_id=turn)
    return json.loads(changed)["peer_update"] if changed is not None else None


def test_fresh_chat_discovers_root_once_without_peer_text_or_writes(install):
    service = import_module("peer-messaging.service")
    pin(install, "recipient")
    pin(install, "elsewhere")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.set_title("sender", "peer-controlled-title")
    notices = service.InboxNotices()
    assert not (install / "runtime" / "peer_messages_v2").exists()
    assert start("recipient", "fresh-turn") == ""
    update = work(notices)
    assert update == {"authority": "peer_data", "neighbors": ["sender"]}
    assert work(notices) is None
    assert work(notices, turn="next-turn") is None
    assert work(notices, sid="elsewhere") is None
    listed = call("peer_sessions", "recipient", scope="roots")
    assert [peer["session_id"] for peer in listed["peers"]] == ["sender"]
    assert listed["peers"][0]["title"] == "peer-controlled-title"
    assert not (install / "runtime" / "peer_messages_v2").exists()
    assert_no_model_work(install, "sender", "recipient", "elsewhere", "managed", "foreign")


def test_twenty_foreign_workers_across_eight_calls_never_update_root_or_child(install, monkeypatch):
    from test_team_delivery import add_child, peers
    service = import_module("peer-messaging.service")
    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "coordinator-a", "recipient")
        add_child(store, "coordinator-b", "sender")
    for sid in ("recipient", "coordinator-a"):
        pin(install, sid)
    clock = [100.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=service.time.time))
    notices = service.InboxNotices()
    assert work(notices)["neighbors"] == ["sender"]
    assert work(notices, "coordinator-a") == {"authority": "peer_data", "parent": "recipient"}
    index = 0
    for batch in range(8):
        with open_persistence(profile_scope(home=install)) as store:
            for _ in range(3 if batch < 4 else 2):
                add_child(store, f"foreign-{index}", "sender")
                index += 1
            if batch:
                store.sessions.end(f"foreign-{batch - 1}", "agent_close")
        clock[0] += service.DISCOVERY_INTERVAL + 1
        assert work(notices) is None
        assert work(notices, "coordinator-a") is None
    assert index == 20
    assert work(notices, turn="another-turn") is None
    for scope in ("all", "team", "roots"):
        assert set(peers("coordinator-a", scope=scope, same_project=False)) == {"recipient"}
    assert not call("peer_send", "coordinator-b", target_session_id="coordinator-a",
                    message="Cannot bypass the coordinators")["success"]
    finding = call("peer_send", "coordinator-b", target_session_id="parent", message="Parser change needs coordination")
    assert call("peer_receive", "sender")["messages"][0]["message_id"] == finding["message_id"]
    forwarded = call("peer_send", "sender", target_session_id="recipient", message="Coordinate parser changes",
                     in_reply_to=finding["message_id"])
    assert call("peer_receive", "recipient")["messages"][0]["message_id"] == forwarded["message_id"]
    assert call("peer_receive", "recipient", action="ack", message_ids=[forwarded["message_id"]])["acknowledged"] == [forwarded["message_id"]]
    sent = call("peer_send", "recipient", target_session_id="coordinator-a", message="Coordinate parser changes",
                in_reply_to=forwarded["message_id"])
    assert sent["status"] == "queued"
    delivery = import_module("peer-messaging.delivery").InboxDelivery()
    receiver = service.InboxNotices(delivery)
    carried = json.loads(receiver.after_tool(tool_name="work", result='{"ok":true}', session_id="coordinator-a",
                                              turn_id="mail-turn", tool_call_id="receive-coordination"))
    assert carried["peer_data"]["messages"][0]["message_id"] == sent["message_id"]
    assert "neighbors" not in carried.get("peer_update", {})
    reported = call("peer_send", "coordinator-a", target_session_id="parent", message="Parser changes are ready")
    assert reported["target_session_id"] == "recipient"
    assert call("peer_receive", "recipient")["messages"][0]["message_id"] == reported["message_id"]


def test_discovery_rate_limit_and_supported_carrier(install, monkeypatch):
    service = import_module("peer-messaging.service")
    pin(install, "recipient", ("peer_sessions",))
    clock, scans = [100.0], []
    real_time, discover = service.time, service.discover
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=real_time.time))

    def counted(*args, **kwargs):
        scans.append((clock[0], kwargs["scope"]))
        return discover(*args, **kwargs)

    monkeypatch.setattr(service, "discover", counted)
    notices = service.InboxNotices()
    identity = {"tool_name": "work", "session_id": "recipient", "turn_id": "one-turn"}
    assert notices.after_tool(result="[]", **identity) is None
    assert notices.after_tool(result='"scalar"', **identity) is None
    assert scans == []
    assert work(notices)["neighbors"] == ["sender"]
    assert scans == [(100.0, "roots")]
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("new-peer", "cli", cwd="/project/new", git_repo_root="/project"))
    clock[0] = 104.0
    assert work(notices) is None
    clock[0] = 106.0
    assert work(notices)["neighbors"] == ["new-peer"]
    assert work(notices, turn="next-turn") is None
    clock[0] = 112.0
    assert work(notices) is None
    assert scans == [(100.0, "roots"), (106.0, "roots"), (112.0, "roots")]


def test_compaction_of_either_root_does_not_repeat_neighbor_notice(install, monkeypatch):
    service = import_module("peer-messaging.service")
    pin(install, "recipient", ("peer_sessions",))
    clock = [100.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=service.time.time))
    notices = service.InboxNotices()
    assert work(notices)["neighbors"] == ["sender"]
    with open_persistence(profile_scope(home=install)) as store:
        for old, tip in (("sender", "sender-tip"), ("recipient", "recipient-tip")):
            store.sessions.end(old, "compression")
            store.sessions.create(NewSession(tip, "desktop", parent_session_id=old, cwd="/project/ui", git_repo_root="/project"))
    pin(install, "recipient-tip", ("peer_sessions",))
    clock[0] += service.DISCOVERY_INTERVAL + 1
    assert work(notices, "recipient-tip", "new-turn") is None
    assert call("peer_sessions", "recipient-tip", scope="roots")["peers"][0]["thread_id"] == "sender"
    assert call("peer_send", "recipient-tip", target_session_id="sender", message="Stable address")["target_session_id"] == "sender-tip"


def test_root_only_scan_is_shared_and_child_awareness_does_not_scan(install, monkeypatch):
    service = import_module("peer-messaging.service")
    from test_team_delivery import add_child

    with open_persistence(profile_scope(home=install)) as store:
        for index in range(105):
            add_child(store, f"worker-{index}", "sender")
    for sid in ("sender", "recipient", "worker-0", "worker-1"):
        pin(install, sid, ("peer_sessions",))
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: 100.0, time=service.time.time))
    with open_persistence_reader(install) as store:
        repository = type(store.sessions)
    real_list, real_owners = repository.list_sessions, service.owners
    counts = {"scans": 0, "owners": 0}

    def counted_list(*args, **kwargs):
        counts["scans"] += 1
        return real_list(*args, **kwargs)

    def counted_owners(*args, **kwargs):
        counts["owners"] += 1
        return real_owners(*args, **kwargs)

    monkeypatch.setattr(repository, "list_sessions", counted_list)
    monkeypatch.setattr(service, "owners", counted_owners)
    notices = service.InboxNotices()
    assert work(notices, "sender")["neighbors"] == ["recipient"]
    assert work(notices, "recipient")["neighbors"] == ["sender"]
    assert work(notices, "worker-0")["parent"] == "sender"
    assert work(notices, "worker-1")["parent"] == "sender"
    assert counts == {"scans": 1, "owners": 1}
    roots = service.sessions({"scope": "roots"}, "recipient")
    assert [peer["session_id"] for peer in roots["peers"]] == ["sender"]
    assert "subagent" not in roots["scan_limited_sources"]
    full = service.sessions({"scope": "all"}, "recipient")
    assert "subagent" not in full["scan_limited_sources"]
    assert counts == {"scans": 3, "owners": 3}
    sibling_directory = service.sessions({"scope": "team"}, "worker-0")
    assert {p["session_id"] for p in sibling_directory["peers"]} == {
        "sender", *(f"worker-{index}" for index in range(1, 105)),
    }
    assert counts["scans"] == 3
    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "old-sibling", "recipient")
        add_child(store, "old-worker", "recipient")
        for index in range(110):
            add_child(store, f"unrelated-new-{index}", "sender")
    listed = service.sessions({"scope": "team"}, "old-worker")
    assert {p["session_id"] for p in listed["peers"]} == {"recipient", "old-sibling"}
    assert counts["scans"] == 3


def test_long_running_owner_not_hidden_by_default_recency_cutoff(install):
    """A live session lease may outlive the last persisted activity stamp."""
    from time import time
    service = import_module("peer-messaging.service")
    teams = import_module("peer-messaging.teams")
    with open_persistence_reader(install) as store:
        recipient = teams.resolve(store, "recipient")
        future_cutoff = time() + 7200  # All stored last-active timestamps are older.
        absent, _ = service.discover(store, recipient, cutoff=future_cutoff, live_owners=set())
        leased, _ = service.discover(store, recipient, cutoff=future_cutoff,
                                    live_owners={"sender"})
    assert not absent
    assert [peer["session_id"] for peer in leased] == ["sender"]
    # A registered owner is NOT evidence of an in-flight LLM request.
    assert leased[0]["last_active"] < future_cutoff


def test_awareness_observes_old_live_owners_without_waiting_for_registry_writer(install, monkeypatch):
    from hermes_cli.active_sessions import _FileLock, _lock_path, _state_path, try_acquire_active_session

    service = import_module("peer-messaging.service")
    pin(install, "recipient", ("peer_sessions",))
    lease, error = try_acquire_active_session(
        session_id="sender", surface="cli", config={}, registry_home=install, track_liveness=True,
    )
    assert error is None and lease is not None
    registry = _state_path(install)
    before, modified = registry.read_bytes(), registry.stat().st_mtime_ns
    clock = service.time
    monkeypatch.setattr(service, "time", SimpleNamespace(
        time=lambda: clock.time() + 7200, monotonic=clock.monotonic,
    ))
    notices = service.InboxNotices()
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with _FileLock(_lock_path(install), timeout=1.0):
                future = executor.submit(
                    copy_context().run, work, notices, "recipient", "old-owner-turn",
                )
                # The registry writer can remain busy while this normal hook
                # discovers a live owner outside the persisted recency window.
                assert future.result(timeout=2.0) == {"authority": "peer_data", "neighbors": ["sender"]}
        assert registry.read_bytes() == before and registry.stat().st_mtime_ns == modified
    finally:
        lease.release()


@pytest.mark.parametrize("compressed", [False, True])
def test_nested_child_gets_only_immediate_parent_without_ancestor_or_contact_leaks(install, compressed):
    service = import_module("peer-messaging.service")
    from test_team_delivery import add_child
    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "parent-child", "sender")
        parent = "parent-child"
        if compressed:
            store.sessions.end(parent, "compression")
            store.sessions.create(NewSession("parent-middle", "subagent", parent_session_id=parent,
                                             model_config={"_delegate_from": "sender", "_subagent_id": "task-parent-child"}))
            parent = "parent-middle"
        add_child(store, "reader", parent)
        if compressed:
            store.sessions.end(parent, "compression")
            store.sessions.create(NewSession("parent-tip", "subagent", parent_session_id=parent,
                                             model_config={"_delegate_from": "sender", "_subagent_id": "task-parent-child"}))
        for index in range(12):
            add_child(store, f"recent-{index}", "sender")
        store.sessions.set_title("sender", "peer-controlled-request")
    pin(install, "reader")
    notices = service.InboxNotices()
    assert work(notices, "reader") == {"authority": "peer_data", "parent": "parent-child"}
    assert work(notices, "reader", "next-turn") is None
    listed = call("peer_sessions", "reader", scope="all", same_project=False)
    assert listed["success"] and "links" not in listed
    assert listed["parent_thread_id"] == "parent-child"
    parent_peer, = listed["peers"]
    assert parent_peer["session_id"] == ("parent-tip" if compressed else "parent-child")
    assert parent_peer["relation"] == "parent"
    assert parent_peer["parent_session_id"] is None and parent_peer["parent_thread_id"] is None
    # Visible parent metadata must not become a back door to its own parent,
    # its siblings, or the independent root that offered a contact link.
    encoded = json.dumps(listed)
    for hidden in ("sender", "recipient", "peer-controlled-request", *(f"recent-{index}" for index in range(12))):
        assert json.dumps(hidden) not in encoded
    assert "root_thread_id" not in parent_peer and "root_thread_id" not in listed
    sent = call("peer_send", "reader", target_session_id="parent-child", message="Addressed without directory lookup")
    assert sent["target_session_id"] == ("parent-tip" if compressed else "parent-child")
    assert sent["from_thread_id"] == "reader" and sent["target_thread_id"] == "parent-child"
    assert json.dumps("sender") not in json.dumps(sent)
