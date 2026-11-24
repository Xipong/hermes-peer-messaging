"""Proactive local discovery and incoming-link awareness through native hooks."""
import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from importlib import import_module
from types import SimpleNamespace

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401 -- native temporary profile


def pin(home, sid, names=("peer_sessions", "peer_receive", "list_threads", "link_thread")):
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.update_tool_pin(sid, {"tools": [{"function": {"name": name}} for name in names]})


def start(sid, turn, *, platform="desktop", parent=""):
    from hermes_cli.lifecycle import invoke_hook

    values = invoke_hook("pre_llm_call", session_id=sid, turn_id=turn, platform=platform,
                         task_id=sid, parent_session_id=parent, is_first_turn=True)
    return "\n\n".join(value["context"] for value in values
                        if isinstance(value, dict) and value.get("context"))


def tool(sid, turn, name="list_threads", **args):
    from model_tools import handle_function_call

    return json.loads(handle_function_call(name, args, session_id=sid, turn_id=turn,
                                          tool_call_id=f"observe-{name}", enabled_toolsets=["peer_messaging"]))


def assert_no_model_work(home, *ids):
    with open_persistence_reader(home) as store:
        for sid in ids:
            row = store.sessions.get(sid)
            assert row.message_count == row.api_call_count == 0


def test_fresh_chat_discovers_peers_before_first_contact_without_writes(install):
    service = import_module("peer-messaging.service")
    pin(install, "recipient")
    pin(install, "elsewhere")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.set_title("sender", "peer-controlled-title")
    folder = install / "runtime" / "peer_messages"
    assert not folder.exists()

    context = start("recipient", "fresh-turn")
    assert context == service.DISCOVERY_HINT
    assert "peer-controlled-title" not in context
    # No same-project peer exists for this other project; configured foreign and
    # managed rows also cannot make the current project's discovery eligible.
    assert start("elsewhere", "other-project-turn", platform="tui") == ""
    assert tool("recipient", "fresh-turn")["threads"] == []
    assert "_peer_discovery_notice" not in tool("recipient", "fresh-turn")
    discovered = tool("recipient", "fresh-turn", "peer_sessions")
    assert [peer["session_id"] for peer in discovered["peers"]] == ["sender"]
    assert discovered["peers"][0]["title"] == "peer-controlled-title"
    assert "_peer_discovery_notice" not in discovered
    assert not folder.exists()
    assert_no_model_work(install, "sender", "recipient", "elsewhere", "managed", "foreign")


def test_incoming_offer_notifies_active_and_next_turn_but_only_parent_accepts(install):
    service = import_module("peer-messaging.service")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("child", "subagent", parent_session_id="recipient",
                                         model_config={"_delegate_from": "recipient", "_subagent_id": "task-child"}))
    pin(install, "recipient")
    pin(install, "child")
    assert service.LINK_HINT not in start("recipient", "running-turn")
    assert call("link_thread", "sender", thread_id="recipient")["status"] == "awaiting_other_chat"
    graph = install / "runtime" / "peer_messages" / ".threads.json"
    offered = graph.read_bytes()

    # An unrelated inbox inspection can carry the new offer without executing it.
    noticed = tool("recipient", "running-turn", "peer_receive")
    assert noticed["messages"] == []
    assert noticed["_peer_link_notice"] == service.LINK_HINT
    assert "_peer_link_notice" not in tool("recipient", "running-turn", "peer_receive")
    assert service.LINK_HINT in start("recipient", "next-turn")
    assert service.LINK_HINT not in start("child", "child-turn", platform="subagent", parent="recipient")
    assert [peer["session_id"] for peer in call("peer_sessions", "child")["peers"]] == ["recipient"]
    assert not call("link_thread", "child", thread_id="sender")["success"]
    assert graph.read_bytes() == offered
    assert not (graph.parent / "recipient").exists()
    assert_no_model_work(install, "sender", "recipient", "child")

    incoming = tool("recipient", "next-turn", "list_threads")
    assert "_peer_link_notice" not in incoming
    assert incoming["threads"][0]["link"] == "incoming"
    target = incoming["threads"][0]["thread_id"]
    assert call("link_thread", "recipient", thread_id=target)["status"] == "linked"
    assert {peer["session_id"] for peer in call("peer_sessions", "child")["peers"]} == {"recipient", "sender"}


def test_cross_hook_dedup_does_not_hide_mail_arriving_later_in_same_turn(install):
    service = import_module("peer-messaging.service")
    pin(install, "recipient")
    first = call("peer_send", "sender", target_session_id="recipient", message="private-first-note")
    context = start("recipient", "one-turn")
    assert service.HINT in context and service.DISCOVERY_HINT in context
    assert "private-first-note" not in context
    after_entry = tool("recipient", "one-turn")
    assert "_peer_inbox_notice" not in after_entry and "_peer_discovery_notice" not in after_entry

    second = call("peer_send", "sender", target_session_id="recipient", message="private-second-note")
    later = tool("recipient", "one-turn")
    assert later["_peer_inbox_notice"] == service.HINT
    assert "private-second-note" not in json.dumps(later)
    assert "_peer_inbox_notice" not in tool("recipient", "one-turn")
    pending = call("peer_receive", "recipient")
    assert [message["message_id"] for message in pending["messages"]] == [first["message_id"], second["message_id"]]
    assert service.HINT in start("recipient", "another-turn")
    assert_no_model_work(install, "sender", "recipient")


def test_discovery_waits_for_a_supported_carrier_and_rate_limits_scans(install, monkeypatch):
    service = import_module("peer-messaging.service")
    pin(install, "recipient", ("peer_sessions",))
    clock, scans = [100.0], []
    real_time, discover = service.time, service.discover
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=real_time.time))

    def counted(*args, **kwargs):
        scans.append(clock[0])
        return discover(*args, **kwargs)

    monkeypatch.setattr(service, "discover", counted)
    notices = service.InboxNotices()
    identity = {"tool_name": "some_tool", "session_id": "recipient", "turn_id": "one-turn"}
    assert notices.after_tool(result="[]", **identity) is None
    assert notices.after_tool(result='"scalar"', **identity) is None
    assert scans == []
    first = json.loads(notices.after_tool(result='{"result": 1}', **identity))
    assert first["result"] == 1 and first["_peer_discovery_notice"] == service.DISCOVERY_HINT
    assert scans == [100.0]

    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("new-peer", "cli", cwd="/project/new", git_repo_root="/project"))
    clock[0] = 104.0
    assert notices.after_tool(result='{"result": 2}', **identity) is None
    assert scans == [100.0]
    clock[0] = 106.0
    assert "_peer_discovery_notice" in json.loads(notices.after_tool(result='{"result": 3}', **identity))
    assert scans == [100.0, 106.0]
    clock[0] = 112.0
    assert notices.after_tool(result='{"result": 4}', **identity) is None
    assert scans == [100.0, 106.0, 112.0]
    assert not (install / "runtime" / "peer_messages").exists()


def test_unrelated_subagents_do_not_crowd_local_parent_discovery(install):
    service = import_module("peer-messaging.service")
    with open_persistence(profile_scope(home=install)) as store:
        for index in range(105):
            sid = f"unrelated-{index}"
            store.sessions.create(NewSession(sid, "subagent", parent_session_id="elsewhere",
                                             model_config={"_delegate_from": "elsewhere", "_subagent_id": sid}))
    pin(install, "recipient", ("peer_sessions",))
    assert start("recipient", "busy-profile-turn") == service.DISCOVERY_HINT
    discovered = call("peer_sessions", "recipient")
    assert [peer["session_id"] for peer in discovered["peers"]] == ["sender"]
    assert discovered["scan_may_be_truncated"]
    assert "subagent" in discovered["scan_limited_sources"]
    assert not (install / "runtime" / "peer_messages").exists()
    assert_no_model_work(install, "sender", "recipient", "elsewhere")


def test_long_running_owner_not_hidden_by_default_recency_cutoff(install):
    """A live session lease may outlive the last persisted activity stamp."""
    from time import time
    service = import_module("peer-messaging.service")
    teams = import_module("peer-messaging.teams")
    threads = import_module("peer-messaging.threads")
    with threads.graph(install) as links, open_persistence_reader(install) as store:
        recipient = teams.resolve(store, "recipient")
        future_cutoff = time() + 7200  # All stored last-active timestamps are older.
        absent, _ = service.discover(store, recipient, links, cutoff=future_cutoff, live_owners=set())
        leased, _ = service.discover(store, recipient, links, cutoff=future_cutoff,
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
                    copy_context().run, notices.before_turn,
                    session_id="recipient", turn_id="old-owner-turn", platform="desktop",
                )
                # The registry writer can remain busy while this normal hook
                # discovers a live owner outside the persisted recency window.
                assert future.result(timeout=2.0) == {"context": service.DISCOVERY_HINT}
        assert registry.read_bytes() == before and registry.stat().st_mtime_ns == modified
    finally:
        lease.release()
