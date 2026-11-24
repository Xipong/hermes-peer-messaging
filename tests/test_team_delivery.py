"""Durable coordination of independent conversations and their delegated teams."""
import json
import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import pytest

from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401 -- native profile and dispatcher


def ok(tool, sid, **args):
    reply = call(tool, sid, **args)
    assert reply.get("success"), reply
    return reply


def add_child(store, sid, parent, *, profile=None, cwd="/project/work", git_root: str | None = "/project"):
    store.sessions.create(NewSession(
        sid, "subagent", parent_session_id=parent, cwd=cwd, git_repo_root=git_root,
        model_config={"_delegate_from": parent, "_subagent_id": "task-" + sid},
        profile_name=profile,
    ))


@pytest.fixture
def teams(install):
    with open_persistence(profile_scope(home=install)) as store:
        # The database carries no batch or live Python-agent registry. These can
        # be children of separate delegate_task calls in different host processes.
        for sid, parent in (("a1", "sender"), ("a2", "sender"),
                            ("b1", "recipient"), ("b2", "recipient")):
            add_child(store, sid, parent)
    return install


def link_roots():
    assert ok("link_thread", "sender", thread_id="recipient")["status"] == "awaiting_other_chat"
    assert ok("link_thread", "recipient", thread_id="sender")["status"] == "linked"


def peers(sid, **kwargs):
    return {peer["session_id"]: peer for peer in ok("peer_sessions", sid, **kwargs)["peers"]}


def test_discovery_and_cross_team_messages_require_both_parent_links(teams):
    assert set(peers("sender")) == {"recipient", "a1", "a2"}
    assert set(peers("a1")) == {"sender", "a2"}
    assert not call("peer_send", "a1", target_session_id="b1", message="too early")["success"]
    assert not call("peer_send", "sender", target_session_id="b1", message="too early")["success"]
    assert not call("link_thread", "a1", thread_id="recipient")["success"]
    assert not call("link_thread", "sender", thread_id="b1")["success"]

    ok("link_thread", "sender", thread_id="recipient")
    assert "b1" not in peers("a1")
    assert not call("peer_send", "a1", target_session_id="b1", message="one offer only")["success"]
    ok("link_thread", "recipient", thread_id="sender")
    discovered = peers("a1")
    assert set(discovered) == {"sender", "a2", "recipient", "b1", "b2"}
    assert discovered["b1"]["root_thread_id"] == "recipient"
    assert discovered["b1"]["parent_session_id"] == "recipient"
    assert discovered["b1"]["subagent_id"] == "task-b1"
    assert discovered["b1"]["is_child"] is True

    first = ok("peer_send", "a1", target_session_id="b1", message="I own parser.py; avoid concurrent edits")
    assert first["from_session_id"] == "a1" and first["from_root_thread_id"] == "sender"
    assert first["target_session_id"] == "b1" and first["target_root_thread_id"] == "recipient"
    inbox = ok("peer_receive", "b1")
    assert inbox["authority"] == "peer_data"
    assert inbox["messages"][0]["message_id"] == first["message_id"]
    assert inbox["messages"][0]["from_session_id"] == "a1"
    assert ok("peer_receive", "recipient")["messages"] == []
    assert ok("peer_receive", "b2")["messages"] == []

    reported = ok("peer_send", "b1", target_session_id="recipient", message="Other team owns parser.py",
                  in_reply_to=first["message_id"])
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == reported["message_id"]
    response = ok("peer_send", "recipient", target_session_id="a1", message="Our team will handle the tests",
                  in_reply_to=reported["message_id"])
    answer = ok("peer_receive", "a1")["messages"][0]
    assert answer["message_id"] == response["message_id"]
    assert answer["from_session_id"] == "recipient" and answer["in_reply_to"] == reported["message_id"]
    assert ok("peer_receive", "a2")["messages"] == []
    with open_persistence_reader(teams) as store:
        assert all(store.sessions.get(sid).message_count == 0
                   for sid in ("sender", "recipient", "a1", "a2", "b1", "b2"))


def test_same_team_cross_batch_and_nested_messages_keep_separate_inboxes(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        add_child(store, "a-deep", "a1", cwd="/isolated/worktree", git_root="/isolated/worktree")
    discovered = peers("a-deep")
    assert {"sender", "a1", "a2"} <= set(discovered)
    assert "b1" not in discovered
    assert ok("peer_sessions", "a-deep")["root_thread_id"] == "sender"
    sibling_note = ok("peer_send", "a1", target_session_id="a2", message="Separate batch, same parent")
    assert ok("peer_receive", "a2")["messages"][0]["message_id"] == sibling_note["message_id"]
    parent_note = ok("peer_send", "sender", target_session_id="a-deep", message="Nested review requested earlier")
    assert ok("peer_receive", "a-deep")["messages"][0]["message_id"] == parent_note["message_id"]
    root_note = ok("peer_send", "a-deep", target_session_id="sender", message="Nested finding")
    assert ok("peer_receive", "sender")["messages"][0]["message_id"] == root_note["message_id"]
    assert ok("peer_receive", "a1")["messages"] == []
    assert ok("peer_receive", "a1", action="ack", message_ids=[parent_note["message_id"]])["acknowledged"] == []
    assert ok("peer_receive", "a-deep")["messages"][0]["message_id"] == parent_note["message_id"]
    link_roots()
    cross = ok("peer_send", "a-deep", target_session_id="b2", message="Nested cross-team finding")
    assert ok("peer_receive", "b2")["messages"][0]["message_id"] == cross["message_id"]


def test_parent_alias_targets_immediate_parent_through_compaction(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        add_child(store, "nested", "a1")
        for old, current, parent in (("nested", "nested-tip", "a1"), ("a1", "a1-tip", "sender")):
            store.sessions.end(old, "compression")
            store.sessions.create(NewSession(
                current, "subagent", parent_session_id=old,
                model_config={"_delegate_from": parent, "_subagent_id": "task-" + old},
            ))
        store.sessions.end("sender", "compression")
        store.sessions.create(NewSession("sender-tip", "cli", parent_session_id="sender"))

    sent = ok("peer_send", "nested-tip", target_session_id="parent", message="For my immediate parent")
    assert sent["requested_target_session_id"] == "parent"
    assert sent["target_session_id"] == "a1-tip"
    assert ok("peer_receive", "a1-tip")["messages"][0]["message_id"] == sent["message_id"]
    assert ok("peer_receive", "sender-tip")["messages"] == []
    reported = ok("peer_send", "a1-tip", target_session_id="parent", message="For the root")
    assert reported["target_session_id"] == "sender-tip"
    assert ok("peer_receive", "sender-tip")["messages"][0]["message_id"] == reported["message_id"]
    refused = call("peer_send", "sender-tip", target_session_id="parent", message="No parent exists")
    assert refused["success"] is False and "no delegation parent" in refused["error"]


def test_cross_project_parent_consent_enables_only_peer_mail(install):
    from test_awareness import pin, start
    from test_hermes_projects import project

    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "api-child", "sender")
        add_child(store, "other-child", "elsewhere", cwd="/other/work", git_root="/other")
    args = {"target_session_id": "other-child", "message": "Cross-project coordination"}
    assert not call("peer_send", "api-child", **args)["success"]
    pin(install, "elsewhere")
    assert ok("link_thread", "sender", thread_id="elsewhere")["status"] == "awaiting_other_chat"
    assert "list_threads" in start("elsewhere", "incoming", platform="tui")
    offer, = ok("list_threads", "elsewhere")["threads"]
    assert offer["thread_id"] == "sender" and offer["link"] == "incoming"
    assert "other-child" not in peers("api-child", same_project=False)
    assert not call("peer_send", "api-child", **args)["success"]

    accepted = ok("link_thread", "elsewhere", thread_id="sender")
    assert accepted["status"] == "linked" and accepted["transcript_shared"] is False
    assert "other-child" not in peers("api-child")
    assert "other-child" in peers("api-child", same_project=False)
    sent = ok("peer_send", "api-child", **args)
    assert ok("peer_receive", "other-child")["messages"][0]["message_id"] == sent["message_id"]
    reply = ok("peer_send", "other-child", target_session_id="api-child", message="Received",
               in_reply_to=sent["message_id"])
    assert ok("peer_receive", "api-child")["messages"][0]["message_id"] == reply["message_id"]
    assert not call("read_thread", "sender", thread_id="elsewhere")["success"]
    assert not call("wait_threads", "sender", threads=[{"thread_id": "elsewhere"}], wait_seconds=0)["success"]

    # Changing project membership must not turn a mail-only link into consent to share text.
    project(install, "Combined project", "/project", "/other")
    assert "other-child" in peers("api-child")
    assert not call("read_thread", "sender", thread_id="elsewhere")["success"]
    assert not call("wait_threads", "sender", threads=[{"thread_id": "elsewhere"}], wait_seconds=0)["success"]
    assert ok("list_threads", "sender")["threads"][0]["transcript_shared"] is False
    ok("link_thread", "sender", thread_id="elsewhere", action="unlink")
    assert not call("peer_send", "api-child", **args)["success"]


def test_pathless_team_discovers_other_batches_and_parent_without_project(install):
    from persistence.conversation.sessions import SessionId
    from test_awareness import pin, start, tool

    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession(SessionId("root"), "desktop"))
        add_child(store, "batch-one", "root")
        add_child(store, "batch-two", "root", cwd="/another-worktree", git_root=None)
    pin(install, "batch-two")
    assert "across batches" in start("batch-two", "child-turn", platform="subagent", parent="root")
    found = peers("batch-one")
    assert set(found) == {"root", "batch-two"}
    assert found["batch-two"]["root_thread_id"] == "root"
    sent = ok("peer_send", "batch-one", target_session_id=found["batch-two"]["session_id"], message="Across calls")
    notice = tool("batch-two", "child-turn")
    assert "_peer_inbox_notice" not in notice
    assert notice["peer_data"]["messages"][0]["message_id"] == sent["message_id"]
    assert notice["peer_data"]["messages"][0]["message"] == "Across calls"
    received = ok("peer_receive", "batch-two")
    assert received["authority"] == "peer_data"
    assert received["messages"][0]["message_id"] == sent["message_id"]
    reply = ok("peer_send", "root", target_session_id="batch-one", message="Parent can reply too")
    assert ok("peer_receive", "batch-one")["messages"][0]["message_id"] == reply["message_id"]


def test_unlink_revokes_child_access_without_destroying_already_queued_mail(teams):
    link_roots()
    accepted = ok("peer_send", "a1", target_session_id="b1", message="Accepted while linked")
    ok("link_thread", "recipient", thread_id="sender", action="unlink")
    assert set(peers("a1")) == {"sender", "a2"}
    assert not call("peer_send", "a1", target_session_id="b1", message="after unlink")["success"]
    assert not call("peer_send", "b1", target_session_id="sender", message="after unlink")["success"]
    assert ok("peer_receive", "b1")["messages"][0]["message_id"] == accepted["message_id"]
    assert ok("peer_receive", "b1", action="ack", message_ids=[accepted["message_id"]])["acknowledged"] == [accepted["message_id"]]
    assert ok("peer_send", "a1", target_session_id="a2", message="Own team still reachable")["status"] == "queued"


def test_cross_team_send_holds_consent_until_publication(teams, monkeypatch):
    link_roots()
    service = import_module("peer-messaging.service")
    mailbox = import_module("peer-messaging.mailbox")
    enqueue = mailbox.enqueue
    graph = teams / "runtime" / "peer_messages" / ".threads.json"
    before, modified = graph.read_bytes(), graph.stat().st_mtime_ns
    attempted = []

    def revoke_before_publication(*args, **kwargs):
        # A sender paused after admission must not publish behind a successful
        # unlink. Competing revocation times out retryably on the real file lock.
        attempted.append(call("link_thread", "recipient", thread_id="sender", action="unlink"))
        return enqueue(*args, **kwargs)

    monkeypatch.setattr(mailbox, "enqueue", revoke_before_publication)
    sent = service.send({"target_session_id": "b1", "message": "Admitted before revocation"}, "a1")
    assert len(attempted) == 1 and attempted[0]["success"] is False
    assert attempted[0]["retryable"] is True
    # Locking a graph snapshot for sending does not republish link state.
    assert graph.read_bytes() == before and graph.stat().st_mtime_ns == modified
    assert ok("link_thread", "recipient", thread_id="sender", action="unlink")["status"] == "unlinked"
    assert not call("peer_send", "a1", target_session_id="b1", message="After revocation")["success"]
    assert ok("peer_receive", "b1")["messages"][0]["message_id"] == sent["message_id"]


def test_gateway_foreign_or_unattributed_children_cannot_route_as_local_teams(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        add_child(store, "gateway-child", "managed")
        add_child(store, "foreign-child", "foreign", profile="foreign-profile")
        add_child(store, "foreign-parent-child", "foreign")
        store.sessions.create(NewSession("unattributed", "subagent", cwd="/project/work", git_repo_root="/project"))
    denied = ("managed", "foreign", "gateway-child", "foreign-child", "foreign-parent-child", "unattributed")
    assert not set(denied) & set(peers("sender", same_project=False))
    for sid in denied:
        assert not call("peer_send", "sender", target_session_id=sid, message="not local")["success"]
        assert not call("peer_send", sid, target_session_id="sender", message="not local")["success"]
        assert not call("peer_receive", sid)["success"]
    assert ok("peer_receive", "sender")["messages"] == []


def test_branched_root_and_its_children_do_not_inherit_team_links(teams):
    link_roots()
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.create(NewSession(
            "branched-root", "desktop", parent_session_id="sender", lineage_kind=SessionLineageKind.BRANCH,
            cwd="/project/api", git_repo_root="/project",
        ))
        add_child(store, "branched-child", "branched-root")
    assert ok("peer_sessions", "branched-child")["root_thread_id"] == "branched-root"
    assert set(peers("branched-child")) == {"branched-root"}
    assert ok("list_threads", "branched-root")["threads"] == []
    assert not call("peer_send", "branched-child", target_session_id="b1", message="no inherited link")["success"]
    assert not call("peer_send", "a1", target_session_id="branched-child", message="separate root")["success"]


def test_child_compaction_preserves_only_that_childs_mail_and_team(teams):
    link_roots()
    pending = ok("peer_send", "a1", target_session_id="b1", message="Keep through child rotation")
    parent_note = ok("peer_send", "a1", target_session_id="recipient", message="Parent inbox only")
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.end("b1", "compression")
        store.sessions.create(NewSession(
            "b1-tip", "subagent", parent_session_id="b1", cwd="/project/work", git_repo_root="/project",
            model_config={"_delegate_from": "recipient", "_subagent_id": "task-b1"},
        ))
    received = ok("peer_receive", "b1-tip")
    assert [note["message_id"] for note in received["messages"]] == [pending["message_id"]]
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == parent_note["message_id"]
    assert peers("a1")["b1-tip"]["root_thread_id"] == "recipient"
    followup = ok("peer_send", "a2", target_session_id="b1", message="Resolve old child address")
    assert followup["target_session_id"] == "b1-tip"
    assert ok("peer_receive", "b1-tip", action="ack", message_ids=[pending["message_id"]])["acknowledged"] == [pending["message_id"]]
    assert [note["message_id"] for note in ok("peer_receive", "b1-tip")["messages"]] == [followup["message_id"]]


def process_call(home, tool, sid, **args):
    import hermes_constants

    script = """
import json
import sys
from hermes_cli.plugins import discover_plugins
from model_tools import handle_function_call
from tools import delegate_tool_registry

assert not delegate_tool_registry._active_subagents
discover_plugins(force=True)
result = handle_function_call(sys.argv[1], json.loads(sys.argv[3]), session_id=sys.argv[2],
                              enabled_toolsets=["peer_messaging"])
assert not delegate_tool_registry._active_subagents
print("PEER_RESULT:" + result)
"""
    child = subprocess.run(
        [sys.executable, "-c", script, tool, sid, json.dumps(args)],
        cwd=Path(hermes_constants.__file__).resolve().parent,
        env={**os.environ, "HERMES_HOME": str(home), "HERMES_DISABLE_LAZY_INSTALLS": "1"},
        capture_output=True, text=True, timeout=30, check=True,
    )
    results = [line.removeprefix("PEER_RESULT:") for line in child.stdout.splitlines()
               if line.startswith("PEER_RESULT:")]
    assert len(results) == 1, child.stdout + child.stderr
    reply = json.loads(results[0])
    assert reply.get("success"), reply
    return reply


def test_linked_team_exchange_survives_fresh_os_processes(teams):
    link_roots()
    sent = process_call(teams, "peer_send", "a1", target_session_id="b1", message="Fresh-process finding")
    received = process_call(teams, "peer_receive", "b1")
    assert received["messages"][0]["message_id"] == sent["message_id"]
    assert received["messages"][0]["from_session_id"] == "a1"
    reply = process_call(teams, "peer_send", "b1", target_session_id="a1", message="Fresh-process reply",
                         in_reply_to=sent["message_id"])
    assert ok("peer_receive", "a1")["messages"][0]["message_id"] == reply["message_id"]
    assert ok("peer_receive", "a1")["messages"][0]["in_reply_to"] == sent["message_id"]
    assert process_call(teams, "peer_receive", "b1", action="ack", message_ids=[sent["message_id"]])["acknowledged"] == [sent["message_id"]]
    assert ok("peer_receive", "b1")["messages"] == []
    with open_persistence_reader(teams) as store:
        assert store.sessions.get("a1").message_count == store.sessions.get("b1").message_count == 0
