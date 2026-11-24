"""Durable coordination of independent conversations and their delegated teams."""
import json
import os
import subprocess
import sys
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


def peers(sid, **kwargs):
    return {peer["session_id"]: peer for peer in ok("peer_sessions", sid, **kwargs)["peers"]}


def test_roots_keep_workers_private_and_coordination_goes_through_parents(teams):
    assert set(peers("sender")) == {"recipient", "a1", "a2"}
    assert set(peers("a1")) == {"sender", "a2"}
    assert not call("peer_send", "a1", target_session_ids=["b1"], message="too early")["success"]
    assert not call("peer_send", "sender", target_session_ids=["b1"], message="too early")["success"]
    discovered = peers("a1")
    assert set(discovered) == {"sender", "a2"}
    assert discovered["a2"]["parent_thread_id"] == "sender"
    assert discovered["a2"]["subagent_id"] == "task-a2"
    assert discovered["a2"]["is_child"] is True
    assert set(peers("sender")) == {"recipient", "a1", "a2"}
    for sender, target in (("a1", "b1"), ("a1", "recipient"), ("sender", "b1")):
        assert not call("peer_send", sender, target_session_ids=[target], message="Still outside my family")["success"]

    finding = ok("peer_send", "a1", target_session_ids=["parent"], message="I own parser.py; avoid concurrent edits")["results"][0]
    assert ok("peer_receive", "sender")["messages"][0]["message_id"] == finding["message_id"]
    coordinated = ok("peer_send", "sender", target_session_ids=["recipient"], message="Our worker owns parser.py",
                     in_reply_to=finding["message_id"])["results"][0]
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == coordinated["message_id"]
    first = ok("peer_send", "recipient", target_session_ids=["b1"], message="Avoid parser.py; handle tests",
               in_reply_to=coordinated["message_id"])["results"][0]
    assert first["target_session_id"] == "b1"
    inbox = ok("peer_receive", "b1")
    assert inbox["authority"] == "peer_data"
    assert inbox["messages"][0]["message_id"] == first["message_id"]
    assert inbox["messages"][0]["from_session_id"] == "recipient"
    assert inbox["messages"][0]["in_reply_to"] == coordinated["message_id"]
    assert ok("peer_receive", "b2")["messages"] == []

    reported = ok("peer_send", "b1", target_session_ids=["parent"], message="Tests are ready",
                  in_reply_to=first["message_id"])["results"][0]
    assert [note["message_id"] for note in ok("peer_receive", "recipient")["messages"]] == [
        coordinated["message_id"], reported["message_id"],
    ]
    response = ok("peer_send", "recipient", target_session_ids=["sender"], message="Our team's tests are ready",
                  in_reply_to=reported["message_id"])["results"][0]
    forwarded = ok("peer_send", "sender", target_session_ids=["a1"], message="Other team's tests are ready",
                   in_reply_to=response["message_id"])["results"][0]
    answer = ok("peer_receive", "a1")["messages"][0]
    assert answer["message_id"] == forwarded["message_id"]
    assert answer["from_session_id"] == "sender" and answer["in_reply_to"] == response["message_id"]
    assert ok("peer_receive", "a2")["messages"] == []
    with open_persistence_reader(teams) as store:
        assert all(store.sessions.get(sid).message_count == 0
                   for sid in ("sender", "recipient", "a1", "a2", "b1", "b2"))


def test_nested_family_boundaries_and_separate_inboxes(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        add_child(store, "a-deep", "a1", cwd="/isolated/worktree", git_root="/isolated/worktree")
        add_child(store, "a-deep-sibling", "a1")
        add_child(store, "a-cousin", "a2")
    discovered = peers("a-deep")
    assert set(discovered) == {"a1", "a-deep-sibling"}
    for scope in ("all", "team", "roots"):
        visible = set(peers("a-deep", scope=scope, same_project=False))
        assert visible == (set() if scope == "roots" else {"a1", "a-deep-sibling"})
    for inaccessible in ("sender", "a2", "a-cousin", "recipient", "b1", "b2"):
        assert not call("peer_send", "a-deep", target_session_ids=[inaccessible], message="Outside immediate family")["success"]
        assert not call("peer_send", inaccessible, target_session_ids=["a-deep"], message="Outside immediate family")["success"]
    sibling_note = ok("peer_send", "a1", target_session_ids=["a2"], message="Separate batch, same parent")["results"][0]
    assert ok("peer_receive", "a2")["messages"][0]["message_id"] == sibling_note["message_id"]
    nested_note = ok("peer_send", "a-deep", target_session_ids=["a-deep-sibling"], message="Same immediate parent")["results"][0]
    assert ok("peer_receive", "a-deep-sibling")["messages"][0]["message_id"] == nested_note["message_id"]
    parent_note = ok("peer_send", "a1", target_session_ids=["a-deep"], message="Nested review requested earlier")["results"][0]
    assert ok("peer_receive", "a-deep")["messages"][0]["message_id"] == parent_note["message_id"]
    report = ok("peer_send", "a-deep", target_session_ids=["parent"], message="Nested finding")["results"][0]
    assert ok("peer_receive", "a1")["messages"][0]["message_id"] == report["message_id"]
    assert ok("peer_receive", "sender")["messages"] == []
    assert ok("peer_receive", "a1", action="ack", message_ids=[parent_note["message_id"]])["acknowledged"] == []
    assert ok("peer_receive", "a-deep")["messages"][0]["message_id"] == parent_note["message_id"]
    assert set(peers("a-deep", same_project=False)) == {"a1", "a-deep-sibling"}
    assert not call("peer_send", "a-deep", target_session_ids=["b2"], message="Links do not widen family")["success"]


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
        add_child(store, "nested-later", "a1-tip")

    sent = ok("peer_send", "nested-tip", target_session_ids=["parent"], message="For my immediate parent")["results"][0]
    assert sent["requested_target_session_id"] == "parent"
    assert sent["target_session_id"] == "a1-tip"
    assert ok("peer_receive", "a1-tip")["messages"][0]["message_id"] == sent["message_id"]
    assert ok("peer_receive", "sender-tip")["messages"] == []
    assert set(peers("nested-tip")) == {"a1-tip", "nested-later"}
    sibling = ok("peer_send", "nested-tip", target_session_ids=["nested-later"], message="Sibling born after parent compaction")["results"][0]
    assert ok("peer_receive", "nested-later")["messages"][0]["message_id"] == sibling["message_id"]
    for alias in ("sender", "sender-tip"):
        assert not call("peer_send", "nested-tip", target_session_ids=[alias], message="Cannot skip my parent")["success"]
    reported = ok("peer_send", "a1-tip", target_session_ids=["parent"], message="For the root")["results"][0]
    assert reported["target_session_id"] == "sender-tip"
    assert ok("peer_receive", "sender-tip")["messages"][0]["message_id"] == reported["message_id"]
    refused = call("peer_send", "sender-tip", target_session_ids=["parent"], message="No parent exists")
    assert refused["success"] is False and "no delegation parent" in refused["results"][0]["error"]


def test_cross_project_roots_never_grant_worker_access(install):
    from test_hermes_projects import project

    with open_persistence(profile_scope(home=install)) as store:
        add_child(store, "api-child", "sender")
        add_child(store, "other-child", "elsewhere", cwd="/other/work", git_root="/other")
    args = {"target_session_ids": ["other-child"], "message": "Cross-project coordination"}
    assert not call("peer_send", "api-child", **args)["success"]
    for same_project in (True, False):
        assert set(peers("api-child", same_project=same_project)) == {"sender"}
    assert not call("peer_send", "api-child", **args)["success"]
    sent = ok("peer_send", "sender", target_session_ids=["elsewhere"], message="Cross-project root coordination")["results"][0]
    assert ok("peer_receive", "elsewhere")["messages"][0]["message_id"] == sent["message_id"]
    reply = ok("peer_send", "elsewhere", target_session_ids=["sender"], message="Received",
               in_reply_to=sent["message_id"])["results"][0]
    assert ok("peer_receive", "sender")["messages"][0]["message_id"] == reply["message_id"]
    # Project membership can group coordinators but cannot widen a worker's family.
    project(install, "Combined project", "/project", "/other")
    assert set(peers("api-child")) == {"sender"}
    assert "elsewhere" in peers("sender")
    assert not call("peer_send", "api-child", **args)["success"]
    assert not call("peer_send", "api-child", **args)["success"]
    assert ok("peer_send", "sender", target_session_ids=["elsewhere"], message="Roots remain reachable")["status"] == "queued"


def test_pathless_team_discovers_other_batches_and_parent_without_project(install):
    from persistence.conversation.sessions import SessionId
    from test_awareness import pin, tool

    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession(SessionId("root"), "desktop"))
        add_child(store, "batch-one", "root")
        add_child(store, "batch-two", "root", cwd="/another-worktree", git_root=None)
    pin(install, "batch-two")
    found = peers("batch-one")
    assert set(found) == {"root", "batch-two"}
    assert found["batch-two"]["parent_thread_id"] == "root"
    sent = ok("peer_send", "batch-one", target_session_ids=[found["batch-two"]["session_id"]], message="Across calls")["results"][0]
    notice = tool("batch-two", "child-turn")
    assert notice["peer_update"]["parent"] == "root" and not notice["peer_update"].get("mail_pending")
    assert notice["peer_data"]["messages"][0]["message_id"] == sent["message_id"]
    assert notice["peer_data"]["messages"][0]["message"] == "Across calls"
    received = ok("peer_receive", "batch-two")
    assert received["authority"] == "peer_data"
    assert received["messages"][0]["message_id"] == sent["message_id"]
    reply = ok("peer_send", "root", target_session_ids=["batch-one"], message="Parent can reply too")["results"][0]
    assert ok("peer_receive", "batch-one")["messages"][0]["message_id"] == reply["message_id"]


def test_recipient_can_read_published_mail_after_sender_finishes(teams):
    sent = ok("peer_send", "a1", target_session_ids=["a2"], message="Finding before completion")["results"][0]
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.end("a1", "agent_close")
    assert "a1" not in peers("a2")
    assert not call("peer_send", "a2", target_session_ids=["a1"], message="Closed recipient")["success"]
    assert ok("peer_receive", "a2")["messages"][0]["message_id"] == sent["message_id"]
    assert ok("peer_receive", "a2", action="ack", message_ids=[sent["message_id"]])["acknowledged"] == [sent["message_id"]]


def test_gateway_foreign_or_unattributed_children_cannot_route_as_local_teams(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        add_child(store, "gateway-child", "managed")
        add_child(store, "foreign-child", "foreign", profile="foreign-profile")
        add_child(store, "foreign-parent-child", "foreign")
        store.sessions.create(NewSession("unattributed", "subagent", cwd="/project/work", git_repo_root="/project"))
    denied = ("managed", "foreign", "gateway-child", "foreign-child", "foreign-parent-child", "unattributed")
    assert not set(denied) & set(peers("sender", same_project=False))
    for sid in denied:
        assert not call("peer_send", "sender", target_session_ids=[sid], message="not local")["success"]
        assert not call("peer_send", sid, target_session_ids=["sender"], message="not local")["success"]
        assert not call("peer_receive", sid)["success"]
    assert ok("peer_receive", "sender")["messages"] == []


def test_branched_root_and_its_children_remain_separate(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.create(NewSession(
            "branched-root", "desktop", parent_session_id="sender", lineage_kind=SessionLineageKind.BRANCH,
            cwd="/project/api", git_repo_root="/project",
        ))
        add_child(store, "branched-child", "branched-root")
    assert set(peers("branched-child")) == {"branched-root"}
    assert peers("branched-child")["branched-root"]["thread_id"] == "branched-root"
    assert not call("peer_send", "branched-child", target_session_ids=["b1"], message="different root")["success"]
    assert not call("peer_send", "a1", target_session_ids=["branched-child"], message="separate root")["success"]


def test_child_compaction_preserves_only_that_childs_mail_and_team(teams):
    pending = ok("peer_send", "b2", target_session_ids=["b1"], message="Keep through child rotation")["results"][0]
    parent_note = ok("peer_send", "b2", target_session_ids=["recipient"], message="Parent inbox only")["results"][0]
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.end("b1", "compression")
        store.sessions.create(NewSession(
            "b1-tip", "subagent", parent_session_id="b1", cwd="/project/work", git_repo_root="/project",
            model_config={"_delegate_from": "recipient", "_subagent_id": "task-b1"},
        ))
    received = ok("peer_receive", "b1-tip")
    assert [note["message_id"] for note in received["messages"]] == [pending["message_id"]]
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == parent_note["message_id"]
    assert peers("b2")["b1-tip"]["thread_id"] == "b1"
    followup = ok("peer_send", "b2", target_session_ids=["b1"], message="Resolve old child address")["results"][0]
    assert followup["target_session_id"] == "b1-tip"
    folder = teams / "runtime" / "peer_messages_v2"
    assert (folder / "b1" / f"{followup['message_id']}.json").is_file()
    assert not (folder / "b1-tip").exists()
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


def test_sibling_exchange_survives_fresh_os_processes(teams):
    sent = process_call(teams, "peer_send", "a1", target_session_ids=["a2"], message="Fresh-process finding")["results"][0]
    received = process_call(teams, "peer_receive", "a2")
    assert received["messages"][0]["message_id"] == sent["message_id"]
    assert received["messages"][0]["from_session_id"] == "a1"
    reply = process_call(teams, "peer_send", "a2", target_session_ids=["a1"], message="Fresh-process reply",
                         in_reply_to=sent["message_id"])["results"][0]
    assert ok("peer_receive", "a1")["messages"][0]["message_id"] == reply["message_id"]
    assert ok("peer_receive", "a1")["messages"][0]["in_reply_to"] == sent["message_id"]
    assert process_call(teams, "peer_receive", "a2", action="ack", message_ids=[sent["message_id"]])["acknowledged"] == [sent["message_id"]]
    assert ok("peer_receive", "a2")["messages"] == []
    with open_persistence_reader(teams) as store:
        assert store.sessions.get("a1").message_count == store.sessions.get("a2").message_count == 0


def test_send_many_deduplicates_parent_and_compression_aliases_and_checks_each_target(teams):
    with open_persistence(profile_scope(home=teams)) as store:
        store.sessions.end("sender", "compression")
        store.sessions.create(NewSession("sender-tip", "cli", parent_session_id="sender"))
    requested = ["parent", "sender", "sender-tip", "b1", "a2", "a1", "a2"]
    reply = call("peer_send", "a1", target_session_ids=requested, message="Shared finding")
    assert reply["success"] is False and reply["status"] == "partial"
    assert reply["queued_count"] == 2 and reply["from_thread_id"] == "a1"
    results = reply["results"]
    assert [item["requested_target_session_id"] for item in results] == requested
    assert [item["status"] for item in results] == [
        "queued", "queued", "queued", "failed", "queued", "failed", "queued"]
    assert results[1]["duplicate_of"] == results[2]["duplicate_of"] == "parent"
    assert results[6]["duplicate_of"] == "a2"
    assert results[0]["target_thread_id"] == "sender"
    assert results[0]["target_session_id"] == "sender-tip"
    assert results[0]["message_id"] == results[1]["message_id"] == results[2]["message_id"]
    assert results[4]["message_id"] == results[6]["message_id"]
    assert results[0]["message_id"] != results[4]["message_id"]
    for recipient, item in (("sender-tip", results[0]), ("a2", results[4])):
        messages = ok("peer_receive", recipient)["messages"]
        assert [(m["message_id"], m["message"]) for m in messages] == [(item["message_id"], "Shared finding")]
    assert ok("peer_receive", "b1")["messages"] == []
    assert ok("peer_receive", "a1")["messages"] == []


def test_send_many_full_inbox_does_not_block_other_recipient(teams):
    from importlib import import_module

    box = import_module("peer-messaging.mailbox")
    for index in range(box.MAX_PENDING_MESSAGES):
        assert call("peer_send", "sender", target_session_ids=["a1"], message=str(index))["success"]
    sent = call("peer_send", "sender", target_session_ids=["a1", "a2", "a1"], message="New handoff")
    assert sent["status"] == "partial" and sent["queued_count"] == 1
    first, second, duplicate = sent["results"]
    assert first["status"] == duplicate["status"] == "failed" and "full" in first["error"]
    assert duplicate["duplicate_of"] == "a1"
    assert second["status"] == "queued"
    assert ok("peer_receive", "a1")["pending_count"] == box.MAX_PENDING_MESSAGES
    assert ok("peer_receive", "a2")["messages"][0]["message_id"] == second["message_id"]
