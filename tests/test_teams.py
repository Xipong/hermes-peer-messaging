"""Stored delegation provenance and immediate-family communication boundaries."""
from importlib import import_module

import pytest

from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401 -- temporary native Hermes profile

teams = import_module("peer-messaging.teams")


def child(store, sid, parent, *, subagent_id=None, **fields):
    config = {"_delegate_from": parent, "_subagent_id": subagent_id or f"task-{sid}"}
    store.sessions.create(NewSession(sid, "subagent", parent_session_id=parent,
                                     model_config=config, **fields))


def test_nested_children_keep_team_identity_when_parent_compresses(install):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "a1", "sender")
        child(store, "nested", "a1")
        store.sessions.end("sender", "compression")
        store.sessions.create(NewSession("sender-tip", "cli", parent_session_id="sender"))
    with open_persistence_reader(install) as store:
        worker = teams.resolve(store, "nested")
        root = teams.resolve(store, "sender")
        assert worker.root_id == root.thread_id == "sender"
        assert worker.root.session_id == root.row.session_id == "sender-tip"
        assert worker.parent_session_id == "a1"
        assert worker.parent_thread_id == "a1"
        assert worker.thread_id == "nested"
        assert not teams.allowed(worker, root)
        assert teams.allowed(worker, teams.resolve(store, "a1"))
        assert teams.describe(worker, viewer=worker) == {
            "session_id": "nested", "thread_id": "nested",
            "parent_session_id": "a1", "parent_thread_id": "a1", "subagent_id": "task-nested", "is_child": True,
            "title": None, "source": "subagent", "relation": "self",
        }
        with pytest.raises(ValueError, match="outside"):
            teams.describe(root, viewer=worker)
        visible_parent = teams.describe(teams.resolve(store, "a1"), viewer=worker)
        assert visible_parent["relation"] == "parent"
        assert visible_parent["parent_session_id"] is None
        assert visible_parent["parent_thread_id"] is None
        assert "root_thread_id" not in visible_parent


def test_child_compression_follows_ancestry_not_reused_subagent_id(install):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "a1", "sender", subagent_id="same-short-id")
        store.sessions.end("a1", "compression")
        store.sessions.create(NewSession("a1-tip", "subagent", parent_session_id="a1",
                                         model_config={"_delegate_from": "sender", "_subagent_id": "same-short-id"}))
        child(store, "b1", "recipient", subagent_id="same-short-id")
    with open_persistence_reader(install) as store:
        worker = teams.resolve(store, "a1")
        assert worker.row.session_id == "a1-tip"
        assert worker.thread_id == "a1" and worker.root_id == "sender"
        assert worker.parent_session_id == "sender"
        assert teams.resolve(store, "a1-tip", follow=False) == worker
        with pytest.raises(ValueError, match="rotated"):
            teams.resolve(store, "a1", follow=False)


@pytest.mark.parametrize("kind", [SessionLineageKind.BRANCH, SessionLineageKind.RESET])
def test_forked_child_cannot_inherit_a_team_or_its_inbox(install, kind):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "a1", "sender")
        store.sessions.end("a1", "compression")
        store.sessions.create(NewSession("fork", "subagent", parent_session_id="a1", lineage_kind=kind,
                                         model_config={"_delegate_from": "sender", "_subagent_id": "task-a1"}))
        child(store, "fork-child", "fork")
    with open_persistence_reader(install) as store:
        for sid in ("fork", "fork-child"):
            with pytest.raises(ValueError, match="delegation"):
                teams.resolve(store, sid)


def test_hierarchy_discovery_and_known_address_sends_share_the_same_boundary(install):
    edges = (("a2", "sender"), ("b2", "sender"), ("a3", "a2"), ("a4", "a2"),
             ("b3", "b2"), ("b4", "b2"), ("other-child", "recipient"))
    with open_persistence(profile_scope(home=install)) as store:
        for sid, parent in edges:
            child(store, sid, parent)
    allowed_edges = {frozenset(edge) for edge in (
        *edges, ("a2", "b2"), ("a3", "a4"), ("b3", "b4"), ("sender", "recipient"),
    )}
    participants = ["sender", "recipient", *(sid for sid, _ in edges)]
    with open_persistence_reader(install) as store:
        resolved = {sid: teams.resolve(store, sid) for sid in participants}
        for sender in participants:
            for target in participants:
                assert teams.allowed(resolved[sender], resolved[target]) == (
                    frozenset((sender, target)) in allowed_edges)
    for sender in participants:
        listed = call("peer_sessions", sender, scope="all", same_project=False)
        assert listed.get("success"), listed
        expected = {target for target in participants if frozenset((sender, target)) in allowed_edges}
        # The fixture also has another independent root outside this graph.
        actual = {peer["session_id"] for peer in listed["peers"]} & set(participants)
        assert actual == expected
        for target in participants:
            if target == sender:
                continue
            result = call("peer_send", sender, target_session_ids=[target], message=f"{sender} to {target}")
            assert result.get("success") == (target in expected), (sender, target, result)
    folder = install / "runtime" / "peer_messages_v2"
    for recipient in participants:
        messages = call("peer_receive", recipient)["messages"]
        assert all(frozenset((message["from_session_id"], recipient)) in allowed_edges for message in messages)
        assert len(list((folder / recipient).glob("*.json"))) == sum(
            recipient in edge for edge in allowed_edges)


def test_immediate_family_survives_parent_and_child_compression_across_batches(install):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "coordinator", "sender")
        child(store, "early", "coordinator", subagent_id="worker")
        store.sessions.end("coordinator", "compression")
        store.sessions.create(NewSession("coordinator-tip", "subagent", parent_session_id="coordinator",
                                         model_config={"_delegate_from": "sender", "_subagent_id": "task-coordinator"}))
        child(store, "late", "coordinator-tip")
        store.sessions.end("early", "compression")
        store.sessions.create(NewSession("early-tip", "subagent", parent_session_id="early",
                                         model_config={"_delegate_from": "coordinator", "_subagent_id": "worker"}))
        child(store, "uncle", "sender", subagent_id="worker")
        child(store, "cousin", "uncle", subagent_id="worker")
        store.sessions.end("sender", "compression")
        store.sessions.create(NewSession("sender-tip", "cli", parent_session_id="sender"))
    with open_persistence_reader(install) as store:
        early, late = teams.resolve(store, "early"), teams.resolve(store, "late")
        assert early.parent_session_id == "coordinator"
        assert late.parent_session_id == "coordinator-tip"
        assert early.parent_thread_id == late.parent_thread_id == "coordinator"
        assert teams.relationship(early, late) == "sibling"
        assert teams.relationship(early, teams.resolve(store, "coordinator")) == "parent"
    for sender, target in (("early-tip", "late"), ("late", "early"),
                           ("late", "coordinator"), ("coordinator-tip", "early")):
        assert call("peer_send", sender, target_session_ids=[target], message="same family")["success"]
    sent = call("peer_send", "early-tip", target_session_ids=["parent"], message="report to coordinator")["results"][0]
    assert sent["target_session_id"] == "coordinator-tip"
    for target in ("sender", "sender-tip", "uncle", "cousin", "recipient"):
        assert not call("peer_send", "early-tip", target_session_ids=[target], message="out of scope")["success"]
    listed = call("peer_sessions", "early-tip", scope="all", same_project=False)
    assert {peer["session_id"] for peer in listed["peers"]} == {"coordinator-tip", "late"}


def test_unverified_or_foreign_parent_edges_never_grant_team_membership(install):
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("orphan", "subagent", model_config={"_delegate_from": "sender"}))
        store.sessions.create(NewSession("unmarked", "subagent", parent_session_id="sender"))
        store.sessions.create(NewSession("wrong-parent", "subagent", parent_session_id="sender",
                                         model_config={"_delegate_from": "recipient"}))
        child(store, "foreign-child", "sender", profile_name="foreign-profile")
        child(store, "foreign-parent", "foreign")
        child(store, "managed-parent", "managed")
        store.sessions.create(NewSession("cycle-a", "subagent", model_config={"_delegate_from": "cycle-b"}))
        store.sessions.create(NewSession("cycle-b", "subagent", parent_session_id="cycle-a",
                                         model_config={"_delegate_from": "cycle-a"}))
        store.sessions.create(NewSession("cycle-a", "subagent", parent_session_id="cycle-b"))
    with open_persistence_reader(install) as store:
        for sid in ("orphan", "unmarked", "wrong-parent", "foreign-child", "foreign-parent", "managed-parent", "cycle-a"):
            with pytest.raises(ValueError):
                teams.resolve(store, sid)


def test_ended_child_requires_explicit_observation_and_closed_root_is_not_live(install):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "a1", "sender")
        child(store, "a2", "sender")
        store.sessions.end("a1", "agent_close")
    with open_persistence_reader(install) as store:
        with pytest.raises(ValueError, match="ended"):
            teams.resolve(store, "a1")
        assert teams.resolve(store, "a1", allow_ended=True).row.ended_at is not None
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.end("sender", "agent_close")
    with open_persistence_reader(install) as store:
        with pytest.raises(ValueError, match="ended"):
            teams.resolve(store, "a2")
        assert teams.resolve(store, "a2", allow_ended=True).root.ended_at is not None


@pytest.mark.parametrize("boundary", ["managed", "archived", "foreign"])
def test_local_inbox_stops_at_validated_root_and_keeps_own_mail_usable(install, boundary):
    # Preserve #123218's receive/ACK boundary using native storage: an eligible
    # current session keeps its own inbox without inheriting an ineligible row.
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("old", "telegram" if boundary == "managed" else "cli",
                                         cwd="/project", git_repo_root="/project",
                                         profile_name="foreign-profile" if boundary == "foreign" else None))
        store.sessions.end("old", "compression")
        if boundary == "archived":
            store.sessions.set_archived("old", True)
        store.sessions.create(NewSession("tip", "cli", parent_session_id="old", cwd="/project",
                                         git_repo_root="/project", profile_name="default"))
    box = import_module("peer-messaging.mailbox")
    inbox = install / "runtime" / "peer_messages_v2"
    old_mail = box.enqueue(inbox / "old", sender="sender", target="old", text="ancestor-only mail")
    own_mail = box.enqueue(inbox / "tip", sender="sender", target="tip", text="own mail")
    with open_persistence_reader(install) as store:
        participant = teams.resolve(store, "tip")
        assert participant.root_id == participant.thread_id == "tip"
    received = call("peer_receive", "tip")
    assert received.get("success"), received
    assert received["messages"] == [own_mail]
    receipt = call("peer_receive", "tip", action="ack",
                   message_ids=[own_mail["message_id"], old_mail["message_id"]])
    assert receipt["acknowledged"] == [own_mail["message_id"]]
    assert receipt["not_pending"] == [old_mail["message_id"]]
    assert (inbox / "old" / f"{old_mail['message_id']}.json").exists()
    assert call("peer_send", "sender", target_session_ids=["tip"], message="fresh mail")["success"]
    assert not call("peer_send", "sender", target_session_ids=["old"], message="invalid old alias")["success"]


def test_child_compression_does_not_skip_an_ineligible_ownership_segment(install):
    with open_persistence(profile_scope(home=install)) as store:
        child(store, "a1", "sender")
        store.sessions.end("a1", "compression")
        store.sessions.set_archived("a1", True)
        store.sessions.create(NewSession("a1-tip", "subagent", parent_session_id="a1",
                                         model_config={"_delegate_from": "sender", "_subagent_id": "task-a1"}))
    with open_persistence_reader(install) as store:
        with pytest.raises(ValueError, match="archived"):
            teams.resolve(store, "a1-tip")
