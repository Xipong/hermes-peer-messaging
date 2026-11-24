"""Direct link management, independent of project and session recency."""
from importlib import import_module

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401


def ok(tool, sid, **args):
    reply = call(tool, sid, **args)
    assert reply.get("success"), reply
    return reply


def connect():
    assert ok("link_thread", "sender", thread_id="recipient")["status"] == "awaiting_other_chat"
    assert ok("peer_sessions", "recipient")["links"][0]["link"] == "incoming"
    result = ok("link_thread", "recipient", thread_id="sender")
    assert result["status"] == "linked"


def test_consent_is_bilateral_scoped_and_revocable(install):
    assert ok("peer_sessions", "sender")["links"] == []
    assert not call("link_thread", "sender", thread_id="foreign")["success"]
    assert not call("link_thread", "sender", thread_id="managed")["success"]
    connect()
    assert ok("peer_sessions", "sender", scope="team")["links"][0]["link"] == "linked"
    assert ok("peer_sessions", "elsewhere")["links"] == []
    ok("link_thread", "recipient", thread_id="sender", action="unlink")
    assert ok("peer_sessions", "sender")["links"] == []
    assert ok("peer_send", "sender", target_session_id="recipient", message="Root-to-root mail still allowed")["status"] == "queued"
    with open_persistence_reader(install) as store:
        assert store.sessions.get("recipient").message_count == 0


def test_unlink_accepts_the_ended_current_segment_id(install):
    connect()
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.end("recipient", "compression")
        store.sessions.create(NewSession("recipient-tip", "desktop", parent_session_id="recipient",
                                         cwd="/project/ui", git_repo_root="/project"))
        store.sessions.end("recipient-tip", "agent_close")
    peer = ok("peer_sessions", "sender")["links"][0]
    assert peer["thread_id"] == "recipient" and peer["session_id"] == "recipient-tip"
    unlinked = ok("link_thread", "sender", thread_id=peer["session_id"], action="unlink")
    assert unlinked["status"] == "unlinked" and unlinked["thread_id"] == "recipient"
    assert ok("peer_sessions", "sender")["links"] == []


def test_archived_link_remains_revocable_by_stable_id(install):
    connect()
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.set_archived("recipient", True)
    link, = ok("peer_sessions", "sender", scope="team")["links"]
    assert link["thread_id"] == "recipient" and link["session_id"] is None
    assert ok("link_thread", "sender", thread_id="recipient", action="unlink")["status"] == "unlinked"
    assert ok("peer_sessions", "sender")["links"] == []


def test_removed_transcript_tools_cannot_read_or_observe_other_chats(install):
    connect()
    for tool, args in (("read_thread", {"thread_id": "recipient"}),
                       ("wait_threads", {"threads": [{"thread_id": "recipient"}]}),
                       ("list_threads", {})):
        result = call(tool, "sender", **args)
        assert not result.get("success") and "error" in result, result


def test_removed_create_thread_cannot_create_or_link_session(install):
    # Old frozen tool pins or a direct call must not recover session creation.
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.update_tool_pin("sender", {"tools": [{"function": {
            "name": "create_thread", "description": "Legacy thread creation",
            "parameters": {"type": "object", "properties": {
                "title": {"type": "string"}, "confirm": {"type": "boolean"},
            }},
        }}]})
    result = call("create_thread", "sender", title="Review lane", confirm=True)
    assert "error" in result and not result.get("success"), result
    assert not hasattr(import_module("peer-messaging.threads"), "create_thread")
    with open_persistence_reader(install) as store:
        assert store.sessions.get_by_title("Review lane") is None
    assert not (install / "runtime" / "peer_messages_v2").exists()
