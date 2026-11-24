"""Removed legacy controls cannot reappear via old tool pins or graph data."""
from importlib import import_module

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401


def test_removed_contact_tool_and_old_graph_are_ignored(install):
    from model_tools import get_tool_definitions
    folder = install / "runtime" / "peer_messages_v2"
    folder.mkdir(parents=True)
    graph = folder / ".threads.json"
    graph.write_text("old contact data is deliberately not even valid JSON")
    before = graph.read_bytes()
    result = call("link_thread", "sender", thread_id="recipient")
    assert not result.get("success") and "error" in result
    definitions = get_tool_definitions(enabled_toolsets=["peer_messaging"],
                                       skip_tool_search_assembly=True, quiet_mode=True)
    assert {tool["function"]["name"] for tool in definitions} == {
        "peer_sessions", "peer_send", "peer_receive",
    }
    listed = call("peer_sessions", "sender")
    assert listed["success"] and "links" not in listed
    assert call("peer_send", "sender", target_session_id="recipient", message="root mail")["status"] == "queued"
    assert graph.read_bytes() == before


def test_removed_transcript_tools_cannot_read_or_observe_other_chats(install):
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
    assert not hasattr(import_module("peer-messaging.service"), "create_thread")
    with open_persistence_reader(install) as store:
        assert store.sessions.get_by_title("Review lane") is None
    assert not (install / "runtime" / "peer_messages_v2").exists()
