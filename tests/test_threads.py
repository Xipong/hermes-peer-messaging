"""V2 contracts use temporary native Hermes profiles and its tool dispatcher."""
import time
from copy import deepcopy
from importlib import import_module

import pytest

from persistence.conversation.messages import NewMessage
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401 -- fixture and shared dispatcher


def ok(tool, sid, **args):
    reply = call(tool, sid, **args)
    assert reply.get("success"), reply
    return reply


def append(home, sid, role, content, **fields):
    with open_persistence(profile_scope(home=home)) as store:
        return store.messages.append(sid, NewMessage(role, content, **fields))


def connect():
    assert ok("link_thread", "sender", thread_id="recipient")["status"] == "awaiting_other_chat"
    assert ok("list_threads", "recipient")["threads"][0]["link"] == "incoming"
    result = ok("link_thread", "recipient", thread_id="sender")
    assert result["status"] == "linked" and result["transcript_shared"] is True


def test_consent_is_bilateral_scoped_and_revocable(install):
    assert ok("list_threads", "sender")["threads"] == []
    assert not call("link_thread", "sender", thread_id="foreign")["success"]
    assert not call("link_thread", "sender", thread_id="managed")["success"]
    append(install, "recipient", "assistant", "private pre-link note", timestamp=time.time() - 20, finish_reason="stop")
    assert not call("read_thread", "sender", thread_id="recipient")["success"]
    connect()
    assert not call("read_thread", "elsewhere", thread_id="recipient")["success"]
    assert ok("read_thread", "sender", thread_id="recipient")["messages"] == []
    append(install, "recipient", "tool", "tool-output-token", tool_call_id="hidden")
    append(install, "recipient", "user", "hidden runtime message", display_kind="notification")
    append(install, "recipient", "user", "user text + injected context", api_content="private API prompt",
           display_metadata={"typed_ask": "user text"})
    append(install, "recipient", "assistant", "visible final", finish_reason="stop", reasoning="private reasoning")
    snapshot = ok("read_thread", "sender", thread_id="recipient")
    assert snapshot["authority"] == "peer_data"
    assert [x["text"] for x in snapshot["messages"]] == ["user text", "visible final"]
    assert snapshot["latest_final_row_id"] == snapshot["messages"][-1]["row_id"]
    ok("link_thread", "recipient", thread_id="sender", action="unlink")
    assert ok("list_threads", "sender")["threads"] == []
    assert not call("read_thread", "sender", thread_id="recipient")["success"]


def test_unlink_accepts_the_ended_current_segment_id(install):
    connect()
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.end("recipient", "compression")
        store.sessions.create(NewSession("recipient-tip", "desktop", parent_session_id="recipient",
                                         cwd="/project/ui", git_repo_root="/project"))
        store.sessions.end("recipient-tip", "agent_close")
    peer = ok("list_threads", "sender")["threads"][0]
    assert peer["thread_id"] == "recipient" and peer["session_id"] == "recipient-tip"
    unlinked = ok("link_thread", "sender", thread_id=peer["session_id"], action="unlink")
    assert unlinked["status"] == "unlinked" and unlinked["thread_id"] == "recipient"
    assert ok("list_threads", "sender")["threads"] == []


def test_message_and_wait_never_start_a_thread(install):
    connect()
    first = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    assert first["status"] == "pending" and first["pending"] == ["recipient"]
    queued = ok("peer_send", "sender", target_session_id="recipient", message="coordination, not a prompt")
    assert queued["status"] == "queued"
    with open_persistence_reader(install) as store:
        assert all(store.sessions.get(i).message_count == 0 for i in ("recipient", "elsewhere"))
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == queued["message_id"]
    append(install, "recipient", "assistant", "partial", finish_reason="length")
    assert ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)["status"] == "pending"
    append(install, "recipient", "assistant", "result", finish_reason="stop")
    ready = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    result = ready["ready"][0]["result"]
    assert ready["status"] == "ready" and result["text"] == "result"
    assert ok("wait_threads", "sender", threads=[{"thread_id": "recipient", "after_row_id": result["row_id"]}],
              wait_seconds=0)["status"] == "pending"
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.end("recipient", "agent_close")
    assert ok("read_thread", "sender", thread_id="recipient")["ended_at"] is not None
    assert not call("peer_send", "sender", target_session_id="recipient", message="closed")["success"]
    closed = ok("wait_threads", "sender", threads=[{"thread_id": "recipient", "after_row_id": result["row_id"]}],
                wait_seconds=0)
    assert closed["status"] == "closed" and closed["closed"] == ["recipient"]


@pytest.mark.parametrize("deferred", [False, True], ids=["direct", "tool_call"])
def test_wait_is_one_snapshot_even_with_a_legacy_pin(install, monkeypatch, deferred):
    from model_tools import get_tool_definitions

    connect()
    definitions = get_tool_definitions(enabled_toolsets=["peer_messaging"],
                                       skip_tool_search_assembly=True, quiet_mode=True)
    advertised = next(tool for tool in definitions if tool["function"]["name"] == "wait_threads")
    legacy = deepcopy(advertised)
    legacy["function"]["parameters"]["properties"]["wait_seconds"] = {
        "type": "number", "minimum": 0, "maximum": 30, "default": 15,
    }
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.update_tool_pin("sender", {"tools": [legacy]})

    def no_sleep(_seconds):
        pytest.fail("wait_threads must return one snapshot without waiting for a saved final")

    def observe(**args):
        if deferred:
            return call("tool_call", "sender", calls=[{"name": "wait_threads", "arguments": args}])
        return call("wait_threads", "sender", **args)

    monkeypatch.setattr(time, "sleep", no_sleep)
    empty = observe(threads=[{"thread_id": "recipient"}], wait_seconds=30)
    assert empty.get("success") and empty["status"] == "pending", empty
    assert empty["pending"] == ["recipient"] and empty["wait_seconds_ignored"] is True

    append(install, "recipient", "assistant", "saved after the first snapshot", finish_reason="stop")
    ready = observe(threads=[{"thread_id": "recipient"}])
    assert ready["status"] == "ready" and ready["ready"][0]["result"]["text"] == "saved after the first snapshot"
    assert "wait_seconds_ignored" not in ready
    latest = ready["ready"][0]["result"]["row_id"]
    pending = observe(threads=[{"thread_id": "recipient", "after_row_id": latest}])
    assert pending["status"] == "pending" and pending["ready"] == []

    for args in ({"current_session_id": "recipient"}, {"wait_second": 30}, {"wait_seconds": True}):
        refused = observe(threads=[{"thread_id": "recipient"}], **args)
        assert refused.get("success") is False and "error" in refused, refused
    assert "wait_seconds" not in advertised["function"]["parameters"]["properties"]


def test_tool_traffic_does_not_hide_a_saved_final(install):
    connect()
    append(install, "recipient", "assistant", "finished before the tool traffic", finish_reason="stop")
    with open_persistence(profile_scope(home=install)) as store:
        store.messages.append_batch("recipient", [
            NewMessage("tool", f"private tool result {i}", tool_call_id=f"call-{i}") for i in range(120)
        ] + [NewMessage("user", "private runtime notification", display_kind="notification") for _ in range(90)])
    append(install, "recipient", "user", "new visible ask")
    read = ok("read_thread", "sender", thread_id="recipient")
    assert [message["text"] for message in read["messages"]] == ["finished before the tool traffic", "new visible ask"]
    assert read["latest_final_row_id"] == read["messages"][0]["row_id"]
    result = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    assert result["status"] == "ready"
    assert result["ready"][0]["result"]["text"] == "finished before the tool traffic"


def test_exhausted_final_scan_is_reported_explicitly(install):
    connect()
    append(install, "recipient", "assistant", "older final", finish_reason="stop")
    with open_persistence(profile_scope(home=install)) as store:
        store.messages.append_batch("recipient", [
            NewMessage("assistant", "hidden progress", display_kind="notification", finish_reason="stop")
            for _ in range(81)
        ])
    read = ok("read_thread", "sender", thread_id="recipient")
    assert read["messages"] == [] and read["latest_final_row_id"] is None
    assert read["scan_limit_per_role"] == 80 and read["scan_limited_roles"] == ["assistant"]
    result = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    assert result["status"] == "scan_limit" and result["scan_limited"] == ["recipient"]
    append(install, "recipient", "assistant", "new final", finish_reason="stop")
    ready = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    assert ready["status"] == "ready" and ready["scan_limited"] == []


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
    assert not (install / "runtime" / "peer_messages").exists()
