"""V2 contracts use temporary native Hermes profiles and its tool dispatcher."""
import time
from importlib import import_module
from types import SimpleNamespace

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
    assert ok("link_thread", "recipient", thread_id="sender")["status"] == "linked"


def test_consent_is_bilateral_scoped_and_revocable(install):
    assert ok("list_threads", "sender")["threads"] == []
    assert not call("link_thread", "sender", thread_id="elsewhere")["success"]
    append(install, "recipient", "assistant", "private pre-link note", timestamp=time.time() - 20, finish_reason="stop")
    assert not call("read_thread", "sender", thread_id="recipient")["success"]
    assert not call("send_message_to_thread", "sender", thread_id="recipient", message="not linked")["success"]
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
    assert first["status"] == "timeout" and first["pending"] == ["recipient"]
    queued = ok("send_message_to_thread", "sender", thread_id="recipient", message="coordination, not a prompt")
    assert queued["status"] == "queued"
    with open_persistence_reader(install) as store:
        assert all(store.sessions.get(i).message_count == 0 for i in ("recipient", "elsewhere"))
    assert ok("peer_receive", "recipient")["messages"][0]["message_id"] == queued["message_id"]
    append(install, "recipient", "assistant", "partial", finish_reason="length")
    assert ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)["status"] == "timeout"
    append(install, "recipient", "assistant", "result", finish_reason="stop")
    ready = ok("wait_threads", "sender", threads=[{"thread_id": "recipient"}], wait_seconds=0)
    result = ready["ready"][0]["result"]
    assert ready["status"] == "ready" and result["text"] == "result"
    assert ok("wait_threads", "sender", threads=[{"thread_id": "recipient", "after_row_id": result["row_id"]}],
              wait_seconds=0)["status"] == "timeout"
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.end("recipient", "agent_close")
    assert ok("read_thread", "sender", thread_id="recipient")["ended_at"] is not None
    assert not call("send_message_to_thread", "sender", thread_id="recipient", message="closed")["success"]
    closed = ok("wait_threads", "sender", threads=[{"thread_id": "recipient", "after_row_id": result["row_id"]}],
                wait_seconds=0)
    assert closed["status"] == "closed" and closed["closed"] == ["recipient"]


def test_wait_observes_a_final_saved_after_polling_started(install, monkeypatch):
    connect()
    threads = import_module("peer-messaging.threads")
    clock = [0.0]
    saved = []

    def finish_during_wait(seconds):
        if not saved:
            saved.append(append(install, "recipient", "assistant", "finished during the wait", finish_reason="stop"))
        clock[0] += seconds

    monkeypatch.setattr(threads, "time", SimpleNamespace(
        time=time.time, monotonic=lambda: clock[0], sleep=finish_during_wait,
    ))
    result = threads.wait_threads({"threads": [{"thread_id": "recipient"}], "wait_seconds": 1}, "sender")
    assert saved, "The final must be persisted after the first poll"
    assert result["status"] == "ready", result
    assert result["ready"][0]["result"]["text"] == "finished during the wait"
    assert result["pending"] == []


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


def test_create_is_explicit_independent_and_idle(install):
    assert not call("create_thread", "sender", title="Review lane", confirm=False)["success"]
    result = ok("create_thread", "sender", title="Review lane", confirm=True)
    target = result["thread_id"]
    assert result["status"] == "created_idle" and result["message_count"] == 0
    with open_persistence_reader(install) as store:
        row = store.sessions.get(target)
        assert row.title == "Review lane" and row.cwd == "/project/api" and row.git_repo_root == "/project"
        assert row.parent_session_id is None and row.tool_pin is None and row.system_prompt is None
        assert store.sessions.get_by_title("Review lane").session_id == target
        assert store.messages.read_transcript(target) == ()
    assert ok("list_threads", "sender")["threads"][0]["thread_id"] == target
    assert ok("read_thread", "sender", thread_id=target)["messages"] == []
    assert ok("send_message_to_thread", "sender", thread_id=target, message="for when you open")["status"] == "queued"
    assert ok("wait_threads", "sender", threads=[{"thread_id": target}], wait_seconds=0)["status"] == "timeout"
