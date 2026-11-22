"""V2 contracts use temporary native Hermes profiles and its tool dispatcher."""
import time

from persistence.conversation.messages import NewMessage
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
