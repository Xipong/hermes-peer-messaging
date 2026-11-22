"""New mail becomes visible during a running turn through a content-free tool notice."""
import json
from importlib import import_module

import pytest

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401


def poll(sid="recipient", turn="recipient-turn"):
    from model_tools import handle_function_call

    return json.loads(handle_function_call(
        "peer_sessions", {}, session_id=sid, turn_id=turn, tool_call_id="poll-mail",
        enabled_toolsets=["peer_messaging"],
    ))


def pin_receive(home, sid="recipient"):
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.update_tool_pin(sid, {"tools": [{"function": {"name": "peer_receive"}}]})


def test_new_mail_notifies_a_running_turn_without_payload_or_wake(install):
    pin_receive(install)
    assert "_peer_inbox_notice" not in poll()
    first = call("peer_send", "sender", target_session_id="recipient", message="private-note-one")
    noticed = poll()
    assert noticed["success"] and "peer_receive" in noticed["_peer_inbox_notice"]
    assert "private-note-one" not in json.dumps(noticed)
    assert "_peer_inbox_notice" not in poll()

    second = call("peer_send", "sender", target_session_id="recipient", message="private-note-two")
    assert poll()["_peer_inbox_notice"] == noticed["_peer_inbox_notice"]
    assert poll(turn="next-turn")["_peer_inbox_notice"] == noticed["_peer_inbox_notice"]
    with open_persistence_reader(install) as store:
        assert store.sessions.get("recipient").message_count == 0
    page = call("peer_receive", "recipient")
    assert [message["message_id"] for message in page["messages"]] == [first["message_id"], second["message_id"]]
    call("peer_receive", "recipient", action="ack", message_ids=[first["message_id"], second["message_id"]])
    assert "_peer_inbox_notice" not in poll()


def test_mid_turn_notices_require_the_recipient_tool_pin(install):
    call("peer_send", "sender", target_session_id="recipient", message="pending")
    assert "_peer_inbox_notice" not in poll()
    pin_receive(install)
    assert "_peer_inbox_notice" in poll()
    assert "_peer_inbox_notice" not in poll(sid="elsewhere")
    assert call("peer_receive", "recipient")["pending_count"] == 1


@pytest.mark.parametrize("result", ["[]", '[{"value": 1}]', '"text"', "null", "false", "123"])
def test_notices_preserve_json_array_and_scalar_results(install, result):
    pin_receive(install)
    call("peer_send", "sender", target_session_id="recipient", message="pending")
    notices = import_module("peer-messaging.service").InboxNotices()
    kwargs = {"tool_name": "some_tool", "session_id": "recipient", "turn_id": "one-turn"}
    assert notices.after_tool(result=result, **kwargs) is None
    carried = notices.after_tool(result='{"value": 1}', **kwargs)
    assert json.loads(carried)["value"] == 1
    assert "peer_receive" in json.loads(carried)["_peer_inbox_notice"]
    assert notices.after_tool(result='{"value": 2}', **kwargs) is None


def test_delegated_agent_notices_only_its_own_mail(install):
    from hermes_cli.lifecycle import invoke_hook

    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession(
            "child", "subagent", parent_session_id="sender",
            model_config={"_delegate_from": "sender", "_subagent_id": "child-id"},
        ))
    pin_receive(install, "child")
    call("peer_send", "recipient", target_session_id="sender", message="parent-only")
    assert "_peer_inbox_notice" not in poll(sid="child")
    sent = call("peer_send", "sender", target_session_id="child", message="child-only")
    assert sent["success"]
    assert "_peer_inbox_notice" in poll(sid="child")
    hints = [value for value in invoke_hook("pre_llm_call", session_id="child", platform="subagent") if value]
    assert len(hints) == 1 and "peer_receive" in hints[0]["context"]
    assert [m["message"] for m in call("peer_receive", "child")["messages"]] == ["child-only"]
