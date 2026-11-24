"""A delivery between an operation and its result hook must remain discoverable."""
import json
from importlib import import_module
from types import SimpleNamespace

from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401


def tracker(home, *tools):
    with open_persistence(profile_scope(home=home)) as store:
        store.sessions.update_tool_pin("recipient", {
            "tools": [{"function": {"name": name}} for name in tools],
        })
    return import_module("peer-messaging.service").InboxNotices()


def finish(notices, tool_name, result, args=None):
    return notices.after_tool(
        tool_name=tool_name, result=json.dumps(result), args=args or {},
        session_id="recipient", turn_id="same-running-turn",
    )


def test_mail_arriving_after_read_is_not_marked_as_already_inspected(install):
    service = import_module("peer-messaging.service")
    notices = tracker(install, "peer_receive")
    first = call("peer_send", "sender", target_session_ids=["recipient"], message="first note")["results"][0]
    read = service.receive({}, "recipient")
    assert [item["message_id"] for item in read["messages"]] == [first["message_id"]]

    # This models the dispatcher finishing the read before its transform hook runs.
    late = call("peer_send", "sender", target_session_ids=["recipient"], message="late private note")["results"][0]
    carried = json.loads(finish(notices, "peer_receive", read))
    assert carried["peer_update"]["mail_pending"] is True
    assert carried["messages"] == read["messages"]
    assert "late private note" not in json.dumps(carried)
    assert finish(notices, "other_tool", {"value": 1}) is None
    assert {item["message_id"] for item in service.receive({}, "recipient")["messages"]} == {
        first["message_id"], late["message_id"],
    }


def test_mail_arriving_after_ack_still_announces_itself_in_the_same_turn(install):
    service = import_module("peer-messaging.service")
    notices = tracker(install, "peer_receive")
    first = call("peer_send", "sender", target_session_ids=["recipient"], message="first note")["results"][0]
    read = service.receive({}, "recipient")
    assert finish(notices, "peer_receive", read) is None
    args = {"action": "ack", "message_ids": [first["message_id"]]}
    acknowledged = service.receive(args, "recipient")
    assert acknowledged["acknowledged"] == [first["message_id"]]

    late = call("peer_send", "sender", target_session_ids=["recipient"], message="arrived after ACK")["results"][0]
    carried = json.loads(finish(notices, "peer_receive", acknowledged, args))
    assert carried["peer_update"]["mail_pending"] is True
    assert carried["acknowledged"] == [first["message_id"]]
    assert "arrived after ACK" not in json.dumps(carried)
    assert finish(notices, "other_tool", {"value": 1}) is None
    assert [item["message_id"] for item in service.receive({}, "recipient")["messages"]] == [late["message_id"]]
    with open_persistence_reader(install) as store:
        assert store.sessions.get("recipient").message_count == 0


def test_peer_arriving_after_discovery_is_not_treated_as_a_returned_address(install, monkeypatch):
    service = import_module("peer-messaging.service")
    clock = [100.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=service.time.time))
    notices = tracker(install, "peer_sessions")
    listed = service.sessions({}, "recipient")
    assert {item["session_id"] for item in listed["peers"]} == {"sender"}
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("late-peer", "cli", cwd="/project/other", git_repo_root="/project"))

    # A successful explicit directory is already fresh; its result hook does
    # not do a second full scan or pretend it returned the later arrival.
    assert finish(notices, "peer_sessions", listed) is None
    assert finish(notices, "other_tool", {"value": 1}) is None
    clock[0] += service.DISCOVERY_INTERVAL
    carried = json.loads(finish(notices, "other_tool", {"value": 2}))
    assert carried["peer_update"]["neighbors"] == ["late-peer"]
    assert finish(notices, "other_tool", {"value": 3}) is None
    assert {item["session_id"] for item in service.sessions({}, "recipient")["peers"]} == {"sender", "late-peer"}
