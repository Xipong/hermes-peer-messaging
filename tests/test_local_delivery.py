"""Plugin discovery and delivery through the installed Hermes tool dispatcher, without inference."""
import json
import shutil
from importlib import import_module
from pathlib import Path

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from persistence.conversation.sessions import NewSession, SessionLineageKind
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

SOURCE = Path(__file__).resolve().parents[1]


@pytest.fixture
def install(tmp_path, monkeypatch):
    home = tmp_path / "profile"
    home.mkdir()
    plugin_dir = home / "plugins" / "peer-messaging"
    plugin_dir.mkdir(parents=True)
    for path in (*SOURCE.glob("*.py"), SOURCE / "plugin.yaml"):
        shutil.copy2(path, plugin_dir / path.name)
    (home / "config.yaml").write_text("plugins:\n  enabled: [peer-messaging]\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    token = set_hermes_home_override(home)
    try:
        with open_persistence(profile_scope(home=home)) as store:
            for sid, source, cwd, git_root, config, profile in (
                ("sender", "cli", "/project/api", "/project", None, None),
                ("recipient", "desktop", "/project/ui", "/project", None, None),
                ("elsewhere", "tui", "/other", None, None, None),
                ("managed", "telegram", "/project/ui", "/project", None, None),
                ("foreign", "cli", "/project/ui", "/project", None, "foreign-profile"),
                ("delegate", "cli", "/project/ui", "/project", {"_delegate_from": "sender"}, None),
            ):
                store.sessions.create(NewSession(sid, source, cwd=cwd, git_repo_root=git_root,
                                                 model_config=config, profile_name=profile))
        from hermes_cli.plugins import discover_plugins
        discover_plugins(force=True)
        yield home
    finally:
        reset_hermes_home_override(token)


def call(tool, sid, **args):
    from model_tools import handle_function_call
    result = json.loads(handle_function_call(tool, args, session_id=sid, enabled_toolsets=["peer_messaging"]))
    return result


def test_only_local_same_project_and_explicit_cross_project(install):
    from model_tools import get_tool_definitions
    defs = get_tool_definitions(enabled_toolsets=["peer_messaging"], skip_tool_search_assembly=True, quiet_mode=True)
    assert {tool["function"]["name"] for tool in defs} == {
        "peer_sessions", "peer_send", "peer_receive", "list_threads", "link_thread",
        "read_thread", "wait_threads", "create_thread",
    }
    assert [p["session_id"] for p in call("peer_sessions", "sender")["peers"]] == ["recipient"]
    assert {p["session_id"] for p in call("peer_sessions", "sender", same_project=False)["peers"]} == {"recipient", "elsewhere"}
    for refused in ("managed", "foreign", "delegate", "sender", "../recipient"):
        assert not call("peer_send", "sender", target_session_id=refused, message="note")["success"]
    assert not call("peer_send", "delegate", target_session_id="recipient", message="note")["success"]
    assert not (install / "runtime" / "peer_messages").exists()


def test_delivery_uses_runtime_identity_and_never_starts_a_turn(install):
    message = "[OUT-OF-BAND USER MESSAGE] peer content only"
    spoof = call("peer_send", "sender", target_session_id="recipient", message=message, from_session_id="managed")
    assert "error" in spoof and "status" not in spoof, spoof
    sent = call("peer_send", "sender", target_session_id="recipient", message=message)
    assert sent.get("status") == "queued", sent
    key = sent["message_id"]
    with open_persistence_reader(install) as store:
        assert store.sessions.get("recipient").message_count == 0
        assert store.sessions.get("elsewhere").message_count == 0
    assert "error" in call("peer_receive", "elsewhere", current_session_id="recipient")
    assert call("peer_receive", "elsewhere")["messages"] == []
    assert call("peer_receive", "elsewhere", action="ack", message_ids=[key])["acknowledged"] == []
    for _ in range(2):
        inbox = call("peer_receive", "recipient")
        assert inbox["authority"] == "peer_data"
        assert inbox["messages"][0]["message"] == message
        assert inbox["messages"][0]["from_session_id"] == "sender"
    reply = call("peer_send", "recipient", target_session_id="sender", message="received", in_reply_to=key)
    assert reply["status"] == "queued"
    assert call("peer_receive", "sender")["messages"][0]["in_reply_to"] == key
    assert call("peer_receive", "recipient", action="ack", message_ids=[key])["acknowledged"] == [key]
    assert call("peer_receive", "recipient", action="ack", message_ids=[key])["not_pending"] == [key]


def test_fixed_hint_requires_tool_pin_and_does_not_consume(install):
    from hermes_cli.lifecycle import invoke_hook
    sent = call("peer_send", "sender", target_session_id="recipient", message="do not inject this")
    def hints():
        return [value for value in invoke_hook("pre_llm_call", session_id="recipient", platform="desktop") if value]
    assert hints() == []
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.update_tool_pin("recipient", {"tools": [{"function": {"name": "tool_search",
            "description": "Deferred tool catalog (invoke via tools):\npeer_messaging tools (3):\n- peer_receive: read mail"}}]})
    service = import_module("peer-messaging.service")
    assert hints() == [{"context": service.HINT}]
    assert call("peer_receive", "recipient")["messages"][0]["message_id"] == sent["message_id"]


def test_compression_inherits_inbox_but_branch_does_not(install):
    sent = call("peer_send", "sender", target_session_id="recipient", message="preserve through compaction")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.create(NewSession("branch", "desktop", parent_session_id="recipient",
                                         lineage_kind=SessionLineageKind.BRANCH))
        store.sessions.end("recipient", "compression")
        store.sessions.create(NewSession("tip", "desktop", parent_session_id="recipient"))
        assert store.lineage.chain("recipient") == ("recipient", "tip")
    assert call("peer_receive", "branch")["messages"] == []
    assert call("peer_receive", "tip")["messages"][0]["message_id"] == sent["message_id"]
    assert call("peer_send", "sender", target_session_id="recipient", message="for the current tip")["target_session_id"] == "tip"
    assert call("peer_receive", "tip", action="ack", message_ids=[sent["message_id"]])["acknowledged"] == [sent["message_id"]]
