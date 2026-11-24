"""Named Hermes Projects group multiple independent folders without changing session cwd."""
from importlib import import_module

from hermes_cli import projects_db as pdb
from persistence.conversation.sessions import NewSession
from persistence_assembly import open_persistence, open_persistence_reader, profile_scope

from test_local_delivery import call, install  # noqa: F401


def project(home, name, *folders):
    with pdb.connect_closing(db_path=home / "projects.db") as conn:
        return pdb.create_project(conn, name=name, folders=folders)


def create(home, sid, cwd, root, *, parent=None):
    with open_persistence(profile_scope(home=home)) as store:
        if parent is None:
            store.sessions.create(NewSession(sid, "desktop", cwd=cwd, git_repo_root=root))
        else:
            store.sessions.create(NewSession(sid, "subagent", cwd=cwd, git_repo_root=root,
                                             parent_session_id=parent,
                                             model_config={"_delegate_from": parent, "_subagent_id": "task-" + sid}))


def test_explicit_project_spans_separate_repositories_and_delegated_teams(install):
    project_id = project(install, "One named Desktop project", "/project/api", "/independent/frontend")
    create(install, "frontend", "/independent/frontend", "/independent")
    create(install, "a-child", "/scratch/worktree-a", "/scratch/worktree-a", parent="sender")
    create(install, "b-child", "/other/worktree-b", "/other/worktree-b", parent="frontend")
    peers = call("peer_sessions", "sender")
    assert peers["hermes_project_id"] == project_id
    assert {p["session_id"] for p in peers["peers"]} == {"frontend", "a-child"}
    assert next(p for p in peers["peers"] if p["session_id"] == "frontend")["hermes_project_id"] == project_id
    assert not call("peer_send", "a-child", target_session_id="b-child", message="outside family")["success"]
    for same_project in (True, False):
        child = call("peer_sessions", "a-child", same_project=same_project)
        assert child["hermes_project_id"] == project_id
        assert {p["session_id"] for p in child["peers"]} == {"sender"}
    assert not call("peer_send", "a-child", target_session_id="b-child", message="Project does not widen family")["success"]
    sent = call("peer_send", "sender", target_session_id="frontend", message="We own the API changes")
    assert sent["status"] == "queued"
    assert call("peer_receive", "frontend")["messages"][0]["message_id"] == sent["message_id"]
    forwarded = call("peer_send", "frontend", target_session_id="b-child", message="Handle the frontend changes",
                     in_reply_to=sent["message_id"])
    assert call("peer_receive", "b-child")["messages"][0]["message_id"] == forwarded["message_id"]


def test_nested_named_project_takes_precedence_over_shared_git_root(install):
    project(install, "Umbrella", "/project")
    inner = project(install, "API", "/project/api")
    assert call("peer_sessions", "sender")["hermes_project_id"] == inner
    assert "recipient" not in {p["session_id"] for p in call("peer_sessions", "sender")["peers"]}
    # Independent roots can coordinate without widening family access.
    assert call("peer_send", "sender", target_session_id="recipient", message="heads up")["status"] == "queued"


def test_active_project_pointer_and_pathless_project_do_not_claim_unrelated_chat(install):
    with pdb.connect_closing(db_path=install / "projects.db") as conn:
        active_id = pdb.create_project(conn, name="Active but no folders")
        pdb.set_active(conn, active_id)
    sender = call("peer_sessions", "sender")
    assert sender["hermes_project_id"] is None
    assert {p["session_id"] for p in sender["peers"]} == {"recipient"}
    assert "elsewhere" not in {p["session_id"] for p in sender["peers"]}


def test_same_running_turn_discovers_new_native_project_peer_on_tool_result(install, monkeypatch):
    from hermes_cli.lifecycle import invoke_hook
    from hermes_cli.plugins import iter_hook_callbacks
    from model_tools import handle_function_call
    import json
    from types import SimpleNamespace

    # Patch the installed package behind the hook, not the test-loader alias.
    callback = next(cb for cb in iter_hook_callbacks("pre_llm_call")
                    if cb.__name__ == "before_turn" and cb.__module__.endswith(".service"))
    service = import_module(callback.__module__)
    clock = [100.0]
    monkeypatch.setattr(service, "time", SimpleNamespace(monotonic=lambda: clock[0], time=service.time.time))

    project(install, "One project", "/project/api", "/new-repo/web")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.update_tool_pin("sender", {"tools": [
            {"function": {"name": "peer_sessions"}},
        ]})
    assert not [v for v in invoke_hook("pre_llm_call", session_id="sender",
                                       turn_id="hours-long-turn", platform="cli") if v]
    create(install, "web", "/new-repo/web", "/new-repo")
    clock[0] += service.DISCOVERY_INTERVAL
    result = json.loads(handle_function_call(
        "peer_receive", {}, session_id="sender", turn_id="hours-long-turn",
        tool_call_id="still-same-turn", enabled_toolsets=["peer_messaging"],
    ))
    assert result["peer_update"]["neighbors"] == ["web"]
    assert call("peer_sessions", "sender")["hermes_project_id"] is not None
    with open_persistence_reader(install) as store:
        assert store.sessions.get("web").api_call_count == 0
        assert store.sessions.get("web").message_count == 0
