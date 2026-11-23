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
    assert not call("peer_send", "a-child", target_session_id="b-child", message="not linked")["success"]
    assert call("link_thread", "sender", thread_id="frontend")["status"] == "awaiting_other_chat"
    assert call("link_thread", "frontend", thread_id="sender")["status"] == "linked"
    assert {p["session_id"] for p in call("peer_sessions", "a-child")["peers"]} >= {
        "sender", "frontend", "b-child",
    }
    sent = call("peer_send", "a-child", target_session_id="b-child", message="We own the API changes")
    assert sent["status"] == "queued"
    assert call("peer_receive", "b-child")["messages"][0]["message_id"] == sent["message_id"]


def test_nested_named_project_takes_precedence_over_shared_git_root(install):
    project(install, "Umbrella", "/project")
    inner = project(install, "API", "/project/api")
    assert call("peer_sessions", "sender")["hermes_project_id"] == inner
    assert "recipient" not in {p["session_id"] for p in call("peer_sessions", "sender")["peers"]}
    # First contact parent-to-parent remains permitted, but it does not grant team access.
    assert call("peer_send", "sender", target_session_id="recipient", message="heads up")["status"] == "queued"
    assert not call("link_thread", "sender", thread_id="recipient")["success"]


def test_active_project_pointer_and_pathless_project_do_not_claim_unrelated_chat(install):
    with pdb.connect_closing(db_path=install / "projects.db") as conn:
        active_id = pdb.create_project(conn, name="Active but no folders")
        pdb.set_active(conn, active_id)
    sender = call("peer_sessions", "sender")
    assert sender["hermes_project_id"] is None
    assert {p["session_id"] for p in sender["peers"]} == {"recipient"}
    assert "elsewhere" not in {p["session_id"] for p in sender["peers"]}


def test_same_running_turn_discovers_new_native_project_peer_on_tool_result(install):
    from hermes_cli.lifecycle import invoke_hook
    from model_tools import handle_function_call
    import json

    project(install, "One project", "/project/api", "/new-repo/web")
    with open_persistence(profile_scope(home=install)) as store:
        store.sessions.update_tool_pin("sender", {"tools": [
            {"function": {"name": "peer_sessions"}},
        ]})
    assert not [v for v in invoke_hook("pre_llm_call", session_id="sender",
                                       turn_id="hours-long-turn", platform="cli") if v]
    create(install, "web", "/new-repo/web", "/new-repo")
    result = json.loads(handle_function_call(
        "list_threads", {}, session_id="sender", turn_id="hours-long-turn",
        tool_call_id="still-same-turn", enabled_toolsets=["peer_messaging"],
    ))
    assert result["_peer_discovery_notice"] == import_module("peer-messaging.service").DISCOVERY_HINT
    assert call("peer_sessions", "sender")["hermes_project_id"] is not None
    with open_persistence_reader(install) as store:
        assert store.sessions.get("web").api_call_count == 0
        assert store.sessions.get("web").message_count == 0
