"""Real-file and cross-process contracts for the opt-in peer inbox."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from importlib import import_module
from pathlib import Path

import pytest

from plugin_test_loader import load_plugin

load_plugin()  # Spawned worker interpreters don't import pytest's conftest.
box = import_module("peer-messaging.mailbox")


def _send_worker(args):
    inbox, index = args
    try:
        return box.enqueue(Path(inbox), sender="sender", target="recipient", text=f"message {index}")["message_id"]
    except ValueError:
        return None


def _contended_worker(inbox, operation, key, ready, done, result):
    inbox = Path(inbox)
    ready.set()
    snapshot = None
    try:
        if operation == "send":
            _send(inbox, "must not be queued while busy")
        elif operation == "read":
            snapshot = box.read_pending(inbox)
        else:
            box.acknowledge(inbox, [key])
    except Exception as exc:
        result.put((type(exc).__name__, str(exc)))
    else:
        result.put((None, snapshot))
    finally:
        done.set()


def _send(inbox, text="contract changed", **kwargs):
    return box.enqueue(inbox, sender="sender", target="recipient", text=text, **kwargs)


def test_read_is_non_destructive_and_restart_can_read_same_id(tmp_path):
    item = _send(tmp_path)
    assert box.read_pending(tmp_path)["messages"] == [item]
    # No receiver-local state or claimed lease is needed to retry after a crash.
    assert box.read_pending(Path(str(tmp_path)))["messages"] == [item]
    assert (tmp_path / f"{item['message_id']}.json").exists()
    receipt = box.acknowledge(tmp_path, [item["message_id"]])
    assert receipt["acknowledged"] == [item["message_id"]]
    assert box.read_pending(tmp_path)["messages"] == []
    assert box.acknowledge(tmp_path, [item["message_id"]])["not_pending"] == [item["message_id"]]


def test_reply_correlation_and_ack_do_not_touch_sibling_inbox(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    original = _send(a)
    reply = _send(b, "ack: I will change only the tests", in_reply_to=original["message_id"])
    assert box.read_pending(b)["messages"][0]["in_reply_to"] == original["message_id"]
    assert box.acknowledge(a, [reply["message_id"]])["not_pending"] == [reply["message_id"]]
    assert box.read_pending(b)["messages"] == [reply]


@pytest.mark.parametrize("failure", ["read", "ack"])
def test_io_failure_keeps_valid_message_for_retry(tmp_path, monkeypatch, failure):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        method = "open" if failure == "read" else "unlink"
        real = getattr(Path, method)

        def unavailable(self, *args, **kwargs):
            if self == path:
                raise PermissionError("temporary failure")
            return real(self, *args, **kwargs)

        patch.setattr(Path, method, unavailable)
        if failure == "read":
            assert box.read_pending(tmp_path)["unreadable_count"] == 1
        else:
            assert box.acknowledge(tmp_path, [item["message_id"]])["failed"] == [item["message_id"]]
    assert path.read_bytes() == original
    assert box.read_pending(tmp_path)["messages"] == [item]
    assert box.acknowledge(tmp_path, [item["message_id"]])["success"] is True


@pytest.mark.parametrize("poison", [b"{", b"\xff", b"[]", b'{"message": 12}', b'{"message": ""}'])
def test_poison_does_not_wedge_or_consume_valid_neighbor(tmp_path, poison):
    item = _send(tmp_path)
    bad = tmp_path / "00000000000000000000_deadbeef.json"
    bad.write_bytes(poison)
    assert box.read_pending(tmp_path)["messages"] == [item]
    assert not bad.exists()
    assert (tmp_path / f"{item['message_id']}.json").exists()


@pytest.mark.parametrize("invalid", ["", ".", "..", "../recipient", "a/b", "a\\b", "x" * 129, None])
def test_session_ids_are_not_sanitized_into_other_inboxes(invalid):
    with pytest.raises(ValueError):
        box.session_component(invalid)


@pytest.mark.parametrize("bad_id", ["../foreign", "", "valid.json", None])
def test_ack_validates_entire_batch_before_mutation(tmp_path, bad_id):
    item = _send(tmp_path)
    with pytest.raises(ValueError):
        box.acknowledge(tmp_path, [item["message_id"], bad_id])
    assert box.read_pending(tmp_path)["messages"] == [item]


def test_bounded_read_and_ack_expose_next_page(tmp_path):
    items = [_send(tmp_path, str(n)) for n in range(box.MAX_READ_MESSAGES + 2)]
    first = box.read_pending(tmp_path)
    assert len(first["messages"]) == box.MAX_READ_MESSAGES
    assert first["has_more"] is True
    box.acknowledge(tmp_path, [m["message_id"] for m in first["messages"]])
    assert box.read_pending(tmp_path)["messages"] == items[box.MAX_READ_MESSAGES:]


def test_read_loads_only_requested_fifo_page(tmp_path, monkeypatch):
    items = [_send(tmp_path, str(number)) for number in range(50)]
    loaded = []
    load = box._load

    def observe(path):
        loaded.append(path.stem)
        return load(path)

    monkeypatch.setattr(box, "_load", observe)
    page = box.read_pending(tmp_path)
    assert page["messages"] == items[:5]
    assert loaded == [item["message_id"] for item in items[:5]]
    assert page["has_more"] is True and page["pending_count"] == 50
    assert page["pending_count_is_exact"] is False


def test_uninspected_tail_is_reported_until_its_page_is_validated(tmp_path):
    items = [_send(tmp_path, str(number)) for number in range(6)]
    poison = tmp_path / f"{items[-1]['message_id']}.json"
    poison.write_bytes(b"{")
    first = box.read_pending(tmp_path)
    assert first["messages"] == items[:5]
    assert first["has_more"] and first["pending_count"] == 6
    assert first["pending_count_is_exact"] is False and poison.exists()
    box.acknowledge(tmp_path, [item["message_id"] for item in first["messages"]])
    last = box.read_pending(tmp_path)
    assert last["messages"] == [] and last["pending_count"] == 0
    assert last["pending_count_is_exact"] is True and not last["has_more"]
    assert not poison.exists()


def test_concurrent_processes_cannot_overfill_or_overwrite(tmp_path):
    jobs = [(str(tmp_path), n) for n in range(box.MAX_PENDING_MESSAGES + 8)]
    with ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("spawn")) as pool:
        ids = [key for key in pool.map(_send_worker, jobs) if key is not None]
    assert len(ids) == len(set(ids)) == box.MAX_PENDING_MESSAGES
    assert box.read_pending(tmp_path)["pending_count"] == box.MAX_PENDING_MESSAGES


@pytest.mark.parametrize("operation", ["send", "read", "ack"])
def test_contended_mailbox_keeps_published_mail_readable(tmp_path, operation):
    item = _send(tmp_path)
    context = multiprocessing.get_context("spawn")
    ready, done, result = context.Event(), context.Event(), context.Queue()
    process = context.Process(target=_contended_worker,
                              args=(str(tmp_path), operation, item["message_id"], ready, done, result))
    with box._FileLock(tmp_path / ".lock"):
        process.start()
        try:
            assert ready.wait(10), "mailbox worker did not start"
            assert done.wait(8), "mailbox operation blocked on a lock instead of returning for retry"
            error, outcome = result.get(timeout=2)
            if operation == "read":
                assert error is None, (error, outcome)
                assert outcome["messages"] == [item]
            else:
                assert error == "TimeoutError", (error, outcome)
                assert "retry" in outcome.lower()
            process.join(timeout=5)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
            result.close()
            result.join_thread()
    assert box.read_pending(tmp_path)["messages"] == [item]


def test_concurrent_ack_is_not_reported_as_unreadable(tmp_path, monkeypatch):
    first, second = _send(tmp_path, "first"), _send(tmp_path, "second")
    load = box._load

    def ack_before_open(path):
        if path.stem == first["message_id"]:
            assert box.acknowledge(tmp_path, [path.stem])["acknowledged"] == [path.stem]
        return load(path)

    monkeypatch.setattr(box, "_load", ack_before_open)
    snapshot = box.read_pending(tmp_path)
    assert snapshot["messages"] == [second]
    assert snapshot["pending_count"] == 1
    assert snapshot["unreadable_count"] == 0
    assert (tmp_path / f"{second['message_id']}.json").exists()


def test_invalid_unicode_is_rejected_before_inbox_creation(tmp_path):
    inbox = tmp_path / "absent"
    # Hermes' atomic writer ASCII-escapes the entire JSON on lone surrogates;
    # this otherwise accepted text became a >64KB file deleted on first read.
    text = "\udcff" + "😀" * (box.MAX_MESSAGE_CHARS - 1)
    with pytest.raises(ValueError, match="UTF-8"):
        _send(inbox, text)
    assert not inbox.exists()


@pytest.mark.parametrize("character", ["😀", "\x00"])
def test_maximum_valid_unicode_and_json_escapes_remain_readable(tmp_path, character):
    item = _send(tmp_path, character * box.MAX_MESSAGE_CHARS)
    assert box.read_pending(tmp_path)["messages"] == [item]
    assert (tmp_path / f"{item['message_id']}.json").exists()


def test_storage_projects_only_data_not_attacker_role(tmp_path):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    path.write_text(json.dumps({**item, "role": "system", "authority": "user", "wake": True}), encoding="utf-8")
    assert box.read_pending(tmp_path)["messages"] == [item]


def test_bad_reply_id_does_not_create_inbox(tmp_path):
    inbox = tmp_path / "absent"
    with pytest.raises(ValueError):
        _send(inbox, in_reply_to="../other")
    assert not inbox.exists()


def test_missing_read_does_not_create_mailbox(tmp_path):
    inbox = tmp_path / "absent"
    assert box.read_pending(inbox)["messages"] == []
    assert not inbox.exists()


def test_envelope_without_identity_is_not_accepted(tmp_path):
    key = "00000000000000000001_abcde123"
    path = tmp_path / f"{key}.json"
    path.write_text(json.dumps({"from_session_id": "sender", "message": "invalid", "sent_at": 1.0}), encoding="utf-8")
    assert box.read_pending(tmp_path)["messages"] == []
    assert not path.exists()


@pytest.mark.parametrize("metadata", [
    {"from_session_id": "x" * 200}, {"from_session_id": []},
    {"target_session_id": "../elsewhere"}, {"in_reply_to": "../receipt"},
    {"sent_at": float("inf")}, {"message_id": "mismatch"},
])
def test_bad_metadata_does_not_enter_tool_context(tmp_path, metadata):
    item = _send(tmp_path)
    path = tmp_path / f"{item['message_id']}.json"
    path.write_text(json.dumps({**item, **metadata}), encoding="utf-8")
    assert box.read_pending(tmp_path)["messages"] == []


def test_failed_publication_never_exposes_partial_message(tmp_path, monkeypatch):
    def failed(*args, **kwargs):
        raise OSError("publication unavailable")
    monkeypatch.setattr(box, "atomic_json_write", failed)
    with pytest.raises(OSError):
        _send(tmp_path)
    assert box.read_pending(tmp_path)["messages"] == []


def test_symlink_inbox_cannot_redirect_to_neighbor(tmp_path):
    target = tmp_path / "neighbor"
    target.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        _send(link)
    assert not list(target.iterdir())


def test_archived_envelopes_share_fifo_capacity_and_ack_with_active_mail(tmp_path):
    inbox = tmp_path / "recipient"
    first = _send(inbox, "archived")
    assert box.archive_pending(inbox) == 1
    archived = tmp_path / ".archive" / "recipient" / f"{first['message_id']}.json"
    assert archived.is_file()
    assert not (inbox / archived.name).exists()
    assert box.archive_pending(inbox) == 0
    second = _send(inbox, "new")
    assert box.read_pending(inbox)["messages"] == [first, second]
    assert box.acknowledge(inbox, [first["message_id"]])["acknowledged"] == [first["message_id"]]
    assert not archived.exists()
    assert box.read_pending(inbox)["messages"] == [second]


def test_archive_does_not_bypass_pending_cap_or_lose_mail_after_empty_directory_removed(tmp_path):
    inbox = tmp_path / "recipient"
    sent = [_send(inbox, f"mail {i}") for i in range(box.MAX_PENDING_MESSAGES)]
    assert box.archive_pending(inbox) == len(sent)
    assert box.read_pending(inbox)["pending_count"] == len(sent)
    with pytest.raises(ValueError, match="full"):
        _send(inbox, "above combined cap")
    # Simulate manual removal of only the empty active inbox and its lock.
    import shutil
    shutil.rmtree(inbox)
    assert box.read_pending(inbox)["messages"] == sent[:box.MAX_READ_MESSAGES]
    assert box.acknowledge(inbox, [sent[0]["message_id"]])["acknowledged"] == [sent[0]["message_id"]]
    assert box.read_pending(inbox)["pending_count"] == len(sent) - 1
