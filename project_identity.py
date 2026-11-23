"""Read-only native Hermes Project identity for local coordination.

Hermes stores named multi-folder Projects in projects.db; a session row does not
have a persistent project_id. The desktop groups rows by the longest registered
folder containing their cwd (or recorded repo root). We follow that mapping,
falling back to the existing workspace key for sessions outside those folders.

No active-project pointer, filesystem mutation or inference from project names.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import unicodedata
from pathlib import Path

from hermes_constants import get_hermes_home

LOG = logging.getLogger(__name__)


def _path(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        full = os.path.normcase(os.path.abspath(os.path.expanduser(value)))
        full = unicodedata.normalize("NFC", full).replace("\\", "/")
        return full.rstrip("/") or "/"
    except (OSError, ValueError):
        return None


def _under(folder, target):
    return target == folder or target.startswith(folder.rstrip("/") + "/")


class ProjectIndex:
    """Bounded, read-only snapshot of native Project folders, one per operation."""

    def __init__(self, home=None):
        self.folders = ()
        db = Path(home if home is not None else get_hermes_home()) / "projects.db"
        if not db.is_file() or db.is_symlink():
            return
        try:
            with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=0.25) as conn:
                rows = conn.execute(
                    """SELECT p.id, f.path FROM projects AS p
                       JOIN project_folders AS f ON f.project_id = p.id
                       WHERE p.archived = 0
                       ORDER BY p.created_at ASC, f.added_at ASC"""
                ).fetchall()
            self.folders = tuple((str(project_id), normal) for project_id, path in rows
                                 if (normal := _path(path)) is not None)
        except (sqlite3.DatabaseError, OSError, ValueError):
            LOG.debug("Native Hermes Projects unavailable; using stored workspace identity", exc_info=True)

    def key(self, row):
        """Stable key with explicit Project ID before legacy path fallback."""
        cwd = _path(getattr(row, "cwd", None))
        root = _path(getattr(row, "git_repo_root", None))
        best, selected = None, None
        for candidate_order, candidate in enumerate((cwd, root)):
            if candidate is None:
                continue
            for folder_order, (project_id, folder) in enumerate(self.folders):
                if _under(folder, candidate):
                    # Desktop uses the deepest registered folder and cwd first on
                    # ties; stable registration order breaks any final tie.
                    score = (len(folder.split("/")), len(folder), -candidate_order, -folder_order)
                    if best is None or score > best:
                        best, selected = score, project_id
        if selected is not None:
            return "hermes-project:" + selected
        fallback = root or cwd
        return "workspace-path:" + fallback if fallback is not None else None

    def named_id(self, row):
        key = self.key(row)
        return key.removeprefix("hermes-project:") if key and key.startswith("hermes-project:") else None
