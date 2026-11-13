"""Tests for the permanent purge API and its interaction with rebuilds."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import SETTINGS


def configure_database(path: Path) -> None:
    object.__setattr__(SETTINGS, "db_path", path)
    object.__setattr__(SETTINGS, "rebuild_batch_delay_ms", 120)
    object.__setattr__(SETTINGS, "rebuild_batch_size", 2)
    object.__setattr__(SETTINGS, "rebuild_lease_seconds", 3)


@pytest.fixture
def client(tmp_path: Path):
    configure_database(tmp_path / "archive.db")
    from app.main import app
    from app.rebuild import cancel_local_rebuilds

    with TestClient(app) as test_client:
        yield test_client
    cancel_local_rebuilds()


def _put(client: TestClient, page_id: str, title: str, body: str) -> dict:
    response = client.put(
        f"/api/pages/{page_id}",
        json={"title": title, "body": body},
    )
    assert response.status_code == 200, response.text
    return response.json()


def _search(client: TestClient, query: str) -> dict:
    response = client.get("/api/search", params={"q": query})
    assert response.status_code == 200, response.text
    return response.json()


def _wait_for_terminal(client: TestClient, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = client.get("/api/index/status").json()
        job = status["current_rebuild_job"] or status["latest_rebuild_job"]
        if job and job["job_id"] == job_id and job["state"] in {
            "succeeded",
            "failed",
            "interrupted",
        }:
            return status
        time.sleep(0.02)
    raise AssertionError(f"rebuild {job_id} did not finish within {timeout}s")


def _db_rows(sql: str) -> list:
    from app.db import connect

    conn = connect(query_only=True)
    try:
        return [dict(row) for row in conn.execute(sql).fetchall()]
    finally:
        conn.close()


def test_purge_removes_page_history_and_all_index_copies(client: TestClient):
    first = _put(
        client,
        "ocr-purge-001",
        "Titre secret café",
        "Le café contient un mot-secret alpha.",
    )
    second = _put(
        client,
        "ocr-purge-001",
        "Titre secret corrigé",
        "Le café contient un mot-secret beta.",
    )
    revision = second["revision"]
    assert revision > first["revision"]

    # Another page must remain fully readable and searchable after the purge.
    neighbour = _put(
        client,
        "ocr-keep-001",
        "Voisin café",
        "Le voisin garde son café et ses données.",
    )

    # Create a second index generation so two FTS tables hold the text.
    trigger = client.post("/api/index/rebuild", json={"accent_policy": "fold"})
    assert trigger.status_code == 202
    job_id = trigger.json()["job"]["job_id"]
    _wait_for_terminal(client, job_id)
    assert client.get("/api/index/status").json()["active_policy"] == "fold"

    assert _search(client, "secret")["total"] == 1
    assert _search(client, "cafe")["total"] == 2

    response = client.post(
        "/api/pages/ocr-purge-001/purge",
        json={"revision": revision},
    )
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["purged"] is True
    assert result["replayed"] is False
    assert result["revision"] > revision
    assert result["expected_revision"] == revision

    # The canonical row is gone.
    assert client.get("/api/pages/ocr-purge-001").status_code == 404

    # No searchable text remains in the active index or any retired generation.
    assert _search(client, "secret")["total"] == 0
    assert _search(client, "beta")["total"] == 0
    assert _search(client, "alpha")["total"] == 0
    remaining = _search(client, "cafe")
    assert [hit["page_id"] for hit in remaining["hits"]] == ["ocr-keep-001"]

    # The change log keeps sequence metadata but no title/body text.
    log_rows = _db_rows(
        "SELECT seq, op, title, body, rev, page_ref, changed_at "
        "FROM changes WHERE page_id = 'ocr-purge-001' ORDER BY seq"
    )
    assert len(log_rows) == 3
    assert [row["op"] for row in log_rows] == ["upsert", "upsert", "purge"]
    for row in log_rows:
        assert row["title"] is None
        assert row["body"] is None
        assert row["rev"] >= 1
        assert row["changed_at"]
        # seq values are retained and still unique/ordered.
    assert [row["seq"] for row in log_rows] == sorted(row["seq"] for row in log_rows)

    # The purge row revision equals its own sequence number.
    purge_row = log_rows[-1]
    assert purge_row["rev"] == purge_row["seq"] == result["revision"]

    # Every FTS generation lost the document.
    fts_leftovers = _db_rows(
        "SELECT g.id AS gen_id, g.fts_table "
        "FROM generations AS g WHERE g.fts_table <> ''"
    )
    assert len(fts_leftovers) >= 2
    for gen in fts_leftovers:
        hits = _db_rows(
            f"SELECT page_id FROM {gen['fts_table']} WHERE page_id = 'ocr-purge-001'"
        )
        assert hits == []

    # The idempotency ledger recorded the original result.
    ledger = _db_rows("SELECT * FROM purges WHERE page_id = 'ocr-purge-001'")
    assert len(ledger) == 1
    assert ledger[0]["expected_rev"] == revision
    stored = json.loads(ledger[0]["result_json"])
    assert stored["purged"] is True
    assert stored["revision"] == result["revision"]

    # The neighbour page is untouched.
    assert client.get("/api/pages/ocr-keep-001").json()["revision"] == neighbour["revision"]


def test_stale_revision_returns_409_and_changes_nothing(client: TestClient):
    created = _put(client, "ocr-stale-001", "Ancien titre", "Ancien corps café.")
    old_revision = created["revision"]
    updated = _put(client, "ocr-stale-001", "Nouveau titre", "Nouveau corps café.")
    assert updated["revision"] > old_revision

    response = client.post(
        "/api/pages/ocr-stale-001/purge",
        json={"revision": old_revision},
    )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["error"] == "revision_conflict"
    assert detail["expected_revision"] == old_revision
    assert detail["current_revision"] == updated["revision"]

    # Nothing was purged: the live page and its history remain.
    live = client.get("/api/pages/ocr-stale-001")
    assert live.status_code == 200
    assert live.json()["title"] == "Nouveau titre"
    rows = _db_rows("SELECT op, title, body FROM changes WHERE page_id = 'ocr-stale-001'")
    assert all(row["title"] is not None for row in rows)
    assert _db_rows("SELECT * FROM purges") == []


def test_purge_unknown_page_returns_404(client: TestClient):
    response = client.post(
        "/api/pages/never-seen/purge",
        json={"revision": 99},
    )
    assert response.status_code == 404
    assert response.json()["detail"]["error"] == "page_not_found"


def test_replay_returns_original_result_and_spares_recreated_page(client: TestClient):
    created = _put(client, "ocr-replay-001", "Premier titre", "Premier corps alpha.")
    original_revision = created["revision"]

    first = client.post(
        "/api/pages/ocr-replay-001/purge",
        json={"revision": original_revision},
    )
    assert first.status_code == 200
    first_result = first.json()
    assert first_result["replayed"] is False

    # Recreate a brand-new page with the same ID. Its revision must be larger.
    recreated = _put(
        client,
        "ocr-replay-001",
        "Second titre tout neuf",
        "Second corps beta flambant neuf.",
    )
    assert recreated["revision"] > first_result["revision"]

    # Re-send the exact original purge request: original result, no deletion.
    replay = client.post(
        "/api/pages/ocr-replay-001/purge",
        json={"revision": original_revision},
    )
    assert replay.status_code == 200
    replay_result = replay.json()
    assert replay_result["replayed"] is True
    assert replay_result["revision"] == first_result["revision"]
    assert replay_result["expected_revision"] == original_revision

    live = client.get("/api/pages/ocr-replay-001")
    assert live.status_code == 200
    assert live.json()["title"] == "Second titre tout neuf"
    assert live.json()["revision"] == recreated["revision"]
    assert _search(client, "flambant")["hits"][0]["page_id"] == "ocr-replay-001"
    # The old text is still absent.
    assert _search(client, "alpha")["total"] == 0

    # A purge at the *new* revision does remove the recreated page.
    second_purge = client.post(
        "/api/pages/ocr-replay-001/purge",
        json={"revision": recreated["revision"]},
    )
    assert second_purge.status_code == 200
    assert second_purge.json()["replayed"] is False
    assert client.get("/api/pages/ocr-replay-001").status_code == 404

    # Replaying either original request again keeps returning stored results.
    again = client.post(
        "/api/pages/ocr-replay-001/purge",
        json={"revision": original_revision},
    )
    assert again.status_code == 200
    assert again.json()["replayed"] is True
    assert again.json()["revision"] == first_result["revision"]


def test_purge_during_rebuild_old_snapshot_cannot_restore_text(client: TestClient):
    # Enough pages that the delayed baseline scan is still reading pages when
    # the purge commits.
    for index in range(12):
        _put(
            client,
            f"book-{index:02d}",
            f"Titre {index:02d}",
            f"Le café numéro {index:02d} et le señor.",
        )
    target = _put(
        client,
        "book-target",
        "Titre à purger",
        "Corps contenant le jeton unique purgeable-xray.",
    )
    target_revision = target["revision"]

    trigger = client.post("/api/index/rebuild", json={"accent_policy": "fold"}).json()
    job_id = trigger["job"]["job_id"]

    # Purge while the rebuild is in building/catching_up state.
    purge_response = client.post(
        "/api/pages/book-target/purge",
        json={"revision": target_revision},
    )
    assert purge_response.status_code == 200, purge_response.text

    # Also purge one of the scanned batch pages after the snapshot likely saw it.
    other = client.get("/api/pages/book-00").json()
    purged_other = client.post(
        "/api/pages/book-00/purge",
        json={"revision": other["revision"]},
    )
    assert purged_other.status_code == 200

    status = _wait_for_terminal(client, job_id)
    assert status["latest_rebuild_job"]["state"] == "succeeded", status
    assert status["active_policy"] == "fold"

    # Neither purged page returns from the new generation.
    assert client.get("/api/pages/book-target").status_code == 404
    assert client.get("/api/pages/book-00").status_code == 404
    assert _search(client, "purgeable")["total"] == 0
    assert _search(client, "xray")["total"] == 0
    cafe_hits = _search(client, "cafe")
    ids = [hit["page_id"] for hit in cafe_hits["hits"]]
    assert "book-target" not in ids
    assert "book-00" not in ids
    assert len(ids) == 11

    # A later rebuild also keeps the purged text absent.
    second = client.post("/api/index/rebuild", json={"accent_policy": "keep"}).json()
    _wait_for_terminal(client, second["job"]["job_id"])
    assert _search(client, "purgeable")["total"] == 0
    assert client.get("/api/pages/book-target").status_code == 404


def test_purge_survives_restart_and_other_pages_stay_operational(client: TestClient):
    created = _put(client, "ocr-restart-001", "Titre éphémère", "Jeton restart-delta café.")
    revision = created["revision"]
    _put(client, "ocr-restart-002", "Titre durable", "Jeton durable-epsilon café.")

    purged = client.post(
        "/api/pages/ocr-restart-001/purge",
        json={"revision": revision},
    )
    assert purged.status_code == 200

    # Simulate a process restart: re-run schema initialisation on the same file.
    from app.db import init_db

    init_db()

    assert client.get("/api/pages/ocr-restart-001").status_code == 404
    assert _search(client, "restart-delta")["total"] == 0
    assert _search(client, "durable-epsilon")["total"] == 1

    # The replay ledger survived the restart as well.
    replay = client.post(
        "/api/pages/ocr-restart-001/purge",
        json={"revision": revision},
    )
    assert replay.status_code == 200
    assert replay.json()["replayed"] is True

    # Other pages can still be corrected and searched.
    correction = _put(
        client,
        "ocr-restart-002",
        "Titre durable corrige",
        "Jeton durable-epsilon cafe revise en entier.",
    )
    assert correction["revision"] > revision
    assert _search(client, "revise")["hits"][0]["page_id"] == "ocr-restart-002"


def test_rewrite_same_id_uses_strictly_larger_revision(client: TestClient):
    first = _put(client, "ocr-rewrite-001", "Un", "Contenu initial delta.")
    purged = client.post(
        "/api/pages/ocr-rewrite-001/purge",
        json={"revision": first["revision"]},
    )
    assert purged.status_code == 200

    second = _put(client, "ocr-rewrite-001", "Deux", "Contenu recree delta.")
    third = _put(client, "ocr-rewrite-001", "Trois", "Contenu encore delta.")
    assert first["revision"] < purged.json()["revision"] < second["revision"] < third["revision"]

    live = client.get("/api/pages/ocr-rewrite-001").json()
    assert live["revision"] == third["revision"]
    assert live["title"] == "Trois"


def test_invalid_revision_payload_rejected(client: TestClient):
    response = client.post(
        "/api/pages/ocr-x/purge",
        json={"revision": 0},
    )
    assert response.status_code == 422

    response = client.post(
        "/api/pages/ocr-x/purge",
        json={"revision": 3, "unexpected": 1},
    )
    assert response.status_code == 422


def test_concurrent_purges_leave_consistent_state(client: TestClient):
    import threading

    created = _put(client, "ocr-concurrent-001", "Titre concours", "Corps concours delta.")
    revision = created["revision"]
    outcomes: list[tuple[int, dict]] = []
    barrier = threading.Barrier(4)

    def purge_once() -> None:
        barrier.wait()
        response = client.post(
            "/api/pages/ocr-concurrent-001/purge",
            json={"revision": revision},
        )
        outcomes.append((response.status_code, response.json()))

    threads = [threading.Thread(target=purge_once) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(outcomes) == 4
    assert all(status == 200 for status, _ in outcomes)
    purge_revisions = {body["revision"] for _, body in outcomes}
    # Only one physical purge happened; every call reports the same result.
    assert len(purge_revisions) == 1
    replayed_count = sum(1 for _, body in outcomes if body.get("replayed"))
    assert replayed_count == 3

    assert client.get("/api/pages/ocr-concurrent-001").status_code == 404
    assert _search(client, "concours")["total"] == 0
    ledger = _db_rows("SELECT * FROM purges WHERE page_id = 'ocr-concurrent-001'")
    assert len(ledger) == 1

    # A purge rolled back on conflict leaves no partial state: after the
    # successful purge the log has exactly one purge row and zero text copies.
    purge_rows = _db_rows(
        "SELECT * FROM changes WHERE page_id = 'ocr-concurrent-001' AND op = 'purge'"
    )
    assert len(purge_rows) == 1
    text_rows = _db_rows(
        "SELECT * FROM changes WHERE page_id = 'ocr-concurrent-001' "
        "AND (title IS NOT NULL OR body IS NOT NULL)"
    )
    assert text_rows == []


def test_legacy_schema_migrates_and_accepts_purge_op(tmp_path: Path):
    configure_database(tmp_path / "legacy.db")
    from app.db import connect, init_db

    init_db()
    # Rewrite the changes table to the pre-purge CHECK constraint.
    conn = connect()
    try:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            """
            CREATE TABLE changes_old (
                seq INTEGER PRIMARY KEY AUTOINCREMENT,
                page_id TEXT NOT NULL,
                page_ref INTEGER,
                op TEXT NOT NULL CHECK (op IN ('upsert', 'delete')),
                title TEXT,
                body TEXT,
                rev INTEGER NOT NULL,
                changed_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO changes_old "
            "SELECT seq, page_id, page_ref, op, title, body, rev, changed_at FROM changes"
        )
        conn.execute("DROP TABLE changes")
        conn.execute("ALTER TABLE changes_old RENAME TO changes")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_changes_seq ON changes(seq)")
        conn.commit()
    finally:
        conn.close()

    # Re-initialising must migrate the constraint without losing history.
    init_db()
    schema_sql = _db_rows(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'changes'"
    )[0]["sql"]
    assert "'purge'" in schema_sql

    from app.archive import upsert_page, purge_page

    written = upsert_page("legacy-001", "Titre legacy", "Corps legacy café.")
    result = purge_page("legacy-001", written["revision"])
    assert result["purged"] is True
    ops = [row["op"] for row in _db_rows("SELECT op FROM changes ORDER BY seq")]
    assert "purge" in ops
