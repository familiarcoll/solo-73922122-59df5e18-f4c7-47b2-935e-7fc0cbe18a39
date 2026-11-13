"""Transactional page writes, permanent purges and snapshot-consistent reading."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .accents import build_match_expression, normalize_for_policy
from .db import (
    active_index,
    begin_immediate,
    connect,
    quote_identifier,
    utc_now,
)


class StaleRevisionError(Exception):
    """The purge request names a revision that no longer matches the live page."""

    def __init__(self, page_id: str, expected_revision: int, current_revision: int | None):
        self.page_id = page_id
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        super().__init__(
            f"page {page_id!r} revision conflict: expected revision "
            f"{expected_revision}, current revision is {current_revision}"
        )


class PageNotFoundError(Exception):
    """The page is absent and no purge ledger entry covers the request."""

    def __init__(self, page_id: str):
        self.page_id = page_id
        super().__init__(f"page {page_id!r} not found")


def _fts_ident(generation: sqlite3.Row) -> str:
    return quote_identifier(str(generation["fts_table"]))


def upsert_page(page_id: str, title: str, body: str) -> dict[str, Any]:
    conn = connect()
    try:
        begin_immediate(conn)
        generation = active_index(conn)
        policy = str(generation["policy"])
        now = utc_now()

        existing = conn.execute(
            "SELECT rowid AS page_ref, rev FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        page_ref = int(existing["page_ref"]) if existing is not None else None
        if existing is not None and int(existing["rev"]) < 1:
            conn.rollback()
            raise RuntimeError(
                f"page {page_id!r} has an invalid stored revision; refusing to overwrite"
            )

        cur = conn.execute(
            """
            INSERT INTO changes(page_id, page_ref, op, title, body, rev, changed_at)
            VALUES (?, ?, 'upsert', ?, ?, 0, ?)
            """,
            (page_id, page_ref, title, body, now),
        )
        seq = int(cur.lastrowid)

        conn.execute(
            """
            INSERT INTO pages(page_id, title, body, rev, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(page_id) DO UPDATE SET
                title = excluded.title,
                body = excluded.body,
                rev = excluded.rev,
                updated_at = excluded.updated_at
            """,
            (page_id, title, body, seq, now),
        )
        if page_ref is None:
            page_ref = int(
                conn.execute(
                    "SELECT rowid AS page_ref FROM pages WHERE page_id = ?", (page_id,)
                ).fetchone()["page_ref"]
            )
            conn.execute(
                "UPDATE generations SET doc_count = doc_count + 1, applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        else:
            conn.execute(
                "UPDATE generations SET applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        conn.execute(
            "UPDATE changes SET rev = ?, page_ref = ? WHERE seq = ?",
            (seq, page_ref, seq),
        )

        ident = _fts_ident(generation)
        conn.execute(
            f"""
            INSERT OR REPLACE INTO {ident}(rowid, page_id, title, body)
            VALUES (?, ?, ?, ?)
            """,
            (
                page_ref,
                page_id,
                normalize_for_policy(title, policy),
                normalize_for_policy(body, policy),
            ),
        )
        conn.commit()
        return {
            "page_id": page_id,
            "revision": seq,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "indexed": True,
            "upserted": True,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_page(page_id: str) -> dict[str, Any]:
    conn = connect()
    try:
        begin_immediate(conn)
        generation = active_index(conn)
        policy = str(generation["policy"])
        now = utc_now()

        existing = conn.execute(
            "SELECT rowid AS page_ref FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        page_ref = int(existing["page_ref"]) if existing is not None else None

        cur = conn.execute(
            """
            INSERT INTO changes(page_id, page_ref, op, title, body, rev, changed_at)
            VALUES (?, ?, 'delete', NULL, NULL, 0, ?)
            """,
            (page_id, page_ref, now),
        )
        seq = int(cur.lastrowid)
        conn.execute(
            "UPDATE changes SET rev = ?, page_ref = ? WHERE seq = ?",
            (seq, page_ref, seq),
        )

        conn.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        if page_ref is not None:
            conn.execute(
                "UPDATE generations SET doc_count = doc_count - 1, applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )
        else:
            conn.execute(
                "UPDATE generations SET applied_seq = ? WHERE id = ?",
                (seq, int(generation["id"])),
            )

        ident = _fts_ident(generation)
        if page_ref is not None:
            conn.execute(f"DELETE FROM {ident} WHERE rowid = ?", (page_ref,))
        else:
            # Idempotent delete of an absent page.  This fallback also cleans
            # up any pre-existing orphan from an older schema version.
            conn.execute(f"DELETE FROM {ident} WHERE page_id = ?", (page_id,))

        conn.commit()
        return {
            "page_id": page_id,
            "revision": seq,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "deleted": True,
            "existed": existing is not None,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def purge_page(page_id: str, expected_revision: int) -> dict[str, Any]:
    """Permanently purge a page and every stored copy of its historical text.

    The whole purge is one write transaction, so under either API worker
    process it commits or rolls back atomically:

    * a ``purge`` row is appended to the durable change log so a rebuild that
      already snapshotted the page can never write the old text back;
    * the title/body of every earlier change-log row for the page are cleared
      while the sequence numbers and other metadata are retained;
    * the canonical page row is deleted;
    * the page is removed from every index generation's FTS table, including
      generations still under construction;
    * an idempotency ledger stores the original result keyed by the
      ``(page_id, expected_revision)`` pair.  Replaying the identical request
      returns that stored result and never touches a page later recreated with
      the same ID.

    Only SQLite logical records are affected; no disk-level forensic erasure is
    attempted.
    """

    conn = connect()
    try:
        begin_immediate(conn)
        generation = active_index(conn)
        policy = str(generation["policy"])
        now = utc_now()

        ledger_row = conn.execute(
            "SELECT result_json FROM purges WHERE page_id = ? AND expected_rev = ?",
            (page_id, expected_revision),
        ).fetchone()
        if ledger_row is not None:
            conn.commit()
            result = json.loads(str(ledger_row["result_json"]))
            result["replayed"] = True
            return result

        existing = conn.execute(
            "SELECT rowid AS page_ref, rev FROM pages WHERE page_id = ?", (page_id,)
        ).fetchone()
        if existing is None:
            conn.rollback()
            raise PageNotFoundError(page_id)

        current_revision = int(existing["rev"])
        if current_revision != expected_revision:
            conn.rollback()
            raise StaleRevisionError(page_id, expected_revision, current_revision)

        page_ref = int(existing["page_ref"])

        # 1. Durable purge marker.  Rebuild workers replay this as a delete and
        #    therefore cannot resurrect the page from an older snapshot.
        cur = conn.execute(
            """
            INSERT INTO changes(page_id, page_ref, op, title, body, rev, changed_at)
            VALUES (?, ?, 'purge', NULL, NULL, 0, ?)
            """,
            (page_id, page_ref, now),
        )
        purge_seq = int(cur.lastrowid)
        conn.execute(
            "UPDATE changes SET rev = ?, page_ref = ? WHERE seq = ?",
            (purge_seq, page_ref, purge_seq),
        )

        # 2. Scrub the historical text from every earlier log row.  Sequence
        #    numbers, page ids, op kinds, revisions and timestamps are retained;
        #    only the title/body copies are cleared.  A rebuild reading one of
        #    these rows later treats it as a no-op instead of re-indexing text.
        conn.execute(
            """
            UPDATE changes
               SET title = NULL, body = NULL
             WHERE page_id = ? AND seq < ?
            """,
            (page_id, purge_seq),
        )

        # 3. Delete the canonical logical record.
        conn.execute("DELETE FROM pages WHERE page_id = ?", (page_id,))
        conn.execute(
            "UPDATE generations SET doc_count = doc_count - 1, applied_seq = ? WHERE id = ?",
            (purge_seq, int(generation["id"])),
        )

        # 4. Remove the full-text copy from every generation, including any
        #    index currently being built by the background rebuild.
        fts_tables = conn.execute(
            "SELECT fts_table FROM generations WHERE fts_table <> ''"
        ).fetchall()
        for fts_row in fts_tables:
            ident = quote_identifier(str(fts_row["fts_table"]))
            conn.execute(f"DELETE FROM {ident} WHERE rowid = ?", (page_ref,))
            conn.execute(f"DELETE FROM {ident} WHERE page_id = ?", (page_id,))

        result: dict[str, Any] = {
            "page_id": page_id,
            "revision": purge_seq,
            "expected_revision": expected_revision,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "purged": True,
            "replayed": False,
        }

        # 5. Idempotency ledger: replaying the same (page id, seen revision)
        #    request returns this result without deleting a newer page.
        conn.execute(
            """
            INSERT INTO purges
                (page_id, expected_rev, purge_seq, revision, result_json, purged_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                page_id,
                expected_revision,
                purge_seq,
                purge_seq,
                json.dumps(result),
                now,
            ),
        )
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def search(raw_query: str, *, limit: int, offset: int) -> dict[str, Any]:
    conn = connect()
    try:
        # A single read transaction pins both the active-generation pointer and
        # the page rows.  Hits and returned bodies therefore come from the same
        # committed database state, even if a rebuild swap commits concurrently.
        conn.execute("BEGIN")
        generation = active_index(conn)
        policy = str(generation["policy"])
        expression, terms = build_match_expression(raw_query, policy)
        ident = _fts_ident(generation)

        total = int(
            conn.execute(
                f"""
                SELECT COUNT(*)
                  FROM {ident} AS f
                  JOIN pages AS p ON p.page_id = f.page_id
                 WHERE {ident} MATCH ?
                """,
                (expression,),
            ).fetchone()[0]
        )
        rows = conn.execute(
            f"""
            SELECT p.page_id, p.title, p.body, p.rev, p.updated_at,
                   bm25({ident}) AS score
              FROM {ident} AS f
              JOIN pages AS p ON p.page_id = f.page_id
             WHERE {ident} MATCH ?
             ORDER BY score ASC, p.page_id ASC
             LIMIT ? OFFSET ?
            """,
            (expression, limit, offset),
        ).fetchall()
        conn.execute("COMMIT")
        return {
            "query": raw_query,
            "query_terms": terms,
            "accent_policy": policy,
            "generation_id": int(generation["id"]),
            "total": total,
            "limit": limit,
            "offset": offset,
            "hits": [
                {
                    "page_id": row["page_id"],
                    "title": row["title"],
                    "body": row["body"],
                    "revision": int(row["rev"]),
                    "updated_at": row["updated_at"],
                    "score": float(row["score"]),
                }
                for row in rows
            ],
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_page(page_id: str) -> dict[str, Any] | None:
    conn = connect()
    try:
        conn.execute("BEGIN")
        generation = active_index(conn)
        row = conn.execute(
            "SELECT page_id, title, body, rev, updated_at FROM pages WHERE page_id = ?",
            (page_id,),
        ).fetchone()
        conn.execute("COMMIT")
        if row is None:
            return None
        return {
            "page_id": row["page_id"],
            "title": row["title"],
            "body": row["body"],
            "revision": int(row["rev"]),
            "updated_at": row["updated_at"],
            "accent_policy": str(generation["policy"]),
            "generation_id": int(generation["id"]),
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
