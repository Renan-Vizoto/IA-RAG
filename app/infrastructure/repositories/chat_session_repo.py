import logging
import uuid
from contextlib import contextmanager
from typing import Any
from urllib.parse import urlparse, urlunparse

import psycopg2
import psycopg2.extras
from psycopg2 import sql

from app.infrastructure.configs import settings

logger = logging.getLogger(__name__)

_SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id      VARCHAR(64) PRIMARY KEY,
        input_tokens    BIGINT NOT NULL DEFAULT 0,
        output_tokens   BIGINT NOT NULL DEFAULT 0,
        total_tokens    BIGINT NOT NULL DEFAULT 0,
        message_count   INT NOT NULL DEFAULT 0,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chats (
        chat_id         VARCHAR(64) PRIMARY KEY,
        session_id      VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id),
        title           VARCHAR(255) NOT NULL DEFAULT 'New chat',
        input_tokens    BIGINT NOT NULL DEFAULT 0,
        output_tokens   BIGINT NOT NULL DEFAULT 0,
        total_tokens    BIGINT NOT NULL DEFAULT 0,
        message_count   INT NOT NULL DEFAULT 0,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_messages (
        message_id      VARCHAR(64) PRIMARY KEY,
        chat_id         VARCHAR(64) NOT NULL REFERENCES chats(chat_id),
        role            VARCHAR(16) NOT NULL,
        content         TEXT NOT NULL,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_responses (
        response_id     VARCHAR(64) PRIMARY KEY,
        session_id      VARCHAR(64) NOT NULL REFERENCES chat_sessions(session_id),
        model           VARCHAR(128) NOT NULL,
        user_message    TEXT NOT NULL,
        answer          TEXT NOT NULL,
        input_tokens    INT,
        output_tokens   INT,
        total_tokens    INT,
        response_time_seconds FLOAT,
        confidence_score FLOAT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chat_response_milvus_hits (
        id              SERIAL PRIMARY KEY,
        response_id     VARCHAR(64) NOT NULL REFERENCES chat_responses(response_id),
        milvus_id       VARCHAR(128),
        collection      VARCHAR(64) NOT NULL,
        source          VARCHAR(64),
        distance        FLOAT,
        text_preview    VARCHAR(500)
    )
    """,
]

_MIGRATION_STATEMENTS = [
    "ALTER TABLE chat_responses ADD COLUMN IF NOT EXISTS chat_id VARCHAR(64)",
]

_INDEX_STATEMENTS = [
    "CREATE INDEX IF NOT EXISTS idx_chats_session ON chats(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_messages_chat ON chat_messages(chat_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_responses_session ON chat_responses(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_responses_chat ON chat_responses(chat_id)",
    "CREATE INDEX IF NOT EXISTS idx_milvus_hits_response ON chat_response_milvus_hits(response_id)",
]

_LEGACY_BACKFILL_SQL = """
SELECT DISTINCT cs.session_id,
       cs.input_tokens, cs.output_tokens, cs.total_tokens, cs.message_count
FROM chat_sessions cs
WHERE EXISTS (
    SELECT 1 FROM chat_responses cr WHERE cr.session_id = cs.session_id
)
AND NOT EXISTS (
    SELECT 1 FROM chats c WHERE c.session_id = cs.session_id
)
"""


class ChatSessionRepository:

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or settings.CHAT_DATABASE_URL
        self._schema_ready = False

    def _database_name(self) -> str:
        parsed = urlparse(self._database_url.replace("postgresql://", "postgres://", 1))
        return (parsed.path or "").lstrip("/")

    def _admin_database_url(self) -> str:
        parsed = urlparse(self._database_url.replace("postgresql://", "postgres://", 1))
        db_name = self._database_name()
        admin_db = "mlflow" if db_name != "mlflow" else "postgres"
        return urlunparse(parsed._replace(path=f"/{admin_db}"))

    def _ensure_database_exists(self) -> None:
        db_name = self._database_name()
        if not db_name:
            raise ValueError("CHAT_DATABASE_URL sem nome de database.")

        conn = psycopg2.connect(self._admin_database_url())
        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,),
                )
                if cur.fetchone() is None:
                    cur.execute(
                        sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
                    )
                    logger.info("[CHAT_DB] Database '%s' criado.", db_name)
        finally:
            conn.close()

    @contextmanager
    def _connection(self):
        conn = psycopg2.connect(self._database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _migrate_legacy_sessions(self, cur) -> None:
        cur.execute(_LEGACY_BACKFILL_SQL)
        rows = cur.fetchall()
        for row in rows:
            session_id = row[0]
            chat_id = str(uuid.uuid4())
            cur.execute(
                """
                INSERT INTO chats (
                    chat_id, session_id, title,
                    input_tokens, output_tokens, total_tokens, message_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chat_id,
                    session_id,
                    "Imported chat",
                    row[1] or 0,
                    row[2] or 0,
                    row[3] or 0,
                    row[4] or 0,
                ),
            )
            cur.execute(
                """
                UPDATE chat_responses SET chat_id = %s
                WHERE session_id = %s AND chat_id IS NULL
                """,
                (chat_id, session_id),
            )
            cur.execute(
                """
                SELECT user_message, answer, created_at
                FROM chat_responses
                WHERE chat_id = %s
                ORDER BY created_at
                """,
                (chat_id,),
            )
            for user_msg, answer, _ in cur.fetchall():
                cur.execute(
                    """
                    INSERT INTO chat_messages (message_id, chat_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), chat_id, "user", user_msg),
                )
                cur.execute(
                    """
                    INSERT INTO chat_messages (message_id, chat_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (str(uuid.uuid4()), chat_id, "assistant", answer),
                )
            logger.info(
                "[CHAT_DB] Sessão legada '%s' migrada para chat '%s'.",
                session_id,
                chat_id,
            )

    def _apply_schema(self) -> None:
        self._ensure_database_exists()
        with self._connection() as conn:
            with conn.cursor() as cur:
                for statement in _SCHEMA_STATEMENTS:
                    cur.execute(statement)
                for statement in _MIGRATION_STATEMENTS:
                    cur.execute(statement)
                for statement in _INDEX_STATEMENTS:
                    cur.execute(statement)
                try:
                    self._migrate_legacy_sessions(cur)
                except Exception as exc:
                    logger.warning("[CHAT_DB] Migração legada ignorada: %s", exc)

    def ensure_schema(self) -> None:
        if self._schema_ready:
            return
        self._apply_schema()
        self._schema_ready = True
        logger.info("[CHAT_DB] Schema garantido.")

    def init_schema(self):
        self._apply_schema()
        self._schema_ready = True
        logger.info("[CHAT_DB] Schema inicializado.")

    def ensure_session(self, session_id: str):
        self.ensure_schema()
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (session_id)
                    VALUES (%s)
                    ON CONFLICT (session_id) DO NOTHING
                    """,
                    (session_id,),
                )

    def create_chat(
        self,
        session_id: str,
        chat_id: str,
        title: str = "New chat",
    ) -> dict[str, Any]:
        self.ensure_session(session_id)
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    INSERT INTO chats (chat_id, session_id, title)
                    VALUES (%s, %s, %s)
                    RETURNING chat_id, session_id, title,
                              input_tokens, output_tokens, total_tokens,
                              message_count, created_at, updated_at
                    """,
                    (chat_id, session_id, title),
                )
                return dict(cur.fetchone())

    def list_chats(self, session_id: str) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chat_id, session_id, title,
                           input_tokens, output_tokens, total_tokens,
                           message_count, created_at, updated_at
                    FROM chats
                    WHERE session_id = %s
                    ORDER BY updated_at DESC
                    """,
                    (session_id,),
                )
                return [dict(row) for row in cur.fetchall()]

    def get_chat(self, chat_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT chat_id, session_id, title,
                           input_tokens, output_tokens, total_tokens,
                           message_count, created_at, updated_at
                    FROM chats
                    WHERE chat_id = %s
                    """,
                    (chat_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def ensure_chat(self, session_id: str, chat_id: str | None = None) -> str:
        """Return an existing chat for the session or create one when missing."""
        self.ensure_schema()
        self.ensure_session(session_id)

        if chat_id:
            chat = self.get_chat(chat_id)
            if chat and chat["session_id"] == session_id:
                return chat_id
            if chat and chat["session_id"] != session_id:
                chat_id = str(uuid.uuid4())
            else:
                self.create_chat(session_id, chat_id)
                return chat_id

        chat_id = str(uuid.uuid4())
        self.create_chat(session_id, chat_id)
        return chat_id

    def update_chat_title(self, chat_id: str, title: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET title = %s, updated_at = now() WHERE chat_id = %s",
                    (title[:255], chat_id),
                )

    def touch_chat(self, chat_id: str) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE chats SET updated_at = now() WHERE chat_id = %s",
                    (chat_id,),
                )

    def save_message(
        self,
        message_id: str,
        chat_id: str,
        role: str,
        content: str,
    ) -> None:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (message_id, chat_id, role, content)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (message_id, chat_id, role, content),
                )

    def get_messages(
        self,
        chat_id: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                sql_query = """
                    SELECT message_id, chat_id, role, content, created_at
                    FROM chat_messages
                    WHERE chat_id = %s
                    ORDER BY created_at ASC
                """
                if limit is not None:
                    sql_query += " LIMIT %s"
                    cur.execute(sql_query, (chat_id, limit))
                else:
                    cur.execute(sql_query, (chat_id,))
                return [dict(row) for row in cur.fetchall()]

    def count_messages(self, chat_id: str) -> int:
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM chat_messages WHERE chat_id = %s",
                    (chat_id,),
                )
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def save_response(
        self,
        response_id: str,
        session_id: str,
        model: str,
        user_message: str,
        answer: str,
        input_tokens: int | None,
        output_tokens: int | None,
        total_tokens: int | None,
        response_time_seconds: float,
        confidence_score: float,
        chat_id: str | None = None,
    ):
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_responses (
                        response_id, session_id, chat_id, model, user_message, answer,
                        input_tokens, output_tokens, total_tokens,
                        response_time_seconds, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        response_id,
                        session_id,
                        chat_id,
                        model,
                        user_message,
                        answer,
                        input_tokens,
                        output_tokens,
                        total_tokens,
                        response_time_seconds,
                        confidence_score,
                    ),
                )

    def save_milvus_hits(self, response_id: str, hits: list[dict]):
        if not hits:
            return
        with self._connection() as conn:
            with conn.cursor() as cur:
                for hit in hits:
                    text = hit.get("text", "") or ""
                    cur.execute(
                        """
                        INSERT INTO chat_response_milvus_hits (
                            response_id, milvus_id, collection, source, distance, text_preview
                        ) VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            response_id,
                            hit.get("id") or None,
                            hit.get("collection", "governance"),
                            hit.get("source"),
                            hit.get("distance"),
                            text[:500],
                        ),
                    )

    def add_chat_tokens(
        self,
        chat_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ):
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chats
                    SET input_tokens = input_tokens + %s,
                        output_tokens = output_tokens + %s,
                        total_tokens = total_tokens + %s,
                        message_count = message_count + 1,
                        updated_at = now()
                    WHERE chat_id = %s
                    """,
                    (input_tokens, output_tokens, total_tokens, chat_id),
                )

    def get_chat_tokens(self, chat_id: str) -> dict[str, int]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT input_tokens, output_tokens, total_tokens
                    FROM chats
                    WHERE chat_id = %s
                    """,
                    (chat_id,),
                )
                row = cur.fetchone()
                if not row:
                    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                return {
                    "input_tokens": int(row["input_tokens"]),
                    "output_tokens": int(row["output_tokens"]),
                    "total_tokens": int(row["total_tokens"]),
                }

    def get_response(self, response_id: str) -> dict[str, Any] | None:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT response_id, session_id, chat_id, model, user_message, answer,
                           input_tokens, output_tokens, total_tokens,
                           response_time_seconds, confidence_score, created_at
                    FROM chat_responses
                    WHERE response_id = %s
                    """,
                    (response_id,),
                )
                response = cur.fetchone()
                if not response:
                    return None

                cur.execute(
                    """
                    SELECT milvus_id, collection, source, distance, text_preview
                    FROM chat_response_milvus_hits
                    WHERE response_id = %s
                    ORDER BY id
                    """,
                    (response_id,),
                )
                hits = cur.fetchall()
                return {
                    **dict(response),
                    "milvus_hits": [dict(h) for h in hits],
                }
