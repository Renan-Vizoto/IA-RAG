import logging
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.extras

from app.infrastructure.configs import settings

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id      VARCHAR(64) PRIMARY KEY,
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    total_tokens    BIGINT NOT NULL DEFAULT 0,
    message_count   INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
);

CREATE TABLE IF NOT EXISTS chat_response_milvus_hits (
    id              SERIAL PRIMARY KEY,
    response_id     VARCHAR(64) NOT NULL REFERENCES chat_responses(response_id),
    milvus_id       VARCHAR(128),
    collection      VARCHAR(64) NOT NULL,
    source          VARCHAR(64),
    distance        FLOAT,
    text_preview    VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_responses_session ON chat_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_milvus_hits_response ON chat_response_milvus_hits(response_id);
"""


class ChatSessionRepository:

    def __init__(self, database_url: str | None = None):
        self._database_url = database_url or settings.CHAT_DATABASE_URL

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

    def init_schema(self):
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(_SCHEMA_SQL)
        logger.info("[CHAT_DB] Schema inicializado.")

    def ensure_session(self, session_id: str):
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
    ):
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_responses (
                        response_id, session_id, model, user_message, answer,
                        input_tokens, output_tokens, total_tokens,
                        response_time_seconds, confidence_score
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        response_id,
                        session_id,
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

    def add_session_tokens(
        self,
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ):
        with self._connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET input_tokens = input_tokens + %s,
                        output_tokens = output_tokens + %s,
                        total_tokens = total_tokens + %s,
                        message_count = message_count + 1,
                        updated_at = now()
                    WHERE session_id = %s
                    """,
                    (input_tokens, output_tokens, total_tokens, session_id),
                )

    def get_session_tokens(self, session_id: str) -> dict[str, int]:
        with self._connection() as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT input_tokens, output_tokens, total_tokens
                    FROM chat_sessions
                    WHERE session_id = %s
                    """,
                    (session_id,),
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
                    SELECT response_id, session_id, model, user_message, answer,
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
