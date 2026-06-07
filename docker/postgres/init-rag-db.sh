#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    SELECT 'CREATE DATABASE rag'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'rag')\gexec
EOSQL

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "rag" <<-EOSQL
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
EOSQL
