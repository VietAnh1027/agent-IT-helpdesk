"""
migrate.py — Tạo toàn bộ schema Postgres cho IT Support Ticket AI Agent
theo 03_DATA_MODEL.md, 05_CATEGORY_GRADUATION.md, 06_TICKET_STATUS_LIFECYCLE.md
và 07_TOOLS.md.

Yêu cầu:
    pip install psycopg2-binary python-dotenv

.env cần các biến (xem .env.example đi kèm):
    DB_HOST=localhost
    DB_PORT=5432
    DB_NAME=it_ticket_agent
    DB_USER=postgres
    DB_PASSWORD=your_password
    EMBEDDING_DIM=768        # số chiều embedding, tùy model bạn chọn

Chạy:
    python migrate.py

Script idempotent — chạy lại nhiều lần không lỗi, không tạo trùng bảng/constraint.
"""

import os
import sys

import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}

EMBEDDING_DIM = os.getenv("EMBEDDING_DIM", "768")

REQUIRED_ENV = ["DB_NAME", "DB_USER", "DB_PASSWORD"]


def check_env():
    missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
    if missing:
        print(f"[LỖI] Thiếu biến môi trường trong .env: {', '.join(missing)}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Từng statement được tách riêng để dễ log/debug nếu 1 bước lỗi giữa chừng.
# Thứ tự bắt buộc theo phụ thuộc khóa ngoại (xem ghi chú cạnh từng bảng).
# ---------------------------------------------------------------------------

# list[tuple[str, str]]: [("Tên bước", "CREATE TABLE ...;"), ...].
# Mỗi tuple là nhãn log và một chuỗi SQL, không phải record dữ liệu trong DB.
STATEMENTS = [
    # --- Extensions ---
    ("Bật extension pgvector", """
        CREATE EXTENSION IF NOT EXISTS vector;
    """),
    ("Bật extension pgcrypto (cho gen_random_uuid)", """
        CREATE EXTENSION IF NOT EXISTS pgcrypto;
    """),

    # --- kb_articles (không phụ thuộc bảng nào) ---
    ("Tạo bảng kb_articles", """
        CREATE TABLE IF NOT EXISTS kb_articles (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            title               TEXT NOT NULL,
            category            TEXT NOT NULL,
            content             TEXT NOT NULL,
            content_hash        TEXT NOT NULL,
            date_added          TIMESTAMPTZ NOT NULL DEFAULT now(),  -- ngày record được thêm vào hệ thống, set 1 lần, không đổi sau
            date_created        DATE NOT NULL,                        -- ngày tài liệu THẬT SỰ được soạn, đọc từ metadata đầu PDF
            last_updated        TIMESTAMPTZ NOT NULL DEFAULT now(),
            source              TEXT NOT NULL,
            is_active           BOOLEAN NOT NULL DEFAULT true
        );
    """),
    ("Tạo index kb_articles.content_hash (dùng cho skip-re-embed check)", """
        CREATE INDEX IF NOT EXISTS idx_kb_articles_content_hash
            ON kb_articles (content_hash);
    """),

    # --- kb_chunks (gộp parent-child, self-referencing qua parent_chunk_id) ---
    # parent_chunk_id NULL       => row này là PARENT (embedding phải NULL)
    # parent_chunk_id NOT NULL   => row này là CHILD  (embedding phải NOT NULL)
    # CHECK constraint enforce bất biến này ngay ở DB, không tin ở tầng code.
    (f"Tạo bảng kb_chunks (embedding vector({EMBEDDING_DIM}), self-referencing parent-child)", f"""
        CREATE TABLE IF NOT EXISTS kb_chunks (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            kb_article_id       UUID NOT NULL REFERENCES kb_articles(id) ON DELETE CASCADE,
            parent_chunk_id     UUID REFERENCES kb_chunks(id) ON DELETE CASCADE,
            chunk_index         INT NOT NULL,
            content             TEXT NOT NULL,
            embedding           VECTOR({EMBEDDING_DIM}),
            category            TEXT NOT NULL,   -- denormalize từ kb_articles, dùng filter trước similarity
            is_active           BOOLEAN NOT NULL DEFAULT true,  -- denormalize từ kb_articles
            CONSTRAINT chk_parent_or_child CHECK (
                (parent_chunk_id IS NULL AND embedding IS NULL)
                OR
                (parent_chunk_id IS NOT NULL AND embedding IS NOT NULL)
            )
        );
    """),
    ("Tạo partial index ANN cho kb_chunks.embedding (chỉ child row)", """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_embedding
            ON kb_chunks USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100)
            WHERE parent_chunk_id IS NOT NULL;
    """),
    ("Tạo index kb_chunks.parent_chunk_id (tra parent nhanh sau khi search)", """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_parent_id
            ON kb_chunks (parent_chunk_id);
    """),
    ("Tạo index kb_chunks(category, is_active) cho hybrid filter trước similarity", """
        CREATE INDEX IF NOT EXISTS idx_kb_chunks_category_active
            ON kb_chunks (category, is_active)
            WHERE parent_chunk_id IS NOT NULL;
    """),

    # --- tickets (KHÔNG thêm FK tới incidents ở đây vì incidents chưa tồn tại —
    #     FK đó được thêm ở bước ALTER TABLE bên dưới, sau khi tạo incidents.
    #     external_ticket_id + provider thay thế hoàn toàn parent_thread_id:
    #     nhiều row có thể cùng external_ticket_id, mỗi row là 1 lần AI xử lý,
    #     KHÔNG update status của row cũ khi có event mới — luôn insert row mới) ---
    (f"Tạo bảng tickets (embedding vector({EMBEDDING_DIM}))", f"""
        CREATE TABLE IF NOT EXISTS tickets (
            id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            subject                 TEXT NOT NULL,
            body                    TEXT NOT NULL,
            requester_email         TEXT NOT NULL,
            created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
            category                TEXT,
            status                  TEXT NOT NULL DEFAULT 'new'
                CHECK (status IN (
                    'new', 'classified', 'pending_review',
                    'approved', 'auto_sent', 'awaiting_clarification',
                    'linked_to_incident', 'escalated', 'closed'
                )),
            confidence_score        FLOAT,
            clarification_rounds    INT NOT NULL DEFAULT 0,
            external_ticket_id      TEXT NOT NULL,
            provider                TEXT NOT NULL,
            embedding               VECTOR({EMBEDDING_DIM}),
            incident_id             UUID,  -- FK thêm sau, xem ALTER TABLE bên dưới
            recurring_issue         BOOLEAN NOT NULL DEFAULT false,
            sentiment_flag          TEXT CHECK (sentiment_flag IN ('neutral', 'frustrated', 'angry')),
            multi_issue_suspected   BOOLEAN NOT NULL DEFAULT false
        );
    """),
    ("Tạo index ANN cho tickets.embedding", """
        CREATE INDEX IF NOT EXISTS idx_tickets_embedding
            ON tickets USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
    """),
    ("Tạo index tickets.requester_email + created_at (dùng cho repeat/spam check)", """
        CREATE INDEX IF NOT EXISTS idx_tickets_requester_created
            ON tickets (requester_email, created_at DESC);
    """),
    ("Tạo index tickets.external_ticket_id + provider + created_at (tra row mới nhất theo ticket thật)", """
        CREATE INDEX IF NOT EXISTS idx_tickets_external_id
            ON tickets (external_ticket_id, provider, created_at DESC);
    """),

    # --- incidents (phụ thuộc tickets, giờ đã tồn tại) ---
    ("Tạo bảng incidents", """
        CREATE TABLE IF NOT EXISTS incidents (
            id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            representative_ticket_id    UUID NOT NULL REFERENCES tickets(id) ON DELETE RESTRICT,
            category                    TEXT NOT NULL,
            ticket_count                INT NOT NULL DEFAULT 1,
            status                      TEXT NOT NULL DEFAULT 'open'
                CHECK (status IN ('open', 'acknowledged', 'resolved')),
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),

    # --- Thêm FK còn thiếu của tickets.incident_id, giờ incidents đã tồn tại ---
    # Dùng DO block để idempotent, vì Postgres không hỗ trợ
    # "ADD CONSTRAINT IF NOT EXISTS" trực tiếp.
    ("Thêm khóa ngoại tickets.incident_id -> incidents.id", """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_constraint WHERE conname = 'fk_tickets_incident'
            ) THEN
                ALTER TABLE tickets
                    ADD CONSTRAINT fk_tickets_incident
                    FOREIGN KEY (incident_id) REFERENCES incidents(id) ON DELETE SET NULL;
            END IF;
        END $$;
    """),

    # --- ai_responses (phụ thuộc tickets) ---
    ("Tạo bảng ai_responses", """
        CREATE TABLE IF NOT EXISTS ai_responses (
            id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ticket_id                   UUID NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
            draft_text                  TEXT NOT NULL,
            cited_kb_chunk_ids          UUID[] NOT NULL DEFAULT '{}',  -- trỏ tới kb_chunks.id (chỉ child row)
            retrieval_top_k_scores      JSONB,
            confidence_score            FLOAT NOT NULL,
            model_used                  TEXT NOT NULL,
            latency_ms                  INT,
            created_at                  TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),
    ("Tạo index ai_responses.ticket_id", """
        CREATE INDEX IF NOT EXISTS idx_ai_responses_ticket_id
            ON ai_responses (ticket_id);
    """),

    # --- review_actions (phụ thuộc ai_responses) ---
    ("Tạo bảng review_actions", """
        CREATE TABLE IF NOT EXISTS review_actions (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ai_response_id      UUID NOT NULL REFERENCES ai_responses(id) ON DELETE CASCADE,
            action              TEXT NOT NULL
                CHECK (action IN (
                    'approved_as_is', 'approved_with_edit', 'rejected',
                    'escalated', 'broadcast_from_incident'
                )),
            final_text          TEXT,
            reviewer_note       TEXT,
            reviewed_at         TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),

    # --- eval_set (phụ thuộc kb_articles, nullable) ---
    ("Tạo bảng eval_set", """
        CREATE TABLE IF NOT EXISTS eval_set (
            id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ticket_text                 TEXT NOT NULL,
            expected_category           TEXT NOT NULL,
            expected_kb_article_id      UUID REFERENCES kb_articles(id) ON DELETE SET NULL,
            golden_answer                TEXT NOT NULL,
            difficulty                  TEXT NOT NULL
                CHECK (difficulty IN ('easy', 'medium', 'hard', 'adversarial'))
        );
    """),

    # --- webhook_events (độc lập) ---
    ("Tạo bảng webhook_events", """
        CREATE TABLE IF NOT EXISTS webhook_events (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            provider            TEXT NOT NULL,
            external_event_id   TEXT NOT NULL,
            received_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
            processed           BOOLEAN NOT NULL DEFAULT false,
            UNIQUE (provider, external_event_id)
        );
    """),

    # --- category_automation_state (độc lập, category là PK) ---
    ("Tạo bảng category_automation_state", """
        CREATE TABLE IF NOT EXISTS category_automation_state (
            category                TEXT PRIMARY KEY,
            stage                   TEXT NOT NULL DEFAULT 'shadow'
                CHECK (stage IN ('shadow', 'draft', 'auto')),
            auto_enabled            BOOLEAN NOT NULL DEFAULT false,
            stage_entered_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_evaluated_at       TIMESTAMPTZ,
            last_demotion_reason    TEXT,
            manual_override         BOOLEAN NOT NULL DEFAULT false
        );
    """),

    # --- category_metric_snapshots (phụ thuộc lỏng vào category_automation_state,
    #     không đặt FK cứng để tránh phải seed category trước khi ghi log đầu tiên) ---
    ("Tạo bảng category_metric_snapshots", """
        CREATE TABLE IF NOT EXISTS category_metric_snapshots (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            category             TEXT NOT NULL,
            snapshot_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
            sample_size          INT NOT NULL,
            approval_rate        FLOAT,
            hallucination_rate   FLOAT NOT NULL,
            reopened_rate        FLOAT NOT NULL,
            stage_at_snapshot    TEXT NOT NULL
        );
    """),
    ("Tạo index category_metric_snapshots.category + snapshot_at", """
        CREATE INDEX IF NOT EXISTS idx_category_metric_snapshots_cat_time
            ON category_metric_snapshots (category, snapshot_at DESC);
    """),

    # -----------------------------------------------------------------
    # Tool-use (07_TOOLS.md): mock IT backend + log + trust state
    # -----------------------------------------------------------------

    # --- Mock backend tables (độc lập, mô phỏng hệ thống IT thật) ---
    ("Tạo bảng mock_vpn_gateways", """
        CREATE TABLE IF NOT EXISTS mock_vpn_gateways (
            id                  TEXT PRIMARY KEY,
            status              TEXT NOT NULL DEFAULT 'up'
                CHECK (status IN ('up', 'down', 'maintenance')),
            last_incident_at    TIMESTAMPTZ
        );
    """),
    ("Tạo bảng mock_printers", """
        CREATE TABLE IF NOT EXISTS mock_printers (
            id                  TEXT PRIMARY KEY,
            status              TEXT NOT NULL DEFAULT 'online'
                CHECK (status IN ('online', 'offline', 'jammed', 'low_toner')),
            queue_length        INT NOT NULL DEFAULT 0,
            last_error          TEXT,
            updated_at          TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),
    ("Tạo bảng mock_software_licenses", """
        CREATE TABLE IF NOT EXISTS mock_software_licenses (
            software_name       TEXT PRIMARY KEY,
            total_seats          INT NOT NULL,
            used_seats           INT NOT NULL DEFAULT 0
        );
    """),
    ("Tạo bảng mock_mailboxes", """
        CREATE TABLE IF NOT EXISTS mock_mailboxes (
            email               TEXT PRIMARY KEY,
            last_synced_at      TIMESTAMPTZ,
            sync_status         TEXT NOT NULL DEFAULT 'ok'
                CHECK (sync_status IN ('ok', 'delayed', 'failed'))
        );
    """),

    # --- tool_execution_log (phụ thuộc tickets, ai_responses) ---
    ("Tạo bảng tool_execution_log", """
        CREATE TABLE IF NOT EXISTS tool_execution_log (
            id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            ticket_id           UUID NOT NULL REFERENCES tickets(id) ON DELETE CASCADE,
            ai_response_id      UUID REFERENCES ai_responses(id) ON DELETE SET NULL,
            tool_name           TEXT NOT NULL,
            tier                INT NOT NULL CHECK (tier IN (0, 1)),
            input_params        JSONB NOT NULL,
            output_result       JSONB,
            status              TEXT NOT NULL
                CHECK (status IN ('success', 'failed', 'blocked')),
            blocked_reason      TEXT,
            backend_type        TEXT NOT NULL CHECK (backend_type IN ('real', 'mock')),
            called_at           TIMESTAMPTZ NOT NULL DEFAULT now()
        );
    """),
    ("Tạo index tool_execution_log.ticket_id", """
        CREATE INDEX IF NOT EXISTS idx_tool_execution_log_ticket_id
            ON tool_execution_log (ticket_id);
    """),
    ("Tạo index tool_execution_log.tool_name + called_at (dùng cho job đánh giá trust)", """
        CREATE INDEX IF NOT EXISTS idx_tool_execution_log_tool_time
            ON tool_execution_log (tool_name, called_at DESC);
    """),

    # --- tool_trust_state (độc lập, tool_name là PK) ---
    ("Tạo bảng tool_trust_state", """
        CREATE TABLE IF NOT EXISTS tool_trust_state (
            tool_name            TEXT PRIMARY KEY,
            mode                 TEXT NOT NULL DEFAULT 'propose_only'
                CHECK (mode IN ('propose_only', 'auto_execute')),
            auto_execute_enabled BOOLEAN NOT NULL DEFAULT false,
            stage_entered_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            last_evaluated_at    TIMESTAMPTZ,
            manual_override      BOOLEAN NOT NULL DEFAULT false
        );
    """),
]


def run_migration():
    check_env()
    print(f"Kết nối tới database '{DB_CONFIG['dbname']}' @ {DB_CONFIG['host']}:{DB_CONFIG['port']} ...")

    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            for description, sql in STATEMENTS:
                print(f"  -> {description}")
                cur.execute(sql)
        conn.commit()
        print("\n✅ Migration hoàn tất, toàn bộ schema đã sẵn sàng.")
    except Exception as e:
        conn.rollback()
        print(f"\n❌ Migration thất bại, đã rollback. Lỗi: {e}")
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()
