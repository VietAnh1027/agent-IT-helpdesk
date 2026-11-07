"""Chạy lại riêng judge trên answer/context đã lưu; không sửa report gốc.

python -m eval.harness.diagnose_groundedness --report eval/results/report.json --ticket-id TC_SYNC_001
"""

import argparse
from datetime import datetime
import json
from pathlib import Path

from src.confidence import evaluate_groundedness


def replay(report_path, output, ticket_ids=None):
    """Ghi JSONL riêng, mỗi dòng là một ticket; flush để giữ ticket đã hoàn tất."""
    report_path, output = Path(report_path), Path(output)
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    tickets = report["results"]
    if ticket_ids:
        missing = set(ticket_ids) - {t.get("ticket_id", t.get("id")) for t in tickets}
        if missing:
            raise ValueError(f"Không có ticket: {sorted(missing)}")
        tickets = [t for t in tickets if t.get("ticket_id", t.get("id")) in ticket_ids]
    if not tickets:
        raise ValueError("Report không có ticket để chẩn đoán")
    num_ctx = report.get("run", {}).get("num_ctx")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Không ghi đè report gốc hoặc debug đã có. JSONL chứa dữ liệu ticket/evidence.
    with output.open("x", encoding="utf-8") as handle:
        for ticket in tickets:
            generation = ticket.get("generation") or {}
            contexts = ticket.get("kb_contexts")
            if contexts is None:
                contexts = [
                    {"parent_chunk_id": p["parent_chunk_id"], "source": p["source"], "content": p["content"]}
                    for p in ticket["retrieval"]["parents"] if p["used_for_generation"]
                ]
            diagnostic = {}
            result = evaluate_groundedness(
                ticket["question"], generation.get("answer", ""), contexts,
                answer_complete=generation.get("complete", False),
                num_ctx=num_ctx, diagnostics=diagnostic,
            )
            record = {
                "ticket_id": ticket.get("ticket_id", ticket.get("id")),
                "original_report": str(report_path.resolve()),
                "created_at": datetime.now().astimezone().isoformat(),
                "original_judge": report.get("run", {}).get("groundedness_judge"),
                "original_groundedness": ticket.get("groundedness"),
                "replayed_groundedness": result, "diagnostic": diagnostic,
            }
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            print(f"{record['ticket_id']}: {result['status']} — {result.get('error')}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="File JSONL mới; mặc định nằm cạnh report")
    parser.add_argument("--ticket-id", action="append", help="Có thể truyền nhiều lần; bỏ qua để chạy tất cả")
    args = parser.parse_args()
    destination = args.output or args.report.with_name(
        f"{args.report.stem}_judge_debug_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl")
    print(f"Đã lưu debug: {replay(args.report, destination, args.ticket_id)}")
