"""Ops triage agent — triages failed booking-queue (dead-letter) messages.

Read-only: it reads the failed messages and their transactions, classifies each
failure, and writes a report. It never replays, retries, books, or changes
anything.
"""
from __future__ import annotations

from datetime import date

from config import get_settings
from runtime import (
    AgentBase, AgentContext, AgentRunResult, ToolCall, VerificationResult, build_llm,
)

ROLE = "ops_triage_agent"
GRANTED_MODULES = ("bookings", "queries", "ops")
ALLOWED_TOOLS = frozenset({
    "get_queue_summary", "get_failed_messages", "get_message_detail",
    "get_transaction", "run_named_query", "list_transactions", "record_ops_pattern",
})
MEM_TRIAGE_SUMMARY = "triage_summary"
MEM_SEEN_PATTERNS = "seen_patterns"
MEM_LAST_REPORT = "last_report"
MEM_RUN_COUNT = "run_count"
MEM_CLASS_COUNTS = "classification_counts"
MEM_COUNTED_IDS = "counted_message_ids"

CLASSIFICATIONS = ("supplier_timeout", "validation_error", "pnr_conflict",
                   "payment_failure", "unknown")

# Tools this agent must never reach (also enforced by their absence from the set).
_WRITE_TOOLS = frozenset({"book_hotel", "cancel_hotel", "send_hotel_cancel_request",
                          "refresh_hotel_price"})
_REPLAY_TOOLS = frozenset({"replay_message", "requeue_message", "approve_message"})


def classify_error(error_message: str | None) -> str:
    """Map a queue error to one of CLASSIFICATIONS by keyword."""
    text = (error_message or "").lower()
    if "timeout" in text or "timed out" in text or "supplier" in text:
        return "supplier_timeout"
    if "pnr" in text:
        return "pnr_conflict"
    if "payment" in text or "card" in text or "charge" in text or "declin" in text:
        return "payment_failure"
    if "valid" in text or "invalid" in text or "required" in text or "schema" in text:
        return "validation_error"
    return "unknown"


def _signature(message: dict) -> str:
    return f"{message.get('OperationType')}:{classify_error(message.get('ErrorMessage'))}"


def check_no_booking_created(tool_calls: list[ToolCall]) -> list[str]:
    return [f"write tool called: {c.name}" for c in tool_calls if c.name in _WRITE_TOOLS]


def check_no_auto_approve(tool_calls: list[ToolCall]) -> list[str]:
    return [f"queue-replay tool called: {c.name}" for c in tool_calls if c.name in _REPLAY_TOOLS]


def check_report_is_actionable(report: str | None, classified_count: int) -> list[str]:
    """When failures were fetched and classified, the report must give the ops
    team something to act on. Classification itself is recorded deterministically
    (on_tool_result), so we don't grade the model's wording — only that it left a
    recommendation."""
    if not classified_count:
        return []
    text = (report or "").lower()
    if "recommend" in text or "action" in text:
        return []
    return ["report is missing a recommended action"]


class OpsTriageAgent(AgentBase):
    def get_role(self) -> str:
        return ROLE

    def allowed_tools(self) -> frozenset[str]:
        return ALLOWED_TOOLS

    def build_prompt(self, ctx: AgentContext) -> str:
        known = f"\n\n{ctx.memory_context}\n" if ctx.memory_context else ""
        return f"""You are the ops triage agent for TripOn/Rihla. The async booking queue routes BookAndIssue operations; a message that fails 3 retries lands in the dead-letter queue. You read the failed messages, work out why each one failed, and produce a triage report ops staff use to decide what to do. You are read-only: you never replay, retry, book, or change anything.

Organization ID: {ctx.org_id}. It is attached automatically. Never ask for it. Today is {date.today().isoformat()}.

Start every triage with get_queue_summary() for overall queue health. Then call get_failed_messages() to get the actual failed messages and their MessageId values — the summary only has counts, so you must call get_failed_messages to see the failures; never conclude there are none from the summary alone. For each failed message, call run_named_query('triage_context', {{"message_id": <its MessageId>}}) to pull the message plus its linked transaction. Classify each failure as exactly one of: {', '.join(CLASSIFICATIONS)}. Reuse the error signatures already in memory so you do not re-report a pattern you already classified this session as new.

Never suggest replaying or retrying a failed message — that decision belongs to a human.

Output a structured triage report with: total counts by status, a table classifying each failure with its reason, and a recommended action per failure.{known}
For each distinct failure signature call record_ops_pattern with "operation:classification". It tells you whether that signature was already seen in an earlier session — report those as known and recurring, and only call something new when it is new."""

    def on_run_start(self, ctx: AgentContext) -> None:
        ctx.remember(MEM_RUN_COUNT, (ctx.recall(MEM_RUN_COUNT) or 0) + 1)
        ctx.remember(MEM_CLASS_COUNTS, {})   # per-run tally; seen_patterns persists across runs
        ctx.remember(MEM_COUNTED_IDS, [])    # per-run, so a later run counts afresh

    def on_run_end(self, ctx: AgentContext, output: str) -> None:
        ctx.remember(MEM_LAST_REPORT, output)

    def _record_failure(self, ctx: AgentContext, message: dict) -> None:
        # The prescribed flow reads each failure twice — once from
        # get_failed_messages and again from run_named_query('triage_context')
        # — so one message was tallied as two and the report doubled its counts.
        message_id = message.get("MessageId")
        if message_id:
            counted = list(ctx.recall(MEM_COUNTED_IDS) or [])
            if message_id in counted:
                return
            counted.append(message_id)
            ctx.remember(MEM_COUNTED_IDS, counted)
        cls = classify_error(message.get("ErrorMessage"))
        counts = dict(ctx.recall(MEM_CLASS_COUNTS) or {})
        counts[cls] = counts.get(cls, 0) + 1
        ctx.remember(MEM_CLASS_COUNTS, counts)
        seen = list(ctx.recall(MEM_SEEN_PATTERNS) or [])
        sig = _signature(message)
        if sig not in seen:
            seen.append(sig)
            ctx.remember(MEM_SEEN_PATTERNS, seen)

    def on_tool_result(self, ctx: AgentContext, call: ToolCall) -> None:
        result = call.result
        if call.name == "get_queue_summary" and isinstance(result, dict):
            ctx.remember(MEM_TRIAGE_SUMMARY, result.get("counts"))
        elif call.name == "get_failed_messages" and isinstance(result, list):
            for m in result:
                if isinstance(m, dict):
                    self._record_failure(ctx, m)
        elif call.name == "get_message_detail" and isinstance(result, dict):
            self._record_failure(ctx, result)
        elif call.name == "run_named_query" and isinstance(result, dict):
            msg = result.get("message")
            if isinstance(msg, dict):
                self._record_failure(ctx, msg)

    async def verify(self, ctx: AgentContext) -> VerificationResult:
        result = VerificationResult()
        for issue in check_no_booking_created(ctx.tool_calls):
            result.add_issue(issue)
        for issue in check_no_auto_approve(ctx.tool_calls):
            result.add_issue(issue)
        for call in ctx.tool_calls:
            if call.name not in ALLOWED_TOOLS:
                result.add_issue(f"called tool outside the ops read set: {call.name}")
        if not (ctx.recall(MEM_LAST_REPORT) or "").strip():
            result.add_issue("the run produced no written report")
        classified = sum((ctx.recall(MEM_CLASS_COUNTS) or {}).values())
        for issue in check_report_is_actionable(ctx.recall(MEM_LAST_REPORT), classified):
            result.add_issue(issue)
        return result


async def answer_triage(user_message: str, *, org_id: str, llm=None) -> AgentRunResult:
    """Run the ops triage agent for one message."""
    s = get_settings()
    ctx = AgentContext(org_id=org_id)
    return await OpsTriageAgent().run(ctx, user_message, llm or build_llm(s),
                                      max_iterations=s.agent_max_iterations)
