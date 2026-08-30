import pytest
import ops_tools as ops
from runtime import AgentContext, ToolCall, LLMResponse, LLMToolCall
from agents.ops_triage_agent import (
    OpsTriageAgent, ROLE as OPS_ROLE, ALLOWED_TOOLS as OPS_ALLOWED_TOOLS, classify_error,
    MEM_CLASS_COUNTS, MEM_SEEN_PATTERNS, MEM_RUN_COUNT, MEM_LAST_REPORT, MEM_TRIAGE_SUMMARY,
)
from tests.conftest import Seq

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def _msg(err, mid="M1", op="hotel_book", txn="T-1"):
    return {"Id": 1, "MessageId": mid, "Status": "failed", "OperationType": op,
            "ErrorMessage": err, "RetryCount": 3, "MaxRetries": 3, "TransactionId": txn,
            "CreatedAt": "2026-08-01T00:00:00Z"}


class ScriptedLLM:
    def __init__(self, responses):
        self._responses = list(responses)
    async def complete(self, messages, tools=None):
        return self._responses.pop(0)
    async def aclose(self):
        return None


# ---- classifier ----------------------------------------------------------

def test_classify_error_maps_categories():
    assert classify_error("Supplier timeout after 30s") == "supplier_timeout"
    assert classify_error("Invalid passenger data") == "validation_error"
    assert classify_error("PNR already cancelled") == "pnr_conflict"
    assert classify_error("Payment declined") == "payment_failure"
    assert classify_error("weird thing") == "unknown"
    assert classify_error(None) == "unknown"


# ---- ops tools -----------------------------------------------------------

@pytest.mark.asyncio
async def test_get_queue_summary_counts_in_python(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [
        {"Status": "failed"}, {"Status": "failed"}, {"Status": "complete"}, {"Status": "pending"}]
    out = await ops.get_queue_summary()
    assert out.counts == {"failed": 2, "complete": 1, "pending": 1}
    assert out.total == 4


@pytest.mark.asyncio
async def test_get_failed_messages_filter_and_order(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [_msg("Supplier timeout")]
    out = await ops.get_failed_messages(limit=5)
    assert out[0].Status == "failed"
    v = fake_hasura.calls[0]["variables"]
    assert v["where"] == {"Status": {"_ilike": "failed"}}
    assert v["order_by"] == [{"CreatedAt": "desc"}]


@pytest.mark.asyncio
async def test_run_named_query_triage_context_joins_transaction(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = [_msg("Payment declined", txn="TG-9")]
    fake_hasura.responses["Core_Transactions"] = [{"TransactionGuid": "TG-9", "TransactionStatus": "FAILED"}]
    out = await ops.run_named_query("triage_context", {"message_id": "M1"})
    assert out.message.MessageId == "M1"
    assert out.transaction.TransactionGuid == "TG-9"


@pytest.mark.asyncio
async def test_run_named_query_unknown_raises(fake_hasura):
    with pytest.raises(KeyError):
        await ops.run_named_query("nope", {})


@pytest.mark.asyncio
async def test_list_transactions_enforces_org_scope(fake_hasura):
    fake_hasura.responses["Core_Transactions"] = []
    await ops.list_transactions(organizationId=ORG, where={"TravelDate": {"_gte": "2026-08-01"}})
    where = fake_hasura.calls[0]["variables"]["where"]
    assert where["OrganizationId"] == {"_eq": ORG}
    assert where["TravelDate"] == {"_gte": "2026-08-01"}


# ---- ops triage agent ----------------------------------------------------

def test_role_modules_and_read_only():
    a = OpsTriageAgent()
    assert a.get_role() == OPS_ROLE == "ops_triage_agent"
    assert OPS_ALLOWED_TOOLS.isdisjoint({"book_hotel", "cancel_hotel", "replay_message"})


def test_classification_count_per_type_and_pattern_dedup():
    a, ctx = OpsTriageAgent(), AgentContext(org_id=ORG)
    a.on_run_start(ctx)
    failures = [_msg("Supplier timeout", mid="M1"), _msg("Invalid data", mid="M2"),
                _msg("Payment declined", mid="M3"), _msg("Supplier timeout", mid="M4")]
    a.on_tool_result(ctx, ToolCall("get_failed_messages", {}, failures))
    # one count per failure, grouped by type
    assert ctx.recall(MEM_CLASS_COUNTS) == {"supplier_timeout": 2, "validation_error": 1, "payment_failure": 1}
    # the repeated supplier_timeout signature is stored once
    assert len(ctx.recall(MEM_SEEN_PATTERNS)) == 3


@pytest.mark.asyncio
async def test_verify_flags_write_tool():
    a, ctx = OpsTriageAgent(), AgentContext(org_id=ORG)
    ctx.tool_calls = [ToolCall("book_hotel", {}, {})]
    res = await a.verify(ctx)
    assert not res.passed and any("book_hotel" in i for i in res.issues)


@pytest.mark.asyncio
async def test_loop_produces_report_and_counts(fake_hasura):
    fake_hasura.responses["Core_BookingQueueStatus"] = Seq([
        [{"Status": "failed"}, {"Status": "failed"}, {"Status": "complete"}],   # get_queue_summary
        [_msg("Supplier timeout", mid="M1"), _msg("Invalid data", mid="M2"),
         _msg("Payment declined", mid="M3")],                                   # get_failed_messages
    ])
    llm = ScriptedLLM([
        LLMResponse(tool_calls=[LLMToolCall("c1", "get_queue_summary", {})]),
        LLMResponse(tool_calls=[LLMToolCall("c2", "get_failed_messages", {"limit": 20})]),
        LLMResponse(content="Triage: 3 failed. supplier_timeout x1, validation_error x1, "
                            "payment_failure x1. Recommend: escalate timeouts, fix validation."),
    ])
    ctx = AgentContext(org_id=ORG)
    result = await OpsTriageAgent().run(ctx, "triage the queue", llm)
    assert result.verification.passed, result.verification.issues
    assert ctx.recall(MEM_TRIAGE_SUMMARY) == {"failed": 2, "complete": 1}
    assert ctx.recall(MEM_CLASS_COUNTS) == {"supplier_timeout": 1, "validation_error": 1, "payment_failure": 1}
    assert ctx.recall(MEM_RUN_COUNT) == 1


@pytest.mark.asyncio
async def test_run_count_increments_across_runs(fake_hasura):
    ctx = AgentContext(org_id=ORG)
    agent = OpsTriageAgent()
    for _ in range(2):
        llm = ScriptedLLM([LLMResponse(content="No failures to triage.")])
        await agent.run(ctx, "triage", llm)
    assert ctx.recall(MEM_RUN_COUNT) == 2


def test_report_actionable_check():
    from agents.ops_triage_agent import check_report_is_actionable
    # prose report with a recommendation passes regardless of exact label wording
    assert check_report_is_actionable("a validation error and an unknown ref. Recommended actions below.",
                                      classified_count=3) == []
    # no recommendation -> flagged
    assert check_report_is_actionable("just a summary, no next steps", classified_count=3) == [
        "report is missing a recommended action"]
    # nothing classified -> nothing to check
    assert check_report_is_actionable("anything", classified_count=0) == []
