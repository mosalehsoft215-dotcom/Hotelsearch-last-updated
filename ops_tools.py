from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from hotel_tools import _request, log_tool_call, mcp


class _Row(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class QueueMessage(_Row):
    Id: Any | None = None
    MessageId: str | None = None
    Status: str | None = None
    OperationType: str | None = None
    ErrorMessage: str | None = None
    RetryCount: Any | None = None
    MaxRetries: Any | None = None
    TransactionId: str | None = None
    CreatedAt: str | None = None
    CompletedAt: str | None = None
    ProcessedAt: str | None = None
    RequestData: str | None = None
    ResponseData: str | None = None


class TransactionRow(_Row):
    Id: Any | None = None
    TransactionGuid: str | None = None
    BookingId: str | None = None
    OrganizationId: str | None = None
    CustomerId: str | None = None
    TransactionStatus: str | None = None
    PaymentStatus: Any | None = None
    TotalPrice: Any | None = None
    ChargedCurrency: str | None = None
    TravelDate: str | None = None
    CreatedAt: str | None = None
    short_id: str | None = None
    TransactionType: Any | None = None


class QueueSummary(_Row):
    counts: dict[str, int] | None = None
    total: int | None = None


class TriageContext(_Row):
    message: QueueMessage | None = None
    transaction: TransactionRow | None = None


FAILED_STATUS = "failed"  # matched case-insensitively (_ilike) — real value may be "Failed"

_MSG_FIELDS = ("Id MessageId Status OperationType ErrorMessage RetryCount MaxRetries "
               "TransactionId CreatedAt CompletedAt ProcessedAt")
_MSG_FIELDS_FULL = _MSG_FIELDS + " RequestData ResponseData"

# get_transaction field selection: the model can request a subset by name; only
# these columns are allowed (no arbitrary GraphQL from the caller).
_TXN_FIELDS = ("Id", "TransactionGuid", "BookingId", "OrganizationId", "CustomerId",
               "TransactionStatus", "PaymentStatus", "TotalPrice", "ChargedCurrency",
               "TravelDate", "CreatedAt", "short_id", "TransactionType")

_Q_SUMMARY = "query Core_BookingQueueStatus($limit: Int) { Core_BookingQueueStatus(limit: $limit) { Status } }"

_Q_FAILED = f"""
query Core_BookingQueueStatus($where: Core_BookingQueueStatus_bool_exp, $limit: Int,
                              $order_by: [Core_BookingQueueStatus_order_by!]) {{
  Core_BookingQueueStatus(where: $where, limit: $limit, order_by: $order_by) {{ {_MSG_FIELDS} }}
}}""".strip()

_Q_MSG = f"""
query Core_BookingQueueStatus($where: Core_BookingQueueStatus_bool_exp, $limit: Int) {{
  Core_BookingQueueStatus(where: $where, limit: $limit) {{ {_MSG_FIELDS_FULL} }}
}}""".strip()


def _txn_query(fields: list[str] | None) -> str:
    cols = [f for f in (fields or _TXN_FIELDS) if f in _TXN_FIELDS] or list(_TXN_FIELDS)
    if "TransactionGuid" not in cols:
        cols.append("TransactionGuid")
    return (f"query Core_Transactions($where: Core_Transactions_bool_exp, $limit: Int, "
            f"$order_by: [Core_Transactions_order_by!]) {{ Core_Transactions(where: $where, "
            f"limit: $limit, order_by: $order_by) {{ {' '.join(cols)} }} }}")


@mcp.tool()
async def get_queue_summary(limit: int = 1000) -> QueueSummary:
    """Count booking-queue messages by Status. Fetches Status rows and counts in
    Python (the JWT role can read the table but not run aggregates on it)."""
    with log_tool_call("get_queue_summary", args={"limit": limit}):
        rows = await _request(_Q_SUMMARY, {"limit": limit}, "Core_BookingQueueStatus") or []
        counts: dict[str, int] = {}
        for r in rows:
            s = r.get("Status") or "unknown"
            counts[s] = counts.get(s, 0) + 1
        return QueueSummary(counts=counts, total=len(rows))


@mcp.tool()
async def get_failed_messages(limit: int = 20) -> list[QueueMessage]:
    """Dead-letter messages (Status=failed), newest first."""
    with log_tool_call("get_failed_messages", args={"limit": limit}):
        rows = await _request(_Q_FAILED, {
            "where": {"Status": {"_ilike": FAILED_STATUS}}, "limit": limit,
            "order_by": [{"CreatedAt": "desc"}],
        }, "Core_BookingQueueStatus") or []
        return [QueueMessage.model_validate(r) for r in rows]


@mcp.tool()
async def get_message_detail(message_id: str) -> QueueMessage | None:
    """Full detail for one queue message, including the error trace and payloads."""
    with log_tool_call("get_message_detail", args={"message_id": message_id}):
        rows = await _request(_Q_MSG, {"where": {"MessageId": {"_eq": message_id}}, "limit": 1},
                              "Core_BookingQueueStatus") or []
        return QueueMessage.model_validate(rows[0]) if rows else None


@mcp.tool()
async def get_transaction(guid: str, fields: list[str] | None = None) -> TransactionRow | None:
    """Load one transaction by its TransactionGuid. `fields` optionally selects a
    subset of allowed columns."""
    with log_tool_call("get_transaction", args={"guid": guid, "fields": fields}):
        rows = await _request(_txn_query(fields),
                              {"where": {"TransactionGuid": {"_eq": guid}}, "limit": 1,
                               "order_by": [{"CreatedAt": "desc"}]},
                              "Core_Transactions") or []
        return TransactionRow.model_validate(rows[0]) if rows else None


@mcp.tool()
async def list_transactions(organizationId: str, where: dict[str, Any] | None = None,
                            limit: int = 50) -> list[TransactionRow]:
    """Transactions for an organization, newest first. `where` adds extra Hasura
    filters (e.g. a CreatedAt window); the org scope is always enforced."""
    scoped: dict[str, Any] = {"OrganizationId": {"_eq": organizationId}}
    if where:
        scoped.update(where)
    with log_tool_call("list_transactions", args={"where": scoped, "limit": limit},
                       organization_id=organizationId):
        rows = await _request(_txn_query(None),
                              {"where": scoped, "limit": limit, "order_by": [{"CreatedAt": "desc"}]},
                              "Core_Transactions") or []
        return [TransactionRow.model_validate(r) for r in rows]


@mcp.tool()
async def run_named_query(name: str, variables: dict[str, Any] | None = None) -> TriageContext:
    """Run a pre-defined named query. `triage_context` returns a failed message
    plus its linked transaction in one call."""
    variables = variables or {}
    if name != "triage_context":
        raise KeyError(f"unknown named query {name!r}")
    message_id = variables.get("message_id") or variables.get("messageId")
    with log_tool_call("run_named_query", args={"name": name, "message_id": message_id}):
        msg = await get_message_detail(message_id) if message_id else None
        txn = None
        if msg is not None and msg.TransactionId:
            txn = await get_transaction(guid=str(msg.TransactionId))
        return TriageContext(message=msg, transaction=txn)
