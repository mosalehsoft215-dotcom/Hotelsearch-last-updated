from __future__ import annotations

"""Logging + PII masking. Every tool runs its body inside log_tool_call: args
are masked, and only derived signals (status, supplierCode, latency) are logged,
never the result body."""

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

logger = logging.getLogger("tripon.hotels")

_PII_KEY_SUBSTRINGS: tuple[str, ...] = (
    "passport", "passenger", "guest", "traveler", "traveller",
    "card", "cardholder", "cvv", "cvc", "ccv", "payment", "stripe",
    "secret", "password", "pwd", "credential", "token", "authorization",
    "bearer", "email", "phone", "mobile", "dob", "dateofbirth",
    "nationalid", "iban",
)
_MASK = "***"


def _sensitive(key: str) -> bool:
    k = key.lower()
    return any(s in k for s in _PII_KEY_SUBSTRINGS)


def mask_pii(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {k: (_MASK if isinstance(k, str) and _sensitive(k) else mask_pii(v))
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [mask_pii(v) for v in value]
    if hasattr(value, "model_dump"):
        return mask_pii(value.model_dump())
    try:
        return mask_pii(dict(value))
    except Exception:
        return repr(value)


def _status(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("status")
    return getattr(result, "status", None)


def _supplier(result: Any) -> Any:
    if isinstance(result, dict):
        return result.get("supplierCode") or result.get("SupplierCode")
    return getattr(result, "supplierCode", None) or getattr(result, "SupplierCode", None)


@contextmanager
def log_tool_call(tool_name: str, *, args: dict[str, Any] | None = None,
                  organization_id: Any = None, currency: Any = None,
                  nationality: Any = None) -> Iterator[dict[str, Any]]:
    record: dict[str, Any] = {
        "tool": tool_name,
        "organizationId": organization_id,
        "currency": currency,
        "nationality": nationality,
        "args": mask_pii(args or {}),
        "result": None,
    }
    start = time.perf_counter()
    try:
        yield record
        record["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        record["graphqlStatus"] = _status(record.get("result"))
        record["supplierCode"] = _supplier(record.get("result"))
        record.pop("result", None)
        logger.info("hotels_tool_call", extra={"hotels": record})
    except Exception as exc:
        record["latency_ms"] = round((time.perf_counter() - start) * 1000.0, 2)
        record["error"] = f"{type(exc).__name__}: {exc}"
        record.pop("result", None)
        logger.exception("hotels_tool_call_failed", extra={"hotels": record})
        raise


"""Pre-flight checks, run before any Hasura call: tenant context is present,
nationality is ISO-3166 alpha-2, a booking only proceeds after a fresh reprice
with the same markup and price, and mutation envelopes report success."""

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

from hasura import is_success


class VerifyError(ValueError):
    def __init__(self, violations: list[str]) -> None:
        super().__init__("; ".join(violations))
        self.violations = list(violations)


_ISO_ALPHA2 = re.compile(r"^[A-Z]{2}$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def check_common_args(*, organization_id: Any, currency: Any, nationality: Any) -> list[str]:
    v: list[str] = []
    if organization_id in (None, ""):
        v.append("organizationId is required and must be non-null")
    if currency in (None, ""):
        v.append("currency is required")
    if nationality in (None, ""):
        v.append("nationality is required")
    elif not isinstance(nationality, str) or not _ISO_ALPHA2.match(nationality):
        v.append("nationality must be uppercase ISO 3166-1 alpha-2 (e.g. 'SA', 'AE')")
    return v


def require_common_args(*, organization_id: Any, currency: Any, nationality: Any) -> None:
    v = check_common_args(
        organization_id=organization_id, currency=currency, nationality=nationality
    )
    if v:
        raise VerifyError(v)


def check_response_status(envelope: Any) -> list[str]:
    if not isinstance(envelope, dict):
        return [f"GraphQLResponse must be a dict; got {type(envelope).__name__}"]
    if not is_success(envelope.get("status")):
        return [f"GraphQLResponse.status not success (message: {envelope.get('message') or 'none'})"]
    return []


def require_response_status(envelope: Any) -> None:
    v = check_response_status(envelope)
    if v:
        raise VerifyError(v)


@dataclass
class _RefreshRecord:
    at: datetime
    price: Any | None = None
    apply_markup: Any | None = None


@dataclass
class RefreshFreshnessTracker:
    """Records recent reprices keyed by optionRefId. Enforces, before booking:
    a reprice exists, is within the window, price is unchanged (if supplied),
    and applyMarkup matches what was used at reprice/search time.

    NOTE: in-process state. For multi-worker deployments back this with a
    shared store (Redis) keyed the same way; the interface stays identical.
    """
    window: timedelta = timedelta(minutes=30)
    clock: Callable[[], datetime] = field(default=_utcnow)
    _records: dict[str, _RefreshRecord] = field(default_factory=dict)

    def record_refresh(self, option_ref_id: str, *, price: Any | None = None,
                       apply_markup: Any | None = None) -> None:
        self._records[option_ref_id] = _RefreshRecord(
            at=self.clock(), price=price, apply_markup=apply_markup
        )

    def check_book(self, *, option_ref_id: str, current_price: Any | None = None,
                   apply_markup: Any | None = None) -> list[str]:
        rec = self._records.get(option_ref_id)
        if rec is None:
            return [f"book requires a prior refresh_hotel_price for optionRefId={option_ref_id!r}"]
        age = self.clock() - rec.at
        if age > self.window:
            return [f"reprice for optionRefId={option_ref_id!r} is stale (age {age}, window {self.window})"]
        if current_price is not None and rec.price is not None and current_price != rec.price:
            return [f"price changed for optionRefId={option_ref_id!r} (was {rec.price!r}, now {current_price!r}); surface before booking"]
        if apply_markup is not None and rec.apply_markup is not None and apply_markup != rec.apply_markup:
            return [f"applyMarkup mismatch for optionRefId={option_ref_id!r} (reprice {rec.apply_markup!r}, book {apply_markup!r})"]
        return []

    def require_book(self, *, option_ref_id: str, current_price: Any | None = None,
                     apply_markup: Any | None = None) -> None:
        v = self.check_book(option_ref_id=option_ref_id, current_price=current_price,
                            apply_markup=apply_markup)
        if v:
            raise VerifyError(v)


"""Hotels module for tripon-mcp-service.

Wraps the Yarvel hotel operations one tool per operation. Search uses the same
ops the rihla-os frontend uses (destinationSearcher / search / getSearchResults);
reads hit the Core_Hotel* tables; writes go through the GraphQLResponse envelope.

When this drops into the shared FastMCP server, MODULE below is the capability
name the permission layer checks (granted to quotation / repricing / human roles).
"""

import asyncio
import difflib
from datetime import date, timedelta
from typing import Any

from mcp.server.fastmcp import FastMCP

from hasura import get_forwarded_token
from config import get_settings
from hasura import unwrap
from hasura import HasuraClient
from hasura import (
    CancelPolicy, HotelRoomOption, HotelOptionsResult, HotelStaticData,
    CoreMapperInput,
    Core_HotelBookings,
    Core_HotelCancels,
    Core_HotelMarkups,
    Destination,
    HotelAvailabilityResult,
    HotelBookingJobRef,
    HotelMutationPayload,
    HotelRequestInput,
    HotelSearchResults,
    HotelSearchStart,
    HotelSessionDataInput,
)

MODULE = "hotels"
# Tools that move money/state — the harness/UI must get human confirmation first.
CONFIRM_GATE_TOOLS = {"book_hotel", "send_hotel_cancel_request", "cancel_hotel"}

mcp = FastMCP("tripon-hotels")
_settings = get_settings()
_client = HasuraClient(
    endpoint=_settings.yarvel_url,
    auth_mode=_settings.auth_mode,
    admin_secret=_settings.yarvel_secret,
    sender_ip=_settings.sender_ip,
    timeout=_settings.request_timeout,
)
refresh_tracker = RefreshFreshnessTracker(window=timedelta(minutes=_settings.refresh_window_minutes))

# Confirmed via live introspection (see scripts/introspect.py + scripts/schema_full.json):
# the four mutations do NOT share a wrapping input arg. Each has its own args and shape.
_MUTATION_ARGS = {
    "refreshHotelComponentSession": ("hotelBookingId", "sessionData", "transactionId"),
    "bookAndIssue_Queue": ("coreMapper",),
    "cancelHotel_V2": ("hotelBookingId",),
    "sendCancelRequestHotel_V2": ("voidBooking",),
}


async def _request(query: str, variables: dict[str, Any], op: str) -> Any:
    # forward_jwt mode: pass the caller token. admin_secret mode: token is None
    # and the client falls back to the server secret.
    return await _client.request(
        query=query, variables=variables, operation_name=op, token=get_forwarded_token()
    )


# order_by is passed as a typed variable ([Core_*_order_by!]), so Hasura takes
# {"CreatedAt": "desc"} directly — building it inline as a string is what breaks
# the enum (MCP issues #1), so we never do that.

# Operation names below intentionally match the field they select — Hasura
# rejects the doc as "no such operation found" when the operationName arg
# doesn't match a `query/mutation <name>` in the document (confirmed live).

_Q_DESTINATION = """
query destinationSearcher($criteria: DestinationSearcherInput!) {
  destinationSearcher(criteria: $criteria) { id code title subtitle type hotelCount }
}
""".strip()

_Q_START_SEARCH = """
query search(
  $checkIn: Date!, $checkOut: Date!, $occupancies: [RoomInput!]!,
  $organizationId: String!, $destinations: [String!], $hotels: [String!],
  $pageSize: Int, $language: Language, $currency: Currency,
  $nationality: Country, $applyMarkup: Boolean
) {
  search(
    criteria: {
      checkIn: $checkIn, checkOut: $checkOut, occupancies: $occupancies,
      destinations: $destinations, hotels: $hotels,
      language: $language, currency: $currency, nationality: $nationality
    }
    organizationId: $organizationId, pageSize: $pageSize, applyMarkup: $applyMarkup
  ) {
    uuid count
    hotels {
      hotelName hotelCode available
      price { totalPrice net currency }
      location { city country }
      categoryCode
    }
  }
}
""".strip()

_Q_GET_RESULTS = """
query getSearchResults($uuid: String!, $organizationId: String!,
                       $pageNumber: Int, $pageSize: Int, $sort: SearchSortOption) {
  getSearchResults(uuid: $uuid, organizationId: $organizationId,
                   pageNumber: $pageNumber, pageSize: $pageSize, sort: $sort) {
    isComplete count hasMorePages
    hotels {
      hotelName hotelCode available
      price { totalPrice net gross currency }
      location { city country }
      categoryCode amenities
    }
  }
}
""".strip()

# org_id is a UUID string in Hasura, not an int (MCP issues #3) — the bool_exp
# variables below carry it as a plain string; we never int() it.
_Q_BOOKING_BY_PK = """
query Core_HotelBookings_by_pk($Id: bigint!) {
  Core_HotelBookings_by_pk(Id: $Id) {
    Id HotelBookingId BookingStatus TransactionStatus TransactionId
    OrganizationId CustomerId HotelCode HotelName CheckIn CheckOut Duration
    RoomConfig RoomDetail SelectRoom Price SubTotal TotalPrice
    AdminNetPrice AgencyNetPrice ChargedCurrency Refundable SupplierCode
    BookingMethod IsManual isPaid PriceUpdatedAt BookingCompletedAt
    CreatedAt UpdatedAt UUID short_id OrderNo OptionRefId
  }
}
""".strip()

_Q_LIST_BOOKINGS = """
query Core_HotelBookings($where: Core_HotelBookings_bool_exp, $limit: Int,
                         $offset: Int, $order_by: [Core_HotelBookings_order_by!]) {
  Core_HotelBookings(where: $where, limit: $limit, offset: $offset, order_by: $order_by) {
    Id HotelBookingId BookingStatus TransactionStatus TransactionId
    OrganizationId CustomerId HotelCode HotelName CheckIn CheckOut Duration
    ChargedCurrency Refundable SupplierCode BookingMethod IsManual isPaid
    PriceUpdatedAt BookingCompletedAt CreatedAt UpdatedAt UUID short_id
    OrderNo OptionRefId TotalPrice
  }
}
""".strip()

_Q_LIST_MARKUPS = """
query Core_HotelMarkups($where: Core_HotelMarkups_bool_exp, $limit: Int,
                        $offset: Int, $order_by: [Core_HotelMarkups_order_by!]) {
  Core_HotelMarkups(where: $where, limit: $limit, offset: $offset, order_by: $order_by) {
    id ruleName amount isPercentage isPerNight isPerPax hotelCode hotelName
    cityName countryCode countryName giataCityId organizationId supplierId
    created_at updated_at short_id
  }
}
""".strip()

_Q_LIST_CANCELS = """
query Core_HotelCancels($where: Core_HotelCancels_bool_exp, $limit: Int,
                        $offset: Int, $order_by: [Core_HotelCancels_order_by!]) {
  Core_HotelCancels(where: $where, limit: $limit, offset: $offset, order_by: $order_by) {
    Id HotelBookingId Status Reason CancelAmount CancelFee Notes
    CustomerId OrganizationId TransactionId CreatedBy ResolvedBy UserId
    CreatedAt UpdatedAt short_id
  }
}
""".strip()

_M_REFRESH = """
mutation refreshHotelComponentSession(
  $hotelBookingId: String, $sessionData: HotelSessionDataInput, $transactionId: String
) {
  refreshHotelComponentSession(
    hotelBookingId: $hotelBookingId, sessionData: $sessionData, transactionId: $transactionId
  ) { status message data }
}
""".strip()

_M_BOOK = """
mutation bookAndIssue_Queue($coreMapper: CoreMapperInput) {
  bookAndIssue_Queue(coreMapper: $coreMapper) { status message data }
}
""".strip()

_M_SEND_CANCEL = """
mutation sendCancelRequestHotel_V2($voidBooking: HotelRequestInput) {
  sendCancelRequestHotel_V2(voidBooking: $voidBooking) { status message data }
}
""".strip()

_M_CANCEL = """
mutation cancelHotel_V2($hotelBookingId: String) {
  cancelHotel_V2(hotelBookingId: $hotelBookingId) { status message data }
}
""".strip()


def _occupancies(adults: int, children_ages: list[int] | None, raw: list[dict[str, Any]] | None):
    if raw:
        return raw
    paxes = [{"age": 30} for _ in range(max(adults, 1))]
    paxes += [{"age": a} for a in (children_ages or [])]
    return [{"paxes": paxes}]


@mcp.tool()
async def resolve_destination(query: str, language: str = "en", limit: int = 10) -> list[Destination]:
    """Turn a city or hotel name into destination code(s); prefer type CITY."""
    with log_tool_call("resolve_destination", args={"query": query, "limit": limit}):
        rows = await _request(
            _Q_DESTINATION, {"criteria": {"query": query, "language": language, "limit": limit}},
            "destinationSearcher",
        )
        return [Destination.model_validate(d) for d in (rows or [])]


@mcp.tool()
async def start_hotel_search(
    organizationId: str, currency: str, nationality: str,
    checkIn: str, checkOut: str,
    destinations: list[str] | None = None,
    adults: int = 1, childrenAges: list[int] | None = None,
    occupancies: list[dict[str, Any]] | None = None,
    hotels: list[str] | None = None,
    language: str = "en", pageSize: int = 10, applyMarkup: bool = False,
) -> HotelSearchStart:
    """Start an availability search; returns a uuid plus the first page. Poll
    get_hotel_search_results with the uuid until isComplete."""
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    variables = {
        "checkIn": checkIn, "checkOut": checkOut,
        "occupancies": _occupancies(adults, childrenAges, occupancies),
        "organizationId": organizationId, "destinations": destinations or [],
        "hotels": hotels or [], "pageSize": pageSize, "language": language,
        "currency": currency, "nationality": nationality, "applyMarkup": applyMarkup,
    }
    with log_tool_call("start_hotel_search", args=variables,
                       organization_id=organizationId, currency=currency, nationality=nationality):
        data = await _request(_Q_START_SEARCH, variables, "search")
        return HotelSearchStart.model_validate(data or {})


@mcp.tool()
async def get_hotel_search_results(
    organizationId: str, currency: str, nationality: str,
    uuid: str, pageNumber: int = 0, pageSize: int = 10,
    checkIn: str | None = None, checkOut: str | None = None,
    sortField: str = "PRICE", sortOrder: str = "asc",
    minPrice: float | None = None, maxPrice: float | None = None,
    minStars: int | None = None, maxStars: int | None = None,
    amenities: list[str] | None = None,
) -> HotelSearchResults:
    """Fetch one page of an existing search, sorted and filtered.

    sortField is PRICE, RATING or RECOMMENDED; sortOrder is asc or desc.
    isComplete tells you when to stop polling; hasMorePages and count describe
    the paging. Filters narrow the page by price, star rating and amenities.

    Pass the same checkIn/checkOut the search used and every hotel comes back
    with pricePerNight filled in, as search_hotel_availability returns it.
    Without them the field is null on this path only, which left the agent
    dividing totals by hand for the same answer.
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    sort = build_sort(sortField, sortOrder)
    variables = {
        "uuid": uuid, "organizationId": organizationId,
        "pageNumber": pageNumber, "pageSize": pageSize, "sort": sort,
    }
    with log_tool_call("get_hotel_search_results",
                       args={**variables, "minPrice": minPrice, "maxPrice": maxPrice,
                             "minStars": minStars, "maxStars": maxStars, "amenities": amenities},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        page = HotelSearchResults.model_validate(
            await _request(_Q_GET_RESULTS, variables, "getSearchResults") or {})
        hotels = apply_filters(page.hotels or [], min_price=minPrice, max_price=maxPrice,
                               min_stars=minStars, max_stars=maxStars, amenities=amenities)
        page.hotels = sort_hotels(hotels, sort)
        page.nights = _nights_between(checkIn, checkOut)
        if page.nights:
            for hotel in page.hotels:
                total = _hotel_price(hotel)
                if total is not None:
                    hotel.pricePerNight = round(total / page.nights, 2)
        return page


@mcp.tool()
async def get_hotel_booking(organizationId: str, currency: str, nationality: str,
                            Id: int) -> Core_HotelBookings | None:
    """One booking by primary key."""
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    with log_tool_call("get_hotel_booking", args={"Id": Id, "organizationId": organizationId},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        row = await _request(_Q_BOOKING_BY_PK, {"Id": Id}, "Core_HotelBookings_by_pk")
        if row is None:
            return None
        # _by_pk takes only the primary key, and admin_secret bypasses Hasura's
        # row permissions — so a bare numeric Id read any tenant's booking.
        # Every other read here is org-scoped in the query; this one cannot be,
        # so it is scoped on the way out.
        if row.get("OrganizationId") not in (None, organizationId):
            logger.warning("hotels_booking_cross_tenant_blocked",
                           extra={"hotels": {"Id": Id, "organizationId": organizationId}})
            return None
        return Core_HotelBookings.model_validate(row)


@mcp.tool()
async def list_hotel_bookings(organizationId: str, currency: str, nationality: str,
                              bookingStatus: str | None = None, transactionStatus: str | None = None,
                              customerId: str | None = None, limit: int = 50, offset: int = 0
                              ) -> list[Core_HotelBookings]:
    """Bookings for the org, newest first, with optional status/customer filters."""
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    where: dict[str, Any] = {"OrganizationId": {"_eq": organizationId}}
    if bookingStatus is not None:
        where["BookingStatus"] = {"_eq": bookingStatus}
    if transactionStatus is not None:
        where["TransactionStatus"] = {"_eq": transactionStatus}
    if customerId is not None:
        where["CustomerId"] = {"_eq": customerId}
    variables = {"where": where, "limit": limit, "offset": offset, "order_by": [{"CreatedAt": "desc"}]}
    with log_tool_call("list_hotel_bookings", args={"where": where, "limit": limit, "offset": offset},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        rows = await _request(_Q_LIST_BOOKINGS, variables, "Core_HotelBookings")
        return [Core_HotelBookings.model_validate(r) for r in (rows or [])]


@mcp.tool()
async def list_hotel_markups(organizationId: str, currency: str, nationality: str,
                             supplierId: str | None = None, countryCode: str | None = None,
                             limit: int = 50, offset: int = 0) -> list[Core_HotelMarkups]:
    """Markup rules for the org (read only — markup writes aren't exposed here)."""
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    where: dict[str, Any] = {"organizationId": {"_eq": organizationId}}
    if supplierId is not None:
        where["supplierId"] = {"_eq": supplierId}
    if countryCode is not None:
        where["countryCode"] = {"_eq": countryCode}
    variables = {"where": where, "limit": limit, "offset": offset, "order_by": [{"updated_at": "desc"}]}
    with log_tool_call("list_hotel_markups", args={"where": where, "limit": limit, "offset": offset},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        rows = await _request(_Q_LIST_MARKUPS, variables, "Core_HotelMarkups")
        return [Core_HotelMarkups.model_validate(r) for r in (rows or [])]


@mcp.tool()
async def list_hotel_cancellations(organizationId: str, currency: str, nationality: str,
                                   hotelBookingId: str | None = None, status: str | None = None,
                                   limit: int = 50, offset: int = 0) -> list[Core_HotelCancels]:
    """Cancellation records for the org, newest first."""
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    where: dict[str, Any] = {"OrganizationId": {"_eq": organizationId}}
    if hotelBookingId is not None:
        where["HotelBookingId"] = {"_eq": hotelBookingId}
    if status is not None:
        where["Status"] = {"_eq": status}
    variables = {"where": where, "limit": limit, "offset": offset, "order_by": [{"CreatedAt": "desc"}]}
    with log_tool_call("list_hotel_cancellations", args={"where": where, "limit": limit, "offset": offset},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        rows = await _request(_Q_LIST_CANCELS, variables, "Core_HotelCancels")
        return [Core_HotelCancels.model_validate(r) for r in (rows or [])]


@mcp.tool()
async def refresh_hotel_price(organizationId: str, currency: str, nationality: str,
                              optionRefId: str, applyMarkup: bool = False,
                              sessionData: dict[str, Any] | None = None,
                              hotelBookingId: str | None = None,
                              transactionId: str | None = None,
                              payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Reprice a selected option (no booking). Records the result so a later
    book_hotel for the same option passes the freshness and markup checks.

    Wire shape (confirmed live): mutation takes hotelBookingId / sessionData
    (HotelSessionDataInput) / transactionId. HotelSessionDataInput has required
    Decimal fields (extraMarkup, subtotal, total, transactionSubTotal,
    transactionTotal) and Int serviceType — the caller supplies these via
    `sessionData` (or the legacy `payload` alias).
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    session_extras = dict(sessionData or payload or {})
    session_extras.setdefault("optionRefId", optionRefId)
    session_input = HotelSessionDataInput.model_validate(session_extras).model_dump(exclude_none=True)
    variables = {
        "hotelBookingId": hotelBookingId,
        "sessionData": session_input,
        "transactionId": transactionId,
    }
    with log_tool_call("refresh_hotel_price", args={"optionRefId": optionRefId, "applyMarkup": applyMarkup},
                       organization_id=organizationId, currency=currency, nationality=nationality) as rec:
        env = await _request(_M_REFRESH, variables, "refreshHotelComponentSession")
        rec["result"] = env
        require_response_status(env)
        parsed = unwrap(env, HotelMutationPayload).model_dump()
        refresh_tracker.record_refresh(optionRefId, price=parsed.get("price"), apply_markup=applyMarkup)
        return parsed


@mcp.tool()
async def book_hotel(organizationId: str, currency: str, nationality: str,
                     optionRefId: str, applyMarkup: bool = False,
                     currentPrice: Any | None = None,
                     coreMapper: dict[str, Any] | None = None,
                     payload: dict[str, Any] | None = None) -> HotelBookingJobRef:
    """Needs confirmation. Book a hotel (async — returns a reference to poll via
    get_hotel_booking). Blocked unless there's a fresh, same-markup, same-price
    reprice for this option.

    Wire shape (confirmed live): mutation takes a single coreMapper
    (CoreMapperInput). The caller supplies the ~30 CoreMapperInput fields via
    `coreMapper` (or the legacy `payload` alias); organizationId is injected if
    missing so the call is always scoped to the current tenant.
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    refresh_tracker.require_book(option_ref_id=optionRefId, current_price=currentPrice, apply_markup=applyMarkup)
    mapper = dict(coreMapper or payload or {})
    mapper.setdefault("organizationId", organizationId)
    core_input = CoreMapperInput.model_validate(mapper).model_dump(exclude_none=True)
    with log_tool_call("book_hotel", args={"optionRefId": optionRefId, "applyMarkup": applyMarkup},
                       organization_id=organizationId, currency=currency, nationality=nationality) as rec:
        env = await _request(_M_BOOK, {"coreMapper": core_input}, "bookAndIssue_Queue")
        rec["result"] = env
        require_response_status(env)
        p = unwrap(env, HotelMutationPayload).model_dump() or {}
        msg = env.get("message") if isinstance(env, dict) else None
        # Poll the result via HotelBookingId (confirmed by QM) — see poll_hotel_booking.
        return HotelBookingJobRef(
            jobId=p.get("jobId"),
            hotelBookingId=p.get("hotelBookingId") or p.get("HotelBookingId"),
            transactionId=p.get("transactionId") or p.get("TransactionId"),
            uuid=p.get("uuid") or p.get("UUID"),
            shortId=p.get("short_id") or p.get("shortId"),
            message=msg, raw=p,
        )


@mcp.tool()
async def send_hotel_cancel_request(organizationId: str, currency: str, nationality: str,
                                    hotelBookingId: str, transactionId: str | None = None,
                                    createdBy: int | None = None,
                                    reason: str | None = None, notes: str | None = None,
                                    voidBooking: dict[str, Any] | None = None,
                                    payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Needs confirmation. Open a cancellation *request* for a booking (V2).

    Non-terminal — the request enters a review flow (see `Core_HotelCancels`
    Status field). Use `cancel_hotel` when the intent is to actually void.

    Wire shape (confirmed live): mutation takes voidBooking (HotelRequestInput)
    with required createdBy/hotelBookingId/organizationId/transactionId
    (UUIDs). Extras go via `voidBooking` (or the legacy `payload` alias).
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    request = dict(voidBooking or payload or {})
    request.setdefault("hotelBookingId", hotelBookingId)
    request.setdefault("organizationId", organizationId)
    if transactionId is not None:
        request.setdefault("transactionId", transactionId)
    if createdBy is not None:
        request.setdefault("createdBy", createdBy)
    if reason is not None:
        request.setdefault("reason", reason)
    if notes is not None:
        request.setdefault("notes", notes)
    request_input = HotelRequestInput.model_validate(request).model_dump(exclude_none=True)
    with log_tool_call("send_hotel_cancel_request", args={"hotelBookingId": hotelBookingId},
                       organization_id=organizationId, currency=currency, nationality=nationality) as rec:
        env = await _request(_M_SEND_CANCEL, {"voidBooking": request_input}, "sendCancelRequestHotel_V2")
        rec["result"] = env
        require_response_status(env)
        return unwrap(env, HotelMutationPayload).model_dump()


@mcp.tool()
async def cancel_hotel(organizationId: str, currency: str, nationality: str,
                       hotelBookingId: str) -> dict[str, Any]:
    """Needs confirmation. Cancel a booking (V2) — terminal.

    Wire shape (confirmed live): mutation takes a single scalar
    `hotelBookingId: String` — no wrapping input. Use `send_hotel_cancel_request`
    when a review-required workflow is wanted instead.
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    with log_tool_call("cancel_hotel", args={"hotelBookingId": hotelBookingId},
                       organization_id=organizationId, currency=currency, nationality=nationality) as rec:
        env = await _request(_M_CANCEL, {"hotelBookingId": hotelBookingId}, "cancelHotel_V2")
        rec["result"] = env
        require_response_status(env)
        return unwrap(env, HotelMutationPayload).model_dump()


@mcp.tool()
async def poll_hotel_booking(organizationId: str, currency: str, nationality: str,
                             hotelBookingId: str) -> Core_HotelBookings | None:
    """Poll `Core_HotelBookings` for the row a `book_hotel` job produced.
    Filters on the `HotelBookingId` string column (confirmed by QM as the field
    the async result maps to), not the numeric `Id` primary key.
    """
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    variables = {
        "where": {
            "OrganizationId": {"_eq": organizationId},
            "HotelBookingId": {"_eq": hotelBookingId},
        },
        "limit": 1, "offset": 0, "order_by": [{"CreatedAt": "desc"}],
    }
    with log_tool_call("poll_hotel_booking", args={"hotelBookingId": hotelBookingId},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        rows = await _request(_Q_LIST_BOOKINGS, variables, "Core_HotelBookings")
        rows = rows or []
        return Core_HotelBookings.model_validate(rows[0]) if rows else None


# Minimum poll attempts to let the async supplier fan-out price hotels,
# even if HOTELS_AVAILABILITY_MAX_POLLS is set low. Poll interval stays the
# configured HOTELS_AVAILABILITY_POLL_SECONDS.
_READY_MAX_ATTEMPTS = 20


def _rooms_occupancy(room_count: int, adults: int, children_ages: list[int] | None,
                     raw: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if raw:
        return raw
    one = [{"age": 30} for _ in range(max(adults, 1))] + [{"age": a} for a in (children_ages or [])]
    return [{"paxes": list(one)} for _ in range(max(room_count, 1))]


def _nights_between(check_in: str | None, check_out: str | None) -> int | None:
    """Nights between two ISO dates, or None when either is missing or unusable."""
    if not (check_in and check_out):
        return None
    try:
        nights = (date.fromisoformat(check_out) - date.fromisoformat(check_in)).days
    except (TypeError, ValueError):
        return None
    return nights if nights > 0 else None


def _hotel_price(h: SearchHotel) -> float | None:
    p = h.price
    if p is None:
        return None
    val = p.totalPrice if p.totalPrice is not None else p.net
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _is_priced(h: SearchHotel) -> bool:
    # A hotel is bookable-ready once the async fan-out has filled its price in
    # (until then search returns available/price as null placeholders).
    return h.available is not False and _hotel_price(h) is not None


SEARCH_SORT_FIELDS = ("PRICE", "RATING", "RECOMMENDED")
SEARCH_SORT_ORDERS = ("asc", "desc")


def build_sort(field: str = "PRICE", order: str = "asc") -> dict[str, str]:
    """Build the SearchSortOption the API expects. RATING is the star/rating
    field — there is no STARS value. Order is lowercase."""
    f = (field or "PRICE").strip().upper()
    o = (order or "asc").strip().lower()
    if f not in SEARCH_SORT_FIELDS:
        raise ValueError(f"sortField must be one of {', '.join(SEARCH_SORT_FIELDS)}; got {field!r}")
    if o not in SEARCH_SORT_ORDERS:
        raise ValueError(f"sortOrder must be 'asc' or 'desc'; got {order!r}")
    return {"field": f, "order": o}


def _stars(h: SearchHotel) -> int | None:
    try:
        return int(float(h.categoryCode))
    except (TypeError, ValueError):
        return None


def _has_amenities(h: SearchHotel, wanted: list[str]) -> bool:
    have = [str(a).lower() for a in (h.amenities or [])]
    return all(any(w.strip().lower() in a for a in have)
               for w in wanted if w and w.strip())


def apply_filters(hotels: list[SearchHotel], *, min_price: float | None = None,
                  max_price: float | None = None, min_stars: int | None = None,
                  max_stars: int | None = None,
                  amenities: list[str] | None = None) -> list[SearchHotel]:
    """Keep the hotels that match on the fields a search result carries: total
    price, star rating (categoryCode) and amenities.

    getSearchResults also takes a server-side `filters` argument, but its input
    type is not confirmed by the backend team yet, so filtering happens here
    where the returned data is known. Cancellation policy and board/meal plan are
    not in a search result — they live on the room options, so they cannot be
    filtered at this level.
    """
    kept: list[SearchHotel] = []
    for h in hotels:
        price = _hotel_price(h)
        if min_price is not None and (price is None or price < min_price):
            continue
        if max_price is not None and (price is None or price > max_price):
            continue
        if min_stars is not None or max_stars is not None:
            st = _stars(h)
            if st is None:
                continue
            if min_stars is not None and st < min_stars:
                continue
            if max_stars is not None and st > max_stars:
                continue
        if amenities and not _has_amenities(h, amenities):
            continue
        kept.append(h)
    return kept


def sort_hotels(hotels: list[SearchHotel], sort: dict[str, str]) -> list[SearchHotel]:
    """Order a filtered list the same way the caller asked the API to sort.
    RECOMMENDED keeps the order the supplier returned."""
    field, order = sort.get("field"), sort.get("order", "asc")
    reverse = order == "desc"
    if field == "PRICE":
        return sorted(hotels, key=lambda h: (_hotel_price(h) is None, _hotel_price(h) or 0.0),
                      reverse=reverse)
    if field == "RATING":
        return sorted(hotels, key=lambda h: (_stars(h) is None, _stars(h) or 0), reverse=reverse)
    return list(hotels)


def _match_score(title: str, query: str) -> float:
    t, q = title.lower().strip(), query.lower().strip()
    if t == q:
        return 3.0
    if t.startswith(q) or q.startswith(t):
        return 2.0
    if q in t or t in q:
        return 1.0
    return difflib.SequenceMatcher(None, t, q).ratio()


def _pick_destination(dests: list[Destination], query: str, destination_code: str | None):
    """Choose the destination to search. An explicit code wins. Otherwise pick
    the CITY whose name best matches the query; a single city result is used
    even when its name differs slightly. Other cities are returned as alternatives
    for disambiguation."""
    if destination_code:
        chosen = next((d for d in dests if d.code == destination_code), None)
        others = [d for d in dests if (d.type or "").upper() == "CITY" and d.code != destination_code]
        return chosen, others
    cities = [d for d in dests if (d.type or "").upper() == "CITY"]
    if not cities:
        # No city match — only stray hotel-name hits. Report not-found rather
        # than searching a wrong place.
        return None, []
    ranked = sorted(cities, key=lambda d: _match_score(d.title or "", query), reverse=True)
    return ranked[0], ranked[1:]


@mcp.tool()
async def search_hotel_availability(
    organizationId: str, city: str, checkIn: str, checkOut: str,
    adults: int = 2, childrenAges: list[int] | None = None, roomCount: int = 1,
    occupancies: list[dict[str, Any]] | None = None,
    currency: str | None = None, nationality: str | None = None,
    destinationCode: str | None = None,
    hotels: list[str] | None = None, applyMarkup: bool = False,
    language: str = "en", limit: int = 5, pageNumber: int = 0,
    sortField: str = "PRICE", sortOrder: str = "asc",
    minPrice: float | None = None, maxPrice: float | None = None,
    minStars: int | None = None, maxStars: int | None = None,
    amenities: list[str] | None = None,
) -> HotelAvailabilityResult:
    """Resolve a destination, run an availability search, and return the matching
    priced hotels.

    Chains destinationSearcher -> search -> (poll) getSearchResults. The search is
    async: hotels come back with null price/availability until suppliers answer,
    so this polls until they are priced and returns only priced ones. Pass
    `destinationCode` to force a specific destination. `currency`/`nationality`
    fall back to the org defaults.

    sortField is PRICE, RATING or RECOMMENDED; sortOrder is asc or desc. Filter
    with minPrice/maxPrice (total), minStars/maxStars and amenities. Page through
    a large result set with pageNumber; `limit` caps how many come back.
    """
    currency = currency or _settings.default_currency
    nationality = nationality or _settings.default_nationality
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    sort = build_sort(sortField, sortOrder)

    with log_tool_call("search_hotel_availability",
                       args={"city": city, "destinationCode": destinationCode,
                             "checkIn": checkIn, "checkOut": checkOut,
                             "roomCount": roomCount, "adults": adults,
                             "childrenAges": childrenAges, "limit": limit,
                             "pageNumber": pageNumber, "sort": sort,
                             "minPrice": minPrice, "maxPrice": maxPrice,
                             "minStars": minStars, "maxStars": maxStars,
                             "amenities": amenities},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        rows = await _request(
            _Q_DESTINATION,
            {"criteria": {"query": city, "language": language, "limit": 25}},
            "destinationSearcher",
        )
        dests = [Destination.model_validate(d) for d in (rows or [])]
        chosen, alternatives = _pick_destination(dests, city, destinationCode)
        if chosen is None or not chosen.code:
            return HotelAvailabilityResult(destination=None, alternatives=[], uuid=None,
                                           isComplete=None, pageNumber=pageNumber,
                                           sort=sort, hotels=[])

        start = HotelSearchStart.model_validate(await _request(_Q_START_SEARCH, {
            "checkIn": checkIn, "checkOut": checkOut,
            "occupancies": _rooms_occupancy(roomCount, adults, childrenAges, occupancies),
            "organizationId": organizationId, "destinations": [chosen.code],
            "hotels": hotels or [], "pageSize": max(limit, 20), "language": language,
            "currency": currency, "nationality": nationality, "applyMarkup": applyMarkup,
        }, "search") or {})

        priced: list[SearchHotel] = []
        is_complete = None
        total_count = None
        has_more = None
        if start.uuid:
            attempts = max(_settings.availability_max_polls, _READY_MAX_ATTEMPTS)
            for attempt in range(attempts):
                page = HotelSearchResults.model_validate(await _request(_Q_GET_RESULTS, {
                    "uuid": start.uuid, "organizationId": organizationId,
                    "pageNumber": pageNumber, "pageSize": max(limit, 20), "sort": sort,
                }, "getSearchResults") or {})
                is_complete, total_count, has_more = page.isComplete, page.count, page.hasMorePages
                priced = apply_filters(
                    [h for h in (page.hotels or []) if _is_priced(h)],
                    min_price=minPrice, max_price=maxPrice,
                    min_stars=minStars, max_stars=maxStars, amenities=amenities,
                )
                if len(priced) >= limit or page.isComplete:
                    break
                if attempt < attempts - 1:
                    await asyncio.sleep(_settings.availability_poll_seconds)

        top = sort_hotels(priced, sort)[:limit]
        nights = None
        try:
            nights = (date.fromisoformat(checkOut) - date.fromisoformat(checkIn)).days
        except (TypeError, ValueError):
            nights = None
        if nights and nights > 0:
            for h in top:
                tp = _hotel_price(h)
                if tp is not None:
                    h.pricePerNight = round(tp / nights, 2)
        return HotelAvailabilityResult(
            destination=chosen, alternatives=alternatives, uuid=start.uuid,
            isComplete=is_complete, nights=nights, count=total_count,
            hasMorePages=has_more, pageNumber=pageNumber, sort=sort, hotels=top,
        )


# --- hotel detail: content, availability options, priced options -------------
# Argument shapes below were confirmed by the backend team.

# HotelObj.medias and HotelObj.facilities are [String], not object lists — a
# subselection on either is rejected as "unexpected subselection set for
# non-object field". street is often null; addressLines carries the address.
_Q_STATIC_CORE = """hotelCode hotelId hotelName chainId rating cityName country countryCode
    street addressLines postalCode geoLocation checkInTime checkOutTime giataCityId lastUpdate
    medias phones { tech value }"""

# descriptions is HotelDescription { lang lastUpdate sections { title para type } }.
# Both extras are opt-in because they add the long text to every response.
_Q_STATIC_EXTRAS = {
    "descriptions": "descriptions { lang lastUpdate sections { title para type } }",
    "facilities": "facilities",
}

# getSearchHotelAvailability returns [RoomSearch] — each element carries the
# rooms and, separately, the bookable options for them.
_Q_AVAILABILITY = """
query getSearchHotelAvailability($uuid: String, $hotelCode: String,
                                 $organizationId: String, $language: Language) {
  getSearchHotelAvailability(uuid: $uuid, hotelCode: $hotelCode,
                             organizationId: $organizationId, language: $language) {
    rooms {
      code description refundable medias
      amenities { code type value texts }
    }
    roomsOptions {
      id accessCode boardCodeSupplier boardText status rateRules remarks
      price { totalPrice net gross markup supplierPrice currency }
      cancelPolicy {
        refundable
        cancelPenalties { currency value amount penaltyType deadline hoursBefore
                          isFullAmount isCalculatedDeadline }
      }
      surcharges { chargeType code description mandatory price { net currency } }
    }
  }
}
""".strip()


def flatten_room_options(groups: list[dict[str, Any]]) -> list[HotelRoomOption]:
    """Pair each room with each of its options so one record describes one
    bookable choice: room type, board, price and cancellation policy."""
    flat: list[HotelRoomOption] = []
    for group in groups or []:
        rooms = group.get("rooms") or [{}]
        room = rooms[0] if rooms else {}
        for option in (group.get("roomsOptions") or []):
            policy = option.get("cancelPolicy") or {}
            flat.append(HotelRoomOption.model_validate({
                "roomCode": room.get("code"),
                "roomDescription": room.get("description"),
                "roomAmenities": room.get("amenities"),
                "optionId": option.get("id"),
                "accessCode": option.get("accessCode"),
                "boardText": option.get("boardText"),
                "boardCodeSupplier": option.get("boardCodeSupplier"),
                "status": option.get("status"),
                "rateRules": option.get("rateRules"),
                "remarks": option.get("remarks"),
                "refundable": policy.get("refundable"),
                "price": option.get("price"),
                "cancelPolicy": policy,
                "surcharges": option.get("surcharges"),
            }))
    return flat

# HotelFullOptionsResult.options is [RoomSearch] — the same rooms/roomsOptions
# pair getSearchHotelAvailability returns, so it flattens the same way. The
# option's `id` is the value refresh_hotel_price takes as optionRefId.
_Q_FULL_OPTIONS = """
query getHotelFullOptions($applyMarkup: Boolean!, $criteria: HotelCriteriaSearchInput!,
                          $organizationId: String) {
  getHotelFullOptions(applyMarkup: $applyMarkup, criteria: $criteria,
                      organizationId: $organizationId) {
    hotelId uuid
    errors { code type description }
    options {
      rooms {
        code description refundable medias
        amenities { code type value texts }
      }
      roomsOptions {
        id accessCode boardCodeSupplier boardText status rateRules remarks
        price { totalPrice net gross markup supplierPrice currency }
        cancelPolicy {
          refundable description
          cancelPenalties { currency value amount penaltyType deadline hoursBefore
                            isFullAmount isCalculatedDeadline }
        }
        surcharges { chargeType code description mandatory price { net currency } }
      }
    }
  }
}
""".strip()


def _refundable(policy: CancelPolicy | None) -> bool | None:
    return None if policy is None else policy.refundable


def _matches_board(option: Any, meal_plan: list[str] | None) -> bool:
    if not meal_plan:
        return True
    text = " ".join(str(x).lower() for x in
                    (getattr(option, "boardText", None), getattr(option, "boardCodeSupplier", None)) if x)
    return any(m.strip().lower() in text for m in meal_plan if m and m.strip())


def filter_options(options: list[Any], *, refundable_only: bool = False,
                   meal_plan: list[str] | None = None, min_price: float | None = None,
                   max_price: float | None = None) -> list[Any]:
    """Narrow room options by cancellation policy, meal plan (board) and price.
    These are the option-level filters — the data does not exist on a search
    result, only here."""
    kept = []
    for o in options:
        refundable = getattr(o, "refundable", None)
        if refundable is None:
            refundable = _refundable(getattr(o, "cancelPolicy", None))
        if refundable_only and refundable is not True:
            continue
        if meal_plan and not _matches_board(o, meal_plan):
            continue
        total = getattr(getattr(o, "price", None), "totalPrice", None)
        if min_price is not None and (total is None or float(total) < min_price):
            continue
        if max_price is not None and (total is None or float(total) > max_price):
            continue
        kept.append(o)
    return kept


@mcp.tool()
async def get_hotel_static_data(hotelCode: str, language: str = "en",
                                extras: list[str] | None = None) -> HotelStaticData | None:
    """Hotel content for one hotel: name, address, star rating, media, phones.

    `extras` may include "descriptions" and/or "facilities" (facilities are the
    hotel amenities). Wraps `getSearchHotelStaticData`.
    """
    selection = _Q_STATIC_CORE
    for extra in (extras or []):
        block = _Q_STATIC_EXTRAS.get(extra)
        if block is None:
            raise ValueError(f"extras must be from {sorted(_Q_STATIC_EXTRAS)}; got {extra!r}")
        selection += "\n    " + block
    query = ("query getSearchHotelStaticData($hotelCode: String, $language: Language) "
             "{ getSearchHotelStaticData(hotelCode: $hotelCode, language: $language) { "
             + selection + " } }")
    with log_tool_call("get_hotel_static_data", args={"hotelCode": hotelCode, "extras": extras}):
        row = await _request(query, {"hotelCode": hotelCode, "language": language},
                             "getSearchHotelStaticData")
        return None if row is None else HotelStaticData.model_validate(row)


@mcp.tool()
async def get_hotel_availability_options(
    organizationId: str, uuid: str, hotelCode: str, language: str = "en",
    refundableOnly: bool = False, mealPlan: list[str] | None = None,
    minPrice: float | None = None, maxPrice: float | None = None,
) -> list[HotelRoomOption]:
    """Room options for one hotel inside a running search: room type, board (meal
    plan), price and cancellation policy, one record per bookable choice.

    Uses the search `uuid`, so it only works inside the ~30 minute search session.
    Filter with refundableOnly, mealPlan and the price bounds. Wraps
    `getSearchHotelAvailability`.
    """
    with log_tool_call("get_hotel_availability_options",
                       args={"uuid": uuid, "hotelCode": hotelCode, "refundableOnly": refundableOnly,
                             "mealPlan": mealPlan, "minPrice": minPrice, "maxPrice": maxPrice},
                       organization_id=organizationId):
        groups = await _request(_Q_AVAILABILITY,
                                {"uuid": uuid, "hotelCode": hotelCode,
                                 "organizationId": organizationId, "language": language},
                                "getSearchHotelAvailability")
        options = flatten_room_options(groups or [])
        return filter_options(options, refundable_only=refundableOnly, meal_plan=mealPlan,
                              min_price=minPrice, max_price=maxPrice)


@mcp.tool()
async def get_hotel_options(
    organizationId: str, hotelCode: str, checkIn: str, checkOut: str,
    adults: int = 2, childrenAges: list[int] | None = None, roomCount: int = 1,
    occupancies: list[dict[str, Any]] | None = None,
    currency: str | None = None, nationality: str | None = None,
    language: str = "en", applyMarkup: bool = False,
    refundableOnly: bool = False, minPrice: float | None = None,
    maxPrice: float | None = None,
) -> HotelOptionsResult:
    """Priced room options for one hotel, each with the option id that
    refresh_hotel_price takes as optionRefId. Not tied to a search session, so
    use it when the search uuid has expired. Wraps `getHotelFullOptions`.
    """
    currency = currency or _settings.default_currency
    nationality = nationality or _settings.default_nationality
    require_common_args(organization_id=organizationId, currency=currency, nationality=nationality)
    criteria = {
        "checkIn": checkIn, "checkOut": checkOut,
        "occupancies": _rooms_occupancy(roomCount, adults, childrenAges, occupancies),
        "hotels": [hotelCode], "language": language,
        "currency": currency, "nationality": nationality,
    }
    with log_tool_call("get_hotel_options",
                       args={"hotelCode": hotelCode, "checkIn": checkIn, "checkOut": checkOut,
                             "refundableOnly": refundableOnly, "minPrice": minPrice,
                             "maxPrice": maxPrice},
                       organization_id=organizationId, currency=currency, nationality=nationality):
        data = await _request(_Q_FULL_OPTIONS, {"applyMarkup": applyMarkup, "criteria": criteria,
                                                "organizationId": organizationId},
                              "getHotelFullOptions") or {}
        options = flatten_room_options(data.get("options") or [])
        return HotelOptionsResult(
            hotelId=data.get("hotelId"), uuid=data.get("uuid"),
            errors=data.get("errors"),
            options=filter_options(options, refundable_only=refundableOnly,
                                   min_price=minPrice, max_price=maxPrice))
