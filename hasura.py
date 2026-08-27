from __future__ import annotations

"""Per-request auth context for the hotels MCP server.

Two supported credentials, selected by `config.auth_mode`:

  * forward_jwt  (production): the caller's JWT/API key, forwarded verbatim
                 to Hasura via `Authorization: Bearer`. Never elevated.
  * admin_secret (dev):        a server-held `x-hasura-admin-secret` from
                 config. Sourced from the server, NEVER from the agent.

`get_forwarded_token()` returns the per-request JWT (set by the transport
middleware). The Hasura client decides which credential to attach based on
mode + token presence.
"""

from contextvars import ContextVar, Token

_forwarded_token: ContextVar[str | None] = ContextVar(
    "tripon_mcp_forwarded_token", default=None
)


def get_forwarded_token() -> str | None:
    """Return the caller JWT/API key for the active request, or None."""
    return _forwarded_token.get()


def set_forwarded_token(token: str | None) -> Token:
    """Install the forwarded token for one request. Returns a reset handle."""
    return _forwarded_token.set(token)


def reset_forwarded_token(handle: Token) -> None:
    """Restore the previous token (counterpart to set_forwarded_token)."""
    _forwarded_token.reset(handle)


"""Async GraphQL transport. One named operation per call, no raw-query path.

Credential: forward the caller's JWT when present (prod), otherwise use the
server's admin secret in dev. Refuse if neither is available.
"""

from typing import Any

import httpx


class HasuraTransportError(RuntimeError):
    pass


class HasuraGraphQLError(RuntimeError):
    def __init__(self, errors: list[dict[str, Any]]) -> None:
        first = errors[0] if errors else {}
        super().__init__(first.get("message", "Hasura returned errors"))
        self.errors = errors


class HasuraAuthError(RuntimeError):
    pass


class HasuraClient:
    def __init__(
        self,
        endpoint: str,
        *,
        auth_mode: str = "admin_secret",
        admin_secret: str | None = None,
        sender_ip: str | None = None,
        timeout: float = 30.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.endpoint = endpoint
        self.auth_mode = auth_mode
        self._admin_secret = admin_secret
        self._sender_ip = sender_ip
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "HasuraClient":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    def _auth_headers(self, token: str | None) -> dict[str, str]:
        if token:
            return {"Authorization": f"Bearer {token}"}
        if self.auth_mode == "admin_secret" and self._admin_secret:
            return {"x-hasura-admin-secret": self._admin_secret}
        raise HasuraAuthError(
            "no credential: forward_jwt needs a caller token, admin_secret needs YARVEL_SECRET"
        )

    async def request(
        self,
        *,
        query: str,
        variables: dict[str, Any] | None,
        operation_name: str,
        token: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        headers = {"Content-Type": "application/json", **self._auth_headers(token)}
        # loginRihla / some mutations validate this header (see MCP issues #6).
        if self._sender_ip:
            headers.setdefault("sender-ip", self._sender_ip)
        if extra_headers:
            headers.update(extra_headers)

        body = {"query": query, "variables": variables or {}, "operationName": operation_name}
        try:
            resp = await self._client.post(self.endpoint, json=body, headers=headers)
        except httpx.HTTPError as exc:
            raise HasuraTransportError(f"Hasura request failed: {exc}") from exc

        if resp.status_code >= 400:
            raise HasuraTransportError(f"Hasura HTTP {resp.status_code}: {resp.text[:500]}")

        payload = resp.json()
        if payload.get("errors"):
            raise HasuraGraphQLError(payload["errors"])

        data = payload.get("data") or {}
        if operation_name not in data:
            raise KeyError(f"Hasura response missing data.{operation_name}; keys={list(data)}")
        return data[operation_name]


"""Unwrap the GraphQLResponse envelope every hotel mutation returns:
{ status, message, data } where data is a JSON string. Fails loud on a
non-success status, bad JSON, or a payload that doesn't match the model."""

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class GraphQLResponseError(RuntimeError):
    def __init__(self, message: str | None, *, status: Any = None, payload: Any = None) -> None:
        super().__init__(message or "GraphQL hotel operation failed")
        self.status = status
        self.message = message
        self.payload = payload


def is_success(status: Any) -> bool:
    """Interpret the envelope `status`. Schema does not declare its type, so
    accept the common success shapes and refuse to guess on anything else."""
    if status is True:
        return True
    if isinstance(status, str):
        return status.strip().lower() in {"success", "succeeded", "ok", "true", "200"}
    return False


def unwrap(envelope: dict[str, Any] | BaseModel, model: type[T]) -> T:
    if isinstance(envelope, BaseModel):
        env = envelope.model_dump()
    elif isinstance(envelope, dict):
        env = envelope
    else:
        raise GraphQLResponseError(
            f"envelope must be dict or BaseModel, got {type(envelope).__name__}"
        )

    status, message, raw = env.get("status"), env.get("message"), env.get("data")

    if not is_success(status):
        raise GraphQLResponseError(message, status=status, payload=raw)
    if raw is None:
        raise GraphQLResponseError("status=success but data was null", status=status)
    if not isinstance(raw, str):
        raise GraphQLResponseError(
            f"data must be a JSON string per schema; got {type(raw).__name__}",
            status=status, payload=raw,
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GraphQLResponseError(
            f"data was not valid JSON: {exc.msg}", status=status, payload=raw
        ) from exc
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        raise GraphQLResponseError(
            f"data did not match {model.__name__}: {exc.error_count()} error(s)",
            status=status, payload=payload,
        ) from exc


"""Pydantic models for the hotel operations. Search models match the real
destinationSearcher / search / getSearchResults responses; the rest mirror the
Core_Hotel* tables and the mutation envelope. extra=allow keeps unknown fields."""

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)


class Destination(_Loose):
    id: str | None = None
    code: str | None = None
    title: str | None = None
    subtitle: str | None = None
    type: str | None = None          # "CITY" | "HOTEL" | ...
    hotelCount: int | None = None


class HotelPrice(_Loose):
    totalPrice: float | None = None
    net: float | None = None
    gross: float | None = None
    currency: str | None = None
    markup: float | None = None
    supplierPrice: float | None = None


class HotelLocation(_Loose):
    city: str | None = None
    country: str | None = None


class SearchHotel(_Loose):
    hotelName: str | None = None
    hotelCode: str | None = None
    available: bool | None = None
    price: HotelPrice | None = None
    location: HotelLocation | None = None
    categoryCode: str | None = None
    amenities: list[Any] | None = None
    pricePerNight: float | None = None   # computed by the tool (total / nights)


class HotelSearchStart(_Loose):
    """Return of `search` — uuid + first page of hotels."""
    uuid: str | None = None
    count: int | None = None
    hotels: list[SearchHotel] | None = None


class HotelSearchResults(_Loose):
    """Return of `getSearchResults` — poll until isComplete."""
    isComplete: bool | None = None
    count: int | None = None
    hasMorePages: bool | None = None
    hotels: list[SearchHotel] | None = None


class Core_HotelBookings(_Loose):
    Id: Any | None = None
    HotelBookingId: str | None = None
    BookingStatus: str | None = None
    TransactionStatus: str | None = None
    TransactionId: str | None = None
    OrganizationId: str | None = None
    CustomerId: str | None = None
    HotelCode: str | None = None
    HotelName: str | None = None
    CheckIn: str | None = None
    CheckOut: str | None = None
    Duration: Any | None = None
    RoomConfig: Any | None = None
    RoomDetail: Any | None = None
    SelectRoom: Any | None = None
    Price: Any | None = None
    SubTotal: Any | None = None
    TotalPrice: Any | None = None
    AdminNetPrice: Any | None = None
    AgencyNetPrice: Any | None = None
    ChargedCurrency: str | None = None
    Refundable: bool | None = None
    SupplierCode: str | None = None
    BookingMethod: str | None = None
    IsManual: bool | None = None
    isPaid: bool | None = None
    PriceUpdatedAt: str | None = None
    BookingCompletedAt: str | None = None
    CreatedAt: str | None = None
    UpdatedAt: str | None = None
    UUID: str | None = None
    short_id: str | None = None
    OrderNo: str | None = None
    OptionRefId: str | None = None


class Core_HotelMarkups(_Loose):
    id: Any | None = None
    ruleName: str | None = None
    amount: Any | None = None
    isPercentage: bool | None = None
    isPerNight: bool | None = None
    isPerPax: bool | None = None
    hotelCode: Any | None = None
    hotelName: str | None = None
    cityName: str | None = None
    countryCode: str | None = None
    countryName: str | None = None
    giataCityId: int | None = None
    organizationId: str | None = None
    supplierId: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    short_id: str | None = None


class Core_HotelCancels(_Loose):
    Id: Any | None = None
    HotelBookingId: str | None = None
    Status: str | None = None
    Reason: str | None = None
    CancelAmount: Any | None = None
    CancelFee: Any | None = None
    Notes: str | None = None
    CustomerId: str | None = None
    OrganizationId: str | None = None
    TransactionId: str | None = None
    CreatedBy: Any | None = None
    ResolvedBy: Any | None = None
    UserId: Any | None = None
    CreatedAt: str | None = None
    UpdatedAt: str | None = None
    short_id: str | None = None


class GraphQLResponseEnvelope(_Loose):
    status: Any
    message: str | None = None
    data: str | None = None          # JSON string; parsed by graphql_response.unwrap


class HotelMutationPayload(_Loose):
    """Permissive holder for the *output* JSON that comes back inside
    GraphQLResponse.data. The output shape isn't in the introspected schema
    (data is `String`), so this stays permissive. Per-operation *input* models
    live below."""


class HotelSessionDataInput(_Loose):
    """`refreshHotelComponentSession(sessionData: ...)` input.

    Server marks extraMarkup / serviceType / subtotal / total /
    transactionSubTotal / transactionTotal as required (Decimal!/Int!) — kept
    optional here so partial callers still validate; the server rejects on
    missing required fields.
    """
    extraMarkup: float | None = None
    optionRefId: str | None = None
    roomDetail: str | None = None
    selectedRoom: str | None = None
    serviceType: int | None = None
    subtotal: float | None = None
    total: float | None = None
    transactionSubTotal: float | None = None
    transactionTotal: float | None = None
    uuid: str | None = None


class CoreMapperInput(_Loose):
    """`bookAndIssue_Queue(coreMapper: ...)` input.

    ~30 fields; server-required ones flagged in the field type. Left mostly
    permissive (extra="allow" via _Loose) because CoreMapperInput composes
    several nested inputs (passengers, transferBooking, flightBooking …) that
    aren't hotel-specific — the caller assembles them.
    """
    adminMarkup: float | None = None
    cancellationFee: float | None = None
    chargedCurrency: str | None = None
    customerId: str | None = None
    extraMarkup: float | None = None
    isManual: bool | None = None
    isQuotation: bool | None = None
    organizationId: str | None = None      # UUID string
    organizationMarkup: float | None = None
    origin: int | None = None
    parentTransactionId: str | None = None
    paymentSettlementType: int | None = None
    paymentType: int | None = None
    price: float | None = None
    subTotal: float | None = None
    totalMarkup: float | None = None
    totalPax: str | None = None
    totalPrice: float | None = None
    transactionGuid: str | None = None     # UUID
    transactionStatus: str | None = None
    transactionType: int | None = None
    travelDate: str | None = None
    userId: int | None = None


class HotelRequestInput(_Loose):
    """`sendCancelRequestHotel_V2(voidBooking: ...)` input.

    createdBy / hotelBookingId / organizationId / transactionId are all
    required by the server; kept optional here for the same reason as above.
    """
    createdBy: int | None = None
    hotelBookingId: str | None = None      # UUID
    notes: str | None = None
    organizationId: str | None = None      # UUID
    reason: str | None = None
    transactionId: str | None = None       # UUID


class HotelBookingJobRef(_Loose):
    """Reference returned by the async `book_hotel`. Poll
    get_hotel_booking / list_hotel_bookings until BookingStatus is terminal."""
    jobId: str | None = None
    hotelBookingId: str | None = None
    transactionId: str | None = None
    uuid: str | None = None
    shortId: str | None = None
    message: str | None = None
    raw: dict[str, Any] | None = None


class CancelPenalty(_Loose):
    hoursBefore: Any | None = None
    penaltyType: str | None = None
    value: Any | None = None
    amount: Any | None = None
    currency: str | None = None
    deadline: str | None = None
    isFullAmount: bool | None = None
    isCalculatedDeadline: bool | None = None


class CancelPolicy(_Loose):
    refundable: bool | None = None
    description: str | None = None
    cancelPenalties: list[CancelPenalty] | None = None


class HotelMedia(_Loose):
    url: str | None = None
    type: str | None = None
    code: str | None = None
    order: str | None = None


class HotelPhone(_Loose):
    tech: str | None = None
    value: str | None = None


class HotelStaticData(_Loose):
    """`getSearchHotelStaticData` — hotel content: name, address, rating, media."""
    hotelCode: str | None = None
    hotelId: str | None = None
    hotelName: str | None = None
    chainId: str | None = None
    rating: str | None = None
    cityName: str | None = None
    country: str | None = None
    countryCode: str | None = None
    street: str | None = None
    postalCode: str | None = None
    geoLocation: str | None = None
    checkInTime: str | None = None
    checkOutTime: str | None = None
    giataCityId: int | None = None
    lastUpdate: str | None = None
    medias: list[HotelMedia] | None = None
    phones: list[HotelPhone] | None = None
    descriptions: list[Any] | None = None
    facilities: list[Any] | None = None


class RoomAmenity(_Loose):
    code: str | None = None
    type: str | None = None
    value: str | None = None
    texts: list[Any] | None = None


class RoomsInOption(_Loose):
    """A room in an availability result (`RoomsInOption`)."""
    code: str | None = None
    description: str | None = None
    refundable: bool | None = None
    medias: list[Any] | None = None
    amenities: list[RoomAmenity] | None = None


class HotelSurcharge(_Loose):
    chargeType: str | None = None
    code: str | None = None
    description: str | None = None
    mandatory: bool | None = None
    price: HotelPrice | None = None


class HotelRoomOption(_Loose):
    """One room paired with one of its bookable options, flattened from
    `getSearchHotelAvailability` (which returns rooms and roomsOptions side by
    side). `optionId` is the option identifier the availability call returns."""
    roomCode: str | None = None
    roomDescription: str | None = None
    roomAmenities: list[RoomAmenity] | None = None
    optionId: str | None = None
    accessCode: str | None = None
    boardText: str | None = None
    boardCodeSupplier: str | None = None
    status: str | None = None
    rateRules: list[Any] | None = None
    remarks: str | None = None
    refundable: bool | None = None
    price: HotelPrice | None = None
    cancelPolicy: CancelPolicy | None = None
    surcharges: list[HotelSurcharge] | None = None




class HotelOptionQuote(_Loose):
    """One priced option from `getHotelFullOptions`. `optionRefId` is what
    refresh_hotel_price and the booking step need."""
    optionRefId: str | None = None
    status: str | None = None
    remarks: str | None = None
    price: HotelPrice | None = None
    searchPrice: HotelPrice | None = None
    cancelPolicy: CancelPolicy | None = None


class HotelOptionsResult(_Loose):
    """Return of `getHotelFullOptions`."""
    hotelId: str | None = None
    uuid: str | None = None
    options: list[HotelOptionQuote] | None = None


class HotelAvailabilityResult(_Loose):
    """Return of `search_hotel_availability` — the resolved destination, any
    other same-name candidates (for disambiguation), the search uuid, whether
    the supplier fan-out finished, and the price-sorted hotels."""
    destination: Destination | None = None
    alternatives: list[Destination] | None = None
    uuid: str | None = None
    isComplete: bool | None = None
    nights: int | None = None
    count: int | None = None
    hasMorePages: bool | None = None
    pageNumber: int | None = None
    sort: dict[str, str] | None = None
    hotels: list[SearchHotel] | None = None
