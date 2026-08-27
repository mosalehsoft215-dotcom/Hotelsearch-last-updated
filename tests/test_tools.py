from datetime import datetime, timedelta, timezone
import pytest
from hotel_tools import (
    check_common_args, require_common_args, VerifyError,
    check_response_status, RefreshFreshnessTracker,
)


def test_common_args_ok():
    assert check_common_args(organization_id="o", currency="USD", nationality="SA") == []


def test_common_args_collects_all_violations():
    v = check_common_args(organization_id="", currency=None, nationality="sa")
    assert len(v) == 3


def test_nationality_must_be_alpha2_upper():
    assert check_common_args(organization_id="o", currency="USD", nationality="SAU")
    assert check_common_args(organization_id="o", currency="USD", nationality="sa")
    assert check_common_args(organization_id="o", currency="USD", nationality="SA") == []


def test_require_raises():
    with pytest.raises(VerifyError):
        require_common_args(organization_id=None, currency="USD", nationality="SA")


def test_response_status_check():
    assert check_response_status({"status": "success"}) == []
    assert check_response_status({"status": "error", "message": "boom"})
    assert check_response_status("notadict")


def _clock(t):
    return lambda: t


def test_book_requires_prior_refresh():
    tr = RefreshFreshnessTracker()
    assert tr.check_book(option_ref_id="O1")  # no refresh -> violation


def test_book_passes_after_fresh_refresh():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = RefreshFreshnessTracker(window=timedelta(minutes=30), clock=_clock(now))
    tr.record_refresh("O1", price=100, apply_markup=False)
    assert tr.check_book(option_ref_id="O1", current_price=100, apply_markup=False) == []


def test_book_blocks_when_stale():
    base = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [base, base + timedelta(minutes=31)]
    tr = RefreshFreshnessTracker(window=timedelta(minutes=30), clock=lambda: times[-1])
    times[:] = [base]
    tr.record_refresh("O1", price=100)
    times[:] = [base + timedelta(minutes=31)]
    assert tr.check_book(option_ref_id="O1")


def test_book_blocks_on_price_change():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = RefreshFreshnessTracker(clock=_clock(now))
    tr.record_refresh("O1", price=100)
    assert tr.check_book(option_ref_id="O1", current_price=130)


def test_book_blocks_on_markup_mismatch():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = RefreshFreshnessTracker(clock=_clock(now))
    tr.record_refresh("O1", price=100, apply_markup=False)
    assert tr.check_book(option_ref_id="O1", apply_markup=True)


from hotel_tools import mask_pii, log_tool_call


def test_mask_pii_masks_sensitive_keys():
    out = mask_pii({"password": "x", "email": "a@b.c", "organizationId": "keep", "n": {"cardNumber": "1"}})
    assert out["password"] == "***"
    assert out["email"] == "***"
    assert out["organizationId"] == "keep"
    assert out["n"]["cardNumber"] == "***"


def test_mask_pii_recurses_lists():
    # neutral container key -> recurse into list; sensitive inner key masked
    out = mask_pii({"items": [{"passport": "P1"}, {"age": 30}]})
    assert out["items"][0]["passport"] == "***"
    assert out["items"][1]["age"] == 30


def test_mask_pii_masks_sensitive_container_wholesale():
    # a sensitive *container* key (guests) is masked entirely, not recursed
    out = mask_pii({"guests": [{"passport": "P1"}]})
    assert out["guests"] == "***"


def test_log_tool_call_emits(caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="tripon.hotels"):
        with log_tool_call("t", args={"password": "x"}, organization_id="o") as rec:
            rec["result"] = {"status": "success", "supplierCode": "PK"}
    assert any(r.message == "hotels_tool_call" for r in caplog.records)


def test_log_tool_call_logs_on_error(caplog):
    import logging, pytest
    with caplog.at_level(logging.INFO, logger="tripon.hotels"):
        try:
            with log_tool_call("t", args={}):
                raise ValueError("boom")
        except ValueError:
            pass
    assert any("failed" in r.message for r in caplog.records)


import pytest
import hotel_tools as srv
from tests.conftest import Seq

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"


def _hotel(name, price):
    return {"hotelName": name, "hotelCode": name[:3].upper(), "available": True,
            "price": {"totalPrice": price, "currency": "USD"},
            "location": {"city": "Metroville", "country": "AE"}, "categoryCode": "5"}


@pytest.mark.asyncio
async def test_resolves_city_starts_and_returns_sorted(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CTA", "title": "Metroville", "type": "CITY"},
        {"code": "CTB", "title": "Metroville Airport Area", "type": "CITY"},
        {"code": "H1", "title": "Some Hotel", "type": "HOTEL"},
    ]
    fake_hasura.responses["search"] = {"uuid": "U1", "count": 3, "hotels": []}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True,
        "hotels": [_hotel("Bravo", 200), _hotel("Alpha", 90), _hotel("Cee", 150)],
    }
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20", adults=2,
    )
    assert out.destination.code == "CTA"                 # first CITY chosen
    assert [d.code for d in out.alternatives] == ["CTB"]  # other CITY offered for clarify
    assert out.uuid == "U1"
    assert [h.hotelName for h in out.hotels] == ["Alpha", "Cee", "Bravo"]  # price ascending


@pytest.mark.asyncio
async def test_defaults_currency_and_nationality(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": [_hotel("A", 100)]}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("A", 100)]}
    await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20",
    )
    search_vars = next(c["variables"] for c in fake_hasura.calls if c["op"] == "search")
    assert search_vars["currency"] == srv._settings.default_currency
    assert search_vars["nationality"] == srv._settings.default_nationality


@pytest.mark.asyncio
async def test_occupancy_rooms_and_children(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": []}
    await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20",
        adults=2, childrenAges=[7], roomCount=2,
    )
    occ = next(c["variables"] for c in fake_hasura.calls if c["op"] == "search")["occupancies"]
    assert len(occ) == 2
    assert occ[0]["paxes"] == [{"age": 30}, {"age": 30}, {"age": 7}]


@pytest.mark.asyncio
async def test_polls_until_complete(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = Seq([
        {"isComplete": False, "hotels": []},
        {"isComplete": True, "hotels": [_hotel("A", 100)]},
    ])
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20",
    )
    assert out.isComplete is True
    assert [h.hotelName for h in out.hotels] == ["A"]
    assert sum(1 for c in fake_hasura.calls if c["op"] == "getSearchResults") == 2


@pytest.mark.asyncio
async def test_limit_top_n(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [_hotel(f"H{i}", i * 10) for i in range(1, 8)],
    }
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20", limit=5,
    )
    assert len(out.hotels) == 5


@pytest.mark.asyncio
async def test_unknown_city_returns_empty(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = []
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Nowhereville", checkIn="2026-08-15", checkOut="2026-08-20",
    )
    assert out.destination is None and out.hotels == []
    assert not any(c["op"] == "search" for c in fake_hasura.calls)  # no search fired


@pytest.mark.asyncio
async def test_no_availability(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": []}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20",
    )
    assert out.uuid == "U" and out.hotels == []


@pytest.mark.asyncio
async def test_filters_out_unpriced_hotels(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CTA", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    unpriced = {"hotelName": "Pending", "hotelCode": "P", "available": None, "price": None,
                "location": {"city": "Metroville", "country": "AE"}, "categoryCode": "4"}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "hotels": [_hotel("Alpha", 100), unpriced, _hotel("Cee", 150)]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20")
    assert [h.hotelName for h in out.hotels] == ["Alpha", "Cee"]  # null-priced dropped


@pytest.mark.asyncio
async def test_picks_best_matching_city_not_first(fake_hasura):
    # 'Metrovil' (SA) is listed before 'Metroville' (UAE); name match must win.
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CT2", "title": "Metrovil", "subtitle": "Saudi Arabia", "type": "CITY"},
        {"code": "CTA", "title": "Metroville", "subtitle": "United Arab Emirates", "type": "CITY"},
    ]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("A", 100)]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20")
    assert out.destination.code == "CTA"
    assert [d.code for d in out.alternatives] == ["CT2"]
    search_vars = next(c["variables"] for c in fake_hasura.calls if c["op"] == "search")
    assert search_vars["destinations"] == ["CTA"]


@pytest.mark.asyncio
async def test_destination_code_overrides_matching(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "CT2", "title": "Metrovil", "type": "CITY"},
        {"code": "CTA", "title": "Metroville", "type": "CITY"},
    ]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("A", 100)]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-15", checkOut="2026-08-20",
        destinationCode="CT2")
    assert out.destination.code == "CT2"
    search_vars = next(c["variables"] for c in fake_hasura.calls if c["op"] == "search")
    assert search_vars["destinations"] == ["CT2"]


@pytest.mark.asyncio
async def test_single_city_fallback_when_names_differ(fake_hasura):
    # 'Harborcity' resolves to city 'Bay City', listed after hotels — still chosen.
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "h1", "title": "Harborcity Hotel", "type": "HOTEL"},
        {"code": "20300", "title": "Bay City", "subtitle": "Saudi Arabia", "type": "CITY"},
    ]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("A", 100)]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Harborcity", checkIn="2026-08-15", checkOut="2026-08-20")
    assert out.destination.code == "20300"


import pytest
import hotel_tools as srv

ORG, CUR, NAT = "9f04d2c0", "USD", "SA"


@pytest.mark.asyncio
async def test_resolve_destination(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "MAK", "title": "Harborcity", "type": "CITY", "hotelCount": 20},
        {"code": "H9", "title": "Hilton", "type": "HOTEL"},
    ]
    out = await srv.resolve_destination("Harborcity")
    assert out[0].code == "MAK" and out[0].type == "CITY"
    assert fake_hasura.calls[0]["op"] == "destinationSearcher"
    assert fake_hasura.calls[0]["variables"]["criteria"]["query"] == "Harborcity"


@pytest.mark.asyncio
async def test_start_hotel_search_builds_occupancies(fake_hasura):
    fake_hasura.responses["search"] = {
        "uuid": "U1", "count": 1,
        "hotels": [{"hotelName": "Hilton", "hotelCode": "H1", "available": True,
                    "price": {"totalPrice": 120, "currency": "USD"},
                    "location": {"city": "Harborcity", "country": "SA"}}],
    }
    out = await srv.start_hotel_search(
        organizationId=ORG, currency=CUR, nationality=NAT,
        checkIn="2026-08-10", checkOut="2026-08-17", destinations=["MAK"],
        adults=2, childrenAges=[8],
    )
    assert out.uuid == "U1"
    occ = fake_hasura.calls[0]["variables"]["occupancies"]
    assert occ == [{"paxes": [{"age": 30}, {"age": 30}, {"age": 8}]}]


@pytest.mark.asyncio
async def test_start_hotel_search_rejects_bad_nationality(fake_hasura):
    from hotel_tools import VerifyError
    with pytest.raises(VerifyError):
        await srv.start_hotel_search(
            organizationId=ORG, currency=CUR, nationality="sau",
            checkIn="2026-08-10", checkOut="2026-08-17",
        )


@pytest.mark.asyncio
async def test_get_hotel_search_results_poll(fake_hasura):
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True,
        "hotels": [{"hotelName": "Hilton", "price": {"totalPrice": 99, "currency": "USD"}}],
    }
    out = await srv.get_hotel_search_results(
        organizationId=ORG, currency=CUR, nationality=NAT, uuid="U1",
    )
    assert out.isComplete is True
    v = fake_hasura.calls[0]["variables"]
    assert v["sort"] == {"field": "PRICE", "order": "asc"}


import pytest
import hotel_tools as srv

ORG, CUR, NAT = "9f04d2c0", "USD", "SA"


@pytest.mark.asyncio
async def test_get_hotel_booking(fake_hasura):
    fake_hasura.responses["Core_HotelBookings_by_pk"] = {"Id": 7, "BookingStatus": "CONFIRMED"}
    out = await srv.get_hotel_booking(organizationId=ORG, currency=CUR, nationality=NAT, Id=7)
    assert out.Id == 7 and out.BookingStatus == "CONFIRMED"


@pytest.mark.asyncio
async def test_get_hotel_booking_none(fake_hasura):
    fake_hasura.responses["Core_HotelBookings_by_pk"] = None
    out = await srv.get_hotel_booking(organizationId=ORG, currency=CUR, nationality=NAT, Id=7)
    assert out is None


@pytest.mark.asyncio
async def test_list_hotel_bookings_scopes_org_and_filters(fake_hasura):
    fake_hasura.responses["Core_HotelBookings"] = [{"Id": 1, "BookingStatus": "PENDING"}]
    out = await srv.list_hotel_bookings(
        organizationId=ORG, currency=CUR, nationality=NAT, bookingStatus="PENDING",
    )
    assert len(out) == 1
    where = fake_hasura.calls[0]["variables"]["where"]
    assert where["OrganizationId"] == {"_eq": ORG}
    assert where["BookingStatus"] == {"_eq": "PENDING"}


@pytest.mark.asyncio
async def test_list_hotel_markups(fake_hasura):
    fake_hasura.responses["Core_HotelMarkups"] = [{"id": 1, "amount": 10, "isPercentage": True}]
    out = await srv.list_hotel_markups(organizationId=ORG, currency=CUR, nationality=NAT)
    assert out[0].isPercentage is True


@pytest.mark.asyncio
async def test_list_hotel_cancellations(fake_hasura):
    fake_hasura.responses["Core_HotelCancels"] = [{"Id": 3, "Status": "PENDING"}]
    out = await srv.list_hotel_cancellations(organizationId=ORG, currency=CUR, nationality=NAT)
    assert out[0].Status == "PENDING"


@pytest.mark.asyncio
async def test_poll_hotel_booking_uses_hotelbookingid(fake_hasura):
    # QM confirmed the async book result is polled by the HotelBookingId column.
    fake_hasura.responses["Core_HotelBookings"] = [{"HotelBookingId": "HB-1", "BookingStatus": "CONFIRMED"}]
    out = await srv.poll_hotel_booking(organizationId=ORG, currency=CUR, nationality=NAT, hotelBookingId="HB-1")
    where = fake_hasura.calls[0]["variables"]["where"]
    assert where["HotelBookingId"] == {"_eq": "HB-1"}
    assert where["OrganizationId"] == {"_eq": ORG}
    assert out.BookingStatus == "CONFIRMED"


import json
import pytest
import hotel_tools as srv
from hotel_tools import VerifyError

ORG, CUR, NAT = "9f04d2c0", "USD", "SA"


def _env(status="success", data=None, message=None):
    return {"status": status, "message": message, "data": json.dumps(data or {})}


@pytest.mark.asyncio
async def test_refresh_records_freshness(fake_hasura, fresh_tracker):
    fake_hasura.responses["refreshHotelComponentSession"] = _env(data={"price": 100})
    out = await srv.refresh_hotel_price(
        organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1", applyMarkup=False,
    )
    assert out["price"] == 100
    # the tracker now allows booking the same option
    assert fresh_tracker.check_book(option_ref_id="O1", current_price=100, apply_markup=False) == []


@pytest.mark.asyncio
async def test_book_blocked_without_refresh(fake_hasura, fresh_tracker):
    fake_hasura.responses["bookAndIssue_Queue"] = _env(data={"hotelBookingId": "HB1"})
    with pytest.raises(VerifyError):
        await srv.book_hotel(organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1")


@pytest.mark.asyncio
async def test_book_succeeds_after_refresh(fake_hasura, fresh_tracker):
    fake_hasura.responses["refreshHotelComponentSession"] = _env(data={"price": 100})
    fake_hasura.responses["bookAndIssue_Queue"] = _env(data={"hotelBookingId": "HB1", "UUID": "U9"})
    await srv.refresh_hotel_price(organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1")
    ref = await srv.book_hotel(
        organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1", currentPrice=100,
    )
    assert ref.hotelBookingId == "HB1" and ref.uuid == "U9"


@pytest.mark.asyncio
async def test_book_blocked_on_price_drift(fake_hasura, fresh_tracker):
    fake_hasura.responses["refreshHotelComponentSession"] = _env(data={"price": 100})
    await srv.refresh_hotel_price(organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1")
    with pytest.raises(VerifyError):
        await srv.book_hotel(organizationId=ORG, currency=CUR, nationality=NAT,
                             optionRefId="O1", currentPrice=140)


@pytest.mark.asyncio
async def test_write_tool_surfaces_failed_envelope(fake_hasura, fresh_tracker):
    from hasura import GraphQLResponseError
    fake_hasura.responses["cancelHotel_V2"] = _env(status="error", message="not cancellable")
    with pytest.raises((GraphQLResponseError, VerifyError)) as e:
        await srv.cancel_hotel(organizationId=ORG, currency=CUR, nationality=NAT, hotelBookingId="HB1")
    assert "cancellable" in str(e.value) or "status" in str(e.value)


@pytest.mark.asyncio
async def test_send_cancel_request(fake_hasura, fresh_tracker):
    fake_hasura.responses["sendCancelRequestHotel_V2"] = _env(data={"requestId": "R1"})
    out = await srv.send_hotel_cancel_request(
        organizationId=ORG, currency=CUR, nationality=NAT, hotelBookingId="HB1", reason="changed",
    )
    assert out["requestId"] == "R1"
    # reason + booking id merged into the voidBooking input (confirmed live arg name)
    inp = fake_hasura.calls[0]["variables"]["voidBooking"]
    assert inp["hotelBookingId"] == "HB1" and inp["reason"] == "changed"
    assert inp["organizationId"] == ORG


@pytest.mark.asyncio
async def test_cancel_hotel_passes_bare_scalar(fake_hasura, fresh_tracker):
    # cancelHotel_V2 takes hotelBookingId as a bare String, no wrapping input.
    fake_hasura.responses["cancelHotel_V2"] = _env(data={"ok": True})
    await srv.cancel_hotel(organizationId=ORG, currency=CUR, nationality=NAT, hotelBookingId="HB1")
    v = fake_hasura.calls[0]["variables"]
    assert v == {"hotelBookingId": "HB1"}


@pytest.mark.asyncio
async def test_book_uses_coreMapper_arg(fake_hasura, fresh_tracker):
    # bookAndIssue_Queue takes a single coreMapper (CoreMapperInput) — not `input`.
    fake_hasura.responses["refreshHotelComponentSession"] = _env(data={"price": 100})
    fake_hasura.responses["bookAndIssue_Queue"] = _env(data={"hotelBookingId": "HB1"})
    await srv.refresh_hotel_price(organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1")
    await srv.book_hotel(organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1",
                        coreMapper={"totalPrice": 100})
    v = fake_hasura.calls[1]["variables"]
    assert "coreMapper" in v and "input" not in v
    assert v["coreMapper"]["organizationId"] == ORG  # tenant scoping injected


@pytest.mark.asyncio
async def test_refresh_builds_sessionData(fake_hasura, fresh_tracker):
    # refreshHotelComponentSession takes sessionData (HotelSessionDataInput).
    fake_hasura.responses["refreshHotelComponentSession"] = _env(data={"price": 100})
    await srv.refresh_hotel_price(
        organizationId=ORG, currency=CUR, nationality=NAT, optionRefId="O1",
        sessionData={"extraMarkup": 0.0, "serviceType": 1, "subtotal": 100.0,
                     "total": 100.0, "transactionSubTotal": 100.0, "transactionTotal": 100.0},
    )
    v = fake_hasura.calls[0]["variables"]
    assert "sessionData" in v and "input" not in v
    assert v["sessionData"]["optionRefId"] == "O1"
    assert v["sessionData"]["serviceType"] == 1


"""Guards for the concrete failure modes the team hit before (MCP issues doc)."""
import inspect
import pytest
import hotel_tools as srv

ORG = "9f04d2c0-afe2-42c7-a7b2-4f5bcd2b99f2"
CUR, NAT = "USD", "SA"


def test_org_id_never_cast_to_int():
    # #3: OrganizationId is a uuid string. Nothing in the module may int() it.
    src = inspect.getsource(srv)
    assert "int(org" not in src and "int(organizationId" not in src


@pytest.mark.asyncio
async def test_org_passed_as_string_in_where(fake_hasura):
    # #3: the uuid string flows straight into the bool_exp, unmodified.
    fake_hasura.responses["Core_HotelBookings"] = []
    await srv.list_hotel_bookings(organizationId=ORG, currency=CUR, nationality=NAT)
    where = fake_hasura.calls[0]["variables"]["where"]
    assert where["OrganizationId"] == {"_eq": ORG}
    assert isinstance(where["OrganizationId"]["_eq"], str)


@pytest.mark.asyncio
async def test_order_by_sent_as_variable_not_inline(fake_hasura):
    # #1: order_by goes through as a typed variable, so the enum isn't quoted
    # into the query string. Query text must not contain a literal order_by value.
    fake_hasura.responses["Core_HotelBookings"] = []
    await srv.list_hotel_bookings(organizationId=ORG, currency=CUR, nationality=NAT)
    call = fake_hasura.calls[0]
    assert call["variables"]["order_by"] == [{"CreatedAt": "desc"}]


def test_confirm_gate_set_matches_write_tools():
    assert srv.CONFIRM_GATE_TOOLS == {"book_hotel", "send_hotel_cancel_request", "cancel_hotel"}


def test_module_name():
    assert srv.MODULE == "hotels"


def test_tool_registry_sync_access():
    # #9: mcp.list_tools() is async; tests read the registry synchronously.
    names = set(srv.mcp._tool_manager._tools.keys())
    assert {"resolve_destination", "start_hotel_search", "book_hotel",
            "search_hotel_availability", "poll_hotel_booking"} <= names
    # ops tools register on the same MCP server
    assert {"get_queue_summary", "get_failed_messages", "run_named_query",
            "list_transactions", "get_transaction", "get_message_detail"} <= names
    assert {"get_hotel_static_data", "get_hotel_availability_options",
            "get_hotel_options"} <= names
    # 13 hotel + 6 ops + 3 hotel-detail tools
    assert len(names) == 22


def test_mutation_args_map_matches_live_schema():
    # #10: the four mutations do NOT share a wrapping `input` arg — each has its
    # own args and shape (confirmed via scripts/introspect.py). Pin the map so a
    # regression back to the old `_ACTION_ARG = "input"` design fails loudly.
    assert srv._MUTATION_ARGS == {
        "refreshHotelComponentSession": ("hotelBookingId", "sessionData", "transactionId"),
        "bookAndIssue_Queue": ("coreMapper",),
        "cancelHotel_V2": ("hotelBookingId",),
        "sendCancelRequestHotel_V2": ("voidBooking",),
    }
    assert not hasattr(srv, "_ACTION_ARG"), (
        "_ACTION_ARG was removed — mutations don't share one arg name. See _MUTATION_ARGS."
    )


def test_search_query_operation_names_match_field_names():
    # Hasura rejects the doc as "no such operation found" when operationName
    # doesn't match a `query <name>` in the document. Search-query op names
    # used to be PascalCase (DestinationSearcher) while the field is camelCase
    # (destinationSearcher) — that mismatch broke live calls. Guard it.
    for text, op in [
        (srv._Q_DESTINATION, "destinationSearcher"),
        (srv._Q_START_SEARCH, "search"),
        (srv._Q_GET_RESULTS, "getSearchResults"),
    ]:
        assert f"query {op}" in text, f"expected `query {op}` in the {op} query"


@pytest.mark.asyncio
async def test_sender_ip_header_attached_when_configured():
    # #6: loginRihla / some mutations reject requests without sender-ip.
    import httpx
    from hasura import HasuraClient
    cap = {}
    def handler(request):
        cap["headers"] = dict(request.headers)
        return httpx.Response(200, json={"data": {"op": 1}})
    c = HasuraClient("http://h/v1/graphql", admin_secret="s", sender_ip="1.2.3.4",
                     transport=httpx.MockTransport(handler))
    await c.request(query="q", variables={}, operation_name="op", token=None)
    assert cap["headers"]["sender-ip"] == "1.2.3.4"
    await c.aclose()


@pytest.mark.asyncio
async def test_no_city_match_returns_not_found(fake_hasura):
    # only hotel-type hits (like Bay City hotels for "Lakeside") -> not found, no search
    fake_hasura.responses["destinationSearcher"] = [
        {"code": "h1", "title": "Lakeside Palace Hotel", "type": "HOTEL"},
        {"code": "h2", "title": "Grand Lakeside", "type": "HOTEL"},
    ]
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Lakeside", checkIn="2026-08-15", checkOut="2026-08-17")
    assert out.destination is None and out.hotels == []
    assert not any(c["op"] == "search" for c in fake_hasura.calls)


@pytest.mark.asyncio
async def test_price_per_night_computed_by_tool(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CT1", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_hotel("A", 300)]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-08-10", checkOut="2026-08-13")
    assert out.nights == 3
    assert out.hotels[0].pricePerNight == 100.0


def _h(name, price, stars="4", amenities=None, available=True):
    return {"hotelName": name, "hotelCode": name[:3].upper(), "available": available,
            "price": {"totalPrice": price, "currency": "USD"},
            "location": {"city": "Metroville", "country": "XX"},
            "categoryCode": stars, "amenities": amenities or []}


# ---- sort option (SearchSortField has no STARS) ----

def test_build_sort_valid_fields_and_orders():
    from hotel_tools import build_sort
    assert build_sort("PRICE", "asc") == {"field": "PRICE", "order": "asc"}
    assert build_sort("rating", "DESC") == {"field": "RATING", "order": "desc"}
    assert build_sort("RECOMMENDED", "asc") == {"field": "RECOMMENDED", "order": "asc"}


def test_build_sort_rejects_stars_and_bad_order():
    from hotel_tools import build_sort
    with pytest.raises(ValueError):
        build_sort("STARS", "asc")
    with pytest.raises(ValueError):
        build_sort("PRICE", "ascending")


# ---- filters ----

def test_filters_price_stars_amenities():
    from hasura import SearchHotel
    from hotel_tools import apply_filters
    hotels = [SearchHotel.model_validate(x) for x in [
        _h("Cheap3", 80, "3", ["Free WiFi"]),
        _h("Mid4", 150, "4", ["Free WiFi", "Swimming pool"]),
        _h("High5", 400, "5", ["Spa"]),
    ]]
    assert [h.hotelName for h in apply_filters(hotels, max_price=200)] == ["Cheap3", "Mid4"]
    assert [h.hotelName for h in apply_filters(hotels, min_price=100, max_price=400)] == ["Mid4", "High5"]
    assert [h.hotelName for h in apply_filters(hotels, min_stars=4)] == ["Mid4", "High5"]
    assert [h.hotelName for h in apply_filters(hotels, min_stars=3, max_stars=4)] == ["Cheap3", "Mid4"]
    assert [h.hotelName for h in apply_filters(hotels, amenities=["wifi"])] == ["Cheap3", "Mid4"]
    assert [h.hotelName for h in apply_filters(hotels, amenities=["wifi", "pool"])] == ["Mid4"]
    assert apply_filters(hotels, amenities=["helipad"]) == []


def test_sort_hotels_by_price_and_rating():
    from hasura import SearchHotel
    from hotel_tools import sort_hotels
    hotels = [SearchHotel.model_validate(x) for x in [
        _h("B", 200, "3"), _h("A", 100, "5"), _h("C", 300, "4")]]
    assert [h.hotelName for h in sort_hotels(hotels, {"field": "PRICE", "order": "asc"})] == ["A", "B", "C"]
    assert [h.hotelName for h in sort_hotels(hotels, {"field": "PRICE", "order": "desc"})] == ["C", "B", "A"]
    assert [h.hotelName for h in sort_hotels(hotels, {"field": "RATING", "order": "desc"})] == ["A", "C", "B"]
    # RECOMMENDED keeps the supplier order
    assert [h.hotelName for h in sort_hotels(hotels, {"field": "RECOMMENDED", "order": "asc"})] == ["B", "A", "C"]


# ---- filters/sort/paging through the tools ----

@pytest.mark.asyncio
async def test_availability_applies_filters_and_sort(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CT1", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "count": 3, "hasMorePages": False,
        "hotels": [_h("Cheap3", 80, "3", ["Free WiFi"]),
                   _h("Mid4", 150, "4", ["Free WiFi", "Swimming pool"]),
                   _h("High5", 400, "5", ["Spa"])]}
    out = await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-09-01", checkOut="2026-09-04",
        maxPrice=200, minStars=4, amenities=["pool"], sortField="RATING", sortOrder="desc")
    assert [h.hotelName for h in out.hotels] == ["Mid4"]
    assert out.sort == {"field": "RATING", "order": "desc"}
    assert out.count == 3 and out.hasMorePages is False


@pytest.mark.asyncio
async def test_availability_sends_sort_and_page_to_api(fake_hasura):
    fake_hasura.responses["destinationSearcher"] = [{"code": "CT1", "title": "Metroville", "type": "CITY"}]
    fake_hasura.responses["search"] = {"uuid": "U1", "hotels": []}
    fake_hasura.responses["getSearchResults"] = {"isComplete": True, "hotels": [_h("A", 100)]}
    await srv.search_hotel_availability(
        organizationId=ORG, city="Metroville", checkIn="2026-09-01", checkOut="2026-09-04",
        sortField="RECOMMENDED", sortOrder="desc", pageNumber=2)
    v = next(c["variables"] for c in fake_hasura.calls if c["op"] == "getSearchResults")
    assert v["sort"] == {"field": "RECOMMENDED", "order": "desc"}
    assert v["pageNumber"] == 2


@pytest.mark.asyncio
async def test_availability_rejects_invalid_sort_field(fake_hasura):
    with pytest.raises(ValueError):
        await srv.search_hotel_availability(
            organizationId=ORG, city="Metroville", checkIn="2026-09-01", checkOut="2026-09-04",
            sortField="STARS")


@pytest.mark.asyncio
async def test_get_results_filters_and_reports_paging(fake_hasura):
    fake_hasura.responses["getSearchResults"] = {
        "isComplete": True, "count": 42, "hasMorePages": True,
        "hotels": [_h("Cheap3", 80, "3"), _h("Mid4", 150, "4"), _h("High5", 400, "5")]}
    page = await srv.get_hotel_search_results(
        organizationId=ORG, currency=CUR, nationality=NAT, uuid="U1",
        minStars=4, maxPrice=200, sortField="PRICE", sortOrder="asc", pageNumber=1, pageSize=20)
    assert [h.hotelName for h in page.hotels] == ["Mid4"]
    assert page.count == 42 and page.hasMorePages is True
    v = fake_hasura.calls[0]["variables"]
    assert v["pageNumber"] == 1 and v["pageSize"] == 20


# ---- hotel detail tools (static data, availability options, priced options) ----


@pytest.mark.asyncio
async def test_static_data_core_selection(fake_hasura):
    fake_hasura.responses["getSearchHotelStaticData"] = {
        "hotelCode": "H1", "hotelName": "Central Inn", "rating": "4", "cityName": "Metroville",
        "street": "1 Main St", "medias": [{"url": "http://img/1.jpg", "type": "IMAGE"}],
        "phones": [{"tech": "voice", "value": "+100"}]}
    out = await srv.get_hotel_static_data(hotelCode="H1")
    assert out.hotelName == "Central Inn" and out.rating == "4"
    assert out.medias[0].url == "http://img/1.jpg"
    assert out.phones[0].value == "+100"
    q = fake_hasura.calls[0]
    assert q["variables"] == {"hotelCode": "H1", "language": "en"}
    assert "facilities" not in q["query"] and "descriptions" not in q["query"]


@pytest.mark.asyncio
async def test_static_data_extras_opt_in(fake_hasura):
    fake_hasura.responses["getSearchHotelStaticData"] = {"hotelCode": "H1"}
    await srv.get_hotel_static_data(hotelCode="H1", extras=["facilities"])
    assert "facilities {" in fake_hasura.calls[0]["query"]
    with pytest.raises(ValueError):
        await srv.get_hotel_static_data(hotelCode="H1", extras=["nope"])


@pytest.mark.asyncio
async def test_hotel_options_builds_criteria_and_returns_option_ref(fake_hasura):
    fake_hasura.responses["getHotelFullOptions"] = {
        "hotelId": "H1", "uuid": "U9",
        "options": [
            {"optionRefId": "REF-1", "status": "OK", "price": {"totalPrice": 220, "currency": "USD"},
             "cancelPolicy": {"refundable": True, "description": "free until 48h"}},
            {"optionRefId": "REF-2", "status": "OK", "price": {"totalPrice": 500, "currency": "USD"},
             "cancelPolicy": {"refundable": False}},
        ]}
    out = await srv.get_hotel_options(
        organizationId=ORG, hotelCode="H1", checkIn="2026-09-01", checkOut="2026-09-04",
        adults=2, roomCount=2, refundableOnly=True)
    assert out.uuid == "U9"
    assert [o.optionRefId for o in out.options] == ["REF-1"]
    v = fake_hasura.calls[0]["variables"]
    assert v["applyMarkup"] is False
    assert v["criteria"]["hotels"] == ["H1"]
    assert v["criteria"]["checkIn"] == "2026-09-01" and v["criteria"]["checkOut"] == "2026-09-04"
    assert len(v["criteria"]["occupancies"]) == 2          # one RoomInput per room
    assert v["criteria"]["occupancies"][0]["paxes"] == [{"age": 30}, {"age": 30}]


# ---- getSearchHotelAvailability: [RoomSearch] { rooms, roomsOptions } ----

def _room(code, desc):
    return {"code": code, "description": desc, "refundable": None, "medias": None,
            "amenities": [{"code": "AC", "type": "HOTEL", "value": None, "texts": ["Air conditioning"]}]}


def _room_option(oid, total, board="Room Only", refundable=False, supplier_board="0"):
    return {"id": oid, "accessCode": "38900", "boardCodeSupplier": supplier_board,
            "boardText": board, "status": "OK", "rateRules": None, "remarks": None,
            "price": {"totalPrice": total, "net": total - 1, "gross": total - 1,
                      "markup": 1.0, "supplierPrice": total - 1, "currency": "USD"},
            "cancelPolicy": {"refundable": refundable,
                             "cancelPenalties": [{"currency": "USD", "value": total - 1,
                                                  "amount": total, "penaltyType": "IMPORT",
                                                  "deadline": "2026-08-30T00:00:00.000Z",
                                                  "hoursBefore": 34, "isFullAmount": True,
                                                  "isCalculatedDeadline": True}]},
            "surcharges": None}


def _group(room, options):
    return {"rooms": [room], "roomsOptions": options}


def test_flatten_pairs_each_room_with_each_option():
    from hotel_tools import flatten_room_options
    flat = flatten_room_options([
        _group(_room("R1", "Single Room"), [_room_option("OPT-1", 60.93)]),
        _group(_room("R2", "Double Room"), [_room_option("OPT-2", 63.05),
                                            _room_option("OPT-3", 70.0, "Breakfast")]),
    ])
    assert [(o.roomDescription, o.optionId) for o in flat] == [
        ("Single Room", "OPT-1"), ("Double Room", "OPT-2"), ("Double Room", "OPT-3")]
    first = flat[0]
    assert first.price.totalPrice == 60.93 and first.price.currency == "USD"
    assert first.refundable is False
    assert first.cancelPolicy.cancelPenalties[0].hoursBefore == 34
    assert first.roomAmenities[0].texts == ["Air conditioning"]


@pytest.mark.asyncio
async def test_availability_options_queries_and_flattens(fake_hasura):
    fake_hasura.responses["getSearchHotelAvailability"] = [
        _group(_room("R1", "Single Room"), [_room_option("OPT-1", 60.93)])]
    out = await srv.get_hotel_availability_options(
        organizationId=ORG, uuid="U1", hotelCode="503872")
    assert [o.roomDescription for o in out] == ["Single Room"]
    call = fake_hasura.calls[0]
    assert call["variables"] == {"uuid": "U1", "hotelCode": "503872",
                                 "organizationId": ORG, "language": "en"}
    # the selection must go through rooms / roomsOptions, not a flat option
    assert "roomsOptions {" in call["query"] and "rooms {" in call["query"]


@pytest.mark.asyncio
async def test_availability_options_filters_on_flattened_records(fake_hasura):
    fake_hasura.responses["getSearchHotelAvailability"] = [
        _group(_room("R1", "Single Room"), [_room_option("OPT-1", 100, "Room Only", refundable=False)]),
        _group(_room("R2", "Double Room"), [_room_option("OPT-2", 150, "Breakfast included", refundable=True)]),
        _group(_room("R3", "Suite"), [_room_option("OPT-3", 400, "Half Board", refundable=True)]),
    ]
    refundable = await srv.get_hotel_availability_options(
        organizationId=ORG, uuid="U1", hotelCode="H1", refundableOnly=True)
    assert [o.optionId for o in refundable] == ["OPT-2", "OPT-3"]

    breakfast = await srv.get_hotel_availability_options(
        organizationId=ORG, uuid="U1", hotelCode="H1", mealPlan=["breakfast"])
    assert [o.optionId for o in breakfast] == ["OPT-2"]

    capped = await srv.get_hotel_availability_options(
        organizationId=ORG, uuid="U1", hotelCode="H1", maxPrice=200)
    assert [o.optionId for o in capped] == ["OPT-1", "OPT-2"]
