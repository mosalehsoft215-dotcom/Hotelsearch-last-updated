from config import Settings


def test_yarvel_available_true_when_core_fields_set():
    s = Settings(_env_file=None, YARVEL_URL="http://x/v1/graphql", YARVEL_SECRET="s", YARVEL_ORG_ID="o")
    assert s.yarvel_available is True


def test_yarvel_available_false_without_secret():
    s = Settings(_env_file=None, YARVEL_URL="http://x/v1/graphql", YARVEL_SECRET=None, YARVEL_ORG_ID="o")
    assert s.yarvel_available is False


def test_defaults():
    s = Settings(_env_file=None, YARVEL_SECRET=None, YARVEL_ORG_ID=None)
    assert s.auth_mode in ("admin_secret", "forward_jwt")
    assert s.refresh_window_minutes == 30


from hasura import (
    Destination, HotelSearchStart, HotelSearchResults, SearchHotel,
    Core_HotelBookings, Core_HotelMarkups, Core_HotelCancels,
)


def test_destination_parses():
    d = Destination.model_validate({"code": "MAK", "title": "Harborcity", "type": "CITY", "hotelCount": 12})
    assert d.code == "MAK" and d.type == "CITY" and d.hotelCount == 12


def test_search_start_nested_price_location():
    s = HotelSearchStart.model_validate({
        "uuid": "u1", "count": 1,
        "hotels": [{"hotelName": "Hilton", "hotelCode": "H1", "available": True,
                    "price": {"totalPrice": 120.5, "net": 100, "currency": "USD"},
                    "location": {"city": "Harborcity", "country": "SA"}, "categoryCode": "5"}],
    })
    assert s.uuid == "u1"
    assert s.hotels[0].price.totalPrice == 120.5
    assert s.hotels[0].location.city == "Harborcity"


def test_results_is_complete():
    r = HotelSearchResults.model_validate({"isComplete": False, "hotels": []})
    assert r.isComplete is False


def test_core_tables_tolerate_extra_fields():
    b = Core_HotelBookings.model_validate({"Id": 5, "BookingStatus": "CONFIRMED", "weird": 1})
    assert b.Id == 5 and b.BookingStatus == "CONFIRMED"
    Core_HotelMarkups.model_validate({"id": 1, "amount": 10, "isPercentage": True})
    Core_HotelCancels.model_validate({"Id": 2, "Status": "PENDING"})


import json
import pytest
from hasura import unwrap, is_success, GraphQLResponseError
from hasura import HotelMutationPayload


def test_is_success_variants():
    assert is_success(True)
    assert is_success("success") and is_success("OK") and is_success("200")
    assert not is_success(False) and not is_success("error") and not is_success(0) and not is_success(None)


def test_unwrap_success():
    env = {"status": "success", "message": "ok", "data": json.dumps({"hotelBookingId": "HB1"})}
    out = unwrap(env, HotelMutationPayload).model_dump()
    assert out["hotelBookingId"] == "HB1"


def test_unwrap_non_success_raises_with_message():
    env = {"status": "error", "message": "supplier down", "data": None}
    with pytest.raises(GraphQLResponseError) as e:
        unwrap(env, HotelMutationPayload)
    assert "supplier down" in str(e.value)


def test_unwrap_data_must_be_string():
    env = {"status": "success", "message": None, "data": {"not": "a string"}}
    with pytest.raises(GraphQLResponseError):
        unwrap(env, HotelMutationPayload)


def test_unwrap_malformed_json():
    env = {"status": "success", "message": None, "data": "{not json"}
    with pytest.raises(GraphQLResponseError):
        unwrap(env, HotelMutationPayload)


def test_unwrap_null_data():
    env = {"status": "success", "message": None, "data": None}
    with pytest.raises(GraphQLResponseError):
        unwrap(env, HotelMutationPayload)


import json
import httpx
import pytest
from hasura import (
    HasuraClient, HasuraAuthError, HasuraGraphQLError, HasuraTransportError,
)


def _transport(captured, *, body=None, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["headers"] = dict(request.headers)
        captured["json"] = json.loads(request.content.decode())
        return httpx.Response(status, json=body if body is not None else {"data": {"op": {"ok": 1}}})
    return httpx.MockTransport(handler)


@pytest.mark.asyncio
async def test_forward_jwt_sets_bearer():
    cap = {}
    c = HasuraClient("http://h/v1/graphql", auth_mode="forward_jwt", transport=_transport(cap))
    out = await c.request(query="query op{ op }", variables={}, operation_name="op", token="JWT123")
    assert cap["headers"]["authorization"] == "Bearer JWT123"
    assert out == {"ok": 1}
    await c.aclose()


@pytest.mark.asyncio
async def test_admin_secret_used_when_no_token():
    cap = {}
    c = HasuraClient("http://h/v1/graphql", auth_mode="admin_secret",
                     admin_secret="SECRET", transport=_transport(cap))
    await c.request(query="query op{ op }", variables={}, operation_name="op", token=None)
    assert cap["headers"]["x-hasura-admin-secret"] == "SECRET"
    await c.aclose()


@pytest.mark.asyncio
async def test_no_credential_raises():
    c = HasuraClient("http://h/v1/graphql", auth_mode="forward_jwt", transport=_transport({}))
    with pytest.raises(HasuraAuthError):
        await c.request(query="q", variables={}, operation_name="op", token=None)
    await c.aclose()


@pytest.mark.asyncio
async def test_graphql_errors_raise():
    body = {"errors": [{"message": "field missing"}]}
    c = HasuraClient("http://h/v1/graphql", admin_secret="s", transport=_transport({}, body=body))
    with pytest.raises(HasuraGraphQLError) as e:
        await c.request(query="q", variables={}, operation_name="op", token=None)
    assert "field missing" in str(e.value)
    await c.aclose()


@pytest.mark.asyncio
async def test_http_error_raises():
    c = HasuraClient("http://h/v1/graphql", admin_secret="s",
                     transport=_transport({}, body={"x": 1}, status=500))
    with pytest.raises(HasuraTransportError):
        await c.request(query="q", variables={}, operation_name="op", token=None)
    await c.aclose()


@pytest.mark.asyncio
async def test_missing_operation_key_raises_keyerror():
    body = {"data": {"other": 1}}
    c = HasuraClient("http://h/v1/graphql", admin_secret="s", transport=_transport({}, body=body))
    with pytest.raises(KeyError):
        await c.request(query="q", variables={}, operation_name="op", token=None)
    await c.aclose()
