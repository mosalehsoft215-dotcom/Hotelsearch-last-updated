"""Prove the room-detail endpoints work, without the agent in the way.

    python check_room_options.py <hotelCode> [checkIn] [checkOut]
    python check_room_options.py 140143 2026-09-01 2026-09-04
"""
from __future__ import annotations

import asyncio
import sys

import hotel_tools
from config import get_settings


async def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    code = sys.argv[1]
    check_in = sys.argv[2] if len(sys.argv) > 2 else "2026-09-01"
    check_out = sys.argv[3] if len(sys.argv) > 3 else "2026-09-04"
    org = get_settings().yarvel_org_id
    if not org:
        print("FAIL  YARVEL_ORG_ID is not set in .env")
        return 2

    print(f"hotelCode={code}  {check_in} -> {check_out}  org={org}")
    print("registered tools:", len(hotel_tools.mcp._tool_manager._tools))

    try:
        static = await hotel_tools.get_hotel_static_data(hotelCode=code)
        print(f"OK    static data: {getattr(static, 'hotelName', None)} "
              f"rating={getattr(static, 'rating', None)}")
    except Exception as exc:
        print(f"FAIL  get_hotel_static_data: {type(exc).__name__}: {exc}")

    try:
        res = await hotel_tools.get_hotel_options(
            organizationId=org, hotelCode=code, checkIn=check_in, checkOut=check_out,
            adults=2, roomCount=1)
        options = res.options or []
        print(f"OK    get_hotel_options: uuid={res.uuid} options={len(options)}")
        for o in options[:5]:
            policy = o.cancelPolicy
            print(f"        {o.optionRefId}  {getattr(o.price, 'totalPrice', None)} "
                  f"{getattr(o.price, 'currency', '')}  refundable={getattr(policy, 'refundable', None)}")
    except Exception as exc:
        print(f"FAIL  get_hotel_options: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
