#!/usr/bin/env python3
"""
Generates data.csv with per-tribe/tier counts for the Beasts collection.

What it does:
1) Reads .env for:
   - RPC_URL
   - COLLECTION (default provided if missing)
   - TORII_SQL_ENDPOINT
2) Calls collection.total_supply() on StarkNet to get total supply
3) Queries Torii SQL for all tokens' metadata (LIMIT = total supply)
4) Aggregates counts into data.csv with columns:
   tribe,tier,minted,shiny,animated,special
   - tribe: attributes[].trait_type == "Beast" → value
   - tier:  attributes[].trait_type == "tier" → value (case-insensitive)
   - minted: total tokens for that tribe+tier
   - shiny: shiny==1 and animated==0
   - animated: shiny==0 and animated==1
   - special: shiny==1 and animated==1
"""

import os
import csv
import json
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from starknet_py.net.full_node_client import FullNodeClient
from starknet_py.net.client_models import Call
from starknet_py.hash.selector import get_selector_from_name

# ---------------- Env & Constants ----------------

load_dotenv()

RPC_URL = os.getenv("RPC_URL", "").strip()
COLLECTION_ADDR = (os.getenv("COLLECTION", "") or "0x046dA8955829ADF2bDa310099A0063451923f02E648cF25A1203aac6335CF0e4").strip()
TORII_SQL_ENDPOINT = os.getenv("TORII_SQL_ENDPOINT", "https://api.cartridge.gg/x/pg-beasts/torii/sql").strip()

if not RPC_URL:
    raise SystemExit("RPC_URL is required in .env")
if not COLLECTION_ADDR:
    raise SystemExit("COLLECTION (contract address) is required in .env")

# Normalize address styles
COLLECTION_ADDR_INT = int(COLLECTION_ADDR, 16)
COLLECTION_ADDR_HEX_LOWER = "0x" + format(COLLECTION_ADDR_INT, "x")  # lowercase 0x…


# ---------------- StarkNet helpers ----------------

def u256_join(low: int, high: int) -> int:
    return (high << 128) + low

async def call_view(
    client: FullNodeClient,
    addr_hex: str,
    selector: int,
    calldata: List[int] | Tuple[int, ...] = (),
) -> List[int]:
    call = Call(to_addr=int(addr_hex, 16), selector=selector, calldata=list(calldata))
    return await client.call_contract(call)

async def get_total_supply(client: FullNodeClient, collection_hex: str) -> int:
    """
    Tries common selectors: total_supply (snake) then totalSupply (camel).
    Supports return types: felt or (low, high) u256.
    """
    selectors = [
        get_selector_from_name("total_supply"),
        get_selector_from_name("totalSupply"),
    ]
    last_err: Optional[Exception] = None
    for sel in selectors:
        try:
            res = await call_view(client, collection_hex, sel, [])
            if not res:
                continue
            if len(res) == 1:
                return int(res[0])
            if len(res) >= 2:
                return int(u256_join(int(res[0]), int(res[1])))
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to call total supply on {collection_hex}: {last_err}")


# ---------------- Torii SQL helpers ----------------

def build_sql(total_supply: int) -> str:
    # Use single quotes for string literal (Postgres style)
    return (
        "SELECT metadata, id "
        f"FROM tokens "
        f"LIMIT {int(total_supply)};"
    )

def parse_torii_rows(payload: Any) -> List[Dict[str, Any]]:
    """
    Accepts either:
      - { "columns": [...], "rows": [[...], ...] }
      - { "rows": [ {col:val, ...}, ... ] }
      - [ { "metadata": ..., "id": ... }, ... ]
    Returns a list of dict rows.
    """
    rows_out: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        if "rows" in payload and "columns" in payload and isinstance(payload["rows"], list):
            cols = [str(c) for c in payload.get("columns", [])]
            for r in payload["rows"]:
                if isinstance(r, list):
                    row = {cols[i]: r[i] for i in range(min(len(cols), len(r)))}
                    rows_out.append(row)
            return rows_out
        if "rows" in payload and isinstance(payload["rows"], list):
            for r in payload["rows"]:
                if isinstance(r, dict):
                    rows_out.append(r)
            return rows_out
        if "metadata" in payload or "id" in payload:
            rows_out.append(payload)
            return rows_out

    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                rows_out.append(item)
    return rows_out

async def torii_query_all(sql: str) -> List[Dict[str, Any]]:
    """
    GET with ?query=... first, fallback to POST text/plain.
    """
    async with httpx.AsyncClient(timeout=30.0) as client:
        r = await client.get(TORII_SQL_ENDPOINT, params={"query": sql})
        if r.status_code >= 300:
            r = await client.post(TORII_SQL_ENDPOINT, content=sql, headers={"Content-Type": "text/plain"})
        r.raise_for_status()
        data = r.json()
    return parse_torii_rows(data)


# ---------------- Metadata parsing & aggregation ----------------

def get_attr_value(attrs: Any, trait_name: str) -> Optional[str]:
    """
    Finds attributes[].trait_type == trait_name (case-insensitive) and returns its value as string.
    Supports attributes being a list of objects or a dict mapping.
    """
    if isinstance(attrs, list):
        for a in attrs:
            if isinstance(a, dict):
                t = str(a.get("trait_type", "")).strip()
                if t.lower() == trait_name.lower():
                    return None if a.get("value") is None else str(a.get("value"))
    elif isinstance(attrs, dict):
        for k, v in attrs.items():
            if str(k).lower() == trait_name.lower():
                return None if v is None else str(v)
    return None

def parse_metadata(meta_raw: Any) -> Dict[str, Any]:
    """
    Normalizes metadata into a dict and extracts:
      - tribe (Beast)
      - tier (trait_type "tier")
      - shiny (int 0/1)
      - animated (int 0/1)
    """
    meta: Optional[Dict[str, Any]] = None
    if isinstance(meta_raw, dict):
        meta = meta_raw
    elif isinstance(meta_raw, str):
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = None

    tribe = "Unknown"
    tier = "Unknown"
    shiny_i = 0
    animated_i = 0

    if meta:
        attrs = meta.get("attributes")
        tribe_val = get_attr_value(attrs, "Beast")
        if tribe_val:
            tribe = str(tribe_val)

        tier_val = get_attr_value(attrs, "tier")
        if tier_val:
            tier = str(tier_val)

        shiny_val = get_attr_value(attrs, "Shiny")
        animated_val = get_attr_value(attrs, "Animated")

        def to_int01(v: Optional[str]) -> int:
            if v is None:
                return 0
            s = str(v).strip().lower()
            if s in ("1", "true", "yes"):
                return 1
            if s in ("0", "false", "no"):
                return 0
            try:
                return 1 if int(float(s)) != 0 else 0
            except Exception:
                return 0

        shiny_i = to_int01(shiny_val)
        animated_i = to_int01(animated_val)

    return {"tribe": tribe, "tier": tier, "shiny": shiny_i, "animated": animated_i}

def aggregate_counts(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Returns:
      { (tribe, tier): { minted, shiny, animated, special } }
    """
    agg: Dict[Tuple[str, str], Dict[str, int]] = {}

    for row in rows:
        meta_raw = row.get("metadata")
        parsed = parse_metadata(meta_raw)
        tribe = parsed["tribe"]
        tier = parsed["tier"]
        shiny = int(parsed["shiny"])
        animated = int(parsed["animated"])

        key = (tribe, tier)
        if key not in agg:
            agg[key] = {"minted": 0, "shiny": 0, "animated": 0, "special": 0}

        agg[key]["minted"] += 1
        if shiny == 1 and animated == 0:
            agg[key]["shiny"] += 1
        elif shiny == 0 and animated == 1:
            agg[key]["animated"] += 1
        elif shiny == 1 and animated == 1:
            agg[key]["special"] += 1

    return agg


# ---------------- Main ----------------

async def main():
    # 1) Get total supply from chain
    client = FullNodeClient(node_url=RPC_URL)
    total_supply = await get_total_supply(client, COLLECTION_ADDR)
    print(f"Total supply reported on-chain: {total_supply}")

    # 2) Query Torii SQL for all tokens' metadata (LIMIT total_supply)
    sql = build_sql(total_supply)
    rows = await torii_query_all(sql)
    print(f"Fetched {len(rows)} rows from Torii.")

    if not rows:
        print("No rows returned. Exiting.")
        return

    # 3) Aggregate and write CSV
    agg = aggregate_counts(rows)
    out_path = "data.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tribe", "tier", "minted", "shiny", "animated", "special"])
        # Sort by tribe then tier for deterministic output
        for (tribe, tier) in sorted(agg.keys(), key=lambda x: (x[0].lower(), str(x[1]).lower())):
            stats = agg[(tribe, tier)]
            writer.writerow([
                tribe,
                tier,
                stats["minted"],
                stats["shiny"],
                stats["animated"],
                stats["special"],
            ])

    print(f"Wrote {out_path} with {len(agg)} tribe/tier rows.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())