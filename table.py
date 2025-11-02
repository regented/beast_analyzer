#!/usr/bin/env python3
"""
Generates data.csv with per-tribe/tier counts for the Beasts collection.

Output CSV columns (exact order):
  tribe,tier,minted,shiny,animated,special

Definitions (from metadata.attributes):
  - tribe = value where trait_type == "Beast"
  - tier  = value where trait_type == "Tier" (note capital T)
  - shiny count    = Shiny == "1" and Animated == "0"
  - animated count = Shiny == "0" and Animated == "1"
  - special count  = Shiny == "1" and Animated == "1"
  - minted = total tokens aggregated for that (tribe, tier)

Environment:
  - TORII_SQL_ENDPOINT (default: https://api.cartridge.gg/x/pg-beasts/torii/sql)
  - BATCH_SIZE (optional; default: 5000)

The script queries the tokens table via Torii SQL in pages:
  SELECT metadata FROM tokens ORDER BY token_id LIMIT {BATCH} OFFSET {OFFSET}

We do not use chain RPC or contract address; the entire set is fetched from the tokens table.
"""

import os
import csv
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

# ---------------- Env & constants ----------------

load_dotenv()

TORII_SQL_ENDPOINT = os.getenv(
    "TORII_SQL_ENDPOINT",
    "https://api.cartridge.gg/x/pg-beasts/torii/sql",
).strip()

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5000"))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "60"))  # seconds
RETRY_BACKOFF = float(os.getenv("RETRY_BACKOFF", "1.5"))     # seconds
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))

OUT_CSV = "data.csv"

# ---------------- SQL helpers ----------------

def build_paged_sql(limit: int, offset: int) -> str:
    # Only fetch 'metadata' (TEXT JSON) to keep payload smaller
    # We order by token_id for deterministic pagination.
    return f"""
        SELECT metadata
        FROM tokens
        WHERE contract_address = '0x046da8955829adf2bda310099a0063451923f02e648cf25a1203aac6335cf0e4'
        ORDER BY token_id
        LIMIT {int(limit)}
        OFFSET {int(offset)};
    """.strip()

def parse_torii_payload(payload: Any) -> List[Dict[str, Any]]:
    """
    Normalize Torii SQL results to a list of dict rows with key 'metadata'.
    Supports:
      - { "columns": [...], "rows": [[...], ...] }
      - { "rows": [ {col:val, ...}, ... ] }
      - [ { "metadata": ... }, ... ]
    """
    rows_out: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        # Variant: columns + rows (matrix)
        if "rows" in payload and "columns" in payload and isinstance(payload["rows"], list):
            cols = [str(c) for c in payload.get("columns", [])]
            for r in payload["rows"]:
                if isinstance(r, list):
                    row = {cols[i]: r[i] for i in range(min(len(cols), len(r)))}
                    rows_out.append(row)
            return rows_out
        # Variant: rows as list of dicts
        if "rows" in payload and isinstance(payload["rows"], list):
            for r in payload["rows"]:
                if isinstance(r, dict):
                    rows_out.append(r)
            return rows_out
        # Single row dict
        if "metadata" in payload:
            rows_out.append({"metadata": payload.get("metadata")})
            return rows_out

    # Variant: list of dicts
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                # Keep only metadata key if present
                if "metadata" in item:
                    rows_out.append({"metadata": item["metadata"]})
                else:
                    rows_out.append(item)

    return rows_out

def torii_query(sql: str) -> List[Dict[str, Any]]:
    """
    Execute a SQL query against Torii. Try GET first (with ?query), then POST text/plain.
    Includes basic retry for transient failures.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
                # Try GET
                r = client.get(TORII_SQL_ENDPOINT, params={"query": sql})
                if r.status_code >= 300:
                    # Fallback POST
                    r = client.post(
                        TORII_SQL_ENDPOINT,
                        content=sql,
                        headers={"Content-Type": "text/plain"},
                    )
                r.raise_for_status()
                data = r.json()
                return parse_torii_payload(data)
        except Exception as e:
            last_err = e
            if attempt == MAX_RETRIES:
                break
            sleep_s = RETRY_BACKOFF ** attempt
            print(f"[warn] Torii query failed (attempt {attempt}/{MAX_RETRIES}): {e}. Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)
    raise RuntimeError(f"Torii SQL request failed after {MAX_RETRIES} attempts: {last_err}")

def fetch_all_metadata(batch_size: int = BATCH_SIZE) -> List[Dict[str, Any]]:
    """
    Paginates through the tokens table and returns a list of dicts: { "metadata": <json-or-str> }.
    """
    results: List[Dict[str, Any]] = []
    offset = 0
    total_rows = 0
    while True:
        sql = build_paged_sql(limit=batch_size, offset=offset)
        rows = torii_query(sql)
        if not rows:
            break
        results.extend(rows)
        fetched = len(rows)
        total_rows += fetched
        print(f"Fetched {fetched} rows (total {total_rows})...")
        if fetched < batch_size:
            # last page
            break
        offset += batch_size
    return results

# ---------------- Metadata parsing ----------------

def to_str_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        return str(v)
    except Exception:
        return None

def attr_lookup(attributes: Any, trait_type_exact: str) -> Optional[str]:
    """
    Returns value (as string) for the first attribute where trait_type equals trait_type_exact.
    Only matches exact case (e.g., "Tier" not "tier") per requirement.
    Accepts attributes as a list of objects (OpenSea-like format).
    """
    if isinstance(attributes, list):
        for a in attributes:
            if not isinstance(a, dict):
                continue
            ttype = to_str_or_none(a.get("trait_type"))
            if ttype == trait_type_exact:
                return to_str_or_none(a.get("value"))
    # Some collections store attributes as dicts; keep strict to requirement,
    # but if dict-like and exact key exists, use it.
    if isinstance(attributes, dict) and trait_type_exact in attributes:
        return to_str_or_none(attributes.get(trait_type_exact))
    return None

def parse_token_metadata(meta_raw: Any) -> Tuple[str, str, int, int]:
    """
    Parses a single row's metadata into:
      - tribe (Beast)
      - tier  (Tier)
      - shiny flag (0/1)
      - animated flag (0/1)
    Missing fields default to "Unknown" for tribe/tier and 0 for flags.
    """
    tribe = "Unknown"
    tier = "Unknown"
    shiny = 0
    animated = 0

    meta: Optional[Dict[str, Any]] = None

    if isinstance(meta_raw, dict):
        meta = meta_raw
    elif isinstance(meta_raw, str):
        try:
            meta = json.loads(meta_raw)
        except Exception:
            meta = None

    if meta and isinstance(meta, dict):
        attributes = meta.get("attributes")
        # Required exact-case fields per spec
        tribe_val = attr_lookup(attributes, "Beast")
        tier_val = attr_lookup(attributes, "Tier")
        shiny_val = attr_lookup(attributes, "Shiny")
        animated_val = attr_lookup(attributes, "Animated")

        if tribe_val:
            tribe = tribe_val
        if tier_val:
            tier = tier_val

        def to01(s: Optional[str]) -> int:
            if s is None:
                return 0
            ss = s.strip()
            # Only "1" is true per requirement
            return 1 if ss == "1" else 0

        shiny = to01(shiny_val)
        animated = to01(animated_val)

    return tribe, tier, shiny, animated

# ---------------- Aggregation ----------------

def aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, int]]:
    """
    Aggregates counts per (tribe, tier).

    Returns:
      { (tribe, tier): { minted, shiny, animated, special } }
    """
    agg: Dict[Tuple[str, str], Dict[str, int]] = {}

    for row in rows:
        meta_raw = row.get("metadata")
        tribe, tier, shiny, animated = parse_token_metadata(meta_raw)

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
        # If both 0 → regular; not counted in trait columns.

    return agg

# ---------------- CSV writing ----------------

def write_csv(agg: Dict[Tuple[str, str], Dict[str, int]], out_path: str = OUT_CSV) -> str:
    # Deterministic ordering: tribe (case-insensitive), then tier (string)
    sorted_keys = sorted(agg.keys(), key=lambda x: (str(x[0]).lower(), str(x[1]).lower()))
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tribe", "tier", "minted", "shiny", "animated", "special"])
        for (tribe, tier) in sorted_keys:
            stats = agg[(tribe, tier)]
            w.writerow([
                tribe,
                tier,
                stats["minted"],
                stats["shiny"],
                stats["animated"],
                stats["special"],
            ])
    return out_path

# ---------------- Main ----------------

def main():
    if not TORII_SQL_ENDPOINT:
        raise SystemExit("TORII_SQL_ENDPOINT is required (set in .env or environment).")

    print(f"Using Torii SQL endpoint: {TORII_SQL_ENDPOINT}")
    print(f"Batch size: {BATCH_SIZE}")

    rows = fetch_all_metadata(batch_size=BATCH_SIZE)
    total = len(rows)
    if total == 0:
        print("No rows returned from tokens table. Exiting without writing CSV.")
        return

    agg = aggregate(rows)
    out_path = write_csv(agg, OUT_CSV)

    distinct_pairs = len(agg)
    minted_total = sum(v["minted"] for v in agg.values())
    shiny_total = sum(v["shiny"] for v in agg.values())
    animated_total = sum(v["animated"] for v in agg.values())
    special_total = sum(v["special"] for v in agg.values())

    print(f"Wrote {out_path} with {distinct_pairs} tribe/tier rows.")
    print(f"Totals across all rows fetched: minted={minted_total}, shiny={shiny_total}, animated={animated_total}, special={special_total}")

if __name__ == "__main__":
    main()