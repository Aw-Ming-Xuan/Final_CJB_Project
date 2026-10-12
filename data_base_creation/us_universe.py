"""US market universe: which tickers count as "US stocks" for this project.

Definition used: common stocks, REITs and ADRs listed on NYSE, Nasdaq or NYSE American.
ADRs are foreign companies traded in the US in US dollars (e.g. TSM, BABA, SONY, NVO).
Excluded: ETFs/ETPs, OTC names, warrants, crypto.

The list comes from ONE Finnhub call (/stock/symbol?exchange=US, free tier) and is
cached in the us_universe table for 7 days.

To widen or narrow the definition, edit ALLOWED_TYPES / ALLOWED_MICS below, e.g.
remove "ADR" to go back to US-only companies.
"""

import os
from collections import Counter
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

from data_base_creation.database import USUniverse

load_dotenv()

API_KEY = os.getenv("FINNHUB_API_KEY")
SYMBOL_URL = "https://finnhub.io/api/v1/stock/symbol"

ALLOWED_TYPES = {"Common Stock", "REIT", "ADR"}
ALLOWED_MICS = {"XNAS", "XNYS", "XASE"}   # Nasdaq, NYSE, NYSE American
REFRESH_DAYS = 7

_cache = None   # symbols for this process, so we only query the DB once


def _now():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def refresh_us_universe(session, force: bool = False) -> int:
    """Download the US symbol list if the stored one is missing or stale.
    Returns the number of rows stored (0 if nothing was refreshed)."""
    global _cache

    latest = session.query(USUniverse.updated_at).order_by(USUniverse.updated_at.desc()).first()
    if latest and not force and latest[0] and _now() - latest[0] < timedelta(days=REFRESH_DAYS):
        return 0

    if not API_KEY:
        print("  US universe: no FINNHUB_API_KEY, cannot refresh.")
        return 0

    try:
        res = requests.get(SYMBOL_URL, params={"exchange": "US"},
                           headers={"X-Finnhub-Token": API_KEY}, timeout=60)
    except requests.RequestException as err:
        print(f"  US universe: request failed ({err}); keeping the old list.")
        return 0

    if res.status_code != 200:
        print(f"  US universe: HTTP {res.status_code}; keeping the old list.")
        return 0

    rows = {}
    for item in res.json():
        symbol = (item.get("symbol") or "").upper()
        if not symbol:
            continue
        rows[symbol] = {
            "symbol": symbol,
            "description": (item.get("description") or "")[:255],
            "type": item.get("type") or "",
            "mic": item.get("mic") or "",
            "updated_at": _now(),
        }

    if not rows:
        print("  US universe: empty response; keeping the old list.")
        return 0

    session.query(USUniverse).delete()
    session.bulk_insert_mappings(USUniverse, list(rows.values()))
    session.commit()
    _cache = None

    kept = sum(1 for r in rows.values()
               if r["type"] in ALLOWED_TYPES and r["mic"] in ALLOWED_MICS)
    types = Counter(r["type"] for r in rows.values()).most_common(6)
    print(f"  US universe refreshed: {len(rows)} symbols, {kept} match "
          f"{sorted(ALLOWED_TYPES)} on {sorted(ALLOWED_MICS)}.")
    print(f"  Types seen: {types}")
    return len(rows)


def load_us_symbols(session) -> set:
    """Set of allowed US tickers. Empty set means 'universe not available';
    callers then skip US filtering (and should say so)."""
    global _cache
    if _cache is None:
        rows = (
            session.query(USUniverse.symbol)
            .filter(USUniverse.type.in_(ALLOWED_TYPES), USUniverse.mic.in_(ALLOWED_MICS))
            .all()
        )
        _cache = {r[0] for r in rows}
    return _cache
