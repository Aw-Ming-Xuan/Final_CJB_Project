"""Phase 2: quantitative market and volume hard filters.

Input : the Phase 1 summary (one row per ticker) and mention rows.
Output: (survivors, mentions_for_survivors) for Phase 3.

The four filters from the plan (all are hard filters, so a ticker must pass every one):

  1. PENNY STOCK / SCAM EXCLUSION
       market cap >= MIN_MARKET_CAP (300M), plus extra guards: price floor, a minimum
       average dollar volume, a minimum trading history, and no SPAC / blank-check names.
       Data: Yahoo Finance first, then Finnhub, then Alpha Vantage (only if a key is set).
  2. DISCUSSION VOLUME
       chatter z-score = how unusual today's chatter is. Then a cross-check against real
       trading volume from Yahoo Finance: a big spike in talk with ordinary trading is
       treated as a synthetic social push and rejected.
  3. DISTINCT USERS
       at least MIN_UNIQUE_AUTHOR_RATIO of the StockTwits posts must come from different
       accounts (only judged when there are enough posts to judge).
  4. FUNDAMENTALS
       P/E, quarterly revenue growth (YoY) and debt-to-equity within sane limits.

To save API calls, the cheap local filters (chatter size, z-score, author ratio) run first,
and market data is fetched only for the survivors, best chatter first (MAX_CANDIDATES).

Z-SCORE BASELINE
  - 'history'      : the ticker's own past snapshots in phase1_history (needs
                     MIN_HISTORY_SNAPSHOTS earlier days). This is the real "vs its normal
                     level" measure.
  - 'cross-section': until enough history exists, today's chatter is compared with the
                     other tickers (robust z on log counts). That says "popular compared
                     with the rest of the market today", NOT "unusual for this stock".
  Run the pipeline daily (or weekly) so real history builds up.
  The "chatter spike but normal trading = synthetic push" rule only applies to the
  'history' z-score; in cross-section mode only the general volume floor applies.

Run:  python filtering_of_data/phase2_filter.py             (runs Phase 1 first)
      python filtering_of_data/phase2_filter.py --refresh   (ignore cached market data)
Optional .env keys (data_base_creation/.env): FINNHUB_API_KEY, ALPHAVANTAGE_API_KEY
Requires: pip install yfinance requests python-dotenv pandas numpy
"""

import math
import os
import re
import sys
import time
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, "data_base_creation", ".env"))

from data_base_creation.database import (
    SessionLocal, init_db, Phase1History, MarketSnapshot, USUniverse,
)

try:
    from filtering_of_data.phase1_filter import run_phase1
except ImportError:                       # running from inside filtering_of_data/
    from phase1_filter import run_phase1

# ----------------------------------------------------------------------
# Settings (starting points: tune them from the per-stage counts printed below)
# ----------------------------------------------------------------------
# Filter 1: penny stock / scam exclusion
MIN_MARKET_CAP = 300_000_000
MIN_PRICE = 1.0
MIN_AVG_DOLLAR_VOLUME = 5_000_000      # average daily traded value
MIN_TRADING_DAYS = 30                  # sessions of history found in the last 3 months
SPAC_NAME_RE = re.compile(r"\b(acquisition|blank check|spac)\b", re.I)

# Filter 2: discussion volume
MIN_CHATTER = 5                        # ignore tickers with fewer social + unbiased news mentions
MIN_HISTORY_SNAPSHOTS = 3              # earlier daily snapshots needed for the 'history' z-score
MIN_Z_SCORE = 1.0
SPIKE_Z = 2.0                          # a "massive" jump in chatter...
SPIKE_MIN_VOLUME_RATIO = 1.2           # ...must come with clearly elevated trading volume
MIN_VOLUME_RATIO = 0.8                 # otherwise trading must not be well below its norm

# Filter 3: distinct users
MIN_UNIQUE_AUTHOR_RATIO = 0.70
MIN_SOCIAL_FOR_RATIO = 5               # below this many posts the ratio is not judged

# Filter 4: fundamentals (a missing metric is allowed, but >= 2 of 3 must be known)
MAX_PE = 60.0                          # forward P/E, else trailing
REQUIRE_POSITIVE_PE = True             # loss-making companies have no meaningful P/E
MIN_REVENUE_GROWTH = 0.0               # quarterly YoY, 0.10 = +10%
MAX_DEBT_TO_EQUITY = 200.0             # percent, as Yahoo reports it
MIN_KNOWN_FUNDAMENTALS = 2

MAX_CANDIDATES = 60
MARKET_CACHE_HOURS = 12
PAUSE_SECONDS = 0.4

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
ALPHA_KEY = os.getenv("ALPHAVANTAGE_API_KEY")


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _num(value):
    """Float or None (handles None, NaN, 'N/A', strings)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if (math.isnan(f) or math.isinf(f)) else f


def fmt_money(value) -> str:
    if value is None:
        return "n/a"
    for limit, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M")):
        if abs(value) >= limit:
            return f"${value / limit:.2f}{suffix}"
    return f"${value:,.0f}"


# ----------------------------------------------------------------------
# Filter 2a: chatter z-score
# ----------------------------------------------------------------------
def load_history(session, today=None) -> dict:
    """ticker -> list of earlier daily chatter scores (today's snapshot is excluded)."""
    today = today or _utcnow().date()
    rows = (
        session.query(Phase1History.ticker, Phase1History.chatter_score)
        .filter(Phase1History.snapshot_date < today)
        .all()
    )
    history = {}
    for ticker, score in rows:
        history.setdefault(ticker, []).append(float(score or 0))
    return history


def compute_z_scores(summary: pd.DataFrame, history: dict) -> pd.DataFrame:
    """Adds z_score and z_method. History baseline when available, else cross-section."""
    out = summary.copy()
    logs = np.log1p(out["chatter_score"].astype(float))
    median = float(logs.median())
    mad = float((logs - median).abs().median()) * 1.4826
    spread = mad if mad > 0 else (float(logs.std()) or 1.0)

    z_values, methods = [], []
    for ticker, score, log_score in zip(out["ticker"], out["chatter_score"], logs):
        past = history.get(ticker, [])
        if len(past) >= MIN_HISTORY_SNAPSHOTS:
            mean = float(np.mean(past))
            # floor on the spread: a ticker that is usually quiet must not explode the z-score
            std = max(float(np.std(past, ddof=1)), math.sqrt(max(mean, 1.0)), 1.0)
            z_values.append((float(score) - mean) / std)
            methods.append("history")
        else:
            z_values.append((float(log_score) - median) / spread)
            methods.append("cross-section")
    out["z_score"] = np.round(z_values, 2)
    out["z_method"] = methods
    return out


# ----------------------------------------------------------------------
# Market data: Yahoo Finance, then Finnhub, then Alpha Vantage
# ----------------------------------------------------------------------
def drop_unfinished_session(hist: pd.DataFrame, now_ny=None) -> pd.DataFrame:
    """Yahoo's daily history includes TODAY's bar while the US market is still open.
    Its volume is only a fraction of a day, which drags the recent-volume average down,
    so it is removed until the session has closed (16:00 New York time)."""
    if hist is None or hist.empty:
        return hist
    from zoneinfo import ZoneInfo
    now_ny = now_ny or datetime.now(ZoneInfo("America/New_York"))
    last = hist.index[-1]
    last_date = last.date() if hasattr(last, "date") else pd.Timestamp(last).date()
    closed = (now_ny.hour, now_ny.minute) >= (16, 15)
    if last_date == now_ny.date() and not closed:
        return hist.iloc[:-1]
    return hist


def _yahoo_snapshot(ticker: str) -> dict:
    """Everything Phase 2 needs from Yahoo Finance for one ticker (missing values = None)."""
    import yfinance as yf

    data = {}
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}

    data["name"] = info.get("longName") or info.get("shortName")
    data["market_cap"] = _num(info.get("marketCap"))
    data["price"] = _num(info.get("currentPrice") or info.get("regularMarketPrice"))
    data["pe_ratio"] = _num(info.get("forwardPE")) or _num(info.get("trailingPE"))
    data["revenue_growth"] = _num(info.get("revenueGrowth"))     # quarterly YoY fraction
    data["debt_to_equity"] = _num(info.get("debtToEquity"))      # percent

    try:
        hist = t.history(period="3mo", interval="1d", auto_adjust=False)
        hist = drop_unfinished_session(hist)
    except Exception:
        hist = None
    if hist is not None and not hist.empty and "Volume" in hist:
        vol = hist["Volume"].astype(float).dropna()
        close = hist["Close"].astype(float).dropna()
        data["trading_days"] = int(len(vol))
        if data["price"] is None and len(close):
            data["price"] = float(close.iloc[-1])
        if len(vol) >= 8:
            recent = vol.iloc[-3:].mean()
            baseline = vol.iloc[-23:-3].mean()
            data["recent_volume"] = float(recent)
            data["baseline_volume"] = float(baseline)
            data["volume_ratio"] = float(recent / baseline) if baseline > 0 else None
            tail = min(20, len(vol))
            data["avg_dollar_volume"] = float((vol.iloc[-tail:] * close.iloc[-tail:]).mean())
    return data


def _finnhub_get(path: str, params: dict):
    if not FINNHUB_KEY:
        return None
    try:
        res = requests.get(f"https://finnhub.io/api/v1{path}", params=params,
                           headers={"X-Finnhub-Token": FINNHUB_KEY}, timeout=15)
        time.sleep(1.1)   # free tier: 60 calls/minute
        return res.json() if res.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def _finnhub_fallback(ticker: str, need_cap: bool, need_fund: bool) -> dict:
    out = {}
    if need_cap:
        prof = _finnhub_get("/stock/profile2", {"symbol": ticker}) or {}
        cap_m = _num(prof.get("marketCapitalization"))          # reported in millions
        if cap_m:
            out["market_cap"] = cap_m * 1_000_000
        out["name"] = prof.get("name")
    if need_fund:
        # Units below are assumptions about Finnhub's metric endpoint; verify on first use.
        metric = (_finnhub_get("/stock/metric", {"symbol": ticker, "metric": "all"}) or {}).get("metric", {})
        pe = _num(metric.get("peTTM")) or _num(metric.get("peNormalizedAnnual"))
        growth = _num(metric.get("revenueGrowthQuarterlyYoy"))   # percent -> fraction
        de = _num(metric.get("totalDebt/totalEquityQuarterly"))  # ratio -> percent
        if pe is not None:
            out["pe_ratio"] = pe
        if growth is not None:
            out["revenue_growth"] = growth / 100.0
        if de is not None:
            out["debt_to_equity"] = de * 100.0
    return out


def _alpha_fallback(ticker: str) -> dict:
    if not ALPHA_KEY:
        return {}
    try:
        res = requests.get("https://www.alphavantage.co/query",
                           params={"function": "OVERVIEW", "symbol": ticker, "apikey": ALPHA_KEY},
                           timeout=20)
        time.sleep(12.5)   # free tier: 5 calls/minute
        d = res.json() if res.status_code == 200 else {}
    except (requests.RequestException, ValueError):
        return {}
    out = {}
    cap = _num(d.get("MarketCapitalization"))
    if cap:
        out["market_cap"] = cap
    pe = _num(d.get("ForwardPE")) or _num(d.get("PERatio"))
    if pe is not None:
        out["pe_ratio"] = pe
    growth = _num(d.get("QuarterlyRevenueGrowthYOY"))
    if growth is not None:
        out["revenue_growth"] = growth
    out["name"] = d.get("Name")
    return out


_FIELDS = ["name", "market_cap", "price", "recent_volume", "baseline_volume", "volume_ratio",
           "avg_dollar_volume", "trading_days", "pe_ratio", "revenue_growth", "debt_to_equity"]


def fetch_market_data(session, tickers, verbose=True, refresh=False) -> dict:
    """ticker -> dict of market fields. Cached in market_snapshot for MARKET_CACHE_HOURS
    (refresh=True ignores the cache and downloads everything again)."""
    fresh_after = _utcnow() + timedelta(days=1) if refresh else _utcnow() - timedelta(hours=MARKET_CACHE_HOURS)
    cached = {r.ticker: r for r in session.query(MarketSnapshot)
              .filter(MarketSnapshot.ticker.in_(list(tickers)))
              .filter(MarketSnapshot.fetched_at >= fresh_after).all()}

    result, used = {}, {"cache": 0, "yahoo": 0, "finnhub": 0, "alphavantage": 0, "failed": 0}
    for ticker in tickers:
        if ticker in cached:
            row = cached[ticker]
            result[ticker] = {f: getattr(row, f) for f in _FIELDS}
            result[ticker]["source"] = (row.source or "") + "+cache"
            used["cache"] += 1
            continue

        sources = []
        try:
            data = _yahoo_snapshot(ticker)
            sources.append("yahoo")
        except Exception as err:
            if verbose:
                print(f"  ${ticker}: Yahoo Finance failed ({err})")
            data = {}
        time.sleep(PAUSE_SECONDS)

        need_cap = data.get("market_cap") is None
        known = sum(data.get(k) is not None for k in ("pe_ratio", "revenue_growth", "debt_to_equity"))
        need_fund = known < MIN_KNOWN_FUNDAMENTALS
        if need_cap or need_fund:
            fb = _finnhub_fallback(ticker, need_cap, need_fund)
            if fb:
                sources.append("finnhub")
                for k, v in fb.items():
                    if data.get(k) is None and v is not None:
                        data[k] = v
        if data.get("market_cap") is None:
            av = _alpha_fallback(ticker)
            if av:
                sources.append("alphavantage")
                for k, v in av.items():
                    if data.get(k) is None and v is not None:
                        data[k] = v

        if not sources or data.get("market_cap") is None:
            used["failed"] += 1
        for tag in ("yahoo", "finnhub", "alphavantage"):
            if tag in sources:
                used[tag] += 1
        data["source"] = "+".join(sources)
        result[ticker] = data

        try:
            session.merge(MarketSnapshot(ticker=ticker, fetched_at=_utcnow(),
                                         source=data["source"], **{f: data.get(f) for f in _FIELDS}))
            session.commit()
        except Exception:
            session.rollback()

    if verbose:
        print(f"  Market data: {used}")
    return result


# ----------------------------------------------------------------------
# Decision logic (pure function: easy to test and to tune)
# ----------------------------------------------------------------------
def evaluate_market_and_fundamentals(row, market: dict, universe_name: str = "") -> tuple:
    """Filters 1, 2b and 4 for one ticker. Returns (status, reason)."""
    if not market or market.get("market_cap") is None:
        return "REJECTED", "no market-cap data from Yahoo/Finnhub/Alpha Vantage"

    # --- Filter 1: penny stock / scam exclusion ---
    cap = market["market_cap"]
    if cap < MIN_MARKET_CAP:
        return "REJECTED", f"market cap {fmt_money(cap)} < {fmt_money(MIN_MARKET_CAP)}"
    price = market.get("price")
    if price is not None and price < MIN_PRICE:
        return "REJECTED", f"price ${price:.2f} < ${MIN_PRICE:.2f}"
    name = f"{market.get('name') or ''} {universe_name or ''}"
    if SPAC_NAME_RE.search(name):
        return "REJECTED", f"looks like a SPAC / blank-check company ({name.strip()[:40]})"
    days = market.get("trading_days")
    if days is not None and days < MIN_TRADING_DAYS:
        return "REJECTED", f"only {days} trading days of history (recent listing or merger)"
    dollar_vol = market.get("avg_dollar_volume")
    if dollar_vol is not None and dollar_vol < MIN_AVG_DOLLAR_VOLUME:
        return "REJECTED", f"avg daily traded value {fmt_money(dollar_vol)} too thin"

    # --- Filter 2b: does real trading volume confirm the chatter? ---
    ratio = market.get("volume_ratio")
    if ratio is None:
        return "REJECTED", "no trading-volume data to confirm the chatter"
    # The spike rule needs a real "vs its own normal" z-score. A cross-section z only says the
    # ticker is popular compared with others today (mega-caps always are), so it is not used.
    is_spike = getattr(row, "z_method", "history") == "history" and row.z_score >= SPIKE_Z
    if is_spike and ratio < SPIKE_MIN_VOLUME_RATIO:
        return "REJECTED", (f"chatter spike (z={row.z_score}) but trading volume only "
                            f"{ratio:.2f}x normal: likely synthetic social push")
    if ratio < MIN_VOLUME_RATIO:
        return "REJECTED", f"trading volume {ratio:.2f}x normal, below {MIN_VOLUME_RATIO}x"

    # --- Filter 4: fundamentals ---
    pe = market.get("pe_ratio")
    growth = market.get("revenue_growth")
    de = market.get("debt_to_equity")
    known = sum(v is not None for v in (pe, growth, de))
    if known < MIN_KNOWN_FUNDAMENTALS:
        return "REJECTED", f"only {known} of 3 fundamentals available"
    if pe is not None:
        if REQUIRE_POSITIVE_PE and pe <= 0:
            return "REJECTED", f"unprofitable (P/E {pe:.1f})"
        if pe > MAX_PE:
            return "REJECTED", f"P/E {pe:.1f} > {MAX_PE:.0f}"
    if growth is not None and growth < MIN_REVENUE_GROWTH:
        return "REJECTED", f"quarterly revenue growth {growth:+.1%} < {MIN_REVENUE_GROWTH:+.0%}"
    if de is not None and de > MAX_DEBT_TO_EQUITY:
        return "REJECTED", f"debt/equity {de:.0f}% > {MAX_DEBT_TO_EQUITY:.0f}%"
    return "PASS", "ok"


LAST_REJECTED = []   # every ticker Phase 2 dropped this run: {ticker, check, reason}, read by run_pipeline.py


def local_filters(summary: pd.DataFrame) -> list:
    """Filters 2a and 3 (no API needed). Returns [(row, status, reason)]."""
    out = []
    for r in summary.itertuples(index=False):
        if r.chatter_score < MIN_CHATTER:
            out.append((r, "REJECTED", f"chatter {r.chatter_score} < {MIN_CHATTER} mentions"))
        elif r.z_score < MIN_Z_SCORE:
            out.append((r, "REJECTED", f"z-score {r.z_score} < {MIN_Z_SCORE} ({r.z_method})"))
        elif (r.social_mentions >= MIN_SOCIAL_FOR_RATIO
              and r.unique_author_ratio < MIN_UNIQUE_AUTHOR_RATIO):
            out.append((r, "REJECTED",
                        f"unique authors {r.unique_author_ratio:.0%} < {MIN_UNIQUE_AUTHOR_RATIO:.0%} "
                        f"({r.unique_authors} accounts / {r.social_mentions} posts)"))
        else:
            out.append((r, "PASS", "ok"))
    return out


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_phase2(mentions: pd.DataFrame = None, summary: pd.DataFrame = None, verbose: bool = True,
               refresh_market: bool = False):
    """Returns (survivors, mentions_for_survivors). Runs Phase 1 itself if not given its output."""
    if summary is None or mentions is None:
        mentions, summary = run_phase1(verbose=verbose)
    if summary.empty:
        print("Phase 1 produced no tickers; nothing to filter.")
        return pd.DataFrame(), pd.DataFrame()

    init_db()
    session = SessionLocal()
    try:
        if verbose:
            print("\n=== Phase 2: Quantitative market and volume hard filters ===")
            print(f"Tickers from Phase 1: {len(summary)}")

        history = load_history(session)
        z_summary = compute_z_scores(summary, history)
        n_hist = int((z_summary["z_method"] == "history").sum())
        if verbose:
            print(f"\n[2a] Chatter z-score: {n_hist} tickers use their own history, "
                  f"{len(z_summary) - n_hist} use the cross-section fallback")
            if n_hist == 0:
                print("  NOTE: no history yet. Z-scores compare each ticker with today's others, not "
                      "with its own normal level. Run the pipeline daily to build real baselines.")

        # Cheap local filters first: chatter size, z-score, distinct-user ratio
        decisions = local_filters(z_summary)
        local_pass = [r for r, status, _ in decisions if status == "PASS"]
        LAST_REJECTED.clear()
        LAST_REJECTED.extend({"ticker": r.ticker, "check": "chatter and distinct users", "reason": why}
                             for r, status, why in decisions if status == "REJECTED")
        if verbose:
            for reason_key in ("chatter", "z-score", "unique authors"):
                n = sum(1 for _, st, why in decisions if st == "REJECTED" and why.startswith(reason_key))
                print(f"  rejected by {reason_key:15s}: {n}")
            print(f"  passed local filters: {len(local_pass)}")

        ranked = sorted(local_pass, key=lambda r: r.chatter_score, reverse=True)
        candidates = ranked[:MAX_CANDIDATES]
        LAST_REJECTED.extend({"ticker": r.ticker, "check": "chatter and distinct users",
                              "reason": f"not among the {MAX_CANDIDATES} most-discussed stocks checked for market data"}
                             for r in ranked[MAX_CANDIDATES:])
        if len(local_pass) > MAX_CANDIDATES and verbose:
            print(f"  (checking market data for the top {MAX_CANDIDATES} by chatter only)")

        if verbose:
            print(f"\n[1/2b/4] Market data for {len(candidates)} candidates")
        market = fetch_market_data(session, [r.ticker for r in candidates], verbose, refresh_market)

        names = dict(session.query(USUniverse.symbol, USUniverse.description)
                     .filter(USUniverse.symbol.in_([r.ticker for r in candidates])).all())

        passed = []
        for r in candidates:
            m = market.get(r.ticker, {})
            status, reason = evaluate_market_and_fundamentals(r, m, names.get(r.ticker, ""))
            if verbose:
                tag = "PASS    " if status == "PASS" else "REJECTED"
                print(f"  [{tag}] ${r.ticker:5s} | cap {fmt_money(m.get('market_cap')):>9s} | "
                      f"z {r.z_score:5.2f} | vol {m.get('volume_ratio') or float('nan'):.2f}x | "
                      f"authors {r.unique_author_ratio} | {reason}")
            if status != "PASS":
                LAST_REJECTED.append({"ticker": r.ticker, "check": "market, volume and fundamentals",
                                      "reason": reason})
            if status == "PASS":
                passed.append({
                    "ticker": r.ticker, "market_cap": m.get("market_cap"),
                    "chatter_score": r.chatter_score, "social_mentions": r.social_mentions,
                    "unbiased_news": r.unbiased_news, "z_score": r.z_score, "z_method": r.z_method,
                    "volume_ratio": round(m["volume_ratio"], 2),
                    "unique_author_ratio": r.unique_author_ratio,
                    "pe_ratio": m.get("pe_ratio"), "revenue_growth": m.get("revenue_growth"),
                    "debt_to_equity": m.get("debt_to_equity"),
                })
    finally:
        session.close()

    survivors = pd.DataFrame(passed)
    if not survivors.empty:
        survivors = survivors.sort_values(["z_score", "chatter_score"], ascending=False).reset_index(drop=True)
        keep = mentions[mentions["ticker"].isin(survivors["ticker"])].copy()
    else:
        keep = mentions.iloc[0:0].copy()

    if verbose:
        print(f"\n=== PHASE 2 COMPLETE: {len(survivors)} tickers retained ===")
        if not survivors.empty:
            show = survivors.copy()
            show["market_cap"] = show["market_cap"].map(fmt_money)
            show["revenue_growth"] = show["revenue_growth"].map(lambda v: "n/a" if v is None or pd.isna(v) else f"{v:+.1%}")
            print(show.to_string(index=False))
    return survivors, keep


if __name__ == "__main__":
    # python filtering_of_data/phase2_filter.py --refresh   -> ignore cached market data
    run_phase2(refresh_market="--refresh" in sys.argv)
