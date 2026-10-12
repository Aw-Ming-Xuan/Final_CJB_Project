"""Phase 5: risk checks, diversification, final Top 5 and performance tracking.

Phases 1-4 decide WHAT is being talked about and WHY. Phase 5 adds the checks a careful
investor would still make before acting, then keeps score of how past picks did.

Input: the latest Phase 4 output saved in filtering_of_data/output/ (so Phase 5 runs in
about a minute without redoing Phases 1-4). Use --fresh to run Phases 1-4 first.

  1. OVEREXTENSION (exclude): the stock already jumped more than MAX_5D_RETURN in 5 trading
     days, or RSI(14) is above MAX_RSI. The chatter is probably reacting to a move that has
     already happened ("buying the top").
  2. VOLATILITY (exclude above MAX_VOLATILITY, flag above HIGH_VOLATILITY): annualised
     30-day volatility. Very jumpy stocks can wipe out a week's gain in a day.
  3. EARNINGS SOON (flag; exclude if EXCLUDE_EARNINGS_SOON): an earnings report in the next
     EARNINGS_SOON_DAYS days makes the next week a coin flip.
  4. SHORT INTEREST (flag): more than HIGH_SHORT_INTEREST of the float is sold short, so the
     chatter may be squeeze talk rather than business news.
  5. TREND and ANALYSTS (flag): price below its 200-day average; analysts' consensus is
     "sell" or "underperform".
  6. DIVERSIFICATION (skip and backfill): at most MAX_PER_SECTOR picks per sector (2, so a full
     Top 5 covers at least 3 industries), and no
     two picks whose daily returns move almost identically (correlation above
     MAX_CORRELATION, e.g. GOOGL and GOOG, or two memory-chip makers). The next-ranked
     ticker from Phase 4 takes the freed place.
  7. PERFORMANCE TRACKING: every final pick is stored with its price and SPY's price. Each
     run prints how earlier picks did over 5 and 20 trading days versus SPY. This is the
     only real answer to "are these stocks actually good?".

Outputs in filtering_of_data/output/: final_top5_<date>.md (readable report) and
final_top5_<date>.json (for a dashboard or email).

Run:  python filtering_of_data/phase5_filter.py            (uses the saved Phase 4 output)
      python filtering_of_data/phase5_filter.py --fresh    (runs Phases 1-4 first)
This is a screening tool, not investment advice.
"""

import glob
import json
import math
import os
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

from data_base_creation.database import SessionLocal, init_db, PickHistory

try:
    from filtering_of_data import phase4_filter as p4
    from filtering_of_data.phase2_filter import drop_unfinished_session
except ImportError:
    import phase4_filter as p4
    from phase2_filter import drop_unfinished_session

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(HERE, "output")

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
TOP_N = 5
MAX_5D_RETURN = 0.20            # exclude: already up more than 20% in 5 trading days
MAX_RSI = 80                    # exclude: RSI(14) above this = very overbought
FLAG_ABOVE_MA50 = 0.25          # flag: price more than 25% above its 50-day average
MAX_VOLATILITY = 0.90           # exclude: annualised 30-day volatility above 90%
HIGH_VOLATILITY = 0.60          # flag above 60%
EARNINGS_SOON_DAYS = 7
EXCLUDE_EARNINGS_SOON = False   # True = drop tickers reporting earnings within 7 days
HIGH_SHORT_INTEREST = 0.20      # flag: more than 20% of the float sold short
BAD_ANALYST_VIEWS = {"sell", "strong_sell", "underperform"}
MAX_PER_SECTOR = 2              # so a full Top 5 covers at least 3 industries
MAX_CORRELATION = 0.85          # daily-return correlation over the last 120 sessions
EVAL_HORIZONS = [5, 20]         # trading days after the pick
CHART_SESSIONS = 252            # price history published per pick (~1 year); the app shows 90 by default
BENCHMARK = "SPY"

FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _num(v):
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# Input: latest Phase 4 output
# ----------------------------------------------------------------------
def load_phase4_candidates(path: str = None) -> tuple:
    """(all_candidates DataFrame, path). Fills columns that older Phase 4 files lack."""
    if path is None:
        files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "shortlist_*.json")))
        if not files:
            return None, None
        path = files[-1]
    with open(path) as f:
        payload = json.load(f)
    df = pd.DataFrame(payload.get("all_candidates", []))
    defaults = {"materiality": "medium", "catalyst_direction": "mixed", "catalyst_confidence": 0.5,
                "hype_risk": "medium", "catalyst_summary": "", "catalyst_type": "unknown"}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
    return df, path


# ----------------------------------------------------------------------
# Market data (Yahoo Finance via yfinance; Finnhub for the earnings calendar)
# ----------------------------------------------------------------------
_price_cache, _info_cache = {}, {}


def price_history(ticker: str) -> pd.Series:
    """Daily closes for ~1 year (split/dividend adjusted), today's unfinished bar removed."""
    if ticker in _price_cache:
        return _price_cache[ticker]
    import yfinance as yf
    try:
        hist = yf.Ticker(ticker).history(period="1y", interval="1d", auto_adjust=True)
        hist = drop_unfinished_session(hist)
        close = hist["Close"].astype(float).dropna() if hist is not None and not hist.empty else pd.Series(dtype=float)
    except Exception as err:
        print(f"  ${ticker}: price history failed ({err})")
        close = pd.Series(dtype=float)
    if len(close):
        close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    _price_cache[ticker] = close
    time.sleep(0.3)
    return close


def ticker_info(ticker: str) -> dict:
    if ticker in _info_cache:
        return _info_cache[ticker]
    import yfinance as yf
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    out = {"sector": info.get("sector") or "Unknown",
           "short_pct_float": _num(info.get("shortPercentOfFloat")),
           "analyst_view": (info.get("recommendationKey") or "").lower() or None,
           "name": info.get("longName") or info.get("shortName") or ticker}
    _info_cache[ticker] = out
    time.sleep(0.3)
    return out


def days_to_earnings(ticker: str):
    """Days until the next earnings report within EARNINGS_SOON_DAYS, else None."""
    if not FINNHUB_KEY:
        return None
    today = _utcnow().date()
    try:
        res = requests.get("https://finnhub.io/api/v1/calendar/earnings",
                           params={"symbol": ticker, "from": today.isoformat(),
                                   "to": (today + timedelta(days=EARNINGS_SOON_DAYS)).isoformat()},
                           headers={"X-Finnhub-Token": FINNHUB_KEY}, timeout=15)
        time.sleep(1.1)
        cal = res.json().get("earningsCalendar", []) if res.status_code == 200 else []
    except (requests.RequestException, ValueError):
        return None
    dates = sorted(e.get("date") for e in cal if e.get("date"))
    if not dates:
        return None
    return (datetime.strptime(dates[0], "%Y-%m-%d").date() - today).days


# ----------------------------------------------------------------------
# Risk metrics
# ----------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14):
    if len(close) < period + 1:
        return None
    delta = close.diff().dropna()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    if loss.iloc[-1] == 0:
        return 100.0
    return float(100 - 100 / (1 + gain.iloc[-1] / loss.iloc[-1]))


def risk_metrics(close: pd.Series) -> dict:
    m = {"price": None, "ret_5d": None, "rsi_14": None, "above_ma50": None,
         "below_ma200": None, "volatility_30d": None}
    if close is None or len(close) < 30:
        return m
    m["price"] = float(close.iloc[-1])
    m["ret_5d"] = float(close.iloc[-1] / close.iloc[-6] - 1)
    m["rsi_14"] = rsi(close)
    m["above_ma50"] = float(close.iloc[-1] / close.tail(50).mean() - 1)
    if len(close) >= 200:
        m["below_ma200"] = bool(close.iloc[-1] < close.tail(200).mean())
    log_ret = np.log(close / close.shift(1)).dropna().tail(30)
    m["volatility_30d"] = float(log_ret.std() * math.sqrt(252))
    return m


def evaluate_risk(m: dict, info: dict, earn_days) -> tuple:
    """(exclusion reason or None, list of flags)."""
    flags = []
    if m["price"] is None:
        return "no price history (needs at least 30 trading days)", flags
    if m["ret_5d"] is not None and m["ret_5d"] > MAX_5D_RETURN:
        return f"already up {m['ret_5d']:+.0%} in 5 days (chasing a move that happened)", flags
    if m["rsi_14"] is not None and m["rsi_14"] > MAX_RSI:
        return f"RSI {m['rsi_14']:.0f} > {MAX_RSI} (very overbought)", flags
    if m["volatility_30d"] is not None and m["volatility_30d"] > MAX_VOLATILITY:
        return f"volatility {m['volatility_30d']:.0%} a year > {MAX_VOLATILITY:.0%}", flags
    if earn_days is not None and EXCLUDE_EARNINGS_SOON:
        return f"earnings in {earn_days} days (binary event)", flags

    if earn_days is not None:
        flags.append(f"earnings in {earn_days} days")
    if m["volatility_30d"] is not None and m["volatility_30d"] > HIGH_VOLATILITY:
        flags.append(f"high volatility ({m['volatility_30d']:.0%}/yr)")
    if m["above_ma50"] is not None and m["above_ma50"] > FLAG_ABOVE_MA50:
        flags.append(f"{m['above_ma50']:+.0%} above 50-day average")
    if m["below_ma200"]:
        flags.append("below 200-day average")
    if info.get("short_pct_float") and info["short_pct_float"] > HIGH_SHORT_INTEREST:
        flags.append(f"short interest {info['short_pct_float']:.0%} of float")
    if info.get("analyst_view") in BAD_ANALYST_VIEWS:
        flags.append(f"analyst consensus: {info['analyst_view']}")
    return None, flags


def correlation(a: pd.Series, b: pd.Series, sessions: int = 120):
    ra, rb = a.pct_change().dropna().tail(sessions), b.pct_change().dropna().tail(sessions)
    joined = pd.concat([ra, rb], axis=1, join="inner").dropna()
    if len(joined) < 30:
        return None
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))


def select_final(ranked: pd.DataFrame, verbose=True) -> tuple:
    """Walk Phase 4's ranking; apply risk checks and diversification until TOP_N are chosen.
    Returns (final DataFrame, list of (ticker, reason) exclusions)."""
    chosen, excluded = [], []
    sector_count = {}
    for r in ranked.to_dict("records"):
        if len(chosen) >= TOP_N:
            break
        t = r["ticker"]
        close = price_history(t)
        info = ticker_info(t)
        m = risk_metrics(close)
        reason, flags = evaluate_risk(m, info, days_to_earnings(t))
        if reason:
            excluded.append((t, reason))
            continue
        sector = info.get("sector") or "Unknown"
        if sector != "Unknown" and sector_count.get(sector, 0) >= MAX_PER_SECTOR:
            excluded.append((t, f"already {MAX_PER_SECTOR} picks in {sector}"))
            continue
        twin = None
        for c in chosen:
            rho = correlation(close, price_history(c["ticker"]))
            if rho is not None and rho > MAX_CORRELATION:
                twin = (c["ticker"], rho)
                break
        if twin:
            excluded.append((t, f"moves almost identically to ${twin[0]} (correlation {twin[1]:.2f})"))
            continue
        sector_count[sector] = sector_count.get(sector, 0) + 1
        recent = close.tail(CHART_SESSIONS)
        chosen.append({**r, **m, "sector": sector, "name": info.get("name"),
                       "price_history": {"dates": [d.strftime("%Y-%m-%d") for d in recent.index],
                                         "close": [round(float(v), 2) for v in recent.values]},
                       "short_pct_float": info.get("short_pct_float"),
                       "analyst_view": info.get("analyst_view"),
                       "risk_flags": flags})
    final = pd.DataFrame(chosen)
    if not final.empty:
        final.insert(0, "rank", range(1, len(final) + 1))
    return final, excluded


# ----------------------------------------------------------------------
# Performance tracking
# ----------------------------------------------------------------------
def save_picks(final: pd.DataFrame, pick_date) -> int:
    if final.empty:
        return 0
    spy = price_history(BENCHMARK)
    spy_price = float(spy.iloc[-1]) if len(spy) else None
    session = SessionLocal()
    try:
        session.query(PickHistory).filter(PickHistory.pick_date == pick_date).delete()
        for r in final.itertuples(index=False):
            session.add(PickHistory(pick_date=pick_date, ticker=r.ticker, rank=int(r.rank),
                                    final_score=_num(getattr(r, "final_score", None)),
                                    entry_price=_num(r.price), spy_entry_price=spy_price,
                                    catalyst_summary=str(getattr(r, "catalyst_summary", ""))[:500]))
        session.commit()
        return len(final)
    except Exception as err:
        session.rollback()
        print(f"  Could not save picks: {err}")
        return 0
    finally:
        session.close()


def forward_return(close: pd.Series, pick_date, entry: float, horizon: int):
    """Return from the entry price to the close `horizon` sessions after pick_date.
    (value, complete?) where complete=False means the horizon has not passed yet."""
    if close is None or not len(close) or not entry:
        return None, False
    after = close[close.index.date > pick_date]
    if after.empty:
        return None, False
    if len(after) >= horizon:
        return float(after.iloc[horizon - 1] / entry - 1), True
    return float(after.iloc[-1] / entry - 1), False


def scorecard(today=None, verbose=True) -> pd.DataFrame:
    """How every earlier pick did vs SPY over EVAL_HORIZONS trading days."""
    today = today or _utcnow().date()
    session = SessionLocal()
    try:
        rows = session.query(PickHistory).filter(PickHistory.pick_date < today).all()
        picks = [(r.pick_date, r.ticker, r.rank, r.entry_price, r.spy_entry_price) for r in rows]
    finally:
        session.close()
    if not picks:
        if verbose:
            print("  No earlier picks yet. Scores appear from the next run onwards "
                  "(5-day results after a week, 20-day results after a month).")
        return pd.DataFrame()

    spy = price_history(BENCHMARK)
    out = []
    for pick_date, ticker, rank, entry, spy_entry in picks:
        close = price_history(ticker)
        rec = {"pick_date": pick_date, "ticker": ticker, "rank": rank}
        for h in EVAL_HORIZONS:
            r, done = forward_return(close, pick_date, entry, h)
            b, _ = forward_return(spy, pick_date, spy_entry, h)
            rec[f"ret_{h}d"] = r
            rec[f"excess_{h}d"] = (r - b) if (r is not None and b is not None) else None
            rec[f"done_{h}d"] = done
        out.append(rec)
    df = pd.DataFrame(out).sort_values(["pick_date", "rank"])

    if verbose:
        for h in EVAL_HORIZONS:
            done = df[df[f"done_{h}d"] & df[f"excess_{h}d"].notna()]
            if done.empty:
                print(f"  {h}-day results: none complete yet")
                continue
            print(f"  {h}-day results on {len(done)} picks: average return {done[f'ret_{h}d'].mean():+.2%}, "
                  f"vs SPY {done[f'excess_{h}d'].mean():+.2%}, beat SPY {(done[f'excess_{h}d'] > 0).mean():.0%} of the time")
        if len(df) < 30:
            print(f"  NOTE: {len(df)} picks is far too few to judge the system. "
                  "Treat this as a log until there are dozens of picks over several months.")
    return df


# ----------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------
def _pct(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:+.1%}"


def _vol(v):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0%}/yr"


def write_report(final, excluded, card, source_path) -> tuple:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = _utcnow().strftime("%Y-%m-%d")
    md = [f"# Weekly stock shortlist: {stamp}", "",
          "_Screening output from social and news chatter. Not investment advice._", ""]
    if final.empty:
        md.append("No ticker passed every check this week.")
    else:
        md += ["| # | Ticker | Score | Catalyst | Sentiment | 5-day move | Volatility | Flags |",
               "|---|---|---|---|---|---|---|---|"]
        for r in final.itertuples(index=False):
            md.append(f"| {r.rank} | {r.ticker} | {getattr(r, 'final_score', float('nan')):.2f} | "
                      f"{r.catalyst_type} ({r.catalyst_direction}) | {getattr(r, 'final_sentiment', float('nan')):+.2f} | "
                      f"{_pct(r.ret_5d)} | {_vol(r.volatility_30d)} | {'; '.join(r.risk_flags) or 'none'} |")
        md.append("")
        for r in final.itertuples(index=False):
            md += [f"**{r.rank}. {r.ticker} ({r.name}, {r.sector})**  ",
                   f"Catalyst: {r.catalyst_summary}  ",
                   f"Price ${r.price:,.2f}, P/E {getattr(r, 'pe_ratio', float('nan')):.1f}, "
                   f"revenue growth {_pct(getattr(r, 'revenue_growth', None))}, hype risk {r.hype_risk}.", ""]
    if excluded:
        md += ["## Removed in Phase 5", ""] + [f"- {t}: {why}" for t, why in excluded] + [""]
    if card is not None and not card.empty:
        md += ["## How earlier picks did", "",
               "| Picked | Ticker | 5-day | vs SPY | 20-day | vs SPY |", "|---|---|---|---|---|---|"]
        for r in card.itertuples(index=False):
            md.append(f"| {r.pick_date} | {r.ticker} | {_pct(r.ret_5d)} | {_pct(r.excess_5d)} | "
                      f"{_pct(r.ret_20d)} | {_pct(r.excess_20d)} |")
    md.append(f"\n_Source: {os.path.basename(source_path) if source_path else 'fresh run'}_")

    md_path = os.path.join(OUTPUT_DIR, f"final_top5_{stamp}.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md))
    json_path = os.path.join(OUTPUT_DIR, f"final_top5_{stamp}.json")
    with open(json_path, "w") as f:
        json.dump({"generated_at_utc": _utcnow().isoformat(timespec="seconds"),
                   "note": "Screening output, not investment advice.",
                   "top": json.loads(final.to_json(orient="records", date_format="iso")) if not final.empty else [],
                   "removed": [{"ticker": t, "reason": w} for t, w in excluded],
                   "scorecard": json.loads(card.to_json(orient="records", date_format="iso"))
                   if card is not None and not card.empty else []},
                  f, indent=2, default=str)
    return md_path, json_path


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
LAST_RESULT = {}   # removed list + scorecard of the latest run, read by run_pipeline.py


def run_phase5(fresh: bool = False, verbose: bool = True, candidates: pd.DataFrame = None):
    init_db()
    source = None
    if candidates is not None:
        pass                                   # handed over directly by run_pipeline.py
    elif fresh:
        _, candidates = p4.run_phase4(verbose=verbose)
    else:
        candidates, source = load_phase4_candidates()
        if candidates is None:
            print("No saved Phase 4 output found; running Phases 1-4 first.")
            _, candidates = p4.run_phase4(verbose=verbose)

    if verbose:
        print("\n=== Phase 5: Risk checks, diversification and performance tracking ===")
        if source:
            print(f"Using saved Phase 4 output: {os.path.basename(source)}")
    if candidates is None or candidates.empty or "eligibility" not in candidates.columns:
        print("No Phase 4 candidates to check.")
        return pd.DataFrame()

    ranked = p4.rank(candidates)
    if verbose:
        print(f"Phase 4 eligible tickers, in rank order: {list(ranked['ticker']) if not ranked.empty else []}")
    if ranked.empty:
        print("No ticker passed Phase 4.")
        return pd.DataFrame()

    final, excluded = select_final(ranked, verbose)
    today = _utcnow().date()

    if verbose:
        print("\n[1-6] Risk checks and diversification")
        for t, why in excluded:
            print(f"  [REMOVED] ${t:5s} {why}")
        for r in final.itertuples(index=False):
            print(f"  [KEPT   ] ${r.ticker:5s} 5d {_pct(r.ret_5d)} | RSI {r.rsi_14 or float('nan'):.0f} | "
                  f"vol {_vol(r.volatility_30d)} | {r.sector} | flags: {', '.join(r.risk_flags) or 'none'}")

        print("\n[7] How earlier picks did (vs SPY)")
    card = scorecard(today, verbose)
    saved = save_picks(final, today)
    # Stocks that passed every check but are not in the Top 5, with their industry, so the app
    # can show what else qualified across sectors
    picked = set(final["ticker"]) if not final.empty else set()
    removed = {t for t, _ in excluded}
    others = [{"ticker": r["ticker"], "name": ticker_info(r["ticker"]).get("name"),
               "sector": ticker_info(r["ticker"]).get("sector") or "Unknown",
               "score": _num(r.get("final_score")), "catalyst": str(r.get("catalyst_summary") or "")}
              for r in ranked.to_dict("records") if r["ticker"] not in picked | removed]
    LAST_RESULT.clear()
    LAST_RESULT.update(removed=excluded, scorecard=card, other_eligible=others)
    md_path, json_path = write_report(final, excluded, card, source)

    if verbose:
        print(f"\n=== FINAL TOP {len(final)} (screening output, not investment advice) ===")
        if final.empty:
            print("  No ticker passed every check.")
        else:
            if len(final) < TOP_N:
                print(f"  Only {len(final)} tickers passed; the list is not padded.")
            for r in final.itertuples(index=False):
                print(f"  {r.rank}. ${r.ticker:5s} ${r.price:,.2f}  {r.catalyst_summary[:90]}")
                if r.risk_flags:
                    print(f"           watch: {', '.join(r.risk_flags)}")
        print(f"\n  Saved {saved} picks to pick_history for tracking.")
        print(f"  Report: {md_path}")
        print(f"  JSON:   {json_path}")
    return final


if __name__ == "__main__":
    run_phase5(fresh="--fresh" in sys.argv)
