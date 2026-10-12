"""One command for the whole backend.

    python run_pipeline.py              # WEEKLY: collect data, run Phases 1-5, publish results
    python run_pipeline.py --daily      # DAILY:  collect data + Phase 1 only (builds z-score history)
    python run_pipeline.py --skip-collect   # weekly run on data already stored

The phases stay in their own files (easier to test and fix one at a time). This file runs
them in order and then writes ONE result that the frontend reads:

    filtering_of_data/output/latest.json        (the newest weekly result)
    filtering_of_data/output/run_<id>.json      (a copy per run)
    pipeline_runs table in market_data.db       (every run, for history and the API)

What the result contains, so the frontend can say where the answer comes from, when it
was screened, and why it can be trusted:
    picks         the final Top 5 with catalyst, linked evidence headlines, sentiment,
                  attention, fundamentals and risk flags
    screened_at   when this run finished (UTC and Singapore time)
    data_window   the period the data covers
    sources       every data source and how many items each contributed
    funnel        how many tickers survived each phase (e.g. 332 -> 13 -> 5)
    methodology   what each phase checks, in plain words
    reliability   sentiment self-check, z-score mode and days of history, LLM health,
                  and the track record of earlier picks vs SPY
    limitations   honest caveats, and the disclaimer
    settings      key thresholds, so the app's explanations match the code
    sector_mix    how many picks come from each sector
    other_eligible  stocks that passed Phases 1-4 but are not in the Top 5, with their sector

Run this from the project folder (Final_Project_CJB).
"""

import argparse
import json
import math
import os
import re
import sys
import traceback
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sqlalchemy import func

from data_base_creation.database import (
    SessionLocal, init_db, RawNewsPost, StockTwitsPost, Phase1History, PipelineRun, USUniverse,
)
from data_base_creation.us_universe import ALLOWED_TYPES, ALLOWED_MICS
from filtering_of_data import phase1_filter as p1
from filtering_of_data import phase2_filter as p2
from filtering_of_data import phase3_filter as p3
from filtering_of_data import phase4_filter as p4
from filtering_of_data import phase5_filter as p5

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "filtering_of_data", "output")
LOCAL_TZ = ZoneInfo("Asia/Singapore")
DISCLAIMER = ("This is an automated screen of social-media and news chatter for a student project. "
              "It is not investment advice. Do your own research before buying or selling anything.")

EXCHANGE_NAMES = {"XNAS": "Nasdaq", "XNYS": "NYSE", "XASE": "NYSE American"}
FEED_NAMES = {"CNBC": "CNBC", "MarketWatch": "MarketWatch", "Yahoo": "Yahoo Finance",
              "Investing": "Investing.com", "Nasdaq": "Nasdaq", "Finnhub": "Finnhub"}


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


def clean(value):
    """Make values JSON-safe: NaN -> None, numpy -> Python, timestamps -> ISO strings."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 4)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value


# ----------------------------------------------------------------------
# Provenance pieces
# ----------------------------------------------------------------------
def collect_sources(since) -> tuple:
    """(sources list, data_window dict) from what is stored for the screening window."""
    session = SessionLocal()
    try:
        news = (session.query(RawNewsPost.source, RawNewsPost.publisher, func.count(),
                              func.min(RawNewsPost.published_at), func.max(RawNewsPost.published_at))
                .filter(RawNewsPost.published_at >= since)
                .group_by(RawNewsPost.source, RawNewsPost.publisher).all())
        st = (session.query(func.count(), func.min(StockTwitsPost.created_at), func.max(StockTwitsPost.created_at))
              .filter(StockTwitsPost.created_at >= since).one())
    finally:
        session.close()

    # Group by the real publisher (CNBC, Yahoo, MarketWatch, Motley Fool...), not by our feed names
    grouped, via, first, last = {}, {}, [], []
    for source, publisher, n, lo, hi in news:
        feed = (source or "").split("_")[0]
        name = publisher or FEED_NAMES.get(feed, feed)
        grouped[name] = grouped.get(name, 0) + n
        via.setdefault(name, set()).add({"Finnhub": "Finnhub API", "Nasdaq": "Nasdaq.com RSS"}.get(feed, "RSS"))
        first.append(lo)
        last.append(hi)
    sources = [{"name": "StockTwits", "type": "social posts", "items": int(st[0] or 0),
                "used_for": "what retail traders are discussing, their own Bullish/Bearish tags"}]
    top = sorted(grouped.items(), key=lambda kv: -kv[1])
    for name, n in top[:12]:
        sources.append({"name": name, "type": "news publisher", "items": int(n),
                        "used_for": f"headlines (via {', '.join(sorted(via[name]))})"})
    if len(top) > 12:
        sources.append({"name": f"{len(top) - 12} other publishers", "type": "news publisher",
                        "items": int(sum(n for _, n in top[12:])), "used_for": "headlines"})
    sources += [
        {"name": "Yahoo Finance (yfinance)", "type": "market data",
         "used_for": "market cap, price, trading volume, P/E, revenue growth, debt/equity, volatility"},
        {"name": "Finnhub", "type": "reference data",
         "used_for": "list of US-listed common stocks, earnings calendar, fallback fundamentals"},
        {"name": f"FinBERT ({p3.FINBERT_MODEL})", "type": "AI model", "used_for": "financial sentiment of each post"},
    ]
    llm = p4.LAST_LLM_INFO
    if llm:
        sources.append({"name": f"{llm.get('provider')} / {llm.get('model')}".strip(" /"), "type": "AI model (LLM)",
                        "used_for": "catalyst validation and sarcasm review"})
    if st[1]:
        first.append(st[1])
        last.append(st[2])
    window = {"from": min(first).isoformat() if first else None,
              "to": max(last).isoformat() if last else None,
              "lookback_days": p1.LOOKBACK_DAYS}
    return sources, window


def history_days() -> int:
    session = SessionLocal()
    try:
        return int(session.query(func.count(func.distinct(Phase1History.snapshot_date))).scalar() or 0)
    finally:
        session.close()


def track_record(card: pd.DataFrame) -> dict:
    if card is None or card.empty:
        return {"status": "not enough history yet",
                "explanation": "Picks are stored every run; 5-day results appear after a week."}
    out = {"status": "ok", "picks_logged": int(len(card))}
    for h in p5.EVAL_HORIZONS:
        done = card[card[f"done_{h}d"] & card[f"excess_{h}d"].notna()]
        if len(done):
            out[f"{h}_day"] = {"picks": int(len(done)), "avg_return": float(done[f"ret_{h}d"].mean()),
                               "avg_vs_spy": float(done[f"excess_{h}d"].mean()),
                               "beat_spy_rate": float((done[f"excess_{h}d"] > 0).mean())}
    if len(card) < 30:
        out["caution"] = f"Only {len(card)} picks logged; far too few to judge the system yet."
    return out


METHODOLOGY = [
    {"phase": 1, "name": "Bot elimination and ticker extraction",
     "what": "Drops posts from accounts under 30 days old, with under 3 followers, posting more than 5 times "
             "a minute, or copy-pasting the same text. Finds tickers from cashtags, exchange tags and company "
             "names, limited to stocks listed on NYSE, Nasdaq and NYSE American, including ADRs of "
             "foreign companies such as TSMC or Alibaba.",
     "details": [
         f"Window: every post and article from the last {p1.LOOKBACK_DAYS} days is read. A week is long enough to "
         "collect a meaningful sample from StockTwits and news feeds, and short enough that the chatter is still "
         "about this week's events.",
         f"Bot rules (StockTwits only, news has no accounts): accounts younger than {p1.MIN_ACCOUNT_AGE_DAYS} days, "
         "with fewer than 3 followers, posting more than 5 times a minute, or posting the same text repeatedly "
         "are removed. These are the usual signs of spam and pump accounts.",
         "Tickers are found from cashtags ($AAPL), exchange tags (NASDAQ: AAPL) and company names (Apple). "
         "Only US-listed common stocks, REITs and ADRs count; ETFs and over-the-counter names are excluded.",
         "Each stock gets a chatter score that combines social posts and news articles, and a snapshot is saved "
         "every day so later runs can tell whether this week's chatter is unusual for that stock.",
     ]},
    {"phase": 2, "name": "Market and volume hard filters",
     "what": "Requires market cap of at least $300M, price above $1, real trading liquidity and history; "
             "no SPACs. Chatter must be unusual (z-score), come from many different users (70%+ unique), "
             "and be confirmed by real trading volume. Fundamentals: positive P/E up to 60, growing revenue, "
             "debt/equity up to 200%.",
     "details": [
         f"Size and quality: market value of at least ${p2.MIN_MARKET_CAP / 1e6:,.0f}M, share price above "
         f"${p2.MIN_PRICE:.0f}, at least ${p2.MIN_AVG_DOLLAR_VOLUME / 1e6:,.0f}M traded per day, and at least "
         f"{p2.MIN_TRADING_DAYS} trading days in the last 3 months. Small, thinly traded stocks are the easiest "
         "to pump and dump. SPACs (blank-check companies) are excluded.",
         f"Enough chatter: at least {p2.MIN_CHATTER} posts or independent articles.",
         f"Unusual chatter: a z-score of at least {p2.MIN_Z_SCORE:.0f} (one standard deviation above normal). Once "
         f"{p2.MIN_HISTORY_SNAPSHOTS}+ daily snapshots exist, each stock is compared with its own normal level; "
         "before that, with the other stocks this week.",
         f"Many voices: at least {p2.MIN_UNIQUE_AUTHOR_RATIO:.0%} of posts must come from different accounts, so a "
         "handful of users cannot push a stock onto the list.",
         f"Backed by real trading: the last 3 days' volume must be at least {p2.MIN_VOLUME_RATIO:.0%} of its usual "
         f"level (the 20 days before), and a big chatter spike needs volume {p2.SPIKE_MIN_VOLUME_RATIO:.1f}x normal. "
         "Talk without trading suggests artificial hype.",
         f"Fundamentals: profitable (positive P/E) with P/E up to {p2.MAX_PE:.0f}, revenue not shrinking year on "
         f"year, and debt/equity up to {p2.MAX_DEBT_TO_EQUITY:.0f}%. At least {p2.MIN_KNOWN_FUNDAMENTALS} of the 3 "
         "must be known.",
     ]},
    {"phase": 3, "name": "Sentiment scoring",
     "what": "Normalises emojis and slang, isolates each ticker in multi-ticker posts, scores with FinBERT "
             "plus a trader-slang dictionary, one vote per user.",
     "details": [
         "FinBERT is a language model trained on financial text. It reads each post or headline and scores it "
         "as positive, negative or neutral for the stock.",
         "Trader slang and emojis (\"to the moon\", \"bagholder\", 🚀) are translated first, using a seed "
         "dictionary plus slang the system has learned from posts with Bullish/Bearish tags.",
         "When a post mentions several stocks, only the part about each stock is scored for it.",
         "One vote per user: a single account posting 20 times counts once.",
         "The model is checked every run against traders' own Bullish/Bearish tags; that accuracy is shown in "
         "\"How far to trust it\".",
     ]},
    {"phase": 4, "name": "AI catalyst and sarcasm review",
     "what": "An LLM explains why each stock is discussed and must cite real headlines. Hype, meme and spam "
             "driven interest, negative or minor catalysts are rejected; sarcastic posts are re-scored.",
     "details": [
         "An AI model (LLM) reads the week's headlines for each remaining stock and names the catalyst: "
         "earnings, a deal, a product, an approval, and so on. It must link the real headlines it used.",
         "Stocks are rejected when the catalyst is bad for the share price, too minor to matter, or when the "
         "interest looks driven by hype, memes or pump language rather than news.",
         "Posts the sentiment model may have misread (sarcasm, jokes) are re-checked and re-scored.",
         "Survivors are ranked by a score that combines sentiment, catalyst strength, unusual chatter and "
         "trading volume, each as a percentile against the other finalists.",
         "If the AI review fails for a stock, that stock is not approved. Answers are cached to keep costs low.",
     ]},
    {"phase": 5, "name": "Risk checks and diversification",
     "what": "Removes stocks that already jumped over 20% in 5 days, are very overbought or very volatile; "
             f"flags earnings within 7 days and high short interest; max {p5.MAX_PER_SECTOR} per sector and no "
             "near-duplicates.",
     "details": [
         f"Chasing: a stock already up more than {p5.MAX_5D_RETURN:.0%} in 5 trading days, or with RSI above "
         f"{p5.MAX_RSI} (very overbought), is removed. The chatter is probably reacting to a move that already "
         "happened.",
         f"Volatility: annualised 30-day volatility above {p5.MAX_VOLATILITY:.0%} is removed; above "
         f"{p5.HIGH_VOLATILITY:.0%} is flagged.",
         f"Flags shown on the pick: earnings within {p5.EARNINGS_SOON_DAYS} days, short interest above "
         f"{p5.HIGH_SHORT_INTEREST:.0%} of the float, price more than {p5.FLAG_ABOVE_MA50:.0%} above its 50-day "
         "average or below its 200-day average, and a negative analyst consensus.",
         f"Different industries: at most {p5.MAX_PER_SECTOR} picks per sector, so a full Top {p5.TOP_N} covers at "
         "least 3 industries whenever enough stocks qualify. The next-ranked stock from another sector takes "
         "the freed place.",
         f"No near-duplicates: two stocks whose daily returns moved almost identically over the last 120 "
         f"sessions (correlation above {p5.MAX_CORRELATION}) would be the same bet twice, so only the "
         "higher-ranked one is kept.",
         "Track record: every pick is logged with its price and the S&P 500's, and its 5- and 20-day results "
         "are reported in later runs.",
     ]},
]

# Key thresholds, published so the app's explanations always match the code
SETTINGS = {
    "lookback_days": p1.LOOKBACK_DAYS, "min_market_cap": p2.MIN_MARKET_CAP,
    "min_unique_author_ratio": p2.MIN_UNIQUE_AUTHOR_RATIO, "max_pe": p2.MAX_PE,
    "max_per_sector": p5.MAX_PER_SECTOR, "top_n": p5.TOP_N, "max_5d_return": p5.MAX_5D_RETURN,
    "max_volatility": p5.MAX_VOLATILITY, "chart_sessions": p5.CHART_SESSIONS, "chart_default_sessions": 90,
}


def pick_record(r: dict) -> dict:
    g = lambda k, d=None: r.get(k, d)
    return {
        "rank": g("rank"), "ticker": g("ticker"), "name": g("name"), "sector": g("sector"),
        "price": g("price"), "score": g("final_score"),
        "catalyst": {"type": g("catalyst_type"), "direction": g("catalyst_direction"),
                     "materiality": g("materiality"), "confidence": g("catalyst_confidence"),
                     "summary": g("catalyst_summary"), "evidence": g("evidence") or [],
                     "hype_risk": g("hype_risk"), "hype_signals": g("hype_signals") or []},
        "sentiment": {"net": g("final_sentiment"), "label": g("final_label"),
                      "social_users": g("social_voters"), "news_articles": g("news_articles"),
                      "bullish_pct": g("bullish_pct"), "bearish_pct": g("bearish_pct"),
                      "sarcasm_corrections": g("sarcasm_fixes")},
        "attention": {"chatter_score": g("chatter_score"), "stocktwits_posts": g("social_mentions"),
                      "z_score": g("z_score"), "z_method": g("z_method"),
                      "unique_author_ratio": g("unique_author_ratio"), "volume_vs_normal": g("volume_ratio")},
        "fundamentals": {"market_cap": g("market_cap"), "pe_ratio": g("pe_ratio"),
                         "revenue_growth": g("revenue_growth"), "debt_to_equity_pct": g("debt_to_equity")},
        "risk": {"return_5d": g("ret_5d"), "rsi_14": g("rsi_14"), "volatility_30d": g("volatility_30d"),
                 "flags": g("risk_flags") or []},
        # how the final score was built (percentile ranks among this week's eligible stocks)
        "score_breakdown": {"sentiment": g("c_sentiment"), "catalyst": g("c_catalyst"),
                            "attention": g("c_attention"), "volume": g("c_volume")},
        "price_history": g("price_history") or {"dates": [], "close": []},
    }


def llm_budget() -> dict:
    """Spend vs limit on the LLM key, for relays that expose OpenAI's old billing endpoints (e.g. apiyi).
    Returns {} when the provider does not report it."""
    key, base = os.getenv("LLM_API_KEY"), (os.getenv("LLM_BASE_URL") or "").rstrip("/")
    if not key or not base:
        return {}
    import urllib.request
    def get(path):
        req = urllib.request.Request(base + path, headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    try:
        limit = float(get("/dashboard/billing/subscription")["hard_limit_usd"])
        end = (_utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
        used = float(get(f"/dashboard/billing/usage?start_date=2020-01-01&end_date={end}")["total_usage"]) / 100
    except Exception:
        return {}
    share = used / limit if limit else 0.0
    return {"used_usd": round(used, 4), "limit_usd": limit, "share_used": round(share, 3),
            "status": "exhausted" if share >= 1 else "low" if share >= 0.8 else "ok"}


def report_budget(budget: dict):
    if not budget:
        return
    print(f"\n  LLM budget: ${budget['used_usd']:.2f} of ${budget['limit_usd']:.2f} used "
          f"({budget['share_used']:.0%})")
    if budget["status"] == "exhausted":
        print("  WARNING: the LLM key has hit its spending limit. Phase 4 and the chat will fall back to "
              "simple rules until you top up.")
    elif budget["status"] == "low":
        print("  WARNING: over 80% of the LLM budget is used. Top up soon.")


def company_names(tickers) -> dict:
    session = SessionLocal()
    try:
        return dict(session.query(USUniverse.symbol, USUniverse.description)
                    .filter(USUniverse.symbol.in_(list(tickers))).all())
    finally:
        session.close()


# Plain-English versions of each filter's technical reason (shown in the app and used in answers)
def _num(text, pattern):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def explain_reason(reason: str) -> str:
    """Turn a filter's technical reason into one sentence a non-expert understands."""
    r = str(reason or "")
    low = r.lower()
    if low.startswith("chatter") and "mentions" in low:
        n = _num(r, r"chatter (\d+)")
        return (f"Only {n} posts or articles mentioned it this week. At least 5 are needed before the screen "
                "treats the interest as real.")
    if low.startswith(("not checked", "not among")):
        return "It was discussed, but less than the stocks that were checked in detail this week."
    if low.startswith("z-score"):
        return "The amount of chatter about it was not unusual compared with other stocks."
    if low.startswith("unique authors"):
        pct = _num(r, r"unique authors (\d+%)")
        return (f"Most posts came from the same few accounts (only {pct} were different people; 70% is needed), "
                "which looks more like a few users promoting it than broad interest.")
    if "market cap" in low and "<" in low:
        return (f"The company is too small ({r.split('market cap ')[-1].split(' <')[0]} market value). Stocks under "
                "$300M are excluded because they are easier to pump and dump.")
    if low.startswith("price $"):
        return "It is a penny stock (under $1 a share), which is excluded as too risky."
    if re.search(r"\bspac\b|blank-check", low):
        return "It looks like a SPAC (a blank-check shell company), which is excluded because its value depends on a future deal."
    if "trading days of history" in low:
        return "It started trading too recently to judge its normal trading volume and price behaviour."
    if "traded value" in low:
        return "Too few shares change hands each day, so prices can be moved easily."
    if "synthetic social push" in low:
        return ("Talk about it suddenly spiked but real trading stayed normal, which suggests the chatter was "
                "artificially pushed rather than reflecting real investor interest.")
    if low.startswith("trading volume") or "no trading-volume data" in low:
        v = _num(r, r"([\d.]+)x normal")
        return (f"People were talking about it, but actual trading was below its usual level ({v}x normal), "
                "so the chatter was not backed by real buying or selling." if v else
                "There was no trading-volume data to confirm the chatter.")
    if low.startswith("unprofitable"):
        return "The company is currently losing money (negative earnings), so it fails the profitability check."
    if low.startswith("p/e"):
        pe = _num(r, r"P/E ([\d.]+)")
        return (f"The share price is expensive relative to profits (about {pe} times earnings; "
                "the screen allows up to 60).")
    if "revenue growth" in low:
        g = _num(r, r"growth ([+-][\d.]+%)")
        return f"Its sales shrank compared with a year ago ({g} quarterly revenue growth)."
    if "debt/equity" in low:
        d = _num(r, r"debt/equity (\d+)%")
        return (f"It carries a lot of debt (debt is about {int(d) / 100:.1f} times shareholders' equity; "
                "the limit is 2 times)." if d else "It carries too much debt.")
    if "fundamentals available" in low or "no market-cap data" in low:
        return "Not enough financial data was available to check the company's fundamentals."
    if low.startswith("llm failed"):
        return "The AI review could not be completed for it, so it was not approved."
    if low.startswith("high hype risk"):
        sig = r[r.find("(") + 1:r.rfind(")")] if "(" in r else ""
        sig = sig.replace("_", " ")
        return ("The AI review judged the interest to be driven mainly by hype rather than real news"
                + (f" (signs: {sig})." if sig else "."))
    if low.startswith("catalyst is negative"):
        what = r[r.find("(") + 1:].rstrip(")") if "(" in r else ""
        what = what.replace("_", " ")
        return f"The news behind the interest is bad for the share price ({what})." if what else \
            "The news behind the interest is bad for the share price."
    if low.startswith("no genuine catalyst"):
        if "low materiality" in low:
            return "There was real news, but too minor to move the share price on its own."
        return "No real news event (earnings, deal, product, approval) explains why people are talking about it."
    if "not positive after sarcasm" in low:
        return "Once sarcastic or mocking posts were corrected, overall sentiment about it was not positive."
    if low.startswith("already up"):
        return f"It has {r.split(' (')[0].replace('already up', 'already risen')}, so buying now risks chasing a move that already happened."
    if low.startswith("rsi"):
        return "The price has risen so fast recently that the stock looks overbought."
    if low.startswith("volatility"):
        return "Its price swings are very large, so a week's gain can disappear in a day."
    if low.startswith("earnings in"):
        return f"It reports earnings soon ({r.split('(')[0].strip()}), which makes the next week a coin flip."
    if "picks in" in low:
        sector = r.split("picks in ")[-1]
        n = _num(r, r"already (\d+) picks") or p5.MAX_PER_SECTOR
        return (f"It passed every check, but the list already had {n} {sector} stocks, so it was left out to keep "
                "the picks spread across different industries.")
    if "moves almost identically" in low:
        twin = _num(r, r"\$([A-Z]+)")
        return (f"Its price moves almost exactly like ${twin}, which is already on the list, so picking both "
                "would be the same bet twice.")
    if low.startswith("passed every check but ranked below"):
        return "It passed every check, but other stocks scored higher on sentiment, news strength and attention."
    if low.startswith("not enough scorable"):
        return "There were not enough posts or headlines about it to measure sentiment reliably."
    if "no price history" in low:
        return "Not enough price history was available to check its risk."
    return r



def build_dropped(summary, survivors, candidates, final) -> list:
    """Why every mentioned stock that is not a final pick was left out, at the phase that dropped it."""
    picked = set(final["ticker"]) if final is not None and not final.empty else set()
    reasons = {}
    for d in p2.LAST_REJECTED:
        reasons[d["ticker"]] = (2, f"Phase 2: {d['check']}", d["reason"])
    reviewed = candidates.to_dict("records") if candidates is not None and not candidates.empty else []
    for r in reviewed:
        if r.get("eligibility") != "ELIGIBLE":
            reasons[r["ticker"]] = (4, "Phase 4: AI catalyst and hype review", r["eligibility"])
    removed = dict(p5.LAST_RESULT.get("removed", []))
    for r in reviewed:
        t = r["ticker"]
        if r.get("eligibility") == "ELIGIBLE" and t not in picked:
            reasons[t] = (5, "Phase 5: risk and diversification checks",
                          removed.get(t, "passed every check but ranked below the final picks"))
    survived = set(survivors["ticker"]) if survivors is not None and not survivors.empty else set()
    for t in survived - picked - set(reasons):
        reasons[t] = (3, "Phase 3: sentiment scoring", "not enough scorable posts or headlines")
    names = company_names(summary["ticker"])
    out = []
    for r in summary.to_dict("records"):
        t = r["ticker"]
        if t in picked:
            continue
        phase, stage, why = reasons.get(t, (2, "Phase 2: chatter and distinct users", "did not pass the filters"))
        out.append({"ticker": t, "name": names.get(t), "phase": phase, "stage": stage, "reason": why,
                    "why": explain_reason(why),
                    "stocktwits_posts": r.get("social_mentions"), "distinct_users": r.get("unique_authors"),
                    "news_articles": r.get("unbiased_news"), "chatter_score": r.get("chatter_score")})
    return sorted(out, key=lambda d: (-(d["chatter_score"] or 0), d["ticker"]))


def build_summary(run_id, started, stages, final, candidates, survivors, summary=None) -> dict:
    finished = _utcnow()
    since = finished - timedelta(days=p1.LOOKBACK_DAYS)
    sources, window = collect_sources(since)
    days = history_days()
    z_mode = (survivors["z_method"].mode().iloc[0]
              if survivors is not None and not survivors.empty and "z_method" in survivors else "n/a")
    check = p3.LAST_SELF_CHECK or {}
    used = check.get("FinBERT + slang (used)") or check.get("seed + learned slang") or {}
    card = p5.LAST_RESULT.get("scorecard")
    llm = p4.LAST_LLM_INFO or {}

    limitations = [
        "StockTwits only returns the latest ~30 posts per ticker each time data is collected, so counts are samples.",
        "Catalyst summaries are written by an AI model from the linked headlines; check the sources.",
        "Market data comes from Yahoo Finance and may contain errors.",
    ]
    if z_mode != "history":
        limitations.append(f"Each screen covers the last {p1.LOOKBACK_DAYS} days, but only {days} daily run(s) are saved as a baseline, "
                           "so 'unusual chatter' compares each stock with the other stocks, not with its own "
                           "normal level (needs 3+ daily runs).")
    if used.get("balanced") is not None:
        limitations.append(f"The sentiment model agrees with traders' own Bullish/Bearish tags "
                           f"{used['balanced']:.0%} of the time (balanced accuracy; 50% = guessing).")
    if not final.empty and len(final) < p5.TOP_N:
        limitations.append(f"Only {len(final)} stocks passed every check this week; the list is not padded.")

    local = finished.replace(tzinfo=timezone.utc).astimezone(LOCAL_TZ)
    return clean({
        "run_id": run_id,
        "question_answered": "Top US-listed stocks (including ADRs) discussed this week that pass quality and risk checks",
        "market": "US-listed common stocks, REITs and ADRs (foreign companies traded in the US) on " + ", ".join(
            EXCHANGE_NAMES.get(m, m) for m in sorted(ALLOWED_MICS)),
        "screened_at_utc": finished.isoformat(timespec="seconds"),
        "screened_at_local": local.strftime("%Y-%m-%d %H:%M") + " Singapore time (UTC+8)",
        "started_at_utc": started.isoformat(timespec="seconds"),
        "data_window": window,
        "sources": sources,
        "funnel": stages,
        "methodology": METHODOLOGY,
        "reliability": {
            "sentiment_check": {"balanced_accuracy": used.get("balanced"),
                                "bearish_hit_rate": used.get("bearish_hit"),
                                "finbert_used": check.get("finbert_used")},
            "unusual_chatter_mode": z_mode, "days_of_history": days,
            "llm": {**llm, "budget": llm_budget()}, "track_record": track_record(card),
        },
        "picks": [pick_record(r) for r in final.to_dict("records")] if not final.empty else [],
        "removed_in_final_checks": [{"ticker": t, "reason": w} for t, w in p5.LAST_RESULT.get("removed", [])],
        "excluded_by_ai_review": ([{"ticker": r["ticker"], "reason": r["eligibility"]}
                                   for r in candidates.to_dict("records") if r.get("eligibility") != "ELIGIBLE"]
                                  if candidates is not None and not candidates.empty else []),
        "dropped": build_dropped(summary, survivors, candidates, final) if summary is not None else [],
        "settings": SETTINGS,
        "sector_mix": (final["sector"].fillna("Unknown").value_counts().to_dict() if not final.empty else {}),
        "other_eligible": p5.LAST_RESULT.get("other_eligible", []),
        "limitations": limitations,
        "disclaimer": DISCLAIMER,
    })


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------
def start_run(run_type: str) -> tuple:
    init_db()
    session = SessionLocal()
    try:
        run = PipelineRun(run_type=run_type, started_at=_utcnow(), status="running")
        session.add(run)
        session.commit()
        return run.id, run.started_at
    finally:
        session.close()


def finish_run(run_id: int, status: str, summary: dict = None, error: str = None):
    session = SessionLocal()
    try:
        run = session.get(PipelineRun, run_id)
        run.finished_at = _utcnow()
        run.status = status
        run.error = error
        run.summary_json = json.dumps(summary) if summary else None
        session.commit()
    finally:
        session.close()


def collect_data():
    from data_base_creation.trigger import run_pipeline as run_collection
    run_collection(rebuild=False, skip_collect=False)


def run_daily(skip_collect=False):
    run_id, started = start_run("daily")
    try:
        if not skip_collect:
            collect_data()
        _, summary = p1.run_phase1()
        finish_run(run_id, "success", clean({"run_id": run_id, "run_type": "daily",
                                             "tickers_mentioned": len(summary), "stats": p1.LAST_STATS}))
        print(f"\nDaily run {run_id} done: {len(summary)} tickers saved to today's history snapshot.")
    except Exception as err:
        finish_run(run_id, "failed", error=traceback.format_exc()[-2000:])
        raise err


def run_weekly(skip_collect=False):
    run_id, started = start_run("weekly")
    try:
        if not skip_collect:
            collect_data()
        mentions, summary = p1.run_phase1()
        survivors, keep = p2.run_phase2(mentions, summary)
        p3_result, scored = p3.run_phase3(survivors, keep) if not survivors.empty else (pd.DataFrame(), pd.DataFrame())
        candidates = pd.DataFrame()
        if not p3_result.empty:
            _, candidates = p4.run_phase4(p3_result, scored)
        final = p5.run_phase5(candidates=candidates) if not candidates.empty else pd.DataFrame()

        stats = p1.LAST_STATS
        stages = [
            {"stage": f"StockTwits posts collected (last {p1.LOOKBACK_DAYS} days)", "count": stats.get("social_in_window")},
            {"stage": "StockTwits posts kept after removing bots and spam", "count": stats.get("social_after_bots")},
            {"stage": f"News articles collected (last {p1.LOOKBACK_DAYS} days)", "count": stats.get("news_in_window")},
            {"stage": "US stocks mentioned", "count": len(summary)},
            {"stage": "Passed market, volume, distinct-user and fundamentals filters", "count": len(survivors)},
            {"stage": "Passed AI catalyst, hype and sarcasm review",
             "count": int((candidates["eligibility"] == "ELIGIBLE").sum()) if not candidates.empty else 0},
            {"stage": "Final picks after risk and diversification checks", "count": len(final)},
        ]
        result = build_summary(run_id, started, stages, final, candidates, survivors, summary)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for name in ("latest.json", f"run_{run_id}.json"):
            with open(os.path.join(OUTPUT_DIR, name), "w") as f:
                json.dump(result, f, indent=2)
        finish_run(run_id, "success", result)

        print(f"\n=== RUN {run_id} PUBLISHED ({result['screened_at_local']}) ===")
        for s in stages:
            print(f"  {s['count'] if s['count'] is not None else '?':>6}  {s['stage']}")
        print(f"\n  Frontend file: {os.path.join(OUTPUT_DIR, 'latest.json')}")
        report_budget(result["reliability"]["llm"].get("budget") or {})
        return result
    except Exception as err:
        finish_run(run_id, "failed", error=traceback.format_exc()[-2000:])
        raise err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the stock-chatter backend.")
    parser.add_argument("--daily", action="store_true", help="collect data + Phase 1 only")
    parser.add_argument("--skip-collect", action="store_true", help="use data already stored")
    args = parser.parse_args()
    if args.daily:
        run_daily(args.skip_collect)
    else:
        run_weekly(args.skip_collect)
