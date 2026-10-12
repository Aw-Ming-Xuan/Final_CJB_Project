"""SQLite models and setup for the stock-chatter pipeline.

Changes from the first version:
- DB path is built relative to this file (override with MARKET_DB_PATH in .env),
  so it no longer breaks if the folder is moved or renamed.
- StockTwitsPost gains author_join_date / author_followers / author_post_count
  (filled by stocktwits_collector.py) so Phase 2 can enforce the bot filter.
- Indexes on the columns the collectors and categorizer look up, plus a UNIQUE
  index on ProcessedNews (raw_post_id, ticker): one row per post per ticker,
  never duplicated.
- init_db() also upgrades an EXISTING database in place (adds the new columns
  and indexes), because SQLAlchemy's create_all() skips tables that already exist.
"""

import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, inspect, text,
    Column, Integer, String, Float, DateTime, Date, Text, Index,
)
from sqlalchemy.orm import sessionmaker, declarative_base

_DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data.db")
DB_PATH = os.getenv("MARKET_DB_PATH", _DEFAULT_DB_PATH)
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def _utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)


# =====================================================================
# TABLE DEFINITIONS
# =====================================================================

class RawNewsPost(Base):
    """Headlines from RSS (news_collector) and Finnhub (finnhub_news_collector)."""
    __tablename__ = "raw_news_posts"
    __table_args__ = {'extend_existing': True}

    id = Column(String(100), primary_key=True)
    title = Column(String(255))
    link = Column(String(500), nullable=True)
    url = Column(String(500), nullable=True)
    source = Column(String(100))          # e.g. CNBC_Earnings, Finnhub_NVDA, Finnhub_general, Nasdaq_NVDA
    publisher = Column(String(100), nullable=True)   # who wrote it: CNBC, Yahoo, MarketWatch, Reuters, Motley Fool...
    summary = Column(Text, nullable=True)
    body = Column(Text, nullable=True)    # "title. summary" + appended $CASHTAGS (Finnhub)
    published_at = Column(DateTime)


class StockTwitsPost(Base):
    """Messages from stocktwits_collector."""
    __tablename__ = "stocktwits_posts"
    __table_args__ = {'extend_existing': True}

    id = Column(String(100), primary_key=True)   # st_<message id>
    symbol = Column(String(20))                  # stream the post was found in
    author = Column(String(100))
    body = Column(Text)                          # message text + appended $CASHTAGS
    sentiment = Column(String(20), nullable=True)  # Bullish / Bearish / Unspecified
    created_at = Column(DateTime)                # the message's own timestamp

    # Used by Phase 2 to drop bots / brand-new accounts
    author_join_date = Column(DateTime, nullable=True)
    author_followers = Column(Integer, nullable=True)
    author_post_count = Column(Integer, nullable=True)


class ProcessedNews(Base):
    """Output of categorizer.py: one row per (raw post, ticker)."""
    __tablename__ = "processed_news"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    raw_post_id = Column(String(100), nullable=True)   # id in raw_news_posts OR stocktwits_posts
    ticker = Column(String(20), nullable=True)         # real ticker, or MARKET
    source_type = Column(String(50), nullable=True)    # NEWS / SOCIAL
    source_name = Column(String(100), nullable=True)
    category = Column(String(50), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    sentiment = Column(String(20), nullable=True)
    title = Column(String(255), nullable=True)
    summary = Column(Text, nullable=True)
    source = Column(String(100), nullable=True)
    created_at = Column(DateTime, nullable=True)       # when the post was published
    processed_at = Column(DateTime, default=_utcnow)


class USUniverse(Base):
    """Every US-listed symbol Finnhub knows, with its type and exchange (us_universe.py).
    Used to keep the pipeline on US common stocks: no ETFs, ADRs or OTC names."""
    __tablename__ = "us_universe"
    __table_args__ = {'extend_existing': True}

    symbol = Column(String(20), primary_key=True)
    description = Column(String(255), nullable=True)
    type = Column(String(50), nullable=True)    # Common Stock / REIT / ADR / ETP / ...
    mic = Column(String(10), nullable=True)     # XNAS, XNYS, XASE, ...
    updated_at = Column(DateTime, default=_utcnow)


class Phase1History(Base):
    """One row per (run date, ticker): the Phase 1 summary, saved every run.
    Phase 2 uses it as the 'normal level of chatter' baseline for the z-score."""
    __tablename__ = "phase1_history"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, nullable=False)
    ticker = Column(String(20), nullable=False)
    chatter_score = Column(Integer)          # social mentions + unbiased news mentions
    social_mentions = Column(Integer)
    unbiased_news = Column(Integer)
    unique_authors = Column(Integer)
    unique_author_ratio = Column(Float, nullable=True)


class MarketSnapshot(Base):
    """Cache of market data per ticker (Yahoo Finance, Finnhub or Alpha Vantage),
    so reruns within MARKET_CACHE_HOURS make no API calls."""
    __tablename__ = "market_snapshot"
    __table_args__ = {'extend_existing': True}

    ticker = Column(String(20), primary_key=True)
    name = Column(String(255), nullable=True)
    market_cap = Column(Float, nullable=True)          # USD
    price = Column(Float, nullable=True)
    recent_volume = Column(Float, nullable=True)       # mean of the last 3 sessions
    baseline_volume = Column(Float, nullable=True)     # mean of the 20 sessions before those
    volume_ratio = Column(Float, nullable=True)
    avg_dollar_volume = Column(Float, nullable=True)
    trading_days = Column(Integer, nullable=True)      # sessions of history found (3 months)
    pe_ratio = Column(Float, nullable=True)            # forward P/E, else trailing
    revenue_growth = Column(Float, nullable=True)      # quarterly YoY, as a fraction (0.12 = 12%)
    debt_to_equity = Column(Float, nullable=True)      # percent, as Yahoo reports it
    source = Column(String(100), nullable=True)
    fetched_at = Column(DateTime, default=_utcnow)


class PickHistory(Base):
    """Every final pick from Phase 5, with the price at the time, so later runs can
    measure how the picks actually did against the S&P 500 (SPY)."""
    __tablename__ = "pick_history"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    pick_date = Column(Date, nullable=False)
    ticker = Column(String(20), nullable=False)
    rank = Column(Integer)
    final_score = Column(Float, nullable=True)
    entry_price = Column(Float, nullable=True)       # last close when picked
    spy_entry_price = Column(Float, nullable=True)
    catalyst_summary = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class PipelineRun(Base):
    """One row per run of run_pipeline.py. summary_json holds everything the frontend
    shows: picks, evidence links, data sources, data window, filter funnel, reliability."""
    __tablename__ = "pipeline_runs"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_type = Column(String(20))                  # weekly / daily
    started_at = Column(DateTime)
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(20))                    # running / success / failed
    error = Column(Text, nullable=True)
    summary_json = Column(Text, nullable=True)


# Indexes (speed up the dedupe lookups and the phase filters)
Index("ix_raw_news_url", RawNewsPost.url)
Index("ix_raw_news_published", RawNewsPost.published_at)
Index("ix_raw_news_source", RawNewsPost.source)
Index("ix_st_symbol", StockTwitsPost.symbol)
Index("ix_st_author", StockTwitsPost.author)
Index("ix_st_created", StockTwitsPost.created_at)
Index("ix_us_universe_type", USUniverse.type)
Index("ix_p1hist_date_ticker", Phase1History.snapshot_date, Phase1History.ticker, unique=True)
Index("ix_p1hist_ticker", Phase1History.ticker)
Index("ix_pick_date_ticker", PickHistory.pick_date, PickHistory.ticker, unique=True)
Index("ix_processed_ticker", ProcessedNews.ticker)
Index("ix_processed_created", ProcessedNews.created_at)
Index("ix_processed_raw_ticker", ProcessedNews.raw_post_id, ProcessedNews.ticker, unique=True)


# Columns added after the first version; applied to existing databases by init_db()
_NEW_COLUMNS = {
    "raw_news_posts": {
        "publisher": "VARCHAR(100)",
    },
    "stocktwits_posts": {
        "author_join_date": "DATETIME",
        "author_followers": "INTEGER",
        "author_post_count": "INTEGER",
    },
}


def _ensure_columns():
    insp = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _NEW_COLUMNS.items():
            if not insp.has_table(table):
                continue
            existing = {c["name"] for c in insp.get_columns(table)}
            for name, ddl in columns.items():
                if name not in existing:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
                    print(f"  Added column {table}.{name}")


def _ensure_indexes():
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            try:
                index.create(bind=engine, checkfirst=True)
            except Exception as err:
                print(f"  Could not create index {index.name}: {err}")


def init_db():
    """Create tables, then upgrade an existing database in place."""
    Base.metadata.create_all(bind=engine)
    _ensure_columns()
    _ensure_indexes()


def save_raw_news_posts(db_session, news_posts):
    """
    Safely saves news posts by ignoring duplicate article IDs in-memory and database.
    """
    if not news_posts:
        return 0

    existing_ids = set(res[0] for res in db_session.query(RawNewsPost.id).all())

    seen_ids = set()
    unique_posts = []

    for post in news_posts:
        post_id = getattr(post, 'id', None)
        if post_id and post_id not in existing_ids and post_id not in seen_ids:
            seen_ids.add(post_id)
            unique_posts.append(post)

    if unique_posts:
        db_session.add_all(unique_posts)
        db_session.commit()

    return len(unique_posts)


if __name__ == "__main__":
    init_db()
    print(f"Database initialized successfully at: {DB_PATH}")
