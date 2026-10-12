"""Phase 3: contextual NLP and sentiment scoring.

Input : Phase 2 output (survivors + their mention rows).
Output: (ticker_sentiment, scored_mentions) for Phase 4.

Steps (run in this order, because each one feeds the next):

  4. EMOJI & SYMBOL NORMALISATION (pre-processing)
       Emojis become plain words FinBERT understands (🚀 -> "surge", 📉 -> "falling",
       🐻 -> "bearish"). "+12%" -> "up 12 percent". URLs, @users and stretched words
       ("soooo") are cleaned. Joke/sarcasm emojis (🤡 😂) are counted for Phase 4.
  2. SLANG DICTIONARY
       a) SEED_SLANG: hand-written trader jargon with a weight from -1 to +1
          ("to the moon" +0.9, "rug pull" -0.9, "selling puts" +0.3, "covered calls" -0.2).
          Longest phrase wins, so "short squeeze" is not read as "short".
          Negation flips a phrase ("not bullish", "don't buy").
       b) LEARNED slang: trained on StockTwits posts that users tagged Bullish/Bearish.
          Words and emojis that appear far more often in one class get a weight
          (smoothed log-odds). It grows every time you collect more posts.
  3. TICKER ISOLATION
       A post about several tickers is split into clauses ("but", "while", "vs", ";",
       and before action words like buy/sell/short/long). Each ticker is scored only on
       the clauses that mention it; a clause with no ticker belongs to the last ticker
       named. Comparisons ("$AMD over $INTC", "$X > $Y", "sell $A buy $B") give the
       winner a boost and the loser a penalty.
  1. FINBERT
       ProsusAI/finbert gives positive/neutral/negative probabilities for each isolated
       text; score = P(positive) - P(negative).

Combining (per post, per ticker, from -1 bearish to +1 bullish):
  social: 0.5 * FinBERT + 0.5 * slang, then blended 50/50 with the user's own
          Bullish/Bearish tag when they set one (single-ticker posts only, because the tag
          describes the whole post) (FinBERT is trained on formal news, so it
          misses jargon; the slang score and the user tag cover that).
  news:   0.8 * FinBERT + 0.2 * slang.
Per ticker, each StockTwits account counts once (its posts are averaged first), so one
loud user cannot dominate. Social and news are reported separately and combined 60/40.
The result is then multiplied by confidence = votes / (votes + 10), so a ticker with a
handful of posts cannot outrank one with a hundred on luck alone.

The script also checks itself: on StockTwits posts that users tagged, it measures how
often each method agrees with the tag (learned slang is trained on 80% and tested on the
other 20%, so that number is honest).

Run:  python filtering_of_data/phase3_filter.py         (runs Phases 1 and 2 first)
Install: pip install transformers torch
         (first run downloads the FinBERT model, about 440 MB; without it the script
          falls back to the slang score and says so)
"""

import hashlib
import math
import os
import html
import re
import sys
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd

from data_base_creation.database import SessionLocal, StockTwitsPost
from data_base_creation.us_universe import load_us_symbols

try:
    from filtering_of_data.phase2_filter import run_phase2
except ImportError:
    from phase2_filter import run_phase2

# ----------------------------------------------------------------------
# Settings
# ----------------------------------------------------------------------
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_BATCH = 16
BULLISH_CUTOFF = 0.15
BEARISH_CUTOFF = -0.15

SOCIAL_WEIGHTS = {"finbert": 0.5, "slang": 0.5}
NEWS_WEIGHTS = {"finbert": 0.8, "slang": 0.2}
USER_TAG_WEIGHT = 0.5            # share of the user's Bullish/Bearish tag in the social score
SOCIAL_SHARE = 0.6               # social vs news when a ticker has both
COMPARISON_BONUS = 0.4
SHRINK_VOTES = 10                # confidence = votes / (votes + 10): 5 votes keep 1/3 of the score

LEARN_MIN_COUNT = 8              # a word must appear this often in tagged posts to be learned
LEARN_MIN_CLASS_POSTS = 30       # need at least this many Bullish AND Bearish posts to learn
LEARN_MIN_LOGODDS = 0.7
LEARNED_SCALE = 0.6              # learned weights are trusted less than the seed list
LEARNED_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_slang.csv")

# ----------------------------------------------------------------------
# Step 4: emoji and symbol normalisation
# ----------------------------------------------------------------------
# Longest sequences first (💎🙌 before 💎). Words are chosen so FinBERT understands them.
EMOJI_MAP = [
    ("💎🙌", " holding strong "), ("💎👐", " holding strong "), ("🚀🚀", " surge surge "),
    ("🚀", " surge "), ("🌙", " moon "), ("🌕", " moon "), ("📈", " rising "),
    ("📉", " falling "), ("🐂", " bullish "), ("🐻", " bearish "), ("🔥", " strong "),
    ("💪", " strong "), ("💎", " holding "), ("🙌", " holding "), ("🤑", " profit "),
    ("💰", " profit "), ("💵", " profit "), ("🟢", " up "), ("✅", " up "),
    ("🔴", " down "), ("❌", " down "), ("🩸", " losses "), ("💀", " crash "),
    ("⚰️", " crash "), ("🪦", " crash "), ("🧻", " weak "), ("⚠️", " warning "),
    ("🚨", " warning "), ("😭", " painful loss "), ("😢", " loss "), ("😱", " fear "),
    ("🙏", " hope "), ("🤡", " "), ("😂", " "), ("🤣", " "), ("🙄", " "),
]
SARCASM_EMOJIS = ("🤡", "😂", "🤣", "🙄")
SARCASM_PHRASES = re.compile(r"(/s\b|\bsure thing\b|\bwhat could go wrong\b|\byeah right\b|"
                             r"\bwhat a surprise\b|\btotally\b)", re.I)

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
PCT_RE = re.compile(r"([+-])\s?(\d+(?:\.\d+)?)\s?%")
STRETCH_RE = re.compile(r"([a-zA-Z])\1{2,}")
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
OTHER_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF☀-➿️]")


def sarcasm_hint(text: str) -> bool:
    return any(e in (text or "") for e in SARCASM_EMOJIS) or bool(SARCASM_PHRASES.search(text or ""))


def normalise_text(text: str) -> str:
    """Emojis -> words, % moves -> words, strip URLs/users, keep tickers without the $."""
    t = html.unescape(text or "")                                   # &#39; -> ' , &quot; -> "
    t = t.replace("\u2019", "'").replace("\u2018", "'")           # curly apostrophes (phones)
    t = URL_RE.sub(" ", t)
    t = USER_RE.sub(" ", t)
    for emoji, words in EMOJI_MAP:
        t = t.replace(emoji, words)
    t = OTHER_EMOJI_RE.sub(" ", t)
    t = PCT_RE.sub(lambda m: f" {'up' if m.group(1) == '+' else 'down'} {m.group(2)} percent ", t)
    t = STRETCH_RE.sub(r"\1\1", t)            # "soooo" -> "soo", "mooooon" -> "moon"
    t = CASHTAG_RE.sub(r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------
# Step 2: slang dictionary
# ----------------------------------------------------------------------
SEED_SLANG = {
    # options positions first: these phrases must beat "calls", "puts", "short", "selling"
    "selling puts": 0.3, "sold puts": 0.3, "short puts": 0.3, "cash secured puts": 0.3,
    "selling calls": -0.2, "sold calls": -0.2, "short calls": -0.2, "covered calls": -0.2,
    "buying calls": 0.6, "bought calls": 0.6, "buying puts": -0.6, "bought puts": -0.6,
    # bullish
    "to the moon": 0.9, "mooning": 0.8, "moon": 0.6, "diamond hands": 0.6, "hodl": 0.5,
    "hold the line": 0.5, "holding strong": 0.5, "buy the dip": 0.6, "btfd": 0.6, "btd": 0.5,
    "loading up": 0.7, "load up": 0.7, "loaded": 0.5, "adding more": 0.5, "calls": 0.5,
    "going long": 0.6, "long": 0.4, "bullish": 0.8, "breakout": 0.6, "short squeeze": 0.7,
    "gamma squeeze": 0.6, "squeeze": 0.5, "printing": 0.6, "tendies": 0.6, "lambo": 0.6,
    "send it": 0.6, "lfg": 0.6, "undervalued": 0.6, "strong buy": 0.8, "buy": 0.4,
    "bought": 0.4, "accumulate": 0.5, "accumulating": 0.5, "gap up": 0.5, "rally": 0.6,
    "ripping": 0.6, "runner": 0.5, "all time high": 0.4, "ath": 0.4, "green": 0.4,
    "surge": 0.6, "rising": 0.4, "profit": 0.4, "beat": 0.5, "upgrade": 0.6, "strong": 0.3,
    # bearish
    "puts": -0.5, "shorting": -0.6, "shorted": -0.5, "short": -0.5, "bearish": -0.8,
    "bag holder": -0.6, "bagholder": -0.6, "bagholding": -0.6, "bags": -0.4,
    "rug pull": -0.9, "rugpull": -0.9, "rugged": -0.8, "pump and dump": -0.9,
    "dumping": -0.6, "dumped": -0.5, "dump": -0.6, "tanking": -0.7, "tank": -0.6,
    "crashing": -0.8, "crash": -0.7, "dead cat bounce": -0.7, "falling knife": -0.6,
    "overvalued": -0.6, "bubble": -0.5, "selling": -0.4, "sold": -0.4, "sell": -0.4,
    "bleeding": -0.6, "red": -0.4, "falling": -0.4, "losses": -0.5, "painful loss": -0.6,
    "downgrade": -0.6, "miss": -0.5, "guh": -0.7, "gap down": -0.5, "avoid": -0.6,
    "stay away": -0.7, "scam": -0.8, "fraud": -0.8, "dilution": -0.6, "warning": -0.3,
    "fear": -0.3, "weak": -0.3, "going to zero": -0.8, "to zero": -0.7, "dead money": -0.5,
    "dead": -0.6, "trash": -0.6, "garbage": -0.6, "puke": -0.5, "cooked": -0.6,
}
NEGATORS = {"not", "no", "never", "dont", "don't", "isnt", "isn't", "wasnt", "wasn't",
            "wont", "won't", "cant", "can't", "without", "hardly", "aint", "ain't"}
STOPWORDS = {"the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "it", "this",
             "that", "i", "im", "i'm", "my", "me", "we", "you", "be", "at", "with", "will",
             "just", "so", "are", "was", "have", "has", "but", "if", "its", "it's", "they",
             "what", "all", "from", "up", "down", "here", "now", "today", "stock", "stocks",
             "more", "over", "added", "add", "top", "out", "get", "got", "go", "going", "like",
             "one", "about", "when", "back", "than", "then", "there", "their", "them", "see",
             "can", "do", "did", "does", "how", "who", "why", "been", "being", "would", "could",
             "should", "our", "us", "your", "as", "by", "not", "week", "day", "next", "last",
             "time", "still", "much", "very", "really", "also", "some", "any", "these", "those",
             "after", "before", "into", "off", "shares", "share", "price", "market", "company"}
TOKEN_RE = re.compile(r"[a-z][a-z'']+")


def build_lexicon(learned: dict = None) -> list:
    """[(compiled regex, weight, phrase)] longest phrase first."""
    lex = dict(SEED_SLANG)
    for word, weight in (learned or {}).items():
        lex.setdefault(word, weight)           # the seed list wins on conflicts
    items = sorted(lex.items(), key=lambda kv: (-len(kv[0].split()), -len(kv[0])))
    return [(re.compile(r"(?<![a-z'])" + re.escape(p) + r"(?![a-z'])"), w, p) for p, w in items]


def slang_score(text: str, lexicon: list) -> tuple:
    """(score in -1..1, matched phrases). Longest match wins; negation flips."""
    t = (text or "").lower()
    total, matched = 0.0, []
    for pattern, weight, phrase in lexicon:
        for m in pattern.finditer(t):
            before = re.findall(r"[a-z']+", t[max(0, m.start() - 25):m.start()])[-3:]
            w = -weight if any(b in NEGATORS for b in before) else weight
            total += w
            matched.append(phrase if w == weight else f"not {phrase}")
        t = pattern.sub(" " + "_" * 3 + " ", t)   # stop shorter phrases matching again
    return math.tanh(total), matched


_TICKER_WORDS = None


def ticker_words() -> set:
    """US ticker symbols, so they are never learned as 'slang' (DNA, MRNA, OKLO...)."""
    global _TICKER_WORDS
    if _TICKER_WORDS is None:
        session = SessionLocal()
        try:
            _TICKER_WORDS = set(load_us_symbols(session))
        except Exception:
            _TICKER_WORDS = set()
        finally:
            session.close()
    return _TICKER_WORDS


def tokens_for_learning(text: str, tickers: set = None) -> list:
    tickers = ticker_words() if tickers is None else tickers
    t = normalise_text(text)
    # drop ticker symbols BEFORE lowercasing (afterwards "DNA" and "dna" look the same)
    t = re.sub(r"\b([A-Z]{1,5})\b", lambda m: " " if m.group(1) in tickers else m.group(0), t)
    words = [w for w in TOKEN_RE.findall(t.lower()) if w not in STOPWORDS]
    bigrams = [f"{a} {b}" for a, b in zip(words, words[1:])]
    return list(set(words + bigrams))


def learn_slang(posts: pd.DataFrame, tickers: set = None) -> dict:
    """Word -> weight, learned from StockTwits posts tagged Bullish/Bearish
    (smoothed log-odds, balanced so the bigger class does not dominate)."""
    tagged = posts[posts["st_sentiment"].isin(["Bullish", "Bearish"])]
    bull = tagged[tagged["st_sentiment"] == "Bullish"]["body"]
    bear = tagged[tagged["st_sentiment"] == "Bearish"]["body"]
    if len(bull) < LEARN_MIN_CLASS_POSTS or len(bear) < LEARN_MIN_CLASS_POSTS:
        return {}
    tickers = ticker_words() if tickers is None else tickers
    # also drop the symbols the posts were collected under
    if "symbol" in posts.columns:
        tickers = tickers | set(posts["symbol"].dropna())
    cb, cr = Counter(), Counter()
    for text in bull:
        cb.update(tokens_for_learning(text, tickers))
    for text in bear:
        cr.update(tokens_for_learning(text, tickers))
    nb, nr = len(bull), len(bear)
    learned = {}
    for tok in set(cb) | set(cr):
        if cb[tok] + cr[tok] < LEARN_MIN_COUNT:
            continue
        # share of posts in each class containing the token, with +1 smoothing
        lo = math.log((cb[tok] + 1) / (nb + 2)) - math.log((cr[tok] + 1) / (nr + 2))
        if abs(lo) >= LEARN_MIN_LOGODDS:
            learned[tok] = round(LEARNED_SCALE * math.tanh(lo / 2), 3)
    return learned


def load_tagged_posts() -> pd.DataFrame:
    """All stored StockTwits posts (every run adds more training data)."""
    session = SessionLocal()
    try:
        rows = session.query(StockTwitsPost.id, StockTwitsPost.body, StockTwitsPost.sentiment,
                             StockTwitsPost.symbol).all()
    finally:
        session.close()
    return pd.DataFrame(rows, columns=["post_id", "body", "st_sentiment", "symbol"])


def _in_test_split(post_id: str) -> bool:
    return int(hashlib.md5(str(post_id).encode()).hexdigest(), 16) % 5 == 0


# ----------------------------------------------------------------------
# Step 3: ticker isolation
# ----------------------------------------------------------------------
CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[.!?;])\s+"
    r"|\s+(?:but|while|whereas|however|although|vs\.?|versus|unlike)\s+"
    r"|\s*,\s*(?=(?:and\s+)?(?:sell|selling|sold|short|shorting|buy|buying|bought|long|"
    r"load|loading|dump|dumping|avoid)\b)"
    r"|\s+(?=(?:sell|selling|sold|short|shorting|buy|buying|bought|long|dump|dumping|avoid)\s+\$?[A-Z]{1,5}\b)",
    re.I,
)
COMPARE_RE = r"\b{a}\b\s*(?:>|>>|over|better than|instead of|rather than|beats|crushes)\s*\b{b}\b"


def tickers_in(text: str, candidates) -> list:
    return [t for t in candidates if re.search(r"(?<![A-Za-z])\$?" + re.escape(t) + r"\b", text)]


def isolate_text(raw_text: str, target: str, post_tickers: list) -> tuple:
    """(text about `target` only, comparison adjustment). Single-ticker posts are unchanged."""
    if len(post_tickers) <= 1:
        return raw_text, 0.0

    clauses = [c for c in CLAUSE_SPLIT_RE.split(raw_text) if c and c.strip()]
    owned, current = [], None
    for clause in clauses:
        named = tickers_in(clause, post_tickers)
        if named:
            current = named
        if (current is None) or (target in (current or [])):
            owned.append(clause)            # no ticker named yet = applies to all
    text = " ".join(owned).strip() or raw_text

    adjust = 0.0
    plain = CASHTAG_RE.sub(r"\1", raw_text)
    for other in post_tickers:
        if other == target:
            continue
        if re.search(COMPARE_RE.format(a=re.escape(target), b=re.escape(other)), plain, re.I):
            adjust += COMPARISON_BONUS
        if re.search(COMPARE_RE.format(a=re.escape(other), b=re.escape(target)), plain, re.I):
            adjust -= COMPARISON_BONUS
    return text, adjust


# ----------------------------------------------------------------------
# Step 1: FinBERT
# ----------------------------------------------------------------------
def load_finbert():
    try:
        from transformers import pipeline
        clf = pipeline("text-classification", model=FINBERT_MODEL, top_k=None, truncation=True)
        print(f"  FinBERT loaded ({FINBERT_MODEL}).")
        return clf
    except Exception as err:
        print(f"  WARNING: FinBERT unavailable ({err}).\n"
              "  Falling back to the slang score only. Install with: pip install transformers torch")
        return None


def finbert_scores(clf, texts: list) -> dict:
    """text -> P(positive) - P(negative). Each unique text is scored once."""
    unique = list(dict.fromkeys(t for t in texts if t))
    if clf is None or not unique:
        return {}
    out = {}
    for start in range(0, len(unique), FINBERT_BATCH):
        batch = unique[start:start + FINBERT_BATCH]
        for text, result in zip(batch, clf(batch, batch_size=FINBERT_BATCH)):
            probs = {r["label"].lower(): r["score"] for r in result}
            out[text] = probs.get("positive", 0.0) - probs.get("negative", 0.0)
    return out


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def label(score: float) -> str:
    if score > BULLISH_CUTOFF:
        return "Bullish"
    if score < BEARISH_CUTOFF:
        return "Bearish"
    return "Neutral"


def prepare_mentions(mentions: pd.DataFrame) -> pd.DataFrame:
    """Isolate the text for each (post, ticker) and normalise it."""
    m = mentions[mentions["origin"] != "QUERY"].copy()   # QUERY articles may not be about the ticker
    post_tickers = m.groupby("post_id")["ticker"].apply(list).to_dict()

    iso_texts, adjusts, hints = [], [], []
    for r in m.itertuples(index=False):
        tickers = list(dict.fromkeys(post_tickers.get(r.post_id, []) + CASHTAG_RE.findall(r.text or "")))
        text, adjust = isolate_text(r.text or "", r.ticker, tickers)
        iso_texts.append(text)
        adjusts.append(adjust)
        hints.append(sarcasm_hint(r.text))
    m["post_ticker_count"] = [len(set(post_tickers.get(pid, []))) for pid in m["post_id"]]
    m["isolated_text"] = iso_texts
    m["clean_text"] = [normalise_text(t) for t in iso_texts]
    m["comparison_adjust"] = adjusts
    m["sarcasm_hint"] = hints
    return m


def score_mentions(m: pd.DataFrame, lexicon: list, fin: dict) -> pd.DataFrame:
    slang, matched = zip(*[slang_score(t, lexicon) for t in m["clean_text"]]) if len(m) else ((), ())
    m = m.copy()
    m["slang_score"] = np.round(slang, 3)
    m["slang_terms"] = [", ".join(x[:6]) for x in matched]
    m["finbert_score"] = [round(fin[t], 3) if t in fin else np.nan for t in m["clean_text"]]

    finals, models = [], []
    for r in m.itertuples(index=False):
        f = None if pd.isna(r.finbert_score) else r.finbert_score
        w = SOCIAL_WEIGHTS if r.source_type == "SOCIAL" else NEWS_WEIGHTS
        model = r.slang_score if f is None else w["finbert"] * f + w["slang"] * r.slang_score
        final = model
        # The user's tag describes the whole post, so it is only used when the post is
        # about one ticker ("$NVDA to the moon, $INTC is dead" tagged Bullish says nothing about INTC).
        if (r.source_type == "SOCIAL" and r.st_sentiment in ("Bullish", "Bearish")
                and r.post_ticker_count <= 1):
            tag = 1.0 if r.st_sentiment == "Bullish" else -1.0
            final = (1 - USER_TAG_WEIGHT) * model + USER_TAG_WEIGHT * tag
        final = float(np.clip(final + r.comparison_adjust, -1, 1))
        models.append(round(model, 3))
        finals.append(round(final, 3))
    m["model_score"] = models          # without the user's tag (used for the self-check)
    m["sentiment_score"] = finals
    m["sentiment"] = [label(s) for s in finals]
    return m


def aggregate(scored: pd.DataFrame) -> pd.DataFrame:
    """Per ticker. Social: one vote per account. News: one vote per article."""
    social = scored[scored["source_type"] == "SOCIAL"]
    news = scored[scored["source_type"] == "NEWS"]
    per_author = social.groupby(["ticker", "author"])["sentiment_score"].mean().reset_index()

    out = pd.DataFrame(index=sorted(scored["ticker"].unique()))
    out["social_sentiment"] = per_author.groupby("ticker")["sentiment_score"].mean()
    out["social_voters"] = per_author.groupby("ticker")["author"].nunique()
    out["news_sentiment"] = news.groupby("ticker")["sentiment_score"].mean()
    out["news_articles"] = news.groupby("ticker")["post_id"].nunique()

    votes = pd.concat([per_author[["ticker", "sentiment_score"]],
                       news[["ticker", "sentiment_score"]]])
    votes["lab"] = votes["sentiment_score"].map(label)
    shares = (votes.groupby("ticker")["lab"].value_counts(normalize=True).unstack(fill_value=0)
              .reindex(columns=["Bullish", "Bearish", "Neutral"], fill_value=0))
    for col in ("Bullish", "Bearish", "Neutral"):
        out[f"{col.lower()}_pct"] = (shares[col] * 100).round(0)
    out["sarcasm_hints"] = scored.groupby("ticker")["sarcasm_hint"].sum()

    def net(r):
        s, n = r["social_sentiment"], r["news_sentiment"]
        if pd.notna(s) and pd.notna(n):
            return SOCIAL_SHARE * s + (1 - SOCIAL_SHARE) * n
        return s if pd.notna(s) else n

    out["raw_sentiment"] = out.apply(net, axis=1)
    out[["social_voters", "news_articles", "sarcasm_hints"]] = (
        out[["social_voters", "news_articles", "sarcasm_hints"]].fillna(0).astype(int))
    # Few votes = low confidence: pull the score toward neutral so 5 posts cannot beat 100
    votes_n = out["social_voters"] + out["news_articles"]
    out["confidence"] = (votes_n / (votes_n + SHRINK_VOTES)).round(2)
    out["net_sentiment"] = out["raw_sentiment"] * out["confidence"]
    for col in ("social_sentiment", "news_sentiment", "raw_sentiment", "net_sentiment"):
        out[col] = out[col].round(3)
    out["sentiment_label"] = out["net_sentiment"].map(lambda v: label(v) if pd.notna(v) else "n/a")
    out.index.name = "ticker"
    return out.reset_index().sort_values("net_sentiment", ascending=False).reset_index(drop=True)


def self_check(clf, verbose=True) -> dict:
    """How often each method agrees with users' own Bullish/Bearish tags (held-out 20%)."""
    posts = load_tagged_posts()
    tagged = posts[posts["st_sentiment"].isin(["Bullish", "Bearish"])].copy()
    if len(tagged) < 30:
        if verbose:
            print(f"  Self-check skipped: only {len(tagged)} posts carry a Bullish/Bearish tag. "
                  "Collect again with the updated stocktwits_collector.py to get tagged posts.")
        return {}
    test_mask = tagged["post_id"].map(_in_test_split)
    train_learned = learn_slang(posts[~posts["post_id"].map(_in_test_split)])
    test = tagged[test_mask].copy()
    if len(test) < 10:
        return {}
    truth = np.where(test["st_sentiment"] == "Bullish", 1, -1)
    clean = [normalise_text(t) for t in test["body"]]

    seed_lex = build_lexicon()
    full_lex = build_lexicon(train_learned)
    results = {
        "seed slang": [slang_score(t, seed_lex)[0] for t in clean],
        "seed + learned slang": [slang_score(t, full_lex)[0] for t in clean],
    }
    fin = finbert_scores(clf, clean)
    if fin:
        results["FinBERT"] = [fin.get(t, 0.0) for t in clean]
        results["FinBERT + slang (used)"] = [
            SOCIAL_WEIGHTS["finbert"] * fin.get(t, 0.0) + SOCIAL_WEIGHTS["slang"] * s
            for t, s in zip(clean, results["seed + learned slang"])]

    report = {}
    bull_share = float((truth == 1).mean())
    n_bear = int((truth == -1).sum())
    if verbose:
        print(f"  Tagged test posts: {len(test)} ({bull_share:.0%} Bullish, {n_bear} Bearish).")
        print("  StockTwits users tag far more Bullish than Bearish, so plain accuracy is misleading")
        print("  (always guessing 'Bullish' scores high). Balanced accuracy averages the hit rate on")
        print("  Bullish and on Bearish posts: 50% = no better than guessing, 100% = perfect.")
        print(f"    {'method':24s} {'balanced':>9s} {'bullish hit':>12s} {'bearish hit':>12s} {'no opinion':>11s}")
        print(f"    {'always guess Bullish':24s} {'50%':>9s} {'100%':>12s} {'0%':>12s} {'0%':>11s}")
    for name, scores in results.items():
        sc = np.array(scores)
        pred = np.sign(sc)                         # 0 = no opinion, counts as a miss
        bull_hit = float((pred[truth == 1] == 1).mean()) if (truth == 1).any() else float("nan")
        bear_hit = float((pred[truth == -1] == -1).mean()) if (truth == -1).any() else float("nan")
        balanced = np.nanmean([bull_hit, bear_hit])
        none = float((pred == 0).mean())
        report[name] = {"balanced": balanced, "bullish_hit": bull_hit, "bearish_hit": bear_hit,
                        "no_opinion": none}
        if verbose:
            print(f"    {name:24s} {balanced:>9.0%} {bull_hit:>12.0%} {bear_hit:>12.0%} {none:>11.0%}")
    if verbose and n_bear < 20:
        print(f"  NOTE: only {n_bear} Bearish test posts, so the bearish hit rate is rough.")
    return report


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
LAST_SELF_CHECK = {}   # self-check result of the latest run, read by run_pipeline.py


def run_phase3(survivors: pd.DataFrame = None, mentions: pd.DataFrame = None, verbose: bool = True):
    """Returns (ticker_sentiment, scored_mentions)."""
    if survivors is None or mentions is None:
        survivors, mentions = run_phase2(verbose=verbose)
    if survivors is None or survivors.empty or mentions.empty:
        print("Phase 2 produced no tickers; nothing to score.")
        return pd.DataFrame(), pd.DataFrame()

    if verbose:
        print("\n=== Phase 3: Contextual NLP and sentiment scoring ===")

    # Step 2b: learn slang from every tagged StockTwits post stored so far
    all_posts = load_tagged_posts()
    learned = learn_slang(all_posts)
    if learned:
        pd.Series(learned, name="weight").sort_values().to_csv(LEARNED_PATH, header=True)
    lexicon = build_lexicon(learned)
    if verbose:
        n_tag = int(all_posts["st_sentiment"].isin(["Bullish", "Bearish"]).sum())
        print(f"[2] Slang dictionary: {len(SEED_SLANG)} seed phrases + {len(learned)} learned "
              f"from {n_tag} tagged StockTwits posts (saved to learned_slang.csv)")
        if learned:
            top = sorted(learned.items(), key=lambda kv: kv[1])
            print("    most bearish learned:", ", ".join(f"{k} ({v:+.2f})" for k, v in top[:6]))
            print("    most bullish learned:", ", ".join(f"{k} ({v:+.2f})" for k, v in top[-6:][::-1]))

    # Steps 4 + 3: normalise and isolate
    prepared = prepare_mentions(mentions)
    multi = int((prepared.groupby("post_id")["ticker"].transform("count") > 1).sum())
    if verbose:
        print(f"[4] Normalised {len(prepared)} mentions "
              f"({int(prepared['sarcasm_hint'].sum())} carry sarcasm hints for Phase 4)")
        print(f"[3] Ticker isolation: {multi} mentions come from multi-ticker posts; "
              f"{int((prepared['comparison_adjust'] != 0).sum())} have a comparison adjustment")

    # Step 1: FinBERT
    if verbose:
        print("[1] FinBERT scoring")
    clf = load_finbert()
    fin = finbert_scores(clf, prepared["clean_text"].tolist())

    scored = score_mentions(prepared, lexicon, fin)
    ticker_sentiment = aggregate(scored)
    result = survivors.merge(ticker_sentiment, on="ticker", how="left")
    result = result.sort_values("net_sentiment", ascending=False, na_position="last").reset_index(drop=True)

    if verbose:
        print("\n[check] Agreement with users' own Bullish/Bearish tags (held-out posts)")
    LAST_SELF_CHECK.clear()
    LAST_SELF_CHECK.update(self_check(clf, verbose) or {})
    LAST_SELF_CHECK["finbert_used"] = clf is not None
    if verbose:
        print(f"\n=== PHASE 3 COMPLETE: {len(result)} tickers scored ===")
        cols = ["ticker", "net_sentiment", "confidence", "raw_sentiment", "sentiment_label",
                "social_sentiment", "social_voters",
                "news_sentiment", "news_articles", "bullish_pct", "bearish_pct", "neutral_pct",
                "sarcasm_hints"]
        print(result[[c for c in cols if c in result.columns]].to_string(index=False))
    return result, scored


if __name__ == "__main__":
    run_phase3()
