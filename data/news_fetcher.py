import os
os.environ["TRANSFORMERS_VERBOSITY"]          = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"]          = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"]   = "1" 
import httpx
import logging
from cache.redis_client import cache
from config import settings
from transformers import pipeline


logger = logging.getLogger(__name__)

# ── Load FinBERT safely ──────────────────────────────────
sentiment_model = None
_model_loaded = False      # ADD THIS FLAG

def get_sentiment_model():
    global sentiment_model

    if sentiment_model is not None:
        return sentiment_model

    try:
        os.environ["TRANSFORMERS_VERBOSITY"] = "error"

        logger.info("Loading FinBERT...")

        sentiment_model = pipeline(
            "text-classification",   # NOT sentiment-analysis (more stable)
            model="ProsusAI/finbert",
            device=-1
        )

        logger.info("FinBERT loaded successfully")

    except Exception as e:
        logger.warning(f"Model load failed: {e}")
        sentiment_model = None

    return sentiment_model


# ── Ticker → Company name map ────────────────────────────
TICKER_TO_COMPANY = {
    "RELIANCE":    "Reliance Industries",
    "TCS":         "Tata Consultancy Services",
    "INFY":        "Infosys",
    "HDFCBANK":    "HDFC Bank",
    "ICICIBANK":   "ICICI Bank",
    "SBIN":        "State Bank of India",
    "BHARTIARTL":  "Bharti Airtel",
    "WIPRO":       "Wipro",
    "LT":          "Larsen Toubro",
    "AXISBANK":    "Axis Bank",
    "KOTAKBANK":   "Kotak Mahindra Bank",
    "HINDUNILVR":  "Hindustan Unilever",
    "ITC":         "ITC Limited",
    "SUNPHARMA":   "Sun Pharma",
    "MARUTI":      "Maruti Suzuki",
    "BAJFINANCE":  "Bajaj Finance",
    "TITAN":       "Titan Company",
    "ASIANPAINT":  "Asian Paints",
    "NESTLEIND":   "Nestle India",
    "PNB":         "Punjab National Bank",
}


async def get_stock_news(ticker: str, max_articles: int = 10) -> dict:
    clean = ticker.replace(".NS", "").replace(".BO", "").upper()
    cache_key = f"news:{clean}"

    # TEMPORARILY SKIP CACHE — for debugging only
    # cached = await cache.get(cache_key)
    # if cached:
    #     return cached

    logger.info(f"=== NEWS DEBUG START for {clean} ===")
    logger.info(f"sentiment_model is: {sentiment_model}")
    logger.info(f"sentiment_model type: {type(sentiment_model)}")

    articles = await _fetch_news(clean, max_articles)
    logger.info(f"Raw articles fetched: {len(articles)}")

    if articles:
        logger.info(f"First article title: {articles[0].get('title')}")

    if not articles:
        result = {
            "overall_sentiment": "NEUTRAL",
            "sentiment_score": 0,
            "breakdown": {"positive": 0, "negative": 0, "neutral": 0},
            "articles": [],
            "note": "No news found"
        }
        return result

    analyzed = _analyze_sentiment(articles)
    logger.info(f"=== NEWS DEBUG END: {analyzed['overall_sentiment']} ===")

    await cache.set(cache_key, analyzed, settings.NEWS_TTL)
    return analyzed

async def _fetch_news(ticker: str, max_articles: int) -> list:
    company = TICKER_TO_COMPANY.get(ticker, ticker)

    # Use NewsAPI /v2/top-headlines for India first,
    # then fall back to everything with strict query
    queries = [
        f'"{company}" stock India',      # exact company name
        f'"{company}" NSE shares',
        f'{company} BSE Sensex Nifty',
    ]

    all_articles = []

    async with httpx.AsyncClient(timeout=15) as client:
        for query in queries:
            params = {
                "q":        query,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": max_articles,
                "apiKey":   settings.NEWS_API_KEY,
                # Remove domain filter — free tier ignores it anyway
            }
            try:
                response = await client.get(
                    "https://newsapi.org/v2/everything",
                    params=params
                )
                data = response.json()

                if data.get("status") == "ok":
                    articles = data.get("articles", [])

                    # ── FILTER: only keep relevant articles ──
                    relevant = _filter_relevant(articles, company, ticker)
                    logger.info(
                        f"Query '{query}': {len(articles)} total, "
                        f"{len(relevant)} relevant"
                    )
                    all_articles.extend(relevant)
                else:
                    logger.warning(f"NewsAPI: {data.get('message')}")

            except Exception as e:
                logger.warning(f"News fetch error: {e}")

            if len(all_articles) >= max_articles:
                break

    # Deduplicate
    seen, unique = set(), []
    for a in all_articles:
        title = a.get("title", "")
        if title and title not in seen and title != "[Removed]":
            seen.add(title)
            unique.append(a)

    logger.info(f"Final unique relevant articles for {ticker}: {len(unique)}")
    return unique[:max_articles]


def _filter_relevant(articles: list, company: str, ticker: str) -> list:
    # Build keyword list — more lenient than before
    company_words = company.lower().split()  # ["reliance", "industries"]
    
    keywords = [
        company.lower(),                     # "reliance industries"
        ticker.lower(),                      # "reliance"
        company_words[0],                    # "reliance" (first word only)
    ]

    # Remove generic words that would match everything
    stop_words = {"the", "of", "and", "limited", "ltd", "inc", "corp"}
    keywords = [k for k in keywords if k not in stop_words]

    logger.info(f"Filtering with keywords: {keywords}")

    relevant = []
    for article in articles:
        title       = (article.get("title") or "").lower()
        description = (article.get("description") or "").lower()
        content     = title + " " + description

        matched = [kw for kw in keywords if kw in content]
        if matched:
            logger.info(f"MATCH ({matched}): {title[:60]}")
            relevant.append(article)
        else:
            logger.info(f"SKIP: {title[:60]}")

    return relevant

def _analyze_sentiment(articles: list) -> dict:
    results = []
    model = get_sentiment_model()

    logger.info(f"Analyzing {len(articles)} articles | FinBERT={model is not None}")

    for article in articles:
        title = article.get("title") or ""
        desc  = article.get("description") or ""

        if not title or title == "[Removed]":
            continue

        text = f"{title}. {desc}"[:512]

        try:
            if model is not None:
                raw = model(text)
                logger.info(f"Raw FinBERT output: {raw}")  # ADD THIS temporarily

                # ── Handle different output formats ──────────
                scores = _parse_finbert_output(raw)

                if not scores:
                    # FinBERT failed to parse — use keyword fallback
                    sent = _rule_based_sentiment(text)
                    logger.info(f"Fallback to rule-based: {sent}")
                else:
                    top = max(scores, key=scores.get)
                    logger.info(f"'{title[:50]}' → {top.upper()}")

                    results.append({
                        "title":        title,
                        "source":       (article.get("source") or {}).get("name") or "Unknown",
                        "url":          article.get("url") or "",
                        "published_at": article.get("publishedAt") or "",
                        "sentiment":    top.upper(),
                        "confidence": {
                            "positive": round(scores.get("positive", 0) * 100, 1),
                            "negative": round(scores.get("negative", 0) * 100, 1),
                            "neutral":  round(scores.get("neutral",  0) * 100, 1),
                        }
                    })
                    continue  # skip rule-based below

            # Rule-based fallback (when model is None or parse failed)
            sent = _rule_based_sentiment(text)
            logger.info(f"Rule-based: '{title[:50]}' → {sent}")
            results.append({
                "title":        title,
                "source":       (article.get("source") or {}).get("name") or "Unknown",
                "url":          article.get("url") or "",
                "published_at": article.get("publishedAt") or "",
                "sentiment":    sent,
                "confidence": {
                    "positive": 100.0 if sent == "POSITIVE" else 0.0,
                    "negative": 100.0 if sent == "NEGATIVE" else 0.0,
                    "neutral":  100.0 if sent == "NEUTRAL"  else 0.0,
                }
            })

        except Exception as e:
            logger.warning(f"Sentiment error on '{title[:40]}': {e}")
            # Don't skip — use rule-based fallback instead
            try:
                sent = _rule_based_sentiment(text)
                results.append({
                    "title":        title,
                    "source":       (article.get("source") or {}).get("name") or "Unknown",
                    "url":          article.get("url") or "",
                    "published_at": article.get("publishedAt") or "",
                    "sentiment":    sent,
                    "confidence": {
                        "positive": 100.0 if sent == "POSITIVE" else 0.0,
                        "negative": 100.0 if sent == "NEGATIVE" else 0.0,
                        "neutral":  100.0 if sent == "NEUTRAL"  else 0.0,
                    }
                })
            except:
                continue

    logger.info(f"Total analyzed successfully: {len(results)}")
    return _aggregate_sentiment(results)


def _parse_finbert_output(raw) -> dict:
    """
    Handle all possible FinBERT output formats safely.
    Returns dict like {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
    or empty dict if parsing fails.
    """
    try:
        # Format 1: [[{'label': 'positive', 'score': 0.8}, ...]]
        # This is return_all_scores=True format
        if isinstance(raw, list) and len(raw) > 0:
            inner = raw[0]

            # Format 1a: list of dicts with 'label' and 'score'
            if isinstance(inner, list):
                scores = {}
                for item in inner:
                    if isinstance(item, dict) and 'label' in item and 'score' in item:
                        scores[item['label'].lower()] = item['score']
                if scores:
                    return scores

            # Format 1b: single dict with 'label' and 'score'
            if isinstance(inner, dict):
                if 'label' in inner and 'score' in inner:
                    # Only one label returned — not all scores
                    label = inner['label'].lower()
                    score = inner['score']
                    others = (1 - score) / 2
                    result = {'positive': others, 'negative': others, 'neutral': others}
                    result[label] = score
                    return result

                # Format 1c: dict with label names as keys directly
                # {'positive': 0.8, 'negative': 0.1, 'neutral': 0.1}
                if any(k in inner for k in ['positive', 'negative', 'neutral']):
                    return {k: inner[k] for k in ['positive', 'negative', 'neutral'] if k in inner}

        # Format 2: direct dict
        if isinstance(raw, dict):
            if any(k in raw for k in ['positive', 'negative', 'neutral']):
                return raw

        logger.warning(f"Unknown FinBERT output format: {type(raw)} = {raw}")
        return {}

    except Exception as e:
        logger.warning(f"FinBERT parse error: {e}")
        return {}
    
def _rule_based_sentiment(text: str) -> str:
    t = text.lower()
    pos = ["surge", "gain", "rise", "profit", "growth", "rally", "bull",
           "strong", "beat", "record", "buy", "upgrade", "boost", "jump",
           "soar", "climb", "high", "positive", "outperform", "up"]
    neg = ["fall", "drop", "loss", "down", "decline", "crash", "bear",
           "weak", "miss", "cut", "downgrade", "sell", "risk", "concern",
           "slip", "plunge", "tumble", "negative", "low", "worry"]

    p = sum(1 for w in pos if w in t)
    n = sum(1 for w in neg if w in t)

    if p > n: return "POSITIVE"
    if n > p: return "NEGATIVE"
    return "NEUTRAL"


def _aggregate_sentiment(results: list) -> dict:
    if not results:
        return {
            "overall_sentiment": "NEUTRAL",
            "sentiment_score":   0,
            "breakdown":         {"positive": 0, "negative": 0, "neutral": 0},
            "articles":          []
        }

    pos  = sum(1 for r in results if r["sentiment"] == "POSITIVE")
    neg  = sum(1 for r in results if r["sentiment"] == "NEGATIVE")
    neut = sum(1 for r in results if r["sentiment"] == "NEUTRAL")
    total = len(results)

    score = ((pos - neg) / total) * 100

    if score > 20:   overall = "BULLISH"
    elif score < -20: overall = "BEARISH"
    else:             overall = "NEUTRAL"

    return {
        "overall_sentiment": overall,
        "sentiment_score":   round(score, 1),
        "breakdown":         {"positive": pos, "negative": neg, "neutral": neut},
        "articles":          results
    }