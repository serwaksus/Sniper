from __future__ import annotations
import logging
from datetime import datetime

from utils import load_json
from schema import (
    CACHE_METACULUS, CACHE_NEWS, CACHE_TIMESTAMP, CACHE_LAST_UPDATE,
)
from config import CACHE_FILE

logger = logging.getLogger(__name__)


def _check_news_cache_freshness(cluster_key: str, slug: str | None = None) -> bool:
    news_ttl = 2 * 3600
    cache = load_json(CACHE_FILE, {CACHE_METACULUS: {}, CACHE_NEWS: {}, CACHE_LAST_UPDATE: None})
    news_section = cache.get(CACHE_NEWS, {})
    cache_key = f"{cluster_key}:{slug}" if slug else cluster_key
    entry = news_section.get(cache_key)
    if isinstance(entry, dict) and entry.get(CACHE_TIMESTAMP):
        try:
            cached_time = datetime.fromisoformat(entry[CACHE_TIMESTAMP])
            age_seconds = (datetime.now() - cached_time).total_seconds()
            if age_seconds < news_ttl:
                logger.info(f"[CACHE-FRESH] news slug '{cache_key}' age={age_seconds/3600:.1f}h < 2h, using cache")
                return True
        except (ValueError, TypeError):
            pass
    return False
