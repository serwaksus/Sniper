"""Tests for news_scanner.py — Tavily API, fallback, cache, keywords, sanity check."""
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import news_scanner as ns


class TestFetchRecentNews:
    @patch("news_scanner.requests.post")
    def test_tavily_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {
            "results": [
                {"title": "Breaking news about AI", "url": "https://example.com/1"},
                {"title": "More AI developments", "url": "https://example.com/2"},
            ]
        }
        mock_post.return_value = mock_resp
        os.environ["TAVILY_API_KEY"] = "test-key"
        result = ns.fetch_recent_news(["AI", "safety"])
        assert result["found"] is True
        assert len(result["headlines"]) == 2

    @patch("news_scanner.requests.post", side_effect=Exception("timeout"))
    @patch("news_scanner._fetch_ddg_news_fallback")
    def test_tavily_failure_uses_fallback(self, mock_fallback, mock_post):
        os.environ["TAVILY_API_KEY"] = "test-key"
        mock_fallback.return_value = {"headlines": ["fallback headline"], "sources": [], "query": "", "found": True}
        result = ns.fetch_recent_news(["AI"])
        assert result["found"] is True
        mock_fallback.assert_called()

    @patch("news_scanner._fetch_ddg_news_fallback")
    def test_no_api_key_uses_fallback(self, mock_fallback):
        os.environ["TAVILY_API_KEY"] = ""
        os.environ["TAVILY_API_KEY_BACKUP"] = ""
        mock_fallback.return_value = {"headlines": [], "sources": [], "query": "", "found": False}
        ns.fetch_recent_news(["AI"])
        mock_fallback.assert_called()

    @patch("news_scanner._fetch_ddg_news_fallback")
    def test_placeholder_key_uses_fallback(self, mock_fallback):
        os.environ["TAVILY_API_KEY"] = "your_tavily_key_here"
        os.environ["TAVILY_API_KEY_BACKUP"] = ""
        mock_fallback.return_value = {"headlines": [], "sources": [], "query": "", "found": False}
        ns.fetch_recent_news(["test"])
        mock_fallback.assert_called()


class TestDDGFallback:
    @patch("duckduckgo_search.DDGS")
    def test_parses_html_headlines(self, mock_ddgs_cls):
        mock_ddgs = MagicMock()
        mock_ddgs.__enter__ = MagicMock(return_value=mock_ddgs)
        mock_ddgs.__exit__ = MagicMock(return_value=False)
        mock_ddgs.news.return_value = [
            {"title": "Breaking: Something important happened today", "url": "http://x"}
        ]
        mock_ddgs_cls.return_value = mock_ddgs
        result = ns._fetch_ddg_news_fallback(["AI"], 5)
        assert len(result["headlines"]) >= 1

    @patch("duckduckgo_search.DDGS", side_effect=Exception("err"))
    def test_exception_returns_empty(self, mock_ddgs_cls):
        result = ns._fetch_ddg_news_fallback(["AI"], 5)
        assert result["headlines"] == []
        assert result["found"] is False


class TestExtractKeywords:
    def test_basic_extraction(self):
        result = ns.extract_keywords("Will AI be regulated by the government?")
        assert "regulated" in result
        assert "government" in result

    def test_removes_stop_words(self):
        result = ns.extract_keywords("Will the president be elected?")
        assert "will" not in result
        assert "the" not in result

    def test_max_8_keywords(self):
        result = ns.extract_keywords("This is a very long question with many words to extract from the text")
        assert len(result) <= 8

    def test_short_words_excluded(self):
        result = ns.extract_keywords("Is it ok?")
        assert len(result) == 0


class TestNewsSanityCheck:
    def test_no_headlines_passes(self):
        passed, _reason = ns.news_sanity_check("Q?", [])
        assert passed is True

    @patch("news_scanner.requests.post")
    def test_block_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"choices": [{"message": {"content": "BLOCK"}}]}
        mock_post.return_value = mock_resp
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        passed, _reason = ns.news_sanity_check("Will X happen?", ["Headline 1"])
        assert passed is False

    @patch("news_scanner.requests.post")
    def test_pass_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.json.return_value = {"choices": [{"message": {"content": "PASS"}}]}
        mock_post.return_value = mock_resp
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        passed, _reason = ns.news_sanity_check("Will X happen?", ["Headline 1"])
        assert passed is True

    def test_no_api_key_passes(self):
        os.environ["DEEPSEEK_API_KEY"] = ""
        passed, _reason = ns.news_sanity_check("Q?", ["headline"])
        assert passed is True

    @patch("news_scanner.requests.post", side_effect=Exception("err"))
    def test_exception_passes(self, mock_post):
        os.environ["DEEPSEEK_API_KEY"] = "test-key"
        passed, _reason = ns.news_sanity_check("Q?", ["headline"])
        assert passed is True


class TestCheckMarketNews:
    @patch("news_scanner.news_sanity_check", return_value=(True, "ok"))
    @patch("news_scanner.fetch_recent_news")
    @patch("news_scanner.extract_keywords", return_value=["AI", "safety"])
    @patch("news_scanner.save_json")
    @patch("news_scanner.load_json", return_value={})
    def test_passes_with_news(self, mock_load, mock_save, mock_kw, mock_fetch, mock_sanity):
        mock_fetch.return_value = {"headlines": ["AI news"], "sources": [], "query": "", "found": True}
        market = {"question": "Will AI be regulated?", "clusters": ["ai_tech"]}
        passed, _reason = ns.check_market_news(market)
        assert passed is True

    @patch("news_scanner.extract_keywords", return_value=[])
    def test_no_keywords_passes(self, mock_kw):
        market = {"question": ""}
        passed, _reason = ns.check_market_news(market)
        assert passed is True

    @patch("news_scanner.fetch_recent_news", return_value={"headlines": [], "sources": [], "query": "", "found": False})
    @patch("news_scanner.extract_keywords", return_value=["test"])
    def test_no_news_passes(self, mock_kw, mock_fetch):
        market = {"question": "Will X happen?", "clusters": ["other"]}
        passed, _reason = ns.check_market_news(market)
        assert passed is True


class TestCircuitBreaker:
    """Tavily circuit breaker — stops wasting VPN connections after quota errors."""

    def test_breaker_starts_closed(self):
        ns._tavily_trip_time = 0.0
        assert ns._tavily_circuit_open() is False

    @patch("news_scanner.requests.post")
    def test_429_trips_breaker(self, mock_post):
        ns._tavily_trip_time = 0.0
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 429
        mock_resp.text = "quota exceeded"
        mock_post.return_value = mock_resp
        os.environ["TAVILY_API_KEY"] = "test-key"
        ns._tavily_search("test-key", "query", 5, 30)
        assert ns._tavily_circuit_open() is True

    @patch("news_scanner.requests.post")
    def test_401_trips_breaker(self, mock_post):
        ns._tavily_trip_time = 0.0
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 401
        mock_resp.text = "unauthorized"
        mock_post.return_value = mock_resp
        ns._tavily_search("test-key", "query", 5, 30)
        assert ns._tavily_circuit_open() is True

    @patch("news_scanner._fetch_ddg_news_fallback")
    def test_open_breaker_skips_tavily(self, mock_fallback):
        """When breaker is open, fetch_recent_news goes straight to DDG."""
        import time as _time
        ns._tavily_trip_time = _time.monotonic()  # Tripped just now
        mock_fallback.return_value = {"headlines": [], "sources": [], "query": "", "found": False}
        os.environ["TAVILY_API_KEY"] = "test-key"
        with patch("news_scanner.requests.post") as mock_post:
            ns.fetch_recent_news(["test"])
            mock_post.assert_not_called()  # No Tavily connection at all
        mock_fallback.assert_called_once()
        ns._tavily_trip_time = 0.0  # Reset

    @patch("news_scanner.requests.post")
    @patch("news_scanner._fetch_ddg_news_fallback")
    def test_500_does_not_trip_breaker(self, mock_fallback, mock_post):
        """Server errors (500) should NOT trip the breaker — only quota/auth."""
        ns._tavily_trip_time = 0.0
        mock_resp = MagicMock()
        mock_resp.ok = False
        mock_resp.status_code = 500
        mock_resp.text = "server error"
        mock_post.return_value = mock_resp
        mock_fallback.return_value = {"headlines": [], "sources": [], "query": "", "found": False}
        os.environ["TAVILY_API_KEY"] = "test-key"
        ns.fetch_recent_news(["test"])
        assert ns._tavily_circuit_open() is False  # Still closed
