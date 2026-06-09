"""Tests for tg_sender.py — queue operations, rate limiting, credential loading, flush."""
import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import tg_sender as tg


class TestGetCredentials:
    @patch.dict(os.environ, {"TG_BOT_TOKEN": "tok123", "TG_CHAT_ID": "chat456"}, clear=False)
    def test_from_environment(self):
        token, chat_id = tg._get_credentials()
        assert token == "tok123"
        assert chat_id == "chat456"

    def test_missing_credentials(self, tmp_path):
        env_file = tmp_path / ".env"
        env_file.write_text("")
        orig_env = tg.ENV_FILE
        tg.ENV_FILE = str(env_file)
        with patch.dict(os.environ, {"TG_BOT_TOKEN": "", "TG_CHAT_ID": ""}, clear=False):
            token, chat_id = tg._get_credentials()
            assert token == ""
            assert chat_id == ""
        tg.ENV_FILE = orig_env


class TestSendOnce:
    @patch("tg_sender.requests.post")
    def test_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.ok = True
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        result = tg._send_once("tok", "chat", "hello")
        assert result is True

    @patch("tg_sender.requests.post")
    def test_rate_limited_429(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_resp.ok = False
        mock_resp.headers = {"Retry-After": "1"}
        mock_post.return_value = mock_resp
        result = tg._send_once("tok", "chat", "hello")
        assert result is False

    @patch("tg_sender.requests.post", side_effect=Exception("timeout"))
    def test_exception_returns_false(self, mock_post):
        result = tg._send_once("tok", "chat", "hello")
        assert result is False


class TestSendTelegram:
    @patch("tg_sender._send_once", return_value=True)
    @patch("tg_sender._get_credentials", return_value=("tok", "chat"))
    def test_success(self, mock_cred, mock_send):
        result = tg.send_telegram("hello", max_retries=1)
        assert result is True

    @patch("tg_sender._get_credentials", return_value=("", ""))
    def test_no_credentials_returns_false(self, mock_cred):
        result = tg.send_telegram("hello")
        assert result is False

    @patch("tg_sender._enqueue")
    @patch("tg_sender._send_once", return_value=False)
    @patch("tg_sender._get_credentials", return_value=("tok", "chat"))
    def test_queues_on_failure(self, mock_cred, mock_send, mock_enqueue):
        result = tg.send_telegram("hello", max_retries=1)
        assert result is False
        assert mock_enqueue.called

    @patch("tg_sender._send_once", return_value=False)
    @patch("tg_sender._get_credentials", return_value=("tok", "chat"))
    def test_no_queue_when_disabled(self, mock_cred, mock_send):
        result = tg.send_telegram("hello", max_retries=1, queue_on_fail=False)
        assert result is False


class TestEnqueue:
    @patch("tg_sender.save_json")
    @patch("tg_sender.load_json", return_value=[])
    @patch("tg_sender.file_lock")
    def test_push_message(self, mock_lock, mock_load, mock_save):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        tg._enqueue("test message")
        assert mock_save.called
        saved = mock_save.call_args[0][1]
        assert len(saved) == 1
        assert saved[0]["message"] == "test message"

    @patch("tg_sender.save_json")
    @patch("tg_sender.load_json")
    @patch("tg_sender.file_lock")
    def test_fifo_order(self, mock_lock, mock_load, mock_save):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        existing = [
            {"message": "old", "queued_at": (datetime.now() - timedelta(hours=1)).isoformat(), "attempts": 0},
        ]
        mock_load.return_value = existing
        tg._enqueue("new message")
        saved = mock_save.call_args[0][1]
        assert len(saved) == 2
        assert saved[0]["message"] == "old"
        assert saved[1]["message"] == "new message"

    @patch("tg_sender.save_json")
    @patch("tg_sender.load_json")
    @patch("tg_sender.file_lock")
    def test_max_queue_size(self, mock_lock, mock_load, mock_save):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        old_ts = (datetime.now() - timedelta(hours=1)).isoformat()
        existing = [{"message": f"m{i}", "queued_at": old_ts, "attempts": 0} for i in range(100)]
        mock_load.return_value = existing
        tg._enqueue("overflow")
        saved = mock_save.call_args[0][1]
        assert len(saved) <= 100

    @patch("tg_sender.save_json")
    @patch("tg_sender.load_json")
    @patch("tg_sender.file_lock")
    def test_old_entries_pruned(self, mock_lock, mock_load, mock_save):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        old_ts = (datetime.now() - timedelta(hours=72)).isoformat()
        existing = [{"message": "stale", "queued_at": old_ts, "attempts": 0}]
        mock_load.return_value = existing
        tg._enqueue("fresh")
        saved = mock_save.call_args[0][1]
        assert len(saved) == 1
        assert saved[0]["message"] == "fresh"


class TestFlushQueue:
    @patch("tg_sender._send_once", return_value=True)
    @patch("tg_sender._get_credentials", return_value=("tok", "chat"))
    @patch("tg_sender.save_json")
    @patch("tg_sender.load_json")
    @patch("tg_sender.file_lock")
    def test_flush_delivers(self, mock_lock, mock_load, mock_save, mock_cred, mock_send):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        mock_load.return_value = [
            {"message": "queued msg", "queued_at": datetime.now().isoformat(), "attempts": 0},
        ]
        sent = tg.flush_queue()
        assert sent == 1

    @patch("tg_sender.load_json", return_value=[])
    @patch("tg_sender.file_lock")
    def test_empty_queue_returns_zero(self, mock_lock, mock_load):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        sent = tg.flush_queue()
        assert sent == 0

    @patch("tg_sender._get_credentials", return_value=("", ""))
    @patch("tg_sender.load_json", return_value=[{"message": "x", "queued_at": datetime.now().isoformat(), "attempts": 0}])
    @patch("tg_sender.file_lock")
    def test_no_credentials_returns_zero(self, mock_lock, mock_load, mock_cred):
        mock_lock.return_value.__enter__ = MagicMock()
        mock_lock.return_value.__exit__ = MagicMock(return_value=False)
        sent = tg.flush_queue()
        assert sent == 0


class TestGetQueueSize:
    @patch("tg_sender.load_json", return_value=[{"message": "a"}, {"message": "b"}])
    def test_returns_count(self, mock_load):
        assert tg.get_queue_size() == 2

    @patch("tg_sender.load_json", return_value=[])
    def test_empty_returns_zero(self, mock_load):
        assert tg.get_queue_size() == 0
