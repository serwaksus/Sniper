#!/usr/bin/env python3
"""
Shared utilities for DOTM Sniper ecosystem.
Consolidates load_json/save_json, file locking, key normalization.
"""
import json
import os
import fcntl
import tempfile
import logging

logger = logging.getLogger(__name__)


def _lock_file(fd, exclusive=True):
    try:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, op)
    except (OSError, AttributeError):
        pass


def _unlock_file(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except (OSError, AttributeError):
        pass


def _normalize_keys(obj):
    if isinstance(obj, dict):
        return {k.strip(): _normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    return obj


def _strip_dict_keys_recursive(obj):
    if isinstance(obj, dict):
        return {k.strip() if isinstance(k, str) else k: _strip_dict_keys_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_dict_keys_recursive(item) for item in obj]
    if isinstance(obj, str):
        return obj.strip()
    return obj


def load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            _lock_file(fd, exclusive=False)
            with os.fdopen(fd, 'r') as f:
                return _normalize_keys(json.load(f))
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            return default
    except Exception:
        return default


def save_json(path, data):
    dir_name = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        _lock_file(fd, exclusive=True)
        with os.fdopen(fd, 'w') as f:
            json.dump(_strip_dict_keys_recursive(data), f, indent=2, default=str)
        lock_fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
        try:
            _lock_file(lock_fd, exclusive=True)
            os.replace(tmp_path, path)
        finally:
            _unlock_file(lock_fd)
            try:
                os.close(lock_fd)
            except OSError:
                pass
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_json_versioned(path, default):
    data = load_json(path, default)
    version = 0
    if isinstance(data, dict):
        version = data.pop("__version", 0)
    return data, version


def save_json_versioned(path, data, expected_version=None):
    if expected_version is not None:
        current, _ = load_json_versioned(path, {})
        current_version = current.get("__version", 0) if isinstance(current, dict) else 0
        if current_version != expected_version:
            logger.warning(f"[UTILS] Version mismatch for {path}: expected={expected_version}, actual={current_version}, retrying merge")
            return False
    if isinstance(data, dict):
        data = dict(data)
        data["__version"] = expected_version + 1 if expected_version is not None else data.get("__version", 0) + 1
    save_json(path, data)
    return True


def sanitize_for_prompt(text):
    if not text:
        return ""
    cleaned = text.replace("{", "").replace("}", "").replace("\\", "")
    cleaned = "".join(c for c in cleaned if ord(c) < 0x10000)
    return cleaned[:500]
