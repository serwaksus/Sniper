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
    except OSError:
        return default
    _lock_file(fd, exclusive=False)
    fd_owned = False
    try:
        f = os.fdopen(fd, 'r')
        fd_owned = True
        try:
            return _normalize_keys(json.load(f))
        finally:
            f.close()
    except Exception:
        if not fd_owned:
            try:
                os.close(fd)
            except OSError:
                pass
        return default
    finally:
        try:
            _unlock_file(fd)
        except OSError:
            pass


def save_json(path, data):
    dir_name = os.path.dirname(path) or '.'
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(_strip_dict_keys_recursive(data), f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
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
        version = data.get("__version", 0)
    return data, version


def save_json_versioned(path, data, expected_version=None):
    try:
        if expected_version is not None:
            current_data = load_json(path, {})
            current_version = current_data.get("__version", 0) if isinstance(current_data, dict) else 0
            if current_version != expected_version:
                logger.warning(f"[UTILS] Version mismatch for {path}: expected={expected_version}, actual={current_version}")
                return False
        if isinstance(data, dict):
            data = dict(data)
            data["__version"] = expected_version + 1 if expected_version is not None else data.get("__version", 0) + 1
        save_json(path, data)
        return True
    except Exception as e:
        logger.error(f"[UTILS] save_json_versioned failed for {path}: {e}")
        return False




def check_and_write_pid(pid_file):
    try:
        fd = os.open(pid_file, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        logger.warning(f"[PID] Cannot open {pid_file}: {e}")
        return True
    try:
        _lock_file(fd, exclusive=False)
        try:
            content = os.read(fd, 32).strip()
            if content:
                old_pid = int(content)
                os.kill(old_pid, 0)
                print(f"[PID] Another instance running (PID {old_pid}), exiting")
                return False
        except (OSError, ValueError, ProcessLookupError):
            pass
        _unlock_file(fd)
        _lock_file(fd, exclusive=True)
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode())
        return True
    except OSError as e:
        logger.warning(f"[PID] Cannot write {pid_file}: {e}")
        return True
    finally:
        _unlock_file(fd)
        try:
            os.close(fd)
        except OSError:
            pass


def cleanup_pid_file(pid_file):
    try:
        os.unlink(pid_file)
    except OSError:
        pass


def sanitize_for_prompt(text):
    if not text:
        return ""
    cleaned = text.replace("{", "").replace("}", "").replace("\\", "")
    cleaned = "".join(c for c in cleaned if ord(c) < 0x10000)
    return cleaned[:500]


def load_env_file(path="/root/dotm-sniper/.env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                val = val.strip().strip('"').strip("'")
                os.environ.setdefault(key.strip(), val)
