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
import contextlib
import re
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def file_lock(lock_path, timeout=30):
    """Acquire an exclusive cross-process file lock via fcntl.flock."""
    import time as _time
    with open(lock_path, 'w') as lock_fd:
        deadline = _time.time() + timeout
        while True:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if _time.time() >= deadline:
                    raise TimeoutError(f"Could not acquire lock on {lock_path} within {timeout}s") from None
                _time.sleep(0.1)
        try:
            yield lock_fd
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)


def locked_update_json(json_path, update_fn, default=None, lock_dir="/tmp"):
    """Atomically read-modify-write a JSON file with cross-process locking."""
    lock_path = os.path.join(lock_dir, os.path.basename(json_path) + ".lock")
    with file_lock(lock_path):
        data = load_json(json_path, default if default is not None else {})
        data = update_fn(data)
        if data is not None:
            save_json(json_path, data)
        return data


def _lock_file(fd, exclusive=True):
    try:
        op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(fd, op)
    except (OSError, AttributeError):
        pass


def _unlock_file(fd):
    with contextlib.suppress(OSError, AttributeError):
        fcntl.flock(fd, fcntl.LOCK_UN)


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
        fd = -1
        fd_owned = True
        try:
            return _normalize_keys(json.load(f))
        finally:
            f.close()
    except Exception as e:
        logger.debug(f"[utils] {type(e).__name__}: {e}")
        if not fd_owned:
            with contextlib.suppress(OSError):
                os.close(fd)
        return default
    finally:
        if fd >= 0:
            with contextlib.suppress(OSError):
                _unlock_file(fd)


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
            with contextlib.suppress(OSError):
                os.close(lock_fd)
    except Exception as e:
        logger.debug(f"[utils] {type(e).__name__}: {e}")
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def load_json_versioned(path, default):
    data = load_json(path, default)
    version = 0
    if isinstance(data, dict):
        version = data.get("__version", 0)
    return data, version


def save_json_versioned(path, data, expected_version=None):
    try:
        dir_name = os.path.dirname(path) or '.'
        lock_fd = os.open(path, os.O_RDONLY | os.O_CREAT, 0o644)
        try:
            _lock_file(lock_fd, exclusive=True)
            if expected_version is not None:
                try:
                    with open(path) as f:
                        current_data = json.load(f)
                except Exception as e:
                    logger.debug(f"[utils] {type(e).__name__}: {e}")
                    current_data = {}
                current_version = current_data.get("__version", 0) if isinstance(current_data, dict) else 0
                if current_version != expected_version:
                    logger.warning(f"[UTILS] Version mismatch for {path}: expected={expected_version}, actual={current_version}")
                    return False
            if isinstance(data, dict):
                data = dict(data)
                data["__version"] = expected_version + 1 if expected_version is not None else data.get("__version", 0) + 1
            fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(_strip_dict_keys_recursive(data), f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, path)
            except Exception as e:
                logger.debug(f"[utils] {type(e).__name__}: {e}")
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
                raise
            return True
        finally:
            _unlock_file(lock_fd)
            with contextlib.suppress(OSError):
                os.close(lock_fd)
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
        _lock_file(fd, exclusive=True)
        try:
            content = os.read(fd, 32).strip()
            if content:
                old_pid = int(content)
                os.kill(old_pid, 0)
                print(f"[PID] Another instance running (PID {old_pid}), exiting")
                return False
        except (OSError, ValueError, ProcessLookupError):
            pass
        os.ftruncate(fd, 0)
        os.lseek(fd, 0, os.SEEK_SET)
        os.write(fd, str(os.getpid()).encode())
        return True
    except OSError as e:
        logger.warning(f"[PID] Cannot write {pid_file}: {e}")
        return True
    finally:
        _unlock_file(fd)
        with contextlib.suppress(OSError):
            os.close(fd)


def cleanup_pid_file(pid_file):
    with contextlib.suppress(OSError):
        os.unlink(pid_file)


def sanitize_for_prompt(text):
    if not text:
        return ""
    cleaned = text.replace("{", "").replace("}", "").replace("\\", "")
    cleaned = "".join(c for c in cleaned if ord(c) < 0x10000)
    return cleaned[:500]


def load_env_file(path=None):
    if path is None:
        from config import ENV_FILE
        path = ENV_FILE
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                val = val.strip().strip('"').strip("'")
                os.environ[key.strip()] = val


MAX_LOG_BYTES = 50 * 1024 * 1024

def rotate_log_if_needed(log_path, max_bytes=MAX_LOG_BYTES, keep_bytes=5*1024*1024):
    try:
        size = os.path.getsize(log_path)
        if size < max_bytes:
            return False
        with open(log_path) as f:
            f.seek(max(0, size - keep_bytes))
            f.readline()
            tail = f.read()
        dir_name = os.path.dirname(log_path) or '.'
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix='.tmp')
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(tail)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, log_path)
        except Exception as e:
            logger.debug(f"[utils] {type(e).__name__}: {e}")
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise
        logger.info(f"[LOG-ROTATE] {log_path}: {size/1024/1024:.1f}MB -> {len(tail)/1024/1024:.1f}MB")
        return True
    except Exception as e:
        logger.warning(f"[LOG-ROTATE] Failed for {log_path}: {e}")
        return False


def validate_env_vars(required_vars):
    missing = [v for v in required_vars if not os.environ.get(v)]
    if missing:
        print(f"FATAL: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)


def parse_llm_json(response_text):
    start = response_text.find('{')
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(response_text)):
        c = response_text[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(response_text[start:i + 1])
                except json.JSONDecodeError:
                    pass
                break
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    return None
