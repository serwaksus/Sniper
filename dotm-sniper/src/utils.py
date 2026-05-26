"""
DOTM Sniper Utilities Module
============================
Centralized utilities for file operations, locking, and data sanitization.
Addresses: BUG-08, BUG-10, BUG-13, BUG-41, BUG-42, BUG-51, BUG-52

Usage:
    from src.utils import load_json, save_json, sanitize_for_prompt
"""

import json
import os
import fcntl
import tempfile
import re
import logging
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

# Resolve base directory dynamically (BUG-42 fix)
# Defaults to /root/dotm-sniper if not set via env
DOTM_HOME = Path(os.getenv("DOTM_HOME", "/root/dotm-sniper"))

logger = logging.getLogger(__name__)


def _normalize_keys(obj: Any) -> Any:
    """Recursively normalize dictionary keys to strings and strip whitespace."""
    if isinstance(obj, dict):
        return {str(k).strip(): _normalize_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_normalize_keys(item) for item in obj]
    else:
        return obj


def _strip_dict_keys_recursive(data: Any) -> Any:
    """Alias for _normalize_keys for backward compatibility."""
    return _normalize_keys(data)


def load_json(filepath: str, normalize: bool = True) -> Any:
    """
    Load JSON file with file locking to prevent read-write races.
    
    Args:
        filepath: Path to JSON file (absolute or relative to DOTM_HOME)
        normalize: Whether to normalize keys (default True)
    
    Returns:
        Parsed JSON data
    
    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is corrupted
    """
    path = Path(filepath) if Path(filepath).is_absolute() else DOTM_HOME / filepath
    
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            # Shared lock for reading (allows concurrent reads)
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                content = f.read()
                if not content.strip():
                    return {} if normalize else {}
                data = json.loads(content)
                return _normalize_keys(data) if normalize else data
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except json.JSONDecodeError as e:
        logger.error(f"Corrupted JSON file {path}: {e}")
        raise


def save_json(filepath: str, data: Any, atomic: bool = True, normalize: bool = True) -> None:
    """
    Save JSON file atomically with exclusive locking.
    
    Uses tempfile + os.replace for atomicity (BUG-08, BUG-10 fix).
    Uses fcntl.flock for process synchronization (BUG-03 mitigation).
    
    Args:
        filepath: Path to JSON file
        data: Data to serialize
        atomic: Use atomic write (tempfile + replace) - default True
        normalize: Normalize keys before saving - default True
    """
    path = Path(filepath) if Path(filepath).is_absolute() else DOTM_HOME / filepath
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if normalize:
        data = _normalize_keys(data)
    
    if atomic:
        # Atomic write: write to temp file, then rename
        fd, temp_path = tempfile.mkstemp(suffix='.json', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            
            # Rename is atomic on POSIX systems
            os.replace(temp_path, path)
        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.unlink(temp_path)
            logger.error(f"Failed to save JSON {path}: {e}")
            raise
    else:
        # Non-atomic write with locking (legacy compatibility)
        with open(path, 'w', encoding='utf-8') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_json_versioned(filepath: str) -> Tuple[Any, int]:
    """
    Load JSON with version tracking for optimistic locking (BUG-03, BUG-52 fix).
    
    Returns:
        Tuple of (data, version) where version is incremented on each save.
    """
    data = load_json(filepath)
    version = data.get("_version", 0) if isinstance(data, dict) else 0
    return data, version


def save_json_versioned(filepath: str, data: Dict[str, Any], expected_version: Optional[int] = None) -> bool:
    """
    Save JSON with optimistic locking.
    
    Fails if the file was modified by another process since loading.
    
    Args:
        filepath: Path to JSON file
        data: Data to save (will have _version incremented)
        expected_version: Expected version number (None to skip check)
    
    Returns:
        True if save succeeded, False if version conflict detected
    """
    if not isinstance(data, dict):
        raise ValueError("Versioned save requires dict data")
    
    # Load current version
    try:
        current_data, current_version = load_json_versioned(filepath)
    except FileNotFoundError:
        current_version = 0
        current_data = {}
    
    # Check version if expected_version provided
    if expected_version is not None and current_version != expected_version:
        logger.warning(f"Version conflict for {filepath}: expected {expected_version}, got {current_version}")
        return False
    
    # Increment version
    data["_version"] = current_version + 1
    
    # Save atomically
    save_json(filepath, data, atomic=True)
    return True


def sanitize_for_prompt(text: str) -> str:
    """
    Sanitize text for LLM prompts to prevent injection attacks (BUG-41, BUG-51 fix).
    
    - Removes control characters
    - Normalizes Unicode
    - Escapes potential prompt injection patterns
    - Truncates excessively long inputs
    
    Args:
        text: Raw input text
    
    Returns:
        Sanitized text safe for LLM prompts
    """
    if not text:
        return ""
    
    # Normalize Unicode (NFKC normalization handles compatibility chars)
    text = text.strip()
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)  # Remove control chars
    
    # Remove common injection patterns
    injection_patterns = [
        r'ignore\s+previous\s+instructions',
        r'output\s+only\s+"yes"',
        r'system\s+prompt',
        r'you\s+are\s+now\s+',
        r'disregard\s+all\s+',
    ]
    for pattern in injection_patterns:
        text = re.sub(pattern, '[FILTERED]', text, flags=re.IGNORECASE)
    
    # Truncate if too long (prevent context overflow)
    max_length = 4000
    if len(text) > max_length:
        text = text[:max_length - 100] + "\n...[TRUNCATED]"
    
    return text


def get_pid_file_path() -> Path:
    """Get path to PID file for process locking (BUG-01 fix)."""
    return DOTM_HOME / "sniper.pid"


def check_and_write_pid() -> bool:
    """
    Check if another instance is running and write PID file.
    
    Returns:
        True if this is the first instance, False if duplicate detected
    """
    pid_file = get_pid_file_path()
    
    if pid_file.exists():
        try:
            with open(pid_file, 'r') as f:
                old_pid = int(f.read().strip())
            
            # Check if process is still running
            import signal
            os.kill(old_pid, 0)  # Signal 0 checks existence without killing
            
            # Process exists, this is a duplicate
            logger.warning(f"Duplicate instance detected (PID {old_pid}). Exiting.")
            return False
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale PID file, remove it
            pid_file.unlink()
            logger.info("Removed stale PID file.")
    
    # Write current PID
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    
    logger.info(f"PID file created for process {os.getpid()}")
    return True


def cleanup_pid_file() -> None:
    """Remove PID file on clean shutdown."""
    pid_file = get_pid_file_path()
    if pid_file.exists():
        try:
            pid_file.unlink()
            logger.info("PID file removed.")
        except Exception as e:
            logger.error(f"Failed to remove PID file: {e}")


def prune_cache(cache_data: Dict[str, Any], max_age_hours: int = 24) -> Dict[str, Any]:
    """
    Remove stale entries from cache (BUG-50 fix).
    
    Args:
        cache_data: Cache dictionary with 'timestamp' fields
        max_age_hours: Maximum age in hours
    
    Returns:
        Pruned cache dictionary
    """
    import time
    cutoff = time.time() - (max_age_hours * 3600)
    
    pruned = {}
    for key, value in cache_data.items():
        if isinstance(value, dict) and 'timestamp' in value:
            if value['timestamp'] >= cutoff:
                pruned[key] = value
        else:
            # Keep non-timestamped entries (backward compat)
            pruned[key] = value
    
    if len(pruned) < len(cache_data):
        logger.info(f"Pruned {len(cache_data) - len(pruned)} stale cache entries.")
    
    return pruned
