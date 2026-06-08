#!/usr/bin/env python3
"""Centralized positions access — abstracts JSON vs SQLite storage."""
import sys
import os
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import (
    load_positions as _load, save_positions as _save,
    update_position as _update, delete_position as _delete,
    merge_save_positions as _merge, load_position as _load_one,
    count_positions as _count, position_slugs as _slugs,
    auto_migrate,
)

_initialized = False

def ensure_init() -> None:
    global _initialized
    if not _initialized:
        auto_migrate()
        _initialized = True

def load_all() -> dict[str, dict[str, Any]]:
    """Load all positions as dict. Drop-in for load_json(POSITIONS_FILE, {})."""
    ensure_init()
    return _load()

def save_all(positions: dict[str, dict[str, Any]]) -> None:
    """Save all positions. Drop-in for save_json(POSITIONS_FILE, positions)."""
    ensure_init()
    _save(positions)

def get(slug: str) -> dict[str, Any] | None:
    """Load a single position."""
    ensure_init()
    return _load_one(slug)

load = get

def update(slug: str, data: dict[str, Any]) -> None:
    """Update or insert a single position."""
    ensure_init()
    _update(slug, data)

def delete(slug: str) -> None:
    """Delete a single position."""
    ensure_init()
    _delete(slug)

def merge(updated: dict[str, dict[str, Any]]) -> None:
    """Merge updated positions into existing. Like hermes _merge_save_positions."""
    ensure_init()
    _merge(updated)

def count() -> int:
    ensure_init()
    return _count()

def slugs() -> list[str]:
    ensure_init()
    return _slugs()
