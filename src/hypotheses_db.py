#!/usr/bin/env python3
"""Centralized hypothesis_db access — abstracts JSON vs SQLite storage."""
from __future__ import annotations
import sys
import os
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import (
    load_hypotheses as _load, save_hypotheses as _save,
    update_hypothesis as _update, delete_hypothesis as _delete,
    load_hypothesis as _load_one, count_resolved_hypotheses as _count_resolved,
    hypothesis_slugs as _slugs, auto_migrate,
)
from schema import HYP_SLUG, HYP_DB_HYPOTHESES, HYP_DB_RESOLVED, HYP_RESOLVED

_initialized = False

def ensure_init() -> None:
    global _initialized
    if not _initialized:
        auto_migrate()
        _initialized = True

def load_all() -> dict[str, Any]:
    """Load full hypothesis DB. Drop-in for load_json(HYPOTHESIS_DB, ...).
    Returns {hypotheses: [list], resolved: [list]} matching legacy JSON format."""
    ensure_init()
    raw = _load()
    hypotheses_dict = raw.get("hypotheses", {})

    all_hyps: list[dict[str, Any]] = []
    resolved_list: list[dict[str, Any]] = []
    for slug, data in hypotheses_dict.items():
        h = dict(data)
        h[HYP_SLUG] = slug
        all_hyps.append(h)
        if h.get(HYP_RESOLVED):
            resolved_list.append(h)

    return {HYP_DB_HYPOTHESES: all_hyps, HYP_DB_RESOLVED: resolved_list}

def save_all(db_dict: dict[str, Any]) -> None:
    """Save full hypothesis DB. Drop-in for save_json(HYPOTHESIS_DB, db).
    Accepts {hypotheses: [list], resolved: [list]} legacy format."""
    ensure_init()
    hyp_dict: dict[str, dict[str, Any]] = {}

    for h in db_dict.get(HYP_DB_HYPOTHESES, []):
        slug = h.get(HYP_SLUG)
        if slug:
            hyp_dict[slug] = {k: v for k, v in h.items() if k != HYP_SLUG}

    for h in db_dict.get(HYP_DB_RESOLVED, []):
        slug = h.get(HYP_SLUG)
        if slug:
            hyp_dict[slug] = {k: v for k, v in h.items() if k != HYP_SLUG}

    _save({"hypotheses": hyp_dict})

def get(slug: str) -> dict[str, Any] | None:
    ensure_init()
    result = _load_one(slug)
    if result is not None:
        result[HYP_SLUG] = slug
    return result

def update(slug: str, data: dict[str, Any]) -> None:
    ensure_init()
    _update(slug, data)

def delete(slug: str) -> None:
    ensure_init()
    _delete(slug)

def count_resolved() -> int:
    ensure_init()
    return _count_resolved()

def slugs() -> list[str]:
    ensure_init()
    return _slugs()
