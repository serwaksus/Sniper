#!/usr/bin/env python3
"""Drop-in replacement for load_json/save_json using SQLite backend.

Drop-in replacement: replace 'from utils import load_json, save_json' with
'from db_compat import load_positions_json, save_positions_json, ...' and no other changes needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from db import (
    init_db, load_positions, save_positions, load_hypotheses, save_hypotheses,
)

_initialized = False

def ensure_init():
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True

def load_positions_json(path, default=None):
    """Drop-in for load_json(POSITIONS_FILE, {})."""
    ensure_init()
    return load_positions()

def save_positions_json(path, data):
    """Drop-in for save_json(POSITIONS_FILE, data)."""
    ensure_init()
    save_positions(data)

def load_hypothesis_db_json(path, default=None):
    """Drop-in for load_json(HYPOTHESIS_DB, ...)."""
    ensure_init()
    return load_hypotheses()

def save_hypothesis_db_json(path, data):
    """Drop-in for save_json(HYPOTHESIS_DB, data)."""
    ensure_init()
    save_hypotheses(data)
