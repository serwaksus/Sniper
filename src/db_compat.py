#!/usr/bin/env python3
"""Compatibility wrapper — drop-in replacement for load_json/save_json for positions and hypotheses."""
from db import (
    init_db, load_positions, save_positions, load_hypotheses,
    save_hypotheses
)

POSITIONS_KEY = "positions"
HYPOTHESES_KEY = "hypothesis_db"

_initialized = False

def ensure_init():
    global _initialized
    if not _initialized:
        init_db()
        _initialized = True

def load_positions_compat(path, default=None):
    ensure_init()
    return load_positions()

def save_positions_compat(path, data):
    ensure_init()
    save_positions(data)

def load_hypotheses_compat(path, default=None):
    ensure_init()
    return load_hypotheses()

def save_hypotheses_compat(path, data):
    ensure_init()
    save_hypotheses(data)
