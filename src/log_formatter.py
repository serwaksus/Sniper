#!/usr/bin/env python3
"""Structured log formatter supporting both human-readable and JSON output.

Usage:
    import logging
    from log_formatter import StructuredFormatter

    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter(json_mode=False))  # or True
    logging.root.addHandler(handler)

Enable JSON globally via environment:
    LOG_FORMAT=json python3 src/dotm_sniper.py
"""
from __future__ import annotations
from typing import Any
import json
import logging
import os
from config import sanitize


class StructuredFormatter(logging.Formatter):
    def __init__(self, json_mode: bool = False) -> None:
        super().__init__()
        self.json_mode = json_mode or os.environ.get("LOG_FORMAT") == "json"

    def format(self, record: Any) -> str:
        record.msg = sanitize(str(record.msg))
        if not self.json_mode:
            result = super().format(record)
            if record.exc_info and record.exc_text:
                result = result.replace(record.exc_text, sanitize(record.exc_text))
            return result

        log_entry = {
            "ts": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "module": record.module,
            "msg": sanitize(record.getMessage()),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = sanitize(str(record.exc_info[1]))
        return json.dumps(log_entry, ensure_ascii=False)
