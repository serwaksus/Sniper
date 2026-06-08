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
import json
import logging
import os


class StructuredFormatter(logging.Formatter):
    def __init__(self, json_mode=False):
        super().__init__()
        self.json_mode = json_mode or os.environ.get("LOG_FORMAT") == "json"

    def format(self, record):
        if not self.json_mode:
            return super().format(record)

        log_entry = {
            "ts": self.formatTime(record, self.default_time_format),
            "level": record.levelname,
            "module": record.module,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[1]:
            log_entry["error"] = str(record.exc_info[1])
        return json.dumps(log_entry, ensure_ascii=False)
