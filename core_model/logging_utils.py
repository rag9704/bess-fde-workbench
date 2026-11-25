"""Logging setup: readable text locally, one-JSON-object-per-line in production.

Structured fields passed via `logger.info("msg", extra={...})` are kept, so solve
time, status and objective can be queried in Cloud Logging / Datadog etc.
"""

import json
import logging
import os
import sys

_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update({k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS})
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class KeyValueFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7} {record.name}: {record.getMessage()}"
        extras = {k: v for k, v in vars(record).items() if k not in _STANDARD_ATTRS}
        if extras:
            base += " | " + " ".join(f"{k}={v}" for k, v in extras.items())
        return base


def setup_logging(level: str | None = None, json_format: bool | None = None) -> None:
    """Configure root logging. Defaults come from LOG_LEVEL and LOG_FORMAT=json env vars."""
    level = (level or os.getenv("LOG_LEVEL", "WARNING")).upper()
    if json_format is None:
        json_format = os.getenv("LOG_FORMAT", "").lower() == "json"
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if json_format else KeyValueFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
