# generated-by: codex-agent 2025-02-15T00:58:00Z
"""
Helpers to optionally auto-start a local MongoDB daemon during development.
"""

from __future__ import annotations

import asyncio
import logging
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, IO, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .config import settings

logger = logging.getLogger("chatfleet.mongo")

_mongo_process: Optional[subprocess.Popen[str]] = None
_mongo_started_by_app = False
_mongo_log_handle: Optional[IO[Any]] = None


async def ensure_local_mongo() -> None:
    """Start a local `mongod` if configured and not already running."""

    if not settings.auto_start_mongo:
        logger.debug("AUTO_START_MONGO disabled; skipping mongod bootstrap")
        return

    host, port = _extract_host_port(settings.mongo_uri)
    is_ready = await asyncio.to_thread(_is_port_open, host, port)
    if is_ready:
        logger.info("Detected MongoDB already listening on %s:%s", host, port)
        return

    logger.info(
        "MongoDB not reachable at %s:%s; attempting to launch `%s` automatically",
        host,
        port,
        settings.mongo_bin,
    )

    started = await asyncio.to_thread(_start_mongo_process, host, port)
    if not started:
        return
    await asyncio.to_thread(_wait_for_port, host, port, settings.mongo_startup_timeout_s)


def stop_local_mongo() -> None:
    """Terminate the auto-started MongoDB instance, if we launched one."""

    global _mongo_process, _mongo_started_by_app, _mongo_log_handle

    if not _mongo_started_by_app or _mongo_process is None:
        return

    if _mongo_process.poll() is None:
        logger.info("Stopping auto-started MongoDB process (pid=%s)", _mongo_process.pid)
        _mongo_process.terminate()
        try:
            _mongo_process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("MongoDB process did not exit; sending SIGKILL")
            _mongo_process.kill()
            _mongo_process.wait(timeout=5)

    if _mongo_log_handle:
        try:
            _mongo_log_handle.close()
        except Exception:
            logger.debug("Failed to close MongoDB log handle", exc_info=True)

    _mongo_process = None
    _mongo_started_by_app = False
    _mongo_log_handle = None


def _extract_host_port(uri: str) -> Tuple[str, int]:
    parsed = urlparse(uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 27017
    return host, port


def _is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
        except OSError:
            return False
    return True


def _start_mongo_process(host: str, port: int) -> bool:
    global _mongo_process, _mongo_started_by_app, _mongo_log_handle

    if _mongo_process and _mongo_process.poll() is None:
        return False

    cmd = _build_command(host, port)
    log_path = _log_file_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8")
    log_handle.write(f"\n--- mongo launcher at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
    log_handle.flush()

    try:
        _mongo_process = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            close_fds=True,
        )
    except FileNotFoundError as exc:
        log_handle.close()
        raise RuntimeError(
            f"Unable to auto-start MongoDB; `{settings.mongo_bin}` not found on PATH"
        ) from exc
    except PermissionError as exc:  # e.g. sandbox denied execution
        log_handle.close()
        logger.warning("Failed to launch `%s`: %s; continuing without auto-start", settings.mongo_bin, exc)
        return False

    _mongo_log_handle = log_handle
    _mongo_started_by_app = True
    logger.info("Spawned MongoDB process pid=%s via `%s`", _mongo_process.pid, " ".join(cmd))
    return True


def _wait_for_port(host: str, port: int, timeout_s: int) -> None:
    deadline = time.monotonic() + timeout_s
    delay = 0.5
    while time.monotonic() < deadline:
        if _is_port_open(host, port):
            logger.info("MongoDB is accepting connections on %s:%s", host, port)
            return
        time.sleep(delay)
    raise RuntimeError(f"MongoDB failed to start within {timeout_s}s; check {_log_file_path()} for details.")


def _build_command(host: str, port: int) -> Sequence[str]:
    cmd: list[str] = [settings.mongo_bin]
    if settings.mongo_config_path:
        cmd.extend(["--config", str(settings.mongo_config_path)])
    else:
        cmd.extend(
            [
                "--dbpath",
                str(settings.mongo_db_path),
                "--bind_ip",
                host,
                "--port",
                str(port),
            ]
        )
    return cmd


def _log_file_path() -> Path:
    return settings.mongo_db_path / "mongod.log"
