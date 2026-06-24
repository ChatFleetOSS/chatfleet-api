# generated-by: codex-agent 2026-06-24T00:00:00Z
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import httpx


PROBE_DOCS = {
    "latency-upgrades.txt": """ChatFleet latency probe document.

Fact CF-LAT-001: Warm RAG retrieval should stay below 250 milliseconds at the API service layer.
Fact CF-LAT-002: The first SSE ready event should be sent after retrieval and before LLM generation.
Fact CF-LAT-003: A request slower than 10 seconds should be attributed to embedding, retrieval, prompt build, or LLM timing.
Fact CF-LAT-004: During upgrades, Mongo data, RAG indexes, uploaded documents, and configured secrets must be preserved.
""",
    "latency-operations.txt": """ChatFleet operations probe document.

Fact CF-OPS-010: RAG indexes are stored under /var/lib/chatfleet/faiss in the API container.
Fact CF-OPS-011: Uploaded documents are stored under /var/lib/chatfleet/uploads in the API container.
Fact CF-OPS-012: The latency report must keep the correlation identifier for every measured request.
Fact CF-OPS-013: If the context lacks an answer, the assistant must use the configured fallback answer.
""",
}

PROBE_QUESTIONS = [
    {
        "id": "retrieval_budget",
        "kind": "exact",
        "question": "What does CF-LAT-001 require for warm RAG retrieval?",
    },
    {
        "id": "sse_ready",
        "kind": "exact",
        "question": "When should the first SSE ready event be sent according to CF-LAT-002?",
    },
    {
        "id": "slow_request_attribution",
        "kind": "exact",
        "question": "How should a request slower than 10 seconds be attributed?",
    },
    {
        "id": "upgrade_preservation",
        "kind": "semantic",
        "question": "Which data must be preserved during upgrades?",
    },
    {
        "id": "index_location",
        "kind": "semantic",
        "question": "Where are RAG indexes stored in the API container?",
    },
    {
        "id": "uploads_location",
        "kind": "semantic",
        "question": "Where are uploaded documents stored in the API container?",
    },
    {
        "id": "corr_id_tracking",
        "kind": "semantic",
        "question": "What identifier must the latency report keep for each request?",
    },
    {
        "id": "fallback_unknown",
        "kind": "out_of_context",
        "question": "What is the cafeteria menu for next Friday?",
    },
    {
        "id": "unrelated_policy",
        "kind": "out_of_context",
        "question": "What is the company travel reimbursement rate in Japan?",
    },
]

DEFAULT_CLIENT_QUESTIONS = [
    {
        "id": "client_known_1",
        "kind": "client",
        "question": "Résume les informations principales disponibles dans ce RAG.",
    },
    {
        "id": "client_known_2",
        "kind": "client",
        "question": "Quels éléments factuels peux-tu citer depuis les documents ?",
    },
    {
        "id": "client_out_of_context",
        "kind": "out_of_context",
        "question": "Donne une information absente des documents sur la météo de demain.",
    },
]

METRIC_RE = re.compile(r"chatfleet\.metrics\s+({.*})")


@dataclass
class ChatMeasurement:
    suite: str
    rag_slug: str
    question_id: str
    question_kind: str
    question: str
    mode: str
    run: int
    corr_id: str
    status_code: int | None = None
    error_code: str | None = None
    client_total_ms: float | None = None
    first_sse_event_ms: float | None = None
    ready_sse_ms: float | None = None
    x_response_time_ms: float | None = None
    answer_chars: int = 0
    citations_count: int = 0
    chunks_count: int = 0
    tokens_in: int | None = None
    tokens_out: int | None = None
    server_events: list[dict[str, Any]] = field(default_factory=list)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    return value if value not in {None, ""} else default


def _auth_headers(token: str, corr_id: str | None = None) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {token}"}
    if corr_id:
        headers["x-corr-id"] = corr_id
    return headers


def _parse_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


def _summarize(values: Iterable[float | None]) -> dict[str, float | None]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "min": round(min(clean), 1),
        "p50": round(statistics.median(clean), 1),
        "p95": round(_percentile(clean, 95) or 0, 1),
        "max": round(max(clean), 1),
    }


def _login(client: httpx.Client, base_url: str, email: str, password: str) -> str:
    response = client.post(
        f"{base_url}/auth/login",
        json={"email": email, "password": password},
        timeout=15.0,
    )
    response.raise_for_status()
    token = response.json().get("token")
    if not token:
        raise RuntimeError("Login did not return a token")
    return token


def _llm_config(client: httpx.Client, base_url: str, token: str) -> dict[str, Any]:
    response = client.get(
        f"{base_url}/admin/llm/config",
        headers=_auth_headers(token),
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json().get("config") or {}


def _ensure_probe_rag(
    client: httpx.Client, base_url: str, token: str, slug: str
) -> None:
    payload = {
        "slug": slug,
        "name": "Latency probe RAG",
        "description": "Controlled RAG used by scripts/rag_latency_report.py.",
    }
    response = client.post(
        f"{base_url}/rag",
        json=payload,
        headers=_auth_headers(token),
        timeout=15.0,
    )
    if response.status_code not in {200, 201, 409}:
        response.raise_for_status()


def _upload_probe_docs(
    client: httpx.Client,
    base_url: str,
    token: str,
    slug: str,
) -> str:
    with tempfile.TemporaryDirectory(prefix="chatfleet-latency-docs-") as tmp:
        paths: list[Path] = []
        for filename, content in PROBE_DOCS.items():
            path = Path(tmp) / filename
            path.write_text(content, encoding="utf-8")
            paths.append(path)

        handles = [path.open("rb") for path in paths]
        try:
            files = [
                ("files", (path.name, handle, "text/plain"))
                for path, handle in zip(paths, handles)
            ]
            response = client.post(
                f"{base_url}/rag/upload",
                files=files,
                data={"rag_slug": slug},
                headers=_auth_headers(token),
                timeout=60.0,
            )
        finally:
            for handle in handles:
                handle.close()

    response.raise_for_status()
    payload = response.json()
    skipped = payload.get("skipped") or []
    if skipped:
        raise RuntimeError(f"Probe upload skipped files: {skipped}")
    job_id = payload.get("job_id")
    if not job_id:
        raise RuntimeError(f"Upload did not return job_id: {payload}")
    return str(job_id)


def _poll_job(
    client: httpx.Client,
    base_url: str,
    token: str,
    job_id: str,
    timeout_s: float,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        response = client.get(
            f"{base_url}/jobs/{job_id}",
            headers=_auth_headers(token),
            timeout=15.0,
        )
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") in {"done", "error"}:
            return payload
        time.sleep(1.5)
    raise TimeoutError(f"Job {job_id} did not finish within {timeout_s}s")


def _index_status(
    client: httpx.Client,
    base_url: str,
    token: str,
    slug: str,
) -> dict[str, Any]:
    response = client.get(
        f"{base_url}/rag/index/status",
        params={"rag_slug": slug},
        headers=_auth_headers(token),
        timeout=15.0,
    )
    response.raise_for_status()
    return response.json()


def _load_client_questions(path: str | None) -> list[dict[str, str]]:
    if not path:
        return DEFAULT_CLIENT_QUESTIONS
    raw = Path(path).read_text(encoding="utf-8")
    if path.endswith(".json"):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("Client question JSON must be a list")
        return [
            {
                "id": str(item.get("id") or f"client_{idx + 1}"),
                "kind": str(item.get("kind") or "client"),
                "question": str(item["question"]),
            }
            for idx, item in enumerate(data)
        ]
    questions: list[dict[str, str]] = []
    for idx, line in enumerate(raw.splitlines(), start=1):
        question = line.strip()
        if question and not question.startswith("#"):
            questions.append(
                {"id": f"client_{idx}", "kind": "client", "question": question}
            )
    return questions or DEFAULT_CLIENT_QUESTIONS


def _chat_once(
    client: httpx.Client,
    base_url: str,
    token: str,
    suite: str,
    slug: str,
    question: dict[str, str],
    run: int,
    timeout_s: float,
) -> ChatMeasurement:
    corr_id = f"rag-lat-{suite}-{question['id']}-chat-{run}-{uuid.uuid4()}"
    payload = {
        "rag_slug": slug,
        "messages": [{"role": "user", "content": question["question"]}],
    }
    measurement = ChatMeasurement(
        suite=suite,
        rag_slug=slug,
        question_id=question["id"],
        question_kind=question["kind"],
        question=question["question"],
        mode="chat",
        run=run,
        corr_id=corr_id,
    )
    started = time.perf_counter()
    try:
        response = client.post(
            f"{base_url}/chat",
            json=payload,
            headers=_auth_headers(token, corr_id),
            timeout=timeout_s,
        )
        measurement.client_total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        measurement.status_code = response.status_code
        measurement.x_response_time_ms = _parse_float(
            response.headers.get("x-response-time-ms")
        )
        data = response.json()
        if response.status_code >= 400:
            measurement.error_code = (
                ((data.get("detail") or {}).get("error") or {}).get("code")
                if isinstance(data, dict)
                else None
            )
            return measurement
        answer = data.get("answer") or ""
        citations = data.get("citations") or []
        usage = data.get("usage") or {}
        measurement.answer_chars = len(answer)
        measurement.citations_count = len(citations)
        measurement.tokens_in = usage.get("tokens_in")
        measurement.tokens_out = usage.get("tokens_out")
    except Exception as exc:
        measurement.client_total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        measurement.error_code = exc.__class__.__name__
    return measurement


def _consume_sse_line(
    measurement: ChatMeasurement,
    event: str | None,
    data_text: str,
    elapsed_ms: float,
) -> None:
    if measurement.first_sse_event_ms is None:
        measurement.first_sse_event_ms = round(elapsed_ms, 1)
    if event == "ready" and measurement.ready_sse_ms is None:
        measurement.ready_sse_ms = round(elapsed_ms, 1)
    try:
        payload = json.loads(data_text or "{}")
    except json.JSONDecodeError:
        payload = {}
    if event == "chunk":
        measurement.chunks_count += 1
        measurement.answer_chars += len(payload.get("delta") or "")
    elif event == "citations":
        measurement.citations_count = len(payload.get("citations") or [])
    elif event == "done":
        usage = payload.get("usage") or {}
        measurement.tokens_in = usage.get("tokens_in")
        measurement.tokens_out = usage.get("tokens_out")
    elif event == "error":
        error = payload.get("error") or {}
        measurement.error_code = error.get("code")


def _chat_stream_once(
    client: httpx.Client,
    base_url: str,
    token: str,
    suite: str,
    slug: str,
    question: dict[str, str],
    run: int,
    timeout_s: float,
) -> ChatMeasurement:
    corr_id = f"rag-lat-{suite}-{question['id']}-stream-{run}-{uuid.uuid4()}"
    payload = {
        "rag_slug": slug,
        "messages": [{"role": "user", "content": question["question"]}],
    }
    measurement = ChatMeasurement(
        suite=suite,
        rag_slug=slug,
        question_id=question["id"],
        question_kind=question["kind"],
        question=question["question"],
        mode="stream",
        run=run,
        corr_id=corr_id,
    )
    started = time.perf_counter()
    try:
        with client.stream(
            "POST",
            f"{base_url}/chat/stream",
            json=payload,
            headers=_auth_headers(token, corr_id),
            timeout=timeout_s,
        ) as response:
            measurement.status_code = response.status_code
            measurement.x_response_time_ms = _parse_float(
                response.headers.get("x-response-time-ms")
            )
            event: str | None = None
            data_lines: list[str] = []
            for line in response.iter_lines():
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                if line.startswith("event: "):
                    event = line.removeprefix("event: ").strip()
                elif line.startswith("data: "):
                    data_lines.append(line.removeprefix("data: ").strip())
                elif line == "":
                    if event is not None:
                        _consume_sse_line(
                            measurement,
                            event,
                            "\n".join(data_lines),
                            elapsed_ms,
                        )
                    event = None
                    data_lines = []
        measurement.client_total_ms = round((time.perf_counter() - started) * 1000.0, 1)
    except Exception as exc:
        measurement.client_total_ms = round((time.perf_counter() - started) * 1000.0, 1)
        measurement.error_code = exc.__class__.__name__
    return measurement


def _collect_docker_logs(container: str, since_iso: str) -> str:
    result = subprocess.run(
        ["docker", "logs", "--since", since_iso, container],
        check=False,
        capture_output=True,
        text=True,
    )
    return "\n".join(part for part in [result.stdout, result.stderr] if part)


def _load_metric_events(log_text: str) -> dict[str, list[dict[str, Any]]]:
    events_by_corr: dict[str, list[dict[str, Any]]] = {}
    for line in log_text.splitlines():
        match = METRIC_RE.search(line)
        if not match:
            continue
        try:
            event = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        corr_id = event.get("corr_id")
        if corr_id:
            events_by_corr.setdefault(str(corr_id), []).append(event)
    return events_by_corr


def _attach_server_events(
    measurements: list[ChatMeasurement],
    log_file: str | None,
    docker_container: str | None,
    since_iso: str,
) -> None:
    log_parts: list[str] = []
    if log_file:
        path = Path(log_file)
        if path.exists():
            log_parts.append(path.read_text(encoding="utf-8", errors="replace"))
    if docker_container:
        log_parts.append(_collect_docker_logs(docker_container, since_iso))
    events_by_corr = _load_metric_events("\n".join(log_parts))
    for measurement in measurements:
        measurement.server_events = events_by_corr.get(measurement.corr_id, [])


def _event_value(measurement: ChatMeasurement, event_name: str, key: str) -> Any:
    for event in measurement.server_events:
        if event.get("event") == event_name and key in event:
            return event[key]
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def _status_label(measurement: ChatMeasurement) -> str:
    if measurement.error_code:
        return f"ERR:{measurement.error_code}"
    if measurement.status_code and measurement.status_code >= 400:
        return f"HTTP:{measurement.status_code}"
    return "OK"


def _verdict(measurements: list[ChatMeasurement]) -> str:
    slow = [m for m in measurements if (m.client_total_ms or 0) > 10_000]
    missing_server = [m for m in measurements if not m.server_events]
    retrieval_values = [
        _parse_float(_event_value(m, "chat.retrieval", "retrieval_ms"))
        for m in measurements
    ]
    llm_values = [
        _parse_float(_event_value(m, "chat.llm.timing", "llm_ms")) for m in measurements
    ]
    max_retrieval = max([v for v in retrieval_values if v is not None] or [0.0])
    max_llm = max([v for v in llm_values if v is not None] or [0.0])
    if missing_server:
        return (
            "Données HTTP collectées, mais métriques serveur incomplètes. "
            "Fournir `--api-log-file` ou `--docker-container` pour attribuer précisément les lenteurs."
        )
    if slow and max_llm >= max_retrieval * 5:
        return "Non OK: les requêtes lentes sont majoritairement attribuées au LLM/provider."
    if max_retrieval > 1000:
        return "Non OK: retrieval serveur supérieur à 1s sur au moins une requête."
    if slow:
        return "À investiguer: des requêtes dépassent 10s sans goulot unique évident."
    return "OK: aucune latence critique observée sur la campagne."


def _markdown_report(
    *,
    base_url: str,
    started_at: str,
    llm_config: dict[str, Any],
    probe_slug: str,
    client_slug: str | None,
    index_status: dict[str, Any] | None,
    measurements: list[ChatMeasurement],
) -> str:
    lines = [
        "# Rapport latence RAG réel",
        "",
        f"- Date UTC: `{started_at}`",
        f"- API: `{base_url}`",
        f"- RAG probe: `{probe_slug}`",
        f"- RAG client: `{client_slug or 'non fourni'}`",
        f"- LLM provider: `{llm_config.get('provider')}`",
        f"- Chat model: `{llm_config.get('chat_model')}`",
        f"- Embed provider: `{llm_config.get('embed_provider')}`",
        f"- Embed model: `{llm_config.get('embed_model')}`",
        "",
        "## Verdict",
        "",
        _verdict(measurements),
        "",
    ]
    if index_status:
        lines.extend(
            [
                "## Index probe",
                "",
                "```json",
                json.dumps(index_status, indent=2, ensure_ascii=False, default=str),
                "```",
                "",
            ]
        )

    for suite in sorted({m.suite for m in measurements}):
        suite_items = [m for m in measurements if m.suite == suite]
        lines.extend([f"## Synthèse `{suite}`", ""])
        for mode in ["chat", "stream"]:
            mode_items = [m for m in suite_items if m.mode == mode]
            if not mode_items:
                continue
            total = _summarize(m.client_total_ms for m in mode_items)
            first = _summarize(m.first_sse_event_ms for m in mode_items)
            retrieval = _summarize(
                _parse_float(_event_value(m, "chat.retrieval", "retrieval_ms"))
                for m in mode_items
            )
            llm = _summarize(
                _parse_float(_event_value(m, "chat.llm.timing", "llm_ms"))
                for m in mode_items
            )
            lines.extend(
                [
                    f"### `{mode}`",
                    "",
                    "| Métrique | count | min | p50 | p95 | max |",
                    "|---|---:|---:|---:|---:|---:|",
                    f"| client_total_ms | {total['count']} | {_fmt(total['min'])} | {_fmt(total['p50'])} | {_fmt(total['p95'])} | {_fmt(total['max'])} |",
                    f"| first_sse_event_ms | {first['count']} | {_fmt(first['min'])} | {_fmt(first['p50'])} | {_fmt(first['p95'])} | {_fmt(first['max'])} |",
                    f"| server_retrieval_ms | {retrieval['count']} | {_fmt(retrieval['min'])} | {_fmt(retrieval['p50'])} | {_fmt(retrieval['p95'])} | {_fmt(retrieval['max'])} |",
                    f"| server_llm_ms | {llm['count']} | {_fmt(llm['min'])} | {_fmt(llm['p50'])} | {_fmt(llm['p95'])} | {_fmt(llm['max'])} |",
                    "",
                ]
            )

    lines.extend(
        [
            "## Détail requêtes",
            "",
            "| Suite | Mode | Run | Question | Status | HTTP ms | First SSE ms | Retrieval ms | LLM ms | Total server ms | Citations | Tokens out | Corr ID |",
            "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for item in measurements:
        server_total = _event_value(item, "chat.completion.timing", "total_ms")
        if server_total is None:
            server_total = _event_value(item, "chat.stream.timing", "total_ms")
        lines.append(
            "| "
            + " | ".join(
                [
                    item.suite,
                    item.mode,
                    str(item.run),
                    item.question_id,
                    _status_label(item),
                    _fmt(item.client_total_ms),
                    _fmt(item.first_sse_event_ms),
                    _fmt(_event_value(item, "chat.retrieval", "retrieval_ms")),
                    _fmt(_event_value(item, "chat.llm.timing", "llm_ms")),
                    _fmt(server_total),
                    str(item.citations_count),
                    _fmt(item.tokens_out),
                    f"`{item.corr_id}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `First SSE ms` est renseigné uniquement pour `/chat/stream`.",
            "- Les métriques serveur nécessitent les logs contenant `chatfleet.metrics {...}`.",
            "- Les questions hors contexte doivent retourner un fallback ou une erreur contrôlée, pas une hallucination.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_outputs(
    output_dir: Path,
    started_at: str,
    report: str,
    measurements: list[ChatMeasurement],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stamp = started_at.replace(":", "").replace("-", "").replace("+", "z")
    md_path = output_dir / f"rag-latency-{safe_stamp}.md"
    json_path = output_dir / f"rag-latency-{safe_stamp}.json"
    md_path.write_text(report, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            [measurement.__dict__ for measurement in measurements],
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )
    return md_path, json_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real RAG latency probes and generate a Markdown report."
    )
    parser.add_argument(
        "--base-url", default=_env("BASE_URL", "http://localhost:8080/api")
    )
    parser.add_argument("--admin-email", default=_env("ADMIN_EMAIL"))
    parser.add_argument("--admin-password", default=_env("ADMIN_PASSWORD"))
    parser.add_argument("--client-rag-slug", default=_env("CLIENT_RAG_SLUG"))
    parser.add_argument("--client-question-file", default=_env("CLIENT_QUESTION_FILE"))
    parser.add_argument("--probe-rag-slug", default=_env("PROBE_RAG_SLUG"))
    parser.add_argument(
        "--runs", type=int, default=int(_env("RAG_LATENCY_RUNS", "3") or "3")
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(_env("RAG_LATENCY_TIMEOUT", "90") or "90"),
    )
    parser.add_argument(
        "--poll-timeout",
        type=float,
        default=float(_env("POLL_TIMEOUT", "120") or "120"),
    )
    parser.add_argument("--api-log-file", default=_env("API_LOG_FILE"))
    parser.add_argument("--docker-container", default=_env("API_DOCKER_CONTAINER"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_env("OUTPUT_DIR", "reports") or "reports"),
    )
    parser.add_argument("--skip-probe-upload", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.admin_email or not args.admin_password:
        raise SystemExit(
            "Set ADMIN_EMAIL and ADMIN_PASSWORD, or pass --admin-email/--admin-password."
        )
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    base_url = str(args.base_url).rstrip("/")
    started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    probe_slug = args.probe_rag_slug or f"latency-probe-{int(time.time())}"
    measurements: list[ChatMeasurement] = []
    index_status: dict[str, Any] | None = None

    with httpx.Client(timeout=args.timeout) as client:
        print(f"[health] {base_url}")
        client.get(f"{base_url}/health", timeout=15.0).raise_for_status()
        print(f"[login] {args.admin_email}")
        token = _login(client, base_url, args.admin_email, args.admin_password)
        llm_config = _llm_config(client, base_url, token)
        print(
            "[llm]",
            {
                "provider": llm_config.get("provider"),
                "chat_model": llm_config.get("chat_model"),
                "embed_provider": llm_config.get("embed_provider"),
                "embed_model": llm_config.get("embed_model"),
            },
        )

        print(f"[probe] ensure rag {probe_slug}")
        _ensure_probe_rag(client, base_url, token, probe_slug)
        if not args.skip_probe_upload:
            print("[probe] upload controlled documents")
            job_id = _upload_probe_docs(client, base_url, token, probe_slug)
            job = _poll_job(client, base_url, token, job_id, args.poll_timeout)
            if job.get("status") != "done":
                raise RuntimeError(f"Probe indexing failed: {job}")
        index_status = _index_status(client, base_url, token, probe_slug)
        print("[probe] index", index_status)

        suites = [("probe", probe_slug, PROBE_QUESTIONS)]
        if args.client_rag_slug:
            suites.append(
                (
                    "client",
                    args.client_rag_slug,
                    _load_client_questions(args.client_question_file),
                )
            )
        else:
            print("[client] CLIENT_RAG_SLUG not set; skipping client RAG measurements")

        for suite, slug, questions in suites:
            print(f"[suite] {suite} rag={slug} questions={len(questions)}")
            for run in range(1, args.runs + 1):
                for question in questions:
                    measurements.append(
                        _chat_once(
                            client,
                            base_url,
                            token,
                            suite,
                            slug,
                            question,
                            run,
                            args.timeout,
                        )
                    )
                    measurements.append(
                        _chat_stream_once(
                            client,
                            base_url,
                            token,
                            suite,
                            slug,
                            question,
                            run,
                            args.timeout,
                        )
                    )

    _attach_server_events(
        measurements,
        args.api_log_file,
        args.docker_container,
        started_at,
    )
    report = _markdown_report(
        base_url=base_url,
        started_at=started_at,
        llm_config=llm_config,
        probe_slug=probe_slug,
        client_slug=args.client_rag_slug,
        index_status=index_status,
        measurements=measurements,
    )
    md_path, json_path = _write_outputs(
        args.output_dir, started_at, report, measurements
    )
    print(f"[report] markdown={md_path}")
    print(f"[report] json={json_path}")
    print("[verdict]", _verdict(measurements))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
