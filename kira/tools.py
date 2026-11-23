"""Kira's read-only tools. Each one returns a short text summary for the LLM."""
import json
import math
import time
from datetime import datetime, timezone

import requests
from kubernetes import client, config

PROM = "http://127.0.0.1:9090"
LOKI = "http://127.0.0.1:3100"
NAMESPACE = "boutique"
BACKENDS = ["gateway", "auth", "product-service", "order-service", "orders", "user-service"]
APPS = BACKENDS + ["frontend", "postgres"]
MAX_CHARS = 3000
STALE_MIN = 5  # a log line not seen for this many minutes is OLD
QUIET_WHEN_ZERO = {"error_rate", "restarts_15m"}  # zero series are noise for these checks

config.load_kube_config()
core = client.CoreV1Api()
apps_api = client.AppsV1Api()


def _cap(text: str) -> str:
    return text if len(text) <= MAX_CHARS else text[:MAX_CHARS] + "\n...[truncated]"


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _ago(ts) -> str:
    if not ts:
        return "unknown time"
    mins = int((datetime.now(timezone.utc) - ts).total_seconds() // 60)
    return f"{mins} min ago, RECENT" if mins <= 10 else f"{mins} min ago, OLD"


def fetch_health() -> str:
    """Get the Kubernetes health of the boutique app: deployments and statefulsets
    (desired vs ready replicas), pods (phase, ready, restarts, crash reason) and
    recent warning events with their age. Call this first to see what is broken.
    """
    out = [f"NOW {_now()}", "WORKLOADS (desired/ready):"]
    for d in apps_api.list_namespaced_deployment(NAMESPACE).items:
        out.append(f"  deploy/{d.metadata.name} {d.spec.replicas}/{d.status.ready_replicas or 0}")
    for s in apps_api.list_namespaced_stateful_set(NAMESPACE).items:
        out.append(f"  statefulset/{s.metadata.name} {s.spec.replicas}/{s.status.ready_replicas or 0}")

    out.append("PODS (phase, ready, restarts, reason):")
    pods = core.list_namespaced_pod(NAMESPACE).items
    for p in pods:
        for cs in p.status.container_statuses or []:
            reason = ""
            if cs.state.waiting:
                reason = cs.state.waiting.reason or ""
            elif cs.state.terminated:
                reason = cs.state.terminated.reason or ""
            if cs.last_state and cs.last_state.terminated:
                t = cs.last_state.terminated
                reason += f" (last exit: {t.reason}, code {t.exit_code})"
            out.append(f"  {p.metadata.name} {p.status.phase} ready={cs.ready} restarts={cs.restart_count} {reason}".rstrip())

    def _ts(e):
        return e.last_timestamp or e.event_time or e.metadata.creation_timestamp

    warnings = sorted((e for e in core.list_namespaced_event(NAMESPACE).items if e.type == "Warning"),
                      key=_ts, reverse=True)
    out.append("RECENT WARNING EVENTS (newest first):")
    live = {p.metadata.name for p in pods}
    for e in warnings[:8]:
        obj = e.involved_object
        gone = ", that pod no longer exists" if obj.kind == "Pod" and obj.name not in live else ""
        out.append(f"  [{_ago(_ts(e))}{gone}] {obj.kind}/{obj.name}: {e.reason} - {(e.message or '')[:150]}")
    if not warnings:
        out.append("  none")
    return _cap("\n".join(out))


CHECKS = {
    "targets_up": 'up{namespace="boutique"}',
    "request_rate": 'sum by (job) (rate(http_requests_total{namespace="boutique"}[5m]))',
    "error_rate": 'sum by (job, route, status_code) (rate(http_requests_total{namespace="boutique", status_code=~"5.."}[5m]))',
    "error_ratio_pct": '100 * sum by (job) (rate(http_requests_total{namespace="boutique", status_code=~"5.."}[5m])) / sum by (job) (rate(http_requests_total{namespace="boutique"}[5m]))',
    "latency_p95_seconds": 'histogram_quantile(0.95, sum by (le, job) (rate(http_request_duration_seconds_bucket{namespace="boutique"}[5m])))',
    "restarts_15m": 'sum by (pod) (increase(kube_pod_container_status_restarts_total{namespace="boutique"}[15m]))',
    "db_ready_replicas": 'kube_statefulset_status_replicas_ready{namespace="boutique", statefulset="boutique-postgres"}',
}


def fetch_metrics(check: str) -> str:
    """Run a predefined Prometheus check on the boutique app (last 5 to 15 minutes).

    Args:
        check: one of targets_up, request_rate, error_rate, error_ratio_pct,
            latency_p95_seconds, restarts_15m, db_ready_replicas
    """
    query = CHECKS.get(check)
    if not query:
        return f"Unknown check '{check}'. Valid checks: {', '.join(CHECKS)}"
    try:
        r = requests.get(f"{PROM}/api/v1/query", params={"query": query}, timeout=10)
        r.raise_for_status()
        result = r.json()["data"]["result"]
    except Exception as exc:
        return f"ERROR: cannot query Prometheus at {PROM} ({exc}). Is the port-forward running?"

    lines = [f"{check} (now {_now()}):"]
    hidden = 0
    for s in result[:40]:
        labels = {k: v for k, v in s["metric"].items() if k in ("job", "route", "status_code", "pod", "statefulset")}
        value = float(s["value"][1])
        if math.isnan(value):
            lines.append(f"  {labels} = no traffic in window")
        elif value == 0 and check in QUIET_WHEN_ZERO:
            hidden += 1
        else:
            lines.append(f"  {labels} = {value:.3f}")
    if hidden:
        lines.append(f"  ({hidden} series are 0: nothing happening there now)")
    if not result:
        lines.append("  no data (no matching series: e.g. no traffic, or no errors)")
    if check == "targets_up":
        seen = {s["metric"].get("job") for s in result}
        missing = [b for b in BACKENDS if b not in seen]
        if missing:
            lines.append(f"  MISSING scrape targets (no running pods?): {missing}")
    return _cap("\n".join(lines))


def fetch_logs(app: str, pattern: str = "error|exception|fail|ECONN|ENOTFOUND", minutes: int = 15) -> str:
    """Get recent log lines of one boutique service from Loki, filtered by a regex.
    Repeated lines are grouped with a count and time range, and tagged [NOW] if still
    happening or [OLD] if they stopped more than 5 minutes ago (a past incident).

    Args:
        app: one of gateway, auth, product-service, order-service, orders, user-service, frontend, postgres
        pattern: case-insensitive regex to filter lines; the default matches common errors
        minutes: how far back to search, 1 to 120
    """
    if app not in APPS:
        return f"Unknown app '{app}'. Valid apps: {', '.join(APPS)}"
    minutes = max(1, min(int(minutes), 120))
    logql = f'{{namespace="{NAMESPACE}", app="{app}"}} |~ {json.dumps("(?i)" + pattern)}'
    now = time.time_ns()
    params = {"query": logql, "start": now - minutes * 60 * 10**9, "end": now, "limit": 300, "direction": "backward"}
    try:
        r = requests.get(f"{LOKI}/loki/api/v1/query_range", params=params, timeout=10)
        r.raise_for_status()
        streams = r.json()["data"]["result"]
    except Exception as exc:
        return f"ERROR: cannot query Loki at {LOKI} ({exc}). Is the port-forward running?"

    entries = sorted((int(ts), line.strip()) for s in streams for ts, line in s["values"] if line.strip())
    if not entries:
        return f"NOW {_now()}. No log lines for app={app} matching /{pattern}/ in the last {minutes} min"

    grouped = {}  # line -> [first_ts, last_ts, count]; insertion order = oldest first
    for ts, line in entries:
        if line in grouped:
            grouped[line][1] = ts
            grouped[line][2] += 1
        else:
            grouped[line] = [ts, ts, 1]

    def hm(ns):
        return time.strftime("%H:%M:%S", time.localtime(ns / 1e9))

    stale_ns = STALE_MIN * 60 * 10**9
    any_now = any(now - last <= stale_ns for _, last, _ in grouped.values())
    out = [f"NOW {_now()}. {len(entries)} matching lines for app={app} in last {minutes} min, {len(grouped)} distinct.",
           f"[OLD] = stopped more than {STALE_MIN} min ago (past incident). [NOW] = still happening."]
    for line, (first, last, n) in grouped.items():
        tag = "NOW" if now - last <= stale_ns else "OLD"
        span = hm(first) if first == last else f"{hm(first)}-{hm(last)}"
        out.append(f"  [{tag}] {span} x{n} {line[:200]}")
    if not any_now:
        out.append(f"  SUMMARY: all of these are OLD. Nothing matching is happening now for app={app}.")
    return _cap("\n".join(out))


if __name__ == "__main__":
    print(fetch_health(), end="\n\n")
    for name in ("targets_up", "error_rate", "error_ratio_pct", "restarts_15m"):
        print(fetch_metrics(name), end="\n\n")
    print(fetch_logs("product-service", minutes=30))
