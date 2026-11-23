"""Kira: a small read-only AIOps agent.  Usage: python agent.py "your question" """
import re
import sys
import time

import ollama
import requests

from tools import LOKI, NAMESPACE, apps_api, core, fetch_health, fetch_logs, fetch_metrics

MODEL = "qwen2.5:7b"
MAX_STEPS = 6
OPTIONS = {"temperature": 0, "num_ctx": 16384}
TOOLS = {f.__name__: f for f in (fetch_health, fetch_metrics, fetch_logs)}
SENSITIVE = re.compile(r"(?i)pass|secret|token|key")  # never print these env values

# Deterministic triage: the CODE always gathers these, the model only interprets them
TRIAGE = [
    (fetch_health, {}),
    (fetch_metrics, {"check": "targets_up"}),
    (fetch_metrics, {"check": "error_rate"}),
    (fetch_metrics, {"check": "restarts_15m"}),
    (fetch_metrics, {"check": "db_ready_replicas"}),
]

SYSTEM = """You are Kira, an SRE assistant for the boutique microservices app on Kubernetes.
The Kubernetes namespace is exactly "boutique". You can only READ the system through
your tools. You cannot change anything.

Service dependencies (caller -> what it needs):
- frontend -> gateway
- gateway -> auth, product-service, orders, order-service, user-service
- auth, product-service, orders, order-service, user-service -> boutique-postgres (the database)
The ROOT CAUSE is the deepest broken dependency. If the database is down, the database is
the root cause, even when other services crash or return errors because of it.

You receive a TRIAGE SNAPSHOT. It starts with KEY FINDINGS computed by rules, ordered with
the most fundamental problem first. Trust their order.
- If KEY FINDINGS is "none", all targets are up and there are no 5xx errors now, the app is
  healthy. Answer STATUS: healthy without calling tools.
- BAD ROLLOUT means a service stopped working after a change; the change shown is the
  likely cause. Quote the change and the LAST LOGS line that shows the error.
- CRASHING findings include the pod's LAST LOGS. Use them to say WHY it crashes.
- If a service returns errors and the cause is not clear yet, call fetch_logs for it.
- Log lines tagged [OLD] are past incidents. Only [NOW] lines describe the current state.

Rules:
- Only state facts that appear in the snapshot or tool output. Copy names exactly.
- Ignore events marked OLD or about pods that no longer exist.
- The ROOT CAUSE must name a cause (what is broken and why), not repeat a symptom.
- Never claim you fixed anything, and do not suggest commands: the system adds runbook steps.

Final answer format (plain text):
STATUS: healthy | degraded | down
WHAT IS HAPPENING NOW: one or two sentences
ROOT CAUSE: one sentence naming the cause, or 'none found'
EVIDENCE: bullet points quoting the snapshot or tool output
"""

POD_NAME = re.compile(r"\b[a-z0-9-]+-[a-z0-9]{8,10}-[a-z0-9]{5}\b")  # e.g. orders-64d687644b-r99gg
NS_FLAG = re.compile(r"(?:-n|--namespace)[ =]([\w.-]+)")
NO_FINDINGS = "KEY FINDINGS (rule-based, most fundamental first):\n  none"


def runbook(kind: str, name: str) -> list[str]:
    """Tested fix commands per finding type. Code chooses them; a human approves them."""
    ns = f"-n {NAMESPACE}"
    return {
        "db_scaled_to_zero": [f"kubectl {ns} scale statefulset/{name} --replicas=1"],
        "db_not_ready": [f"kubectl {ns} describe pod {name}-0", f"kubectl {ns} logs {name}-0 --tail=50"],
        "scaled_to_zero": [f"kubectl {ns} scale deploy/{name} --replicas=1",
                           f"kubectl {ns} get events --sort-by=.lastTimestamp | grep {name}"],
        "bad_rollout": [f"kubectl {ns} rollout undo deploy/{name}",
                        f"kubectl {ns} rollout history deploy/{name}"],
        "not_ready": [f"kubectl {ns} describe deploy/{name}"],
        "crashing": [f"kubectl {ns} logs {name} --previous --tail=30"],
    }[kind]


def _revision(rs) -> int:
    return int((rs.metadata.annotations or {}).get("deployment.kubernetes.io/revision", "0"))


def template_diff(old, new) -> list[str]:
    """What changed between two ReplicaSet pod templates (image and env only)."""
    oc, nc = old.spec.template.spec.containers[0], new.spec.template.spec.containers[0]
    changes = []
    if oc.image != nc.image:
        changes.append(f"image {oc.image} -> {nc.image}")
    old_env = {e.name: e.value for e in oc.env or []}
    new_env = {e.name: e.value for e in nc.env or []}
    for key in sorted(set(old_env) | set(new_env)):
        if old_env.get(key) != new_env.get(key):
            if SENSITIVE.search(key):
                changes.append(f"env {key} changed (value hidden)")
            else:
                changes.append(f"env {key}: {old_env.get(key)!r} -> {new_env.get(key)!r}")
    return changes


def last_change(deploy: str, replica_sets) -> str:
    """Diff between the current and previous revision of a deployment, or '' if none."""
    owned = sorted((r for r in replica_sets
                    if any(o.kind == "Deployment" and o.name == deploy for o in (r.metadata.owner_references or []))),
                   key=_revision)
    if len(owned) < 2:
        return ""
    changes = template_diff(owned[-2], owned[-1])
    return f"revision {_revision(owned[-1])}: " + "; ".join(changes) if changes else ""


def _loki_pod_logs(pod: str) -> str:
    now = time.time_ns()
    params = {"query": f'{{namespace="{NAMESPACE}", pod="{pod}"}}', "start": now - 15 * 60 * 10**9,
              "end": now, "limit": 6, "direction": "backward"}
    try:
        r = requests.get(f"{LOKI}/loki/api/v1/query_range", params=params, timeout=10)
        r.raise_for_status()
        lines = [line.strip() for s in r.json()["data"]["result"] for _, line in s["values"] if line.strip()]
        return " | ".join(dict.fromkeys(lines))[-400:]
    except Exception:
        return ""


def last_logs(pod: str, container: str) -> str:
    """Last lines of the crashed container, unfiltered: Kubernetes API first, then Loki."""
    for previous in (True, False):
        try:
            text = core.read_namespaced_pod_log(pod, NAMESPACE, container=container, previous=previous, tail_lines=6)
        except Exception:
            continue
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines and not lines[0].startswith("unable to retrieve container logs"):
            return " | ".join(lines)[-400:]
    return _loki_pod_logs(pod) or "(no logs available)"


def key_findings() -> tuple[str, list[str]]:
    """Rule-based detection ordered by dependency depth. Returns (findings text, runbook steps)."""
    found, steps = [], []

    def add(text: str, kind: str, name: str):
        found.append(text)
        for cmd in runbook(kind, name):
            if cmd not in steps:
                steps.append(cmd)

    for s in apps_api.list_namespaced_stateful_set(NAMESPACE).items:
        name, ready, want = s.metadata.name, s.status.ready_replicas or 0, s.spec.replicas or 0
        if want == 0:
            add(f"DATABASE DOWN: statefulset/{name} is scaled to 0 replicas. Every backend service depends on it.",
                "db_scaled_to_zero", name)
        elif ready < want:
            add(f"DATABASE DOWN: statefulset/{name} has {ready} of {want} replicas ready. "
                "Every backend service depends on it.", "db_not_ready", name)

    replica_sets = apps_api.list_namespaced_replica_set(NAMESPACE).items
    for d in apps_api.list_namespaced_deployment(NAMESPACE).items:
        name, ready, want = d.metadata.name, d.status.ready_replicas or 0, d.spec.replicas or 0
        if want == 0:
            add(f"SCALED TO ZERO: deploy/{name} has 0 desired replicas.", "scaled_to_zero", name)
        elif ready < want:
            change = last_change(name, replica_sets)
            if change:
                add(f"BAD ROLLOUT: deploy/{name} has {ready} of {want} ready after a change ({change}).",
                    "bad_rollout", name)
            else:
                add(f"NOT READY: deploy/{name} has {ready} of {want} replicas ready.", "not_ready", name)

    for p in core.list_namespaced_pod(NAMESPACE).items:
        for cs in p.status.container_statuses or []:
            if cs.restart_count > 0 and not cs.ready:
                add(f"CRASHING: pod {p.metadata.name} restarts={cs.restart_count}, not ready. "
                    f"LAST LOGS: {last_logs(p.metadata.name, cs.name)}", "crashing", p.metadata.name)

    lines = [f"  {i}. {text}" for i, text in enumerate(found, 1)] or ["  none"]
    return "KEY FINDINGS (rule-based, most fundamental first):\n" + "\n".join(lines), steps


def triage() -> tuple[str, list[str]]:
    findings, steps = key_findings()
    parts = [findings]
    for fn, args in TRIAGE:
        try:
            parts.append(fn(**args))
        except Exception as exc:
            parts.append(f"{fn.__name__} failed: {exc}")
    return "\n\n".join(parts), steps


def run(question: str) -> tuple[str, str, list[str]]:
    """Returns (answer, evidence, runbook steps). Evidence = every tool output the model saw."""
    snapshot, steps = triage()
    evidence = [snapshot]
    print("[triage] key findings + health + 4 metric checks collected", file=sys.stderr)
    user = (f"{question}\n\nTRIAGE SNAPSHOT (collected just now):\n{snapshot}\n\n"
            "Investigate further with tools if needed. Then answer in the required format.")
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]

    for step in range(1, MAX_STEPS + 1):
        msg = ollama.chat(model=MODEL, messages=msgs, tools=list(TOOLS.values()), options=OPTIONS).message
        msgs.append(msg)
        if not msg.tool_calls:  # no more tools wanted: this is the answer
            return msg.content, "\n".join(evidence), steps
        for call in msg.tool_calls:
            name, args = call.function.name, call.function.arguments or {}
            print(f"[step {step}] {name}({args})", file=sys.stderr)
            fn = TOOLS.get(name)  # allow-list: only our three tools can run
            if fn is None:
                result = f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}"
            else:
                try:
                    result = fn(**args)
                except Exception as exc:  # bad arguments etc. go back to the model as text
                    result = f"Tool error: {exc}"
            evidence.append(result)
            msgs.append({"role": "tool", "content": result, "tool_name": name})

    msgs.append({"role": "user", "content": "Stop investigating. Give your final answer now in the required format."})
    answer = ollama.chat(model=MODEL, messages=msgs, options=OPTIONS).message.content
    return answer, "\n".join(evidence), steps


def past_issues(evidence: str) -> list[str]:
    """Facts the CODE reports, so the model can't drop or invent them."""
    return list(dict.fromkeys(line.strip() for line in evidence.splitlines() if ", OLD" in line))


def grounding_check(answer: str, evidence: str) -> list[str]:
    """Cheap hallucination guard: flag claims that don't match the evidence."""
    problems = []
    for ns in set(NS_FLAG.findall(answer)):
        if ns != NAMESPACE:
            problems.append(f"wrong namespace in a command: '{ns}' (should be '{NAMESPACE}')")
    for pod in set(POD_NAME.findall(answer)):
        if pod not in evidence:
            problems.append(f"pod name not found in any tool output: {pod}")
    return problems


def rules_see_problem(snapshot: str) -> bool:
    """Independent verdict from the rules: findings, missing targets, or 5xx happening now."""
    if NO_FINDINGS not in snapshot or "MISSING scrape targets" in snapshot:
        return True
    if "error_rate (now" in snapshot:
        block = snapshot.split("error_rate (now", 1)[1].split("\n\n", 1)[0]
        return any(" = " in line and "no traffic" not in line for line in block.splitlines())
    return False


def consistency_check(answer: str, snapshot: str) -> list[str]:
    """Compare the model's STATUS with the rules' verdict; they must agree."""
    match = re.search(r"STATUS:\s*(\w+)", answer)
    status = match.group(1).lower() if match else "missing"
    problem = rules_see_problem(snapshot)
    if status != "healthy" and not problem:
        return [f"model says '{status}' but the rules see no current problem (no findings, all targets up, "
                "no 5xx now). It may be reacting to OLD logs. Verify before acting."]
    if status == "healthy" and problem:
        return ["model says 'healthy' but the rules found a current problem. See KEY FINDINGS."]
    return []


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "Is the boutique app healthy right now?"
    started = time.time()
    answer, evidence, steps = run(question)
    print(answer)

    print("\nRUNBOOK STEPS (from rules; a human decides whether to run them):")
    print("\n".join(f"- {s}" for s in steps) or "- none needed")

    print("\nPAST ISSUES (from the data, not the model):")
    print("\n".join(f"- {p}" for p in past_issues(evidence)) or "- none")

    checks = grounding_check(answer, evidence) + consistency_check(answer, evidence)
    print("\nCHECKS:", "ok (grounding + consistency)" if not checks else "")
    for c in checks:
        print(f"- WARNING: {c}")
    print(f"\n({time.time() - started:.1f}s)", file=sys.stderr)
