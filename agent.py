import os
import time
import json
import requests
from datetime import datetime, timezone
from typing import TypedDict, Annotated

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ── Config ─────────────────────────────────────────────────────────────────────
AIRFLOW_BASE_URL = os.getenv("AIRFLOW_BASE_URL", "http://127.0.0.1:8080")
AIRFLOW_USER     = os.getenv("AIRFLOW_USER", "airflow")
AIRFLOW_PASS     = os.getenv("AIRFLOW_PASS", "airflow")
LLM_BASE_URL     = os.getenv("LLM_BASE_URL", "http://0.0.0.0:8000")
POLL_INTERVAL    = int(os.getenv("POLL_INTERVAL_SECONDS", "120"))
CHROMA_DIR       = os.getenv("CHROMA_DIR", "./chroma_db")
INCIDENT_FILE    = "./data/incidents_structured.json"

# ── State ──────────────────────────────────────────────────────────────────────
# class AgentState(TypedDict):
#     messages:      Annotated[list, add_messages]
#     dag_id:        str
#     dag_run_id:    str
#     failed_tasks:  list[dict]   # [{task_id, error_log}]
#     context:       str
#     analysis:      str

class AgentState(TypedDict):
    messages:      Annotated[list, add_messages]
    dag_id:        str
    dag_run_id:    str
    failed_tasks:  list[dict]
    context:       str
    analysis:      str
    parsed:        dict        # ← add this

# ── LLM + Retriever Setup ──────────────────────────────────────────────────────
llm = ChatOpenAI(
    base_url=LLM_BASE_URL,
    api_key="none",
    model="smollm",
    temperature=0.1,    # low — we want deterministic analysis, not creativity
    max_tokens=512,
)

embeddings = HuggingFaceEmbeddings(
    model_name=os.path.join(os.path.dirname(os.path.abspath(__file__)), "all-MiniLM-L6-v2")
)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1}
)

# ── Airflow API Helpers ────────────────────────────────────────────────────────
session = requests.Session()
session.auth = (AIRFLOW_USER, AIRFLOW_PASS)
session.headers.update({"Content-Type": "application/json"})

def get_failed_dag_runs() -> list[dict]:
    """Fetch all DAG runs currently in 'failed' state."""
    url = f"{AIRFLOW_BASE_URL}/api/v1/dags/~/dagRuns"
    params = {"state": "failed", "limit": 25, "order_by": "-start_date"}
    resp = session.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("dag_runs", [])

def get_failed_tasks(dag_id: str, dag_run_id: str) -> list[dict]:
    """Fetch task instances that failed within a DAG run."""
    url = f"{AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}/dagRuns/{dag_run_id}/taskInstances"
    params = {"state": "failed"}
    resp = session.get(url, params=params)
    resp.raise_for_status()
    return resp.json().get("task_instances", [])

def get_task_log(dag_id: str, dag_run_id: str, task_id: str, attempt: int = 1) -> str:
    """Fetch the log for a specific task instance."""
    url = (
        f"{AIRFLOW_BASE_URL}/api/v1/dags/{dag_id}"
        f"/dagRuns/{dag_run_id}/taskInstances/{task_id}/logs/{attempt}"
    )
    resp = session.get(url)
    resp.raise_for_status()
    log = resp.text

    # Trim log — tiny model has limited context, keep the tail (most relevant)
    lines = log.splitlines()
    return "\n".join(lines[-20:])  # last 20 lines — 2048-token context is tight

# ── Nodes ──────────────────────────────────────────────────────────────────────
def retrieve_node(state: AgentState) -> dict:
    """
    Build a retrieval query from dag_id + task_id + error snippet,
    fetch relevant knowledge base chunks.
    """
    dag_id = state["dag_id"]
    failed_tasks = state["failed_tasks"]

    # Use first failed task's error as retrieval signal
    # (if multiple tasks failed, they often share root cause)
    first_task = failed_tasks[0] if failed_tasks else {}
    task_id    = first_task.get("task_id", "")
    error_log  = first_task.get("error_log", "")

    # Extract just the last error lines for retrieval query
    error_snippet = "\n".join(error_log.splitlines()[-5:])
    query = f"DAG {dag_id} task {task_id} error: {error_snippet}"

    docs    = retriever.invoke(query)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    return {"context": context}


def analyze_node(state: AgentState) -> dict:
    dag_id       = state["dag_id"]
    dag_run_id   = state["dag_run_id"]
    failed_tasks = state["failed_tasks"]
    context      = state["context"]

    MAX_LOG_CHARS = 1500  # 2048 ctx - 512 response - ~400 prompt overhead ≈ 1100 tokens
    task_details = ""
    for t in failed_tasks:
        entry = f"\n### Task: {t['task_id']}\nLog:\n{t['error_log']}\n"
        if len(task_details) + len(entry) > MAX_LOG_CHARS:
            task_details += f"\n### Task: {t['task_id']}\n[truncated — context budget exhausted]\n"
            break
        task_details += entry

    system_prompt = f"""You are an Airflow pipeline debugger.
Use ONLY the context below. If context does not help, reason from the log directly.

Knowledge Base Context:
{context}

Rate your CONFIDENCE based on how well the log error matches the knowledge base context:
- High: error directly matches a known pattern in the context
- Medium: partial match or related pattern found
- Low: no match — analysis inferred from log alone

Based on CONFIDENCE and SEVERITY, output one ACTION keyword:
- ALERT_TEAM      : High confidence + High/Critical severity
- NOTIFY_TEAM     : High confidence + Low/Medium severity
- REVIEW_REQUIRED : Medium confidence, any severity
- ADD_KNOWLEDGE   : Low confidence — knowledge base gap detected
- MONITOR_ONLY    : High confidence + Low severity, no action needed
"""

    user_prompt = f"""DAG '{dag_id}' (run: {dag_run_id}) has failed.

Failed Tasks:
{task_details}

Respond EXACTLY in this format, no extra text:
ROOT CAUSE: <one sentence>
AFFECTED TASK(S): <comma separated>
RECOMMENDED FIX: <max 3 bullet points>
SEVERITY: <Low|Medium|High|Critical>
CONFIDENCE: <High|Medium|Low>
ACTION: <ALERT_TEAM|NOTIFY_TEAM|REVIEW_REQUIRED|ADD_KNOWLEDGE|MONITOR_ONLY>
"""

    messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
    response = llm.invoke(messages)

    # Parse into structured dict
    parsed = parse_analysis(response.content)

    return {
        "analysis": response.content,   # raw text, kept for logging
        "parsed":   parsed              # structured dict for action routing
    }

def parse_analysis(raw: str) -> dict:
    result = {
        "root_cause":     "Unknown",
        "affected_tasks": "Unknown",
        "recommended_fix": "Unknown",
        "severity":       "Unknown",
        "confidence":     "Unknown",
        "action":         "ADD_KNOWLEDGE",  # safe default if parsing fails
    }

    field_map = {
        "ROOT CAUSE":       "root_cause",
        "AFFECTED TASK(S)": "affected_tasks",
        "RECOMMENDED FIX":  "recommended_fix",
        "SEVERITY":         "severity",
        "CONFIDENCE":       "confidence",
        "ACTION":           "action",
    }

    valid_actions = {
        "ALERT_TEAM", "NOTIFY_TEAM", "REVIEW_REQUIRED",
        "ADD_KNOWLEDGE", "MONITOR_ONLY"
    }

    for line in raw.splitlines():
        for key, field in field_map.items():
            if line.startswith(f"{key}:"):
                result[field] = line.replace(f"{key}:", "").strip()

    # Validate action — tiny model might output something unexpected
    if result["action"] not in valid_actions:
        result["action"] = "REVIEW_REQUIRED"  # safe fallback

    return result

def output_node(state: AgentState) -> dict:
    dag_id     = state["dag_id"]
    dag_run_id = state["dag_run_id"]
    parsed     = state["parsed"]
    analysis   = state["analysis"]

    action     = parsed["action"]
    severity   = parsed["severity"]
    confidence = parsed["confidence"]

    # Print to stdout (kubectl logs)
    print("\n" + "="*60)
    print(f"DAG: {dag_id} | Run: {dag_run_id}")
    print(f"SEVERITY: {severity} | CONFIDENCE: {confidence} | ACTION: {action}")
    print(analysis)

    # ── Action routing ──────────────────────────────────────────
    if action == "ALERT_TEAM":
        _alert_team(dag_id, dag_run_id, parsed)

    elif action == "NOTIFY_TEAM":
        _notify_team(dag_id, dag_run_id, parsed)

    elif action == "REVIEW_REQUIRED":
        _flag_for_review(dag_id, dag_run_id, parsed)

    elif action == "ADD_KNOWLEDGE":
        _append_to_gap_report(state)

    elif action == "MONITOR_ONLY":
        print(f"[MONITOR_ONLY] No action taken, low severity confirmed.")

    print("="*60 + "\n")

    # Always append to incident log
    first_task = state["failed_tasks"][0] if state["failed_tasks"] else {}
    incident = {
        "date":          datetime.now(timezone.utc).isoformat(),
        "dag_id":        dag_id,
        "dag_run_id":    dag_run_id,
        "task_id":       first_task.get("task_id", ""),
        "error_summary": "\n".join(first_task.get("error_log", "").splitlines()[-5:]),
        "root_cause":    parsed.get("root_cause", ""),
        "severity":      severity,
        "confidence":    confidence,
        "action":        action,
        "resolution":    "pending",
    }
    _append_incident(incident)

    return {}


# ── Action Handlers ─────────────────────────────────────────────────────────

def _alert_team(dag_id, dag_run_id, parsed):
    """High severity — needs immediate attention."""
    print(f"[ALERT_TEAM] Critical failure on {dag_id}, notifying on-call...")
    # TODO: plug in PagerDuty / Slack @channel / email here

def _notify_team(dag_id, dag_run_id, parsed):
    """Normal notification — post to Slack channel."""
    print(f"[NOTIFY_TEAM] Posting analysis to Slack for {dag_id}...")
    # TODO: plug in Slack webhook here

def _flag_for_review(dag_id, dag_run_id, parsed):
    """Medium confidence — human should verify the analysis."""
    print(f"[REVIEW_REQUIRED] Analysis uncertain for {dag_id}, flagging for human review...")
    # TODO: create Jira ticket or post to review channel

def _append_to_gap_report(state: AgentState):
    """Low confidence — knowledge base gap, needs new document."""
    gap = {
        "date":          datetime.now(timezone.utc).isoformat(),
        "dag_id":        state["dag_id"],
        "task_id":       state["failed_tasks"][0].get("task_id", ""),
        "error_snippet": "\n".join(
            state["failed_tasks"][0].get("error_log", "").splitlines()[-5:]
        ),
        "llm_analysis":  state["analysis"],
        "action_needed": "add_document_then_reingest"
    }
    gap_file = "./data/gap_report.json"
    existing = []
    if os.path.exists(gap_file):
        with open(gap_file) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(gap)
    with open(gap_file, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"[ADD_KNOWLEDGE] Gap logged to {gap_file}")

def _append_incident(incident: dict):
    existing = []
    if os.path.exists(INCIDENT_FILE):
        with open(INCIDENT_FILE) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    existing.append(incident)
    with open(INCIDENT_FILE, "w") as f:
        json.dump(existing, f, indent=2)


# ── Graph ──────────────────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("analyze",  analyze_node)
    graph.add_node("output",   output_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "analyze")
    graph.add_edge("analyze",  "output")
    graph.add_edge("output",   END)

    return graph.compile()


# ── Polling Loop ───────────────────────────────────────────────────────────────
def run_agent():
    app = build_graph()
    seen = set()   # track already-processed run IDs within this session

    print(f"Agent started. Polling Airflow every {POLL_INTERVAL}s...")

    while True:
        try:
            failed_runs = get_failed_dag_runs()

            for run in failed_runs:
                dag_id     = run["dag_id"]
                dag_run_id = run["dag_run_id"]
                key        = f"{dag_id}:{dag_run_id}"

                if key in seen:
                    continue  # already processed this run

                print(f"New failure detected: {key}")
                seen.add(key)

                # Fetch failed tasks + their logs
                failed_task_instances = get_failed_tasks(dag_id, dag_run_id)
                failed_tasks = []
                for ti in failed_task_instances:
                    task_id = ti["task_id"]
                    try:
                        log = get_task_log(dag_id, dag_run_id, task_id)
                    except Exception as e:
                        log = f"Could not fetch log: {e}"
                    failed_tasks.append({"task_id": task_id, "error_log": log})

                if not failed_tasks:
                    print(f"No failed tasks found for {key}, skipping.")
                    continue

                # Run the LangGraph agent
                app.invoke({
                    "messages":     [HumanMessage(content=f"Analyze failure: {key}")],
                    "dag_id":       dag_id,
                    "dag_run_id":   dag_run_id,
                    "failed_tasks": failed_tasks,
                    "context":      "",
                    "analysis":     "",
                })

        except Exception as e:
            print(f"[ERROR] Polling error: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_agent()
