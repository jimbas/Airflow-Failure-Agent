# AI Agent for Airflow Failure Monitor

An LLM-powered agent that polls Apache Airflow for failed DAG runs, retrieves relevant context from a vector knowledge base, and routes incidents to the appropriate action (alert, notify, review, or log).

## How It Works

1. **Poll** — queries Airflow API every `POLL_INTERVAL_SECONDS` for failed DAG runs
2. **Retrieve** — embeds the error log and fetches similar past incidents from ChromaDB
3. **Analyze** — sends context + logs to a local LLM, which outputs root cause, severity, confidence, and a recommended action
4. **Route** — dispatches based on the action keyword:
   - `ALERT_TEAM` — high severity, high confidence (PagerDuty / Slack @channel)
   - `NOTIFY_TEAM` — low/medium severity, high confidence (Slack notification)
   - `REVIEW_REQUIRED` — medium confidence, human verification needed
   - `ADD_KNOWLEDGE` — low confidence, gap logged for KB ingestion
   - `MONITOR_ONLY` — low severity, no action taken

## Requirements

- Docker
- A running Airflow instance (REST API v1)
- A local LLM server compatible with the OpenAI API (e.g. llama.cpp)

## Configuration

All settings are passed via environment variables:

| Variable | Default | Description |
|---|---|---|
| `AIRFLOW_BASE_URL` | `http://127.0.0.1:8080` | Airflow base URL |
| `AIRFLOW_USER` | `airflow` | Airflow API username |
| `AIRFLOW_PASS` | `airflow` | Airflow API password |
| `LLM_BASE_URL` | `http://0.0.0.0:8000` | Local LLM server URL |
| `POLL_INTERVAL_SECONDS` | `120` | Polling interval in seconds |
| `CHROMA_DIR` | `./chroma_db` | ChromaDB persistence directory |

## Usage

### Build

```bash
docker build -t afaiagent .
```

### Run

```bash
docker run --rm \
  -e AIRFLOW_BASE_URL=http://your-airflow:8080 \
  -e AIRFLOW_USER=your_user \
  -e AIRFLOW_PASS=your_pass \
  -e LLM_BASE_URL=http://your-llm:8000 \
  -v $(pwd)/chroma_db:/app/chroma_db \
  -v $(pwd)/data:/app/data \
  afaiagent
```

## Data Files

| Path | Description |
|---|---|
| `./data/incidents_structured.json` | Append-only incident log |
| `./data/gap_report.json` | Knowledge base gaps flagged by the agent |
| `./chroma_db/` | ChromaDB vector store |

## Models

The sentence embedding model (`all-MiniLM-L6-v2`) is downloaded automatically during the Docker build into `/app/all-MiniLM-L6-v2`.
