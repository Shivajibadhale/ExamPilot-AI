# AI Engineering Exam Copilot

A modern Streamlit MVP for engineering students. Upload PDF notes and use a set
of LangChain-powered agents for contextual doubt solving, chapter summaries,
MCQ generation, viva preparation, and personalized study planning.

The project is designed to demonstrate agentic educational AI workflows that can
scale on AMD Developer Cloud. In a production deployment, ROCm-enabled AMD GPUs
can accelerate inference and batch generation for quizzes, summaries, and
student-specific study plans.

## Features

- PDF upload and text extraction with PyPDF
- Chunking and semantic retrieval with LangChain and FAISS
- Gemini or OpenAI model provider selection
- Doubt Solver Agent for note-grounded answers
- Summary Generator Agent for high-yield revision
- Quiz Generator Agent with MCQs, answers, and explanations
- Viva Preparation Agent with oral exam questions and model answers
- Study Planner Agent for personalized timetables
- Autonomous Plan -> Act -> Observe reasoning agent foundation
- SQL/PLSQL execution tool with validation, raw database error feedback, and retry-ready observations
- Dark responsive Streamlit interface

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Add either `GOOGLE_API_KEY` or `OPENAI_API_KEY` to `.env`. The app defaults to
`MODEL_PROVIDER=Gemini` and `MODEL_NAME=gemini-2.5-flash-lite` for lower quota
usage. Keep `SHOW_LOCAL_CONFIG=false` for hackathon/demo mode so the UI looks
like a polished product instead of an internal configuration screen.

For Streamlit Cloud or other hosted demos, add secrets in the platform settings
instead of committing real API keys:

```toml
GOOGLE_API_KEY = "your-key"
MODEL_PROVIDER = "Gemini"
MODEL_NAME = "gemini-2.5-flash-lite"
TEMPERATURE = "0.25"
SHOW_LOCAL_CONFIG = "false"
```

If you want local API/model controls while developing, set:

```bash
SHOW_LOCAL_CONFIG=true
```

## Run

```bash
.\.venv\Scripts\python.exe -m streamlit run app.py
```

On Windows, you can also run:

```bash
run_app.bat
```

For hackathon judging, the **Reasoning Agent** page can run without any API key
because it uses a local SQLite demo to show autonomous SQL error correction.

## Autonomous Reasoning Agent

The project also includes a test-ready orchestration script at
`src/reasoning_orchestrator.py`. It implements an async Plan -> Act -> Observe
loop with a pluggable model interface, tool registry, structured JSON console
logs, and a database tool for dynamic SQL/PLSQL execution.

Run the deterministic demo:

```bash
python src/reasoning_orchestrator.py --demo
```

The demo creates an in-memory SQLite database, intentionally runs a query with a
bad column name, records the exact database traceback as an observation, then
retries with corrected SQL. This validates the autonomous error-correction path
without requiring a live production database.

To connect a real relational database, provide `DBAPIAsyncClient` with a
connection factory for your driver, then register `SQLExecutionTool`:

```python
from src.reasoning_orchestrator import (
    AutonomousReasoningAgent,
    DBAPIAsyncClient,
    SQLExecutionTool,
    ToolRegistry,
    configure_logging,
)

logger = configure_logging()
registry = ToolRegistry(logger)

db_client = DBAPIAsyncClient(lambda: your_driver.connect(...))
registry.register(SQLExecutionTool(db_client, logger, allow_commit=False))
```

For tests, mock the `ReasoningModel` and `DatabaseClient` protocols directly.
The SQL tool returns normalized `ToolResult` objects, including `raw_error`
content when validation or database execution fails.

## Project Structure

```text
app.py                    # Streamlit app entrypoint
src/agents.py             # LangChain agent classes
src/document_processor.py # PDF extraction, chunking, FAISS indexing
src/llm_provider.py       # Gemini/OpenAI model factories
src/reasoning_orchestrator.py # Async reasoning loop and SQL/PLSQL tool
src/ui.py                 # Theme and reusable UI helpers
```

## Notes

This MVP uses `faiss-cpu` for local development. For AMD GPU deployments, connect
the LangChain model layer to ROCm-compatible inference services running on AMD
Developer Cloud.


git add .
git commit -m "Save local changes"
git pull origin main --rebase
git push origin main