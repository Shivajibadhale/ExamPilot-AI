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
usage. For local demos, the same settings are also available inside the
collapsed Developer settings panel.

## Run

```bash
streamlit run app.py
```

## Project Structure

```text
app.py                    # Streamlit app entrypoint
src/agents.py             # LangChain agent classes
src/document_processor.py # PDF extraction, chunking, FAISS indexing
src/llm_provider.py       # Gemini/OpenAI model factories
src/ui.py                 # Theme and reusable UI helpers
```

## Notes

This MVP uses `faiss-cpu` for local development. For AMD GPU deployments, connect
the LangChain model layer to ROCm-compatible inference services running on AMD
Developer Cloud.
