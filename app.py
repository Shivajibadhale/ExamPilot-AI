import json
import os
import re
from datetime import date

import streamlit as st
from dotenv import load_dotenv

from src.agents import (
    DoubtSolverAgent,
    NotesAnalysisAgent,
    PlannerAgent,
    QuizGeneratorAgent,
    VivaAgent,
)
from src.document_processor import DocumentProcessor
from src.llm_provider import build_chat_model, build_embedding_model
from src.ui import agent_card, inject_custom_css, render_chat_message


APP_TITLE = "AI Engineering Exam Copilot"

load_dotenv()


def init_state() -> None:
    defaults = {
        "vectorstore": None,
        "doc_text": "",
        "doc_chunks": [],
        "doc_name": "",
        "chat_history": [],
        "last_summary": None,
        "last_quiz": None,
        "quiz_answers": {},
        "quiz_submitted": False,
        "last_viva": None,
        "last_plan": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def api_key_for(provider: str) -> str:
    if provider == "Gemini":
        return st.session_state.get("gemini_api_key") or os.getenv("GOOGLE_API_KEY", "")
    return st.session_state.get("openai_api_key") or os.getenv("OPENAI_API_KEY", "")


def render_sidebar() -> tuple[str, str, float]:
    with st.sidebar:
        st.markdown("## Study Console")
        st.session_state.active_agent = st.radio(
            "AI Agent",
            [
                "Doubt Solver",
                "Summaries",
                "Quiz",
                "Viva Prep",
                "Study Planner",
            ],
            label_visibility="collapsed",
        )

        provider = os.getenv("MODEL_PROVIDER", "Gemini")
        model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

        with st.expander("Developer settings"):
            provider = st.selectbox(
                "Model provider",
                ["Gemini", "OpenAI"],
                index=0 if provider == "Gemini" else 1,
            )

            if provider == "Gemini":
                model_name = st.selectbox(
                    "Gemini model",
                    ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"],
                    index=["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"].index(model_name)
                    if model_name in ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro"]
                    else 0,
                )
                st.session_state.gemini_api_key = st.text_input(
                    "Google API key",
                    value=os.getenv("GOOGLE_API_KEY", ""),
                    type="password",
                    help="For demos only. Production apps should keep this in .env or server secrets.",
                )
            else:
                model_name = st.selectbox(
                    "OpenAI model",
                    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    index=0,
                )
                st.session_state.openai_api_key = st.text_input(
                    "OpenAI API key",
                    value=os.getenv("OPENAI_API_KEY", ""),
                    type="password",
                    help="For demos only. Production apps should keep this in .env or server secrets.",
                )

        st.divider()
        temperature = st.slider("Creativity", 0.0, 1.0, float(os.getenv("TEMPERATURE", "0.25")), 0.05)
        st.caption(
            "Optimized for accelerated inference on AMD GPUs and AMD Developer Cloud "
            "with ROCm-ready scalable AI workflows."
        )

    return provider, model_name, float(temperature)


def process_upload(provider: str, api_key: str) -> None:
    uploaded_files = st.file_uploader(
        "Upload engineering notes or PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs. The notes are embedded into a FAISS vector database.",
    )

    if not uploaded_files:
        return

    if not api_key:
        st.warning("Add an API key in the sidebar before processing PDFs.")
        return

    if st.button("Analyze PDFs", type="primary", use_container_width=True):
        with st.spinner("Reading PDFs, chunking notes, and building FAISS embeddings..."):
            try:
                embeddings = build_embedding_model(provider, api_key)
                processor = DocumentProcessor(embeddings)
                result = processor.process(uploaded_files)
            except Exception as exc:
                st.error(
                    "PDF analysis failed while creating embeddings. "
                    "Check that your API key is valid and that the selected provider supports embeddings."
                )
                st.exception(exc)
                return

        st.session_state.vectorstore = result.vectorstore
        st.session_state.doc_text = result.full_text
        st.session_state.doc_chunks = result.chunks
        st.session_state.doc_name = ", ".join(file.name for file in uploaded_files)
        st.success(
            f"Ready: indexed {len(result.chunks)} note chunks from {len(uploaded_files)} PDF(s)."
        )


def build_agents(provider: str, model_name: str, api_key: str, temperature: float):
    llm = build_chat_model(provider, model_name, api_key, temperature)
    retriever = (
        st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
        if st.session_state.vectorstore
        else None
    )
    return {
        "notes": NotesAnalysisAgent(llm, retriever),
        "doubt": DoubtSolverAgent(llm, retriever),
        "quiz": QuizGeneratorAgent(llm, retriever),
        "viva": VivaAgent(llm, retriever),
        "planner": PlannerAgent(llm),
    }


def render_dashboard() -> None:
    st.markdown(
        f"""
        <section class="hero">
            <div>
                <p class="eyebrow">AMD Developer Cloud Ready | ROCm Accelerated AI Workflow</p>
                <h1>{APP_TITLE}</h1>
                <p>
                    A multi-agent study companion for engineering students: upload notes,
                    ask doubts, generate summaries, practice quizzes, prepare viva answers,
                    and build an exam timetable.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        agent_card("Notes Analysis", "Extracts concepts, exam themes, and chapter signals.")
    with col2:
        agent_card("Doubt Solver", "Answers questions using your uploaded notes.")
    with col3:
        agent_card("Quiz Generator", "Creates MCQs with answers and explanations.")
    with col4:
        agent_card("Viva Agent", "Builds oral exam questions and model answers.")
    with col5:
        agent_card("Planner Agent", "Turns topics and exam date into a timetable.")


def require_notes() -> bool:
    if st.session_state.vectorstore:
        return True
    st.info("Upload and analyze at least one PDF first so the agents can use your notes.")
    return False


def run_with_ai_error_boundary(action):
    try:
        return action()
    except Exception as exc:
        message = str(exc)
        if "RESOURCE_EXHAUSTED" in message or "429" in message or "quota" in message.lower():
            st.error(
                "Gemini quota is exhausted for the selected model. "
                "Choose `gemini-2.5-flash-lite`, wait for the retry window, or use another API key/project."
            )
        elif "NOT_FOUND" in message or "404" in message:
            st.error(
                "The selected model is not available for this API key. "
                "Try `gemini-2.5-flash-lite` or check the model list in Google AI Studio."
            )
        else:
            st.error("The AI request failed. Check your API key, model, and network connection.")
        with st.expander("Technical details"):
            st.exception(exc)
        return None


def render_trace(trace: list[str]) -> None:
    if not trace:
        st.caption("No trace available.")
        return
    for step in trace:
        st.markdown(f"- {step}")


def parse_quiz_response(content: str) -> list[dict]:
    cleaned = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence_match:
        cleaned = fence_match.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not json_match:
            return []
        payload = json.loads(json_match.group(0))

    questions = payload.get("questions", []) if isinstance(payload, dict) else []
    valid_questions = []
    for item in questions:
        options = item.get("options", {})
        if not isinstance(options, dict):
            continue
        labels = ["A", "B", "C", "D"]
        if all(label in options for label in labels) and item.get("answer") in labels:
            valid_questions.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "options": {label: str(options[label]).strip() for label in labels},
                    "answer": item["answer"],
                    "explanation": str(item.get("explanation", "")).strip(),
                }
            )
    return valid_questions


def render_interactive_quiz(questions: list[dict]) -> None:
    with st.form("interactive_quiz_form"):
        selected_answers = {}
        for index, item in enumerate(questions, start=1):
            st.markdown(f"**{index}. {item['question']}**")
            labels = ["A", "B", "C", "D"]
            selected = st.radio(
                "Choose one answer",
                labels,
                format_func=lambda label, options=item["options"]: f"{label}. {options[label]}",
                key=f"quiz_answer_{index}",
                label_visibility="collapsed",
            )
            selected_answers[str(index)] = selected

        submitted = st.form_submit_button("Submit Answers", type="primary")

    if submitted:
        st.session_state.quiz_answers = selected_answers
        st.session_state.quiz_submitted = True
        st.rerun()

    if not st.session_state.quiz_submitted:
        return

    score = 0
    st.markdown("### Results")
    for index, item in enumerate(questions, start=1):
        chosen = st.session_state.quiz_answers.get(str(index))
        correct = item["answer"]
        is_correct = chosen == correct
        score += int(is_correct)
        status = "Correct" if is_correct else "Incorrect"
        st.markdown(f"**{index}. {status}**")
        st.markdown(f"Your answer: **{chosen}. {item['options'].get(chosen, '')}**")
        st.markdown(f"Correct answer: **{correct}. {item['options'][correct]}**")
        st.info(item["explanation"])

    st.success(f"Score: {score}/{len(questions)}")


def run_notes_agent(agent: NotesAnalysisAgent) -> None:
    st.subheader("Summary Generator Agent")
    focus = st.text_input("Chapter/topic focus", placeholder="Example: Thermodynamics Unit 2")
    if st.button("Generate Summary", type="primary"):
        if require_notes():
            with st.spinner("Notes Analysis Agent is extracting the high-yield ideas..."):
                response = run_with_ai_error_boundary(
                    lambda: agent.run(
                        focus or "entire uploaded document",
                        st.session_state.doc_chunks,
                    )
                )
                if response:
                    st.session_state.last_summary = response

    if st.session_state.last_summary:
        st.markdown(st.session_state.last_summary.content)
        with st.expander("Agent workflow trace"):
            render_trace(st.session_state.last_summary.trace)


def run_doubt_agent(agent: DoubtSolverAgent) -> None:
    st.subheader("AI Doubt Solver Agent")
    for message in st.session_state.chat_history:
        render_chat_message(message["role"], message["content"])

    question = st.chat_input("Ask a doubt from your uploaded notes")
    if question and require_notes():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.spinner("Doubt Solver Agent is grounding the answer in your notes..."):
            response = run_with_ai_error_boundary(lambda: agent.run(question))
        if not response:
            return
        st.session_state.chat_history.append({"role": "assistant", "content": response.content})
        st.rerun()


def run_quiz_agent(agent: QuizGeneratorAgent) -> None:
    st.subheader("MCQ & Quiz Generator Agent")
    col1, col2 = st.columns([2, 1])
    with col1:
        topic = st.text_input("Quiz topic", placeholder="Example: Digital modulation")
    with col2:
        count = st.number_input("Questions", min_value=3, max_value=15, value=5, step=1)

    if st.button("Generate Quiz", type="primary"):
        if require_notes():
            with st.spinner("Quiz Generator Agent is creating exam-style MCQs..."):
                response = run_with_ai_error_boundary(
                    lambda: agent.run(
                        topic or "most important topics",
                        count,
                        st.session_state.doc_chunks,
                    )
                )
                if response:
                    st.session_state.last_quiz = response
                    st.session_state.quiz_answers = {}
                    st.session_state.quiz_submitted = False

    if st.session_state.last_quiz:
        questions = parse_quiz_response(st.session_state.last_quiz.content)
        if questions:
            render_interactive_quiz(questions)
        else:
            st.warning("The quiz response was not structured correctly. Showing raw output.")
            st.markdown(st.session_state.last_quiz.content)
        with st.expander("Agent workflow trace"):
            render_trace(st.session_state.last_quiz.trace)


def run_viva_agent(agent: VivaAgent) -> None:
    st.subheader("Viva Preparation Agent")
    topic = st.text_input("Viva focus area", placeholder="Example: Control systems stability")
    difficulty = st.select_slider("Difficulty", ["Basic", "Intermediate", "Advanced"])

    if st.button("Generate Viva Practice", type="primary"):
        if require_notes():
            with st.spinner("Viva Agent is preparing oral exam prompts..."):
                response = run_with_ai_error_boundary(
                    lambda: agent.run(
                        topic or "uploaded syllabus",
                        difficulty,
                        st.session_state.doc_chunks,
                    )
                )
                if response:
                    st.session_state.last_viva = response

    if st.session_state.last_viva:
        st.markdown(st.session_state.last_viva.content)
        with st.expander("Agent workflow trace"):
            render_trace(st.session_state.last_viva.trace)


def run_planner_agent(agent: PlannerAgent) -> None:
    st.subheader("Study Planner Agent")
    col1, col2 = st.columns([1, 2])
    with col1:
        exam_day = st.date_input("Exam date", min_value=date.today())
        daily_hours = st.slider("Hours/day", 1, 12, 4)
    with col2:
        topics = st.text_area(
            "Subjects/topics",
            placeholder="Example:\nSignals and Systems: Fourier series, LTI systems\nDBMS: SQL, normalization, transactions",
            height=150,
        )

    if st.button("Create Study Timetable", type="primary"):
        if not topics.strip():
            st.warning("Enter at least one subject or topic.")
            return
        with st.spinner("Planner Agent is balancing revision, practice, and buffers..."):
            response = run_with_ai_error_boundary(
                lambda: agent.run(str(exam_day), topics, daily_hours)
            )
            if response:
                st.session_state.last_plan = response

    if st.session_state.last_plan:
        st.markdown(st.session_state.last_plan.content)
        with st.expander("Agent workflow trace"):
            render_trace(st.session_state.last_plan.trace)


def render_status() -> None:
    if st.session_state.vectorstore:
        st.markdown(
            f"""
            <div class="status-pill">
                Indexed notes: <strong>{st.session_state.doc_name}</strong> |
                chunks: <strong>{len(st.session_state.doc_chunks)}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-pill muted">No notes indexed yet. Upload PDFs to activate RAG agents.</div>',
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="AI", layout="wide")
    inject_custom_css()
    init_state()

    provider, model_name, temperature = render_sidebar()
    api_key = api_key_for(provider)

    render_dashboard()
    render_status()
    process_upload(provider, api_key)

    if not api_key:
        st.warning("Add your API key in `.env` or in Developer settings to start the AI agents.")
        return

    try:
        agents = build_agents(provider, model_name, api_key, temperature)
    except Exception as exc:
        st.error(f"Could not initialize model provider: {exc}")
        return

    active_agent = st.session_state.get("active_agent", "Doubt Solver")
    if active_agent == "Doubt Solver":
        run_doubt_agent(agents["doubt"])
    elif active_agent == "Summaries":
        run_notes_agent(agents["notes"])
    elif active_agent == "Quiz":
        run_quiz_agent(agents["quiz"])
    elif active_agent == "Viva Prep":
        run_viva_agent(agents["viva"])
    elif active_agent == "Study Planner":
        run_planner_agent(agents["planner"])


if __name__ == "__main__":
    main()
