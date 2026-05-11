from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever


@dataclass
class AgentResponse:
    content: str
    trace: list[str]


class BaseStudyAgent:
    """Shared helper for the educational agents.

    The same workflow can run locally or on AMD Developer Cloud. In production,
    pair ROCm-enabled AMD GPUs with hosted inference endpoints for faster,
    scalable generation across many student sessions.
    """

    name = "Base Study Agent"

    def __init__(self, llm: BaseChatModel, retriever: Optional[BaseRetriever] = None):
        self.llm = llm
        self.retriever = retriever

    def _retrieve_context(self, query: str) -> tuple[str, list[str]]:
        if not self.retriever:
            return "", ["No retriever attached; using direct model reasoning."]

        docs: Iterable[Document] = self.retriever.invoke(query)
        context_blocks = []
        sources = []
        for index, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source", "uploaded notes")
            page = doc.metadata.get("page")
            label = f"{source}" + (f", page {page + 1}" if isinstance(page, int) else "")
            sources.append(f"Context chunk {index}: {label}")
            context_blocks.append(f"[Chunk {index} | {label}]\n{doc.page_content}")

        return "\n\n".join(context_blocks), sources

    def _run_prompt(self, system_prompt: str, user_prompt: str, context_query: str) -> AgentResponse:
        context, trace = self._retrieve_context(context_query)
        return self._run_with_context(system_prompt, user_prompt, context, trace)

    def _run_with_context(
        self,
        system_prompt: str,
        user_prompt: str,
        context: str,
        trace: Sequence[str],
    ) -> AgentResponse:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Student request:\n{request}\n\nRetrieved notes context:\n{context}",
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        content = chain.invoke({"request": user_prompt, "context": context or "No notes context available."})
        return AgentResponse(content=content, trace=[f"{self.name} started retrieval."] + trace)

    @staticmethod
    def _context_from_chunks(chunks: Sequence[Document], limit: int = 8) -> tuple[str, list[str]]:
        selected = list(chunks[:limit])
        context_blocks = []
        trace = []
        for index, doc in enumerate(selected, start=1):
            source = doc.metadata.get("source", "uploaded notes")
            page = doc.metadata.get("page")
            label = f"{source}" + (f", page {page + 1}" if isinstance(page, int) else "")
            trace.append(f"Document chunk {index}: {label}")
            context_blocks.append(f"[Chunk {index} | {label}]\n{doc.page_content}")
        return "\n\n".join(context_blocks), trace


class NotesAnalysisAgent(BaseStudyAgent):
    name = "Notes Analysis Agent"

    def run(self, focus: str, chunks: Optional[Sequence[Document]] = None) -> AgentResponse:
        system_prompt = """
        You are the Notes Analysis Agent for engineering exam preparation.
        Create a concise, high-yield summary from the uploaded notes.
        Include:
        - crisp chapter summary
        - important definitions/formulas/concepts
        - likely exam topics
        - common mistakes
        - 5 quick revision bullets
        Stay grounded in the retrieved notes. If something is missing, say so.
        """
        if chunks:
            context, trace = self._context_from_chunks(chunks, limit=10)
            return self._run_with_context(
                system_prompt,
                f"Summarize this focus area: {focus}",
                context,
                trace,
            )
        return self._run_prompt(system_prompt, f"Summarize this focus area: {focus}", focus)


class DoubtSolverAgent(BaseStudyAgent):
    name = "Doubt Solver Agent"

    def run(self, question: str) -> AgentResponse:
        system_prompt = """
        You are the Doubt Solver Agent. Answer engineering student doubts using
        the uploaded notes as primary evidence.
        Explain step by step, use equations or examples when useful, and end
        with a short exam-tip. If the notes do not contain enough information,
        clearly label any outside reasoning as general knowledge.
        """
        return self._run_prompt(system_prompt, question, question)


class QuizGeneratorAgent(BaseStudyAgent):
    name = "Quiz Generator Agent"

    def run(self, topic: str, count: int, chunks: Optional[Sequence[Document]] = None) -> AgentResponse:
        system_prompt = """
        You are the Quiz Generator Agent. Generate exam-style engineering MCQs
        from the uploaded notes.
        Return ONLY valid JSON. Do not use Markdown.
        Use this exact schema:
        {{
          "questions": [
            {{
              "question": "Question text",
              "options": {{"A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}},
              "answer": "A",
              "explanation": "Short explanation grounded in the notes"
            }}
          ]
        }}
        Generate exactly the requested number of questions. Mix conceptual,
        numerical, and application-style questions when possible.
        """
        request = f"Generate {count} MCQs on: {topic}"
        if chunks:
            context, trace = self._context_from_chunks(chunks, limit=8)
            return self._run_with_context(system_prompt, request, context, trace)
        return self._run_prompt(system_prompt, request, topic)


class VivaAgent(BaseStudyAgent):
    name = "Viva Preparation Agent"

    def run(
        self,
        topic: str,
        difficulty: str,
        chunks: Optional[Sequence[Document]] = None,
    ) -> AgentResponse:
        system_prompt = """
        You are the Viva Preparation Agent. Create oral exam and interview-style
        questions with model answers from the uploaded notes.
        Include follow-up probes, concise answers, and what the examiner is
        testing. Keep answers spoken and easy to rehearse.
        """
        request = f"Create {difficulty.lower()} viva preparation for: {topic}"
        if chunks:
            context, trace = self._context_from_chunks(chunks, limit=8)
            return self._run_with_context(system_prompt, request, context, trace)
        return self._run_prompt(system_prompt, request, topic)


class PlannerAgent:
    name = "Study Planner Agent"

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def run(self, exam_date: str, topics: str, daily_hours: int) -> AgentResponse:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are the Study Planner Agent for engineering students.
                    Build a practical timetable with topic sequencing, revision,
                    quiz practice, viva practice, and buffer time.
                    Prefer focused daily blocks over unrealistic marathon plans.
                    Mention how the plan can scale on AMD Developer Cloud for
                    AI-assisted batch quiz generation and accelerated inference.
                    """,
                ),
                (
                    "human",
                    """
                    Exam date: {exam_date}
                    Available study hours per day: {daily_hours}
                    Subjects/topics:
                    {topics}

                    Create a personalized study timetable in Markdown.
                    """,
                ),
            ]
        )
        chain = prompt | self.llm | StrOutputParser()
        content = chain.invoke(
            {"exam_date": exam_date, "daily_hours": daily_hours, "topics": topics}
        )
        return AgentResponse(
            content=content,
            trace=[
                "Study Planner Agent parsed exam date, workload, and available hours.",
                "Planner balanced learning, revision, quiz practice, viva rehearsal, and buffers.",
            ],
        )
