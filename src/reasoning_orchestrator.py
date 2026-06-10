"""Autonomous Plan -> Act -> Observe reasoning agent with SQL tool support.

This module is intentionally framework-agnostic while following the same
orchestration shape used by Python agent frameworks such as Microsoft Agent
Framework, Semantic Kernel, LangGraph, and LangChain. The key integration points
are simple protocols:

- ``ReasoningModel`` plans the next action from conversation state.
- ``Tool`` instances provide external capabilities.
- ``DatabaseClient`` executes SQL/PLSQL using a DB-API compatible connection.

The design makes each part straightforward to mock in pytest.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sqlite3
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable, Iterable, Mapping, Protocol, Sequence


JsonDict = dict[str, Any]


class AgentEvent(str, Enum):
    RUN_STARTED = "run_started"
    PLAN_CREATED = "plan_created"
    TOOL_SELECTED = "tool_selected"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    OBSERVATION_RECORDED = "observation_recorded"
    RETRY_REQUESTED = "retry_requested"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"


class ToolError(Exception):
    """Raised by tools when execution fails in a way the agent can observe."""


class ValidationError(ToolError):
    """Raised when a tool input payload fails validation."""


@dataclass(frozen=True)
class ToolResult:
    """Normalized result returned to the reasoning loop."""

    ok: bool
    tool_name: str
    output: Any = None
    error: str | None = None
    raw_error: str | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_observation(self) -> JsonDict:
        return {
            "tool": self.tool_name,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "raw_error": self.raw_error,
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class AgentStep:
    """A single model-planned step."""

    thought: str
    action: str
    action_input: JsonDict


@dataclass(frozen=True)
class AgentPlan:
    """The model's next decision."""

    thought: str
    steps: list[AgentStep]
    final_answer: str | None = None


@dataclass
class AgentState:
    task: str
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    iterations: int = 0
    observations: list[JsonDict] = field(default_factory=list)
    transcript: list[JsonDict] = field(default_factory=list)


@dataclass(frozen=True)
class AgentRunResult:
    """Full result returned by detailed agent runs for UI/testing."""

    final_answer: str
    state: AgentState


class ReasoningModel(Protocol):
    """Minimal async interface expected from any LLM/chat model adapter."""

    async def create_plan(self, state: AgentState, tools: Sequence["Tool"]) -> AgentPlan:
        ...


class Tool(Protocol):
    name: str
    description: str

    async def execute(self, payload: JsonDict) -> ToolResult:
        ...


class DatabaseClient(Protocol):
    """Async database client abstraction used by SQLExecutionTool."""

    async def execute(
        self,
        query: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
        *,
        fetch: bool,
        commit: bool,
    ) -> JsonDict:
        ...


class JsonFormatter(logging.Formatter):
    """Console formatter that emits structured JSON log lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key.startswith("_agent_"):
                payload[key.removeprefix("_agent_")] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("reasoning_agent")
    logger.setLevel(level)
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    return logger


class ToolRegistry:
    def __init__(self, logger: logging.Logger):
        self._tools: dict[str, Tool] = {}
        self._logger = logger

    def register(self, tool: Tool) -> None:
        if not tool.name or not tool.name.strip():
            raise ValueError("Tool name must be non-empty.")
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")
        self._tools[tool.name] = tool
        self._logger.info(
            "Registered tool",
            extra={"_agent_tool": tool.name, "_agent_description": tool.description},
        )

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._tools))
            raise ToolError(f"Unknown tool '{name}'. Available tools: {available}") from exc

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())


class DBAPIAsyncClient:
    """Async wrapper for a DB-API 2.0 connection factory.

    The blocking DB-API work is moved to a thread with ``asyncio.to_thread``.
    For Oracle, PostgreSQL, SQL Server, MySQL, or SQLite, pass a factory that
    returns a fresh connection. Fresh connections avoid cross-thread connection
    reuse surprises and make test fixtures deterministic.
    """

    def __init__(self, connection_factory: Callable[[], Any]):
        self._connection_factory = connection_factory

    async def execute(
        self,
        query: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None = None,
        *,
        fetch: bool,
        commit: bool,
    ) -> JsonDict:
        return await asyncio.to_thread(
            self._execute_sync,
            query,
            parameters,
            fetch=fetch,
            commit=commit,
        )

    def _execute_sync(
        self,
        query: str,
        parameters: Mapping[str, Any] | Sequence[Any] | None,
        *,
        fetch: bool,
        commit: bool,
    ) -> JsonDict:
        connection = self._connection_factory()
        cursor = None
        try:
            cursor = connection.cursor()
            if parameters is None:
                cursor.execute(query)
            else:
                cursor.execute(query, parameters)

            rows: list[JsonDict] = []
            columns: list[str] = []
            if fetch and cursor.description:
                columns = [col[0] for col in cursor.description]
                rows = [dict(zip(columns, row)) for row in cursor.fetchall()]

            rowcount = getattr(cursor, "rowcount", None)
            if commit:
                connection.commit()
            else:
                rollback = getattr(connection, "rollback", None)
                if callable(rollback):
                    rollback()

            return {
                "columns": columns,
                "rows": rows,
                "rowcount": rowcount,
                "committed": commit,
            }
        finally:
            if cursor is not None:
                cursor.close()
            connection.close()


class SQLExecutionTool:
    """Execute dynamic SQL or PLSQL against a relational database."""

    name = "sql_plsql_executor"
    description = (
        "Executes validated SQL or PLSQL against a relational database. "
        "Input JSON: query, parameters, fetch, commit, max_rows."
    )

    def __init__(
        self,
        db_client: DatabaseClient,
        logger: logging.Logger,
        *,
        max_query_chars: int = 20_000,
        default_max_rows: int = 200,
        allow_commit: bool = False,
    ):
        self._db_client = db_client
        self._logger = logger
        self._max_query_chars = max_query_chars
        self._default_max_rows = default_max_rows
        self._allow_commit = allow_commit

    async def execute(self, payload: JsonDict) -> ToolResult:
        started_at = datetime.now(timezone.utc)
        try:
            query, parameters, fetch, commit, max_rows = self._validate_payload(payload)
        except ValidationError as exc:
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=str(exc),
                raw_error=traceback.format_exc(),
                metadata={"stage": "validation"},
            )

        self._logger.info(
            "SQL tool executing query",
            extra={
                "_agent_tool": self.name,
                "_agent_query": query,
                "_agent_parameters": parameters,
                "_agent_fetch": fetch,
                "_agent_commit": commit,
            },
        )

        try:
            result = await self._db_client.execute(
                query,
                parameters,
                fetch=fetch,
                commit=commit,
            )
            rows = result.get("rows", [])
            truncated = False
            if isinstance(rows, list) and len(rows) > max_rows:
                result["rows"] = rows[:max_rows]
                truncated = True

            elapsed_ms = self._elapsed_ms(started_at)
            metadata = {
                "elapsed_ms": elapsed_ms,
                "rowcount": result.get("rowcount"),
                "truncated": truncated,
                "max_rows": max_rows,
            }
            return ToolResult(
                ok=True,
                tool_name=self.name,
                output=result,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - exact DB errors must be observable.
            raw_error = traceback.format_exc()
            error_payload = {
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": raw_error,
                "failed_query": query,
                "parameters": parameters,
            }
            self._logger.error(
                "SQL tool failed",
                extra={
                    "_agent_tool": self.name,
                    "_agent_error": error_payload,
                },
            )
            return ToolResult(
                ok=False,
                tool_name=self.name,
                error=f"{type(exc).__name__}: {exc}",
                raw_error=json.dumps(error_payload, default=str),
                metadata={
                    "stage": "database_execution",
                    "elapsed_ms": self._elapsed_ms(started_at),
                },
            )

    def _validate_payload(
        self,
        payload: JsonDict,
    ) -> tuple[str, Mapping[str, Any] | Sequence[Any] | None, bool, bool, int]:
        if not isinstance(payload, dict):
            raise ValidationError("Tool input must be a JSON object.")

        query = payload.get("query")
        if not isinstance(query, str):
            raise ValidationError("'query' must be a string.")
        query = query.strip()
        if not query:
            raise ValidationError("'query' must not be empty.")
        if len(query) > self._max_query_chars:
            raise ValidationError(
                f"'query' exceeds max length of {self._max_query_chars} characters."
            )

        parameters = payload.get("parameters")
        if parameters is not None and not isinstance(parameters, (dict, list, tuple)):
            raise ValidationError("'parameters' must be an object, array, or null.")

        fetch = payload.get("fetch", True)
        if not isinstance(fetch, bool):
            raise ValidationError("'fetch' must be a boolean.")

        commit = payload.get("commit", False)
        if not isinstance(commit, bool):
            raise ValidationError("'commit' must be a boolean.")
        if commit and not self._allow_commit:
            raise ValidationError(
                "Commits are disabled for this SQL tool instance. "
                "Set allow_commit=True only for trusted workflows."
            )

        max_rows = payload.get("max_rows", self._default_max_rows)
        if not isinstance(max_rows, int) or max_rows < 1 or max_rows > 10_000:
            raise ValidationError("'max_rows' must be an integer from 1 to 10000.")

        return query, parameters, fetch, commit, max_rows

    @staticmethod
    def _elapsed_ms(started_at: datetime) -> int:
        return int((datetime.now(timezone.utc) - started_at).total_seconds() * 1000)


class AutonomousReasoningAgent:
    """Plan -> Act -> Observe loop with autonomous retry context."""

    def __init__(
        self,
        model: ReasoningModel,
        tools: ToolRegistry,
        logger: logging.Logger,
        *,
        max_iterations: int = 8,
        stop_on_tool_error: bool = False,
    ):
        self._model = model
        self._tools = tools
        self._logger = logger
        self._max_iterations = max_iterations
        self._stop_on_tool_error = stop_on_tool_error

    async def run(self, task: str) -> str:
        result = await self.run_detailed(task)
        return result.final_answer

    async def run_detailed(self, task: str) -> AgentRunResult:
        state = AgentState(task=task)
        self._log(AgentEvent.RUN_STARTED, state, task=task)

        while state.iterations < self._max_iterations:
            state.iterations += 1
            plan = await self._model.create_plan(state, self._tools.list_tools())
            state.transcript.append(
                {
                    "iteration": state.iterations,
                    "thought": plan.thought,
                    "steps": [step.__dict__ for step in plan.steps],
                    "final_answer": plan.final_answer,
                }
            )
            self._log(
                AgentEvent.PLAN_CREATED,
                state,
                thought=plan.thought,
                steps=[step.__dict__ for step in plan.steps],
                final_answer=plan.final_answer,
            )

            if plan.final_answer:
                self._log(AgentEvent.RUN_COMPLETED, state, final_answer=plan.final_answer)
                return AgentRunResult(final_answer=plan.final_answer, state=state)

            if not plan.steps:
                state.observations.append(
                    {
                        "ok": False,
                        "error": "Model returned no steps and no final answer.",
                    }
                )
                self._log(AgentEvent.RETRY_REQUESTED, state, reason="empty_plan")
                continue

            for step in plan.steps:
                self._log(
                    AgentEvent.TOOL_SELECTED,
                    state,
                    thought=step.thought,
                    tool=step.action,
                    action_input=step.action_input,
                )
                try:
                    tool = self._tools.get(step.action)
                    self._log(
                        AgentEvent.TOOL_STARTED,
                        state,
                        tool=tool.name,
                        action_input=step.action_input,
                    )
                    result = await tool.execute(step.action_input)
                except Exception as exc:  # noqa: BLE001 - loop must observe all failures.
                    result = ToolResult(
                        ok=False,
                        tool_name=step.action,
                        error=f"{type(exc).__name__}: {exc}",
                        raw_error=traceback.format_exc(),
                        metadata={"stage": "tool_dispatch"},
                    )

                observation = result.to_observation()
                state.observations.append(observation)
                self._log(
                    AgentEvent.TOOL_COMPLETED if result.ok else AgentEvent.TOOL_FAILED,
                    state,
                    tool=result.tool_name,
                    observation=observation,
                )
                self._log(AgentEvent.OBSERVATION_RECORDED, state, observation=observation)

                if not result.ok:
                    self._log(
                        AgentEvent.RETRY_REQUESTED,
                        state,
                        reason="tool_error",
                        raw_error=result.raw_error,
                    )
                    if self._stop_on_tool_error:
                        return AgentRunResult(final_answer=self._failure_summary(state), state=state)
                    break

        final = self._failure_summary(state)
        self._log(AgentEvent.RUN_FAILED, state, final_answer=final)
        return AgentRunResult(final_answer=final, state=state)

    def _failure_summary(self, state: AgentState) -> str:
        return (
            "Agent stopped before producing a final answer. "
            f"Iterations: {state.iterations}. "
            f"Last observation: {state.observations[-1] if state.observations else 'none'}"
        )

    def _log(self, event: AgentEvent, state: AgentState, **fields: Any) -> None:
        self._logger.info(
            event.value,
            extra={
                "_agent_event": event.value,
                "_agent_run_id": state.run_id,
                "_agent_iteration": state.iterations,
                **{f"_agent_{key}": value for key, value in fields.items()},
            },
        )


class JsonReasoningModel:
    """Adapter for chat models that can return a JSON plan.

    The injected callable receives a list of chat-style messages and returns the
    model text. This keeps the orchestration independent from any specific SDK.
    """

    def __init__(self, complete: Callable[[list[JsonDict]], Awaitable[str]]):
        self._complete = complete

    async def create_plan(self, state: AgentState, tools: Sequence[Tool]) -> AgentPlan:
        response = await self._complete(self._messages(state, tools))
        return self._parse_plan(response)

    def _messages(self, state: AgentState, tools: Sequence[Tool]) -> list[JsonDict]:
        tool_descriptions = [
            {"name": tool.name, "description": tool.description} for tool in tools
        ]
        return [
            {
                "role": "system",
                "content": (
                    "You are an autonomous database reasoning agent. Use a strict "
                    "Plan -> Act -> Observe loop. Return ONLY JSON matching: "
                    '{"thought": str, "steps": [{"thought": str, "action": str, '
                    '"action_input": object}], "final_answer": str|null}. '
                    "When an observation contains raw database errors, rewrite the "
                    "query and retry with corrected SQL/PLSQL."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": state.task,
                        "tools": tool_descriptions,
                        "observations": state.observations,
                        "transcript": state.transcript[-5:],
                    },
                    default=str,
                ),
            },
        ]

    @staticmethod
    def _parse_plan(response: str) -> AgentPlan:
        try:
            payload = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Model did not return valid JSON: {response}") from exc

        steps = [
            AgentStep(
                thought=str(step.get("thought", "")),
                action=str(step["action"]),
                action_input=dict(step.get("action_input", {})),
            )
            for step in payload.get("steps", [])
        ]
        return AgentPlan(
            thought=str(payload.get("thought", "")),
            steps=steps,
            final_answer=payload.get("final_answer"),
        )


class ScriptedDemoModel:
    """Small deterministic model used for local smoke tests and examples.

    It deliberately emits one broken SQL query first. The second iteration sees
    the database error in observations and corrects the query autonomously.
    """

    async def create_plan(self, state: AgentState, tools: Sequence[Tool]) -> AgentPlan:
        del tools
        if not state.observations:
            return AgentPlan(
                thought="I need to inspect users, but I may need to adapt to schema errors.",
                steps=[
                    AgentStep(
                        thought="Try the requested query first.",
                        action="sql_plsql_executor",
                        action_input={
                            "query": "SELECT id, full_name FROM users",
                            "fetch": True,
                            "max_rows": 50,
                        },
                    )
                ],
            )

        last = state.observations[-1]
        if not last.get("ok"):
            return AgentPlan(
                thought=(
                    "The database reported the selected column does not exist. "
                    "I will retry using the actual column name."
                ),
                steps=[
                    AgentStep(
                        thought="Rewrite failed SQL using the correct 'name' column.",
                        action="sql_plsql_executor",
                        action_input={
                            "query": "SELECT id, name FROM users",
                            "fetch": True,
                            "max_rows": 50,
                        },
                    )
                ],
            )

        return AgentPlan(
            thought="The corrected SQL succeeded; summarize the raw output.",
            steps=[],
            final_answer=f"Query succeeded. Raw output: {json.dumps(last.get('output'), default=str)}",
        )


def build_demo_sqlite_client() -> DBAPIAsyncClient:
    """Create an in-memory SQLite demo DB for the scripted smoke test."""

    database_uri = "file:reasoning_agent_demo?mode=memory&cache=shared"
    keeper = sqlite3.connect(database_uri, uri=True, check_same_thread=False)
    keeper.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, name TEXT)")
    keeper.execute("DELETE FROM users")
    keeper.executemany("INSERT INTO users (name) VALUES (?)", [("Ada",), ("Grace",)])
    keeper.commit()

    def factory() -> sqlite3.Connection:
        return sqlite3.connect(database_uri, uri=True, check_same_thread=False)

    client = DBAPIAsyncClient(factory)
    client._demo_keeper = keeper  # type: ignore[attr-defined]
    return client


async def run_demo() -> str:
    result = await run_demo_detailed()
    return result.final_answer


async def run_demo_detailed() -> AgentRunResult:
    logger = configure_logging()
    registry = ToolRegistry(logger)
    registry.register(SQLExecutionTool(build_demo_sqlite_client(), logger))
    agent = AutonomousReasoningAgent(
        model=ScriptedDemoModel(),
        tools=registry,
        logger=logger,
        max_iterations=4,
    )
    return await agent.run_detailed("List all users with their identifiers.")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the autonomous reasoning agent demo.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic SQLite demo that shows SQL error correction.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.demo:
        parser.print_help()
        return 0

    final_answer = asyncio.run(run_demo())
    print(final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
