"""DeterministicEngine — spec §14, §15, §16, §22.7.

Implements the deterministic loop:

    1. normalize messages
    2. build allowlist + LLM-visible allowlist
    3. ask the LLM (once) for a ToolCall or FinalAnswer
    4. validate and execute the chosen tool
    5. follow EnrichedPointer transitions deterministically
    6. on Observation/dict/list, do ONE synthesis call with tools disabled
    7. respect max_steps

A single turn is at most one chain (§4.7, §31.1).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from collections.abc import Sequence
from dataclasses import is_dataclass
from typing import Any, Callable, Literal

from pydantic import BaseModel

from adjacency_agents.decorators import ToolNodeSpec
from adjacency_agents.errors import (
    AsyncRequiredError,
    InvalidToolCallError,
    InvalidTransitionError,
    MaxStepsExceededError,
    SynthesisError,
    ToolExecutionError,
    ToolNotAllowedError,
    ToolNotFoundError,
)
from adjacency_agents.models import (
    EnrichedPointer,
    FinalAnswer,
    Message,
    Observation,
    ToolCall,
    UserContext,
)
from adjacency_agents.registry import ToolRegistry
from adjacency_agents.router import (
    build_allowlist,
    build_llm_visible_allowlist,
)
from adjacency_agents.schema import (
    build_json_schema,
    resolve_injected_kwargs,
    validate_full_kwargs,
    validate_kwargs,
)
from adjacency_agents.tracing import ExecutionTrace

ResponseMode = Literal["auto", "final", "synthesize"]
ToolErrorMode = Literal["raise", "final", "synthesize"]
SyncToolStrategy = Literal["thread", "direct"]


SYNTHESIS_INSTRUCTION = (
    "You are producing the final user-facing answer based on a tool "
    "observation. Do not call tools. Do not mention tool names, internal "
    "identifiers, or implementation details. Respond in the user's language."
)


class DeterministicEngine:
    """See §14.1 for the public contract."""

    def __init__(
        self,
        *,
        llm: Any,
        tools: list[Callable[..., Any]],
        max_steps: int = 8,
        default_tool_result_mode: ResponseMode = "auto",
        sync_tool_strategy: SyncToolStrategy = "thread",
        tool_error_mode: ToolErrorMode = "raise",
        default_tool_error_message: str = (
            "I could not complete this operation right now."
        ),
    ) -> None:
        self._llm = llm
        self.registry = ToolRegistry(tools)
        self.max_steps = max_steps
        self.default_tool_result_mode: ResponseMode = default_tool_result_mode
        self.sync_tool_strategy: SyncToolStrategy = sync_tool_strategy
        self.tool_error_mode: ToolErrorMode = tool_error_mode
        self.default_tool_error_message = default_tool_error_message
        self._last_trace: ExecutionTrace | None = None

    # ---- public entry points --------------------------------------

    def invoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            raise AsyncRequiredError(
                "invoke() cannot run inside an active event loop; "
                "use 'await engine.ainvoke(...)' instead."
            )
        return asyncio.run(
            self._ainvoke_impl(
                context=context,
                prompt=prompt,
                messages=messages,
                sync_tool_strategy="direct",
            )
        )

    @property
    def last_trace(self) -> ExecutionTrace | None:
        """Trace from the most recent invocation on this engine instance."""
        return self._last_trace

    async def ainvoke(
        self,
        *,
        context: UserContext,
        prompt: str | None = None,
        messages: Sequence[Message] | None = None,
    ) -> FinalAnswer:
        return await self._ainvoke_impl(
            context=context,
            prompt=prompt,
            messages=messages,
            sync_tool_strategy=self.sync_tool_strategy,
        )

    async def _ainvoke_impl(
        self,
        *,
        context: UserContext,
        prompt: str | None,
        messages: Sequence[Message] | None,
        sync_tool_strategy: SyncToolStrategy,
    ) -> FinalAnswer:
        trace = ExecutionTrace()
        self._last_trace = trace
        conversation = _normalize_conversation(prompt=prompt, messages=messages)

        allowlist = build_allowlist(self.registry, context)
        visible_tools = build_llm_visible_allowlist(allowlist)
        schemas = [build_json_schema(spec) for spec in visible_tools]
        visible_names = {spec.name for spec in visible_tools}
        allowed_names = {spec.name for spec in allowlist}
        trace.record(
            "allowlist_built",
            allowed_tools=sorted(allowed_names),
            visible_tools=sorted(visible_names),
            allowed_count=len(allowed_names),
            visible_count=len(visible_names),
        )

        trace.record(
            "llm_called",
            phase="routing",
            message_count=len(conversation),
            tool_count=len(schemas),
            allow_tool_calls=True,
        )
        model_output = await self._call_llm(
            messages=conversation,
            tools=schemas,
            allow_tool_calls=True,
        )

        if isinstance(model_output, FinalAnswer):
            trace.record("final_answer_returned", source="llm")
            return model_output
        if isinstance(model_output, str):
            trace.record("final_answer_returned", source="llm")
            return FinalAnswer(content=model_output)
        if not isinstance(model_output, ToolCall):
            raise InvalidToolCallError(
                f"LLM returned unsupported type: {type(model_output).__name__}"
            )
        trace.record(
            "tool_call_received",
            tool_name=model_output.name,
            kwarg_names=sorted(model_output.kwargs),
        )

        # §15 — validate the LLM-supplied ToolCall.
        try:
            current_spec = self._resolve_initial_call(
                model_output,
                visible_names=visible_names,
                allowed_names=allowed_names,
            )
        except ToolNotAllowedError:
            trace.record(
                "policy_denied",
                phase="tool_call",
                tool_name=model_output.name,
            )
            raise
        current_kwargs = validate_kwargs(current_spec, model_output.kwargs)
        trace.record(
            "tool_call_validated",
            tool_name=current_spec.name,
            kwarg_names=sorted(current_kwargs),
        )

        steps = 0
        while True:
            steps += 1
            if steps > self.max_steps:
                trace.record(
                    "max_steps_exceeded",
                    max_steps=self.max_steps,
                    attempted_step=steps,
                    tool_name=current_spec.name,
                )
                raise MaxStepsExceededError(
                    f"exceeded max_steps={self.max_steps}"
                )

            # Engine-side validation/injection keeps its specific type
            # (§19.2). It happens *before* the user tool body runs.
            try:
                injected = resolve_injected_kwargs(current_spec, context)
            except Exception:
                trace.record(
                    "context_injection_failed",
                    tool_name=current_spec.name,
                    inject_keys=sorted(current_spec.inject),
                )
                raise
            if current_spec.inject:
                trace.record(
                    "context_injection_resolved",
                    tool_name=current_spec.name,
                    inject_keys=sorted(injected),
                )
            full_kwargs = validate_full_kwargs(
                current_spec, {**current_kwargs, **injected}
            )
            try:
                result = await self._execute_tool_body(
                    current_spec,
                    full_kwargs,
                    sync_tool_strategy=sync_tool_strategy,
                )
            except Exception as exc:
                # §14.9.3 / §35.1 — every exception raised *inside* the
                # user tool body, including AdjacencyAgentsError subclasses,
                # is remapped to ToolExecutionError.
                trace.record(
                    "tool_execution_failed",
                    tool_name=current_spec.name,
                    error_type=type(exc).__name__,
                )
                return await self._handle_tool_runtime_error(
                    exc, current_spec.name, conversation, trace
                )
            trace.record(
                "tool_executed",
                tool_name=current_spec.name,
                result_type=type(result).__name__,
            )

            if isinstance(result, EnrichedPointer):
                from_tool = current_spec.name
                trace.record(
                    "pointer_received",
                    from_tool=from_tool,
                    next_tool=result.next_tool,
                    kwarg_names=sorted(result.kwargs),
                    has_reason=bool(result.reason),
                )
                try:
                    current_spec, current_kwargs = self._resolve_transition(
                        pointer=result,
                        current_spec=current_spec,
                        allowed_names=allowed_names,
                    )
                except InvalidTransitionError:
                    if (
                        self.registry.has(result.next_tool)
                        and result.next_tool in current_spec.structural_neighbors
                        and result.next_tool not in allowed_names
                    ):
                        trace.record(
                            "policy_denied",
                            phase="transition",
                            tool_name=result.next_tool,
                        )
                    raise
                trace.record(
                    "pointer_validated",
                    from_tool=from_tool,
                    next_tool=current_spec.name,
                )
                trace.record(
                    "transition_executed",
                    from_tool=from_tool,
                    to_tool=current_spec.name,
                )
                continue

            response_mode = current_spec.response_mode
            if response_mode == "auto":
                response_mode = self.default_tool_result_mode
            normalized = _normalize_tool_result(result, response_mode=response_mode)

            if isinstance(normalized, FinalAnswer):
                trace.record(
                    "final_answer_returned",
                    source="tool",
                    tool_name=current_spec.name,
                )
                return normalized

            if isinstance(normalized, Observation):
                trace.record(
                    "observation_created",
                    tool_name=current_spec.name,
                    data_type=type(normalized.data).__name__,
                    expose_to_llm=normalized.expose_to_llm,
                    has_summary_hint=normalized.summary_hint is not None,
                )
                return await self._synthesize(
                    conversation=conversation,
                    tool_name=current_spec.name,
                    observation=normalized,
                    trace=trace,
                )

            # Unreachable: _normalize_tool_result returns one of the above.
            raise InvalidToolCallError(
                f"tool {current_spec.name!r} returned unsupported type: "
                f"{type(result).__name__}"
            )

    # ---- internal helpers ----------------------------------------

    async def _call_llm(
        self,
        *,
        messages: Sequence[Message],
        tools: list[dict],
        allow_tool_calls: bool,
    ) -> ToolCall | FinalAnswer | str:
        acomplete = getattr(self._llm, "acomplete", None)
        if acomplete is not None:
            return await acomplete(
                messages=messages,
                tools=tools,
                allow_tool_calls=allow_tool_calls,
            )
        complete = getattr(self._llm, "complete", None)
        if complete is None:
            raise TypeError(
                "llm must expose .complete or .acomplete (§18)"
            )
        return complete(
            messages=messages,
            tools=tools,
            allow_tool_calls=allow_tool_calls,
        )

    def _resolve_initial_call(
        self,
        call: ToolCall,
        *,
        visible_names: set[str],
        allowed_names: set[str],
    ) -> ToolNodeSpec:
        if not self.registry.has(call.name):
            raise ToolNotFoundError(call.name)
        if call.name not in allowed_names:
            raise ToolNotAllowedError(
                f"{call.name!r} is not allowed in this context"
            )
        if call.name not in visible_names:
            # LLM-visible allowlist excludes hidden tools (§14.5.4).
            raise ToolNotAllowedError(
                f"{call.name!r} is not LLM-visible; cannot be invoked by the model"
            )
        return self.registry.get(call.name)

    def _resolve_transition(
        self,
        *,
        pointer: EnrichedPointer,
        current_spec: ToolNodeSpec,
        allowed_names: set[str],
    ) -> tuple[ToolNodeSpec, dict[str, Any]]:
        # §16 — validate target exists, is a declared neighbor, is allowed,
        # and kwargs do not include injected params.
        if not self.registry.has(pointer.next_tool):
            raise InvalidTransitionError(
                f"{current_spec.name!r}: pointer target "
                f"{pointer.next_tool!r} not found"
            )
        if pointer.next_tool not in current_spec.structural_neighbors:
            raise InvalidTransitionError(
                f"{current_spec.name!r}: {pointer.next_tool!r} is not a "
                f"declared structural neighbor"
            )
        if pointer.next_tool not in allowed_names:
            raise InvalidTransitionError(
                f"{pointer.next_tool!r} not allowed in this context"
            )
        target = self.registry.get(pointer.next_tool)
        injected = set(target.inject)
        if injected & set(pointer.kwargs):
            raise InvalidTransitionError(
                f"{target.name!r}: pointer supplied injected args "
                f"{sorted(injected & set(pointer.kwargs))}"
            )
        try:
            kwargs = validate_kwargs(target, pointer.kwargs)
        except InvalidToolCallError as exc:
            raise InvalidTransitionError(str(exc)) from exc
        return target, kwargs

    async def _execute_tool_body(
        self,
        spec: ToolNodeSpec,
        full_kwargs: dict[str, Any],
        *,
        sync_tool_strategy: SyncToolStrategy,
    ) -> Any:
        """Invoke the user-supplied callable. Exceptions here are tool
        runtime failures (§14.9)."""
        if spec.is_coroutine:
            return await spec.fn(**full_kwargs)
        if sync_tool_strategy == "thread":
            return await _run_sync_callable_in_thread(
                lambda: spec.fn(**full_kwargs)
            )
        return spec.fn(**full_kwargs)

    async def _synthesize(
        self,
        *,
        conversation: list[Message],
        tool_name: str,
        observation: Observation,
        trace: ExecutionTrace,
    ) -> FinalAnswer:
        if not observation.expose_to_llm:
            raise SynthesisError(
                "observation marked expose_to_llm=False; no synthesis fallback "
                "configured"
            )
        payload = _safe_json(observation.data)
        synthesis_messages: list[Message] = list(conversation)
        synthesis_messages.append(
            Message(role="tool", content=payload, name=tool_name)
        )
        if observation.summary_hint:
            synthesis_messages.append(
                Message(
                    role="system",
                    content=f"Tool hint: {observation.summary_hint}",
                )
            )
        synthesis_messages.append(
            Message(role="system", content=SYNTHESIS_INSTRUCTION)
        )

        trace.record(
            "synthesis_requested",
            message_count=len(synthesis_messages),
            observation_type=type(observation.data).__name__,
        )
        out = await self._call_llm(
            messages=synthesis_messages,
            tools=[],
            allow_tool_calls=False,
        )
        if isinstance(out, ToolCall):
            raise SynthesisError(
                "LLM returned a ToolCall during synthesis (§14.7.3)"
            )
        if isinstance(out, FinalAnswer):
            trace.record("synthesis_completed", result_type="FinalAnswer")
            trace.record("final_answer_returned", source="synthesis")
            return out
        if isinstance(out, str):
            trace.record("synthesis_completed", result_type="str")
            trace.record("final_answer_returned", source="synthesis")
            return FinalAnswer(content=out)
        raise SynthesisError(
            f"synthesis returned unsupported type: {type(out).__name__}"
        )

    async def _handle_tool_runtime_error(
        self,
        exc: Exception,
        tool_name: str,
        conversation: list[Message],
        trace: ExecutionTrace,
    ) -> FinalAnswer:
        wrapped = ToolExecutionError(
            f"tool {tool_name!r} raised {type(exc).__name__}"
        )
        wrapped.__cause__ = exc

        if self.tool_error_mode == "raise":
            raise wrapped
        if self.tool_error_mode == "final":
            trace.record("final_answer_returned", source="tool_error")
            return FinalAnswer(content=self.default_tool_error_message)
        if self.tool_error_mode == "synthesize":
            sanitized = Observation(
                data={"error": "tool_failure"},
                summary_hint=self.default_tool_error_message,
            )
            # Note: tool_name is NOT exposed to synthesis — the role=tool
            # message uses a neutral name (§35.3).
            return await self._synthesize(
                conversation=conversation,
                tool_name="_error",
                observation=sanitized,
                trace=trace,
            )
        raise wrapped  # pragma: no cover


# ---- helpers ----------------------------------------------------------

def _normalize_conversation(
    *,
    prompt: str | None,
    messages: Sequence[Message] | None,
) -> list[Message]:
    if prompt is None and messages is None:
        raise ValueError("either prompt or messages must be provided")
    convo: list[Message] = list(messages) if messages else []
    if prompt is not None:
        convo.append(Message(role="user", content=prompt))
    return convo


async def _run_sync_callable_in_thread(fn: Callable[[], Any]) -> Any:
    """Run a sync callable without blocking the event loop.

    This deliberately avoids the event loop's default executor. Some embedded
    or sandboxed runtimes can hang during default-executor shutdown, while a
    direct thread plus ``asyncio.Future`` is enough for the MVP contract.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[Any] = loop.create_future()

    def runner() -> None:
        try:
            result = fn()
        except BaseException as exc:
            loop.call_soon_threadsafe(_set_future_exception, future, exc)
        else:
            loop.call_soon_threadsafe(_set_future_result, future, result)

    thread = threading.Thread(
        target=runner,
        name="adjacency-agents-tool",
        daemon=True,
    )
    thread.start()
    try:
        return await future
    finally:
        if not thread.is_alive():
            thread.join(timeout=0)


def _set_future_result(future: asyncio.Future[Any], result: Any) -> None:
    if not future.done():
        future.set_result(result)


def _set_future_exception(
    future: asyncio.Future[Any],
    exc: BaseException,
) -> None:
    if not future.done():
        future.set_exception(exc)


def _normalize_tool_result(
    result: Any, *, response_mode: ResponseMode
) -> FinalAnswer | Observation:
    if isinstance(result, FinalAnswer):
        return result
    if isinstance(result, Observation):
        return result
    if isinstance(result, str):
        if response_mode == "synthesize":
            return Observation(data=result)
        return FinalAnswer(content=result)
    # Structured payloads (dict / list / BaseModel / dataclass) → Observation
    # unless explicitly final (§14.4).
    if response_mode == "final":
        return FinalAnswer(content=_safe_json(result))
    return Observation(data=result)


def _safe_json(value: Any) -> str:
    """Serialize tool output for synthesis without leaking internals."""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    if is_dataclass(value) and not isinstance(value, type):
        return json.dumps(dataclasses.asdict(value), default=str, ensure_ascii=False)
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except TypeError:
        return repr(value)
