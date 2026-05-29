"""Tests for ainvoke and async tools — spec §14.3, §24.5.29-31."""

import asyncio
import time

import pytest

from adjacency_agents import (
    DeterministicEngine,
    FinalAnswer,
    ToolCall,
    UserContext,
    tool_node,
)
from adjacency_agents.errors import AsyncRequiredError
from adjacency_agents.llm import FakeLLMClient


def _ctx(*c):
    return UserContext(session_id="s", capabilities=set(c))


@tool_node(requires=["public"])
async def async_hello() -> str:
    await asyncio.sleep(0)
    return "ola async"


@tool_node(requires=["public"])
def slow_sync() -> str:
    time.sleep(0.02)
    return "ola sync"


class TestAinvoke:
    async def test_executes_async_tool(self):
        fake = FakeLLMClient(script=[ToolCall(name="async_hello")])
        eng = DeterministicEngine(llm=fake, tools=[async_hello])
        out = await eng.ainvoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="ola async")

    async def test_sync_tool_runs_via_thread_by_default(self):
        """§14.3.10 — sync_tool_strategy='thread' must not block the loop."""
        fake = FakeLLMClient(script=[ToolCall(name="slow_sync")])
        eng = DeterministicEngine(llm=fake, tools=[slow_sync])

        # Concurrent sleep should overlap with the thread-bound tool.
        start = time.monotonic()
        results = await asyncio.gather(
            eng.ainvoke(prompt="x", context=_ctx("public")),
            asyncio.sleep(0.02),
        )
        elapsed = time.monotonic() - start
        assert results[0] == FinalAnswer(content="ola sync")
        # If the sync tool had blocked the loop the total would be ~0.04+;
        # with threading both 0.02 waits overlap and total stays well below.
        assert elapsed < 0.05


class TestInvokeFromAsync:
    async def test_invoke_inside_loop_raises_async_required(self):
        """§14.3.6 — invoke() must refuse to run inside an active loop."""
        fake = FakeLLMClient(script=[FinalAnswer(content="x")])
        eng = DeterministicEngine(llm=fake, tools=[async_hello])
        with pytest.raises(AsyncRequiredError):
            eng.invoke(prompt="x", context=_ctx("public"))


class TestSyncInvokeStandalone:
    def test_invoke_outside_loop_works(self):
        fake = FakeLLMClient(script=[ToolCall(name="async_hello")])
        eng = DeterministicEngine(llm=fake, tools=[async_hello])
        out = eng.invoke(prompt="x", context=_ctx("public"))
        assert out == FinalAnswer(content="ola async")
