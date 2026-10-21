"""Offline checks for the budget settings in main.py. No API keys, no network:
litellm's mock_response fakes the LLM reply. Run: python test_budget.py"""
import asyncio
import os
import time

for k in ("ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY"):
    os.environ.pop(k, None)

from forecasting_tools.ai_models.resource_managers.hard_limit_manager import (
    HardLimitExceededError,
)

import main


def test_no_paid_research_without_asknews() -> None:
    llms = main.budget_llms()
    assert llms["researcher"] == "no_research"
    os.environ["ASKNEWS_CLIENT_ID"], os.environ["ASKNEWS_SECRET"] = "x", "y"
    try:
        assert main.budget_llms()["researcher"] == "asknews/news-summaries"
    finally:
        os.environ.pop("ASKNEWS_CLIENT_ID")
        os.environ.pop("ASKNEWS_SECRET")
    for role in ("default", "summarizer", "parser"):
        assert llms[role].model == main.CHEAP_MODEL


def test_cost_is_tracked_and_cap_stops_calls() -> None:
    llm = main.CappedLlm(model=main.CHEAP_MODEL, mock_response="Probability: 40%")

    async def run() -> float:
        with main.MonetaryCostManager(1e-9) as cap:
            await llm.invoke("hello " * 1000)
            spent = cap.current_usage
            try:
                await llm.invoke("hello")
            except HardLimitExceededError:
                return spent
            raise AssertionError("cap did not stop the second call")

    spent = asyncio.run(run())
    assert spent > 0, "cost tracker read $0, so the per-run cap would never trigger"


def test_requests_are_spaced_under_the_rate_limit() -> None:
    main.MAX_REQUESTS_PER_MINUTE = 600  # 0.1 s apart, so the test stays fast
    llm = main.CappedLlm(model=main.CHEAP_MODEL, mock_response="ok")

    async def run() -> float:
        start = time.monotonic()
        await asyncio.gather(*(llm.invoke("hi") for _ in range(5)))  # like 3 predictions + parsing at once
        return time.monotonic() - start

    elapsed = asyncio.run(run())
    assert elapsed >= 0.4 - 0.02, f"5 concurrent calls took {elapsed:.2f}s; expected >= 0.4s at 600/min"


if __name__ == "__main__":
    test_no_paid_research_without_asknews()
    test_cost_is_tracked_and_cap_stops_calls()
    test_requests_are_spaced_under_the_rate_limit()
    print("ok")
