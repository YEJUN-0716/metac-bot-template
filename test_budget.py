"""Offline checks for the budget settings in main.py. No API keys, no network:
litellm's mock_response fakes the LLM reply. Run: python test_budget.py"""
import asyncio
import json
import os
import tempfile
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


def _mock_bot(publish: bool) -> "main.FallTemplateBot2026":
    llms = main.budget_llms("none")
    mock = '{"prediction_in_decimal": 0.4} Probability: 40%'
    for role in ("default", "summarizer", "parser"):
        llms[role] = main.CappedLlm(model=main.CHEAP_MODEL, mock_response=mock)
    bot = main.FallTemplateBot2026(
        research_reports_per_question=1,
        predictions_per_research_report=3,
        publish_reports_to_metaculus=publish,
        skip_previously_forecasted_questions=False,
        llms=llms,
    )
    bot.arm_config = main.arm_config_name(llms, 3)
    return bot


def test_shadow_arm_is_logged_and_shares_the_run_cap() -> None:
    from forecasting_tools import BinaryQuestion

    main.MAX_REQUESTS_PER_MINUTE = 60000
    main.FORECAST_LOG_PATH = os.path.join(tempfile.mkdtemp(), "log.jsonl")
    bot = _mock_bot(publish=False)  # never publish in a test
    bot.shadow_bot = _mock_bot(publish=False)
    question = BinaryQuestion(
        question_text="Will X happen?",
        id_of_post=1,
        id_of_question=2,
        page_url="https://example.invalid/questions/1/",
    )

    async def run() -> float:
        with main.MonetaryCostManager(10) as run_cap:
            reports = await bot.forecast_questions([question], return_exceptions=True)
            assert not isinstance(reports[0], BaseException), reports[0]
            return run_cap.current_usage

    run_total = asyncio.run(run())
    rows = [json.loads(line) for line in open(main.FORECAST_LOG_PATH, encoding="utf-8")]
    assert len(rows) == 1
    row = rows[0]
    assert (row["post_id"], row["question_id"], row["type"]) == (1, 2, "BinaryQuestion")
    main_arm, shadow_arm = row["arms"]
    assert (main_arm["role"], shadow_arm["role"]) == ("main", "shadow")
    assert shadow_arm["published"] is False
    for arm in (main_arm, shadow_arm):
        assert arm["ok"] and arm["final"] == 0.4 and arm["individual"] == [0.4] * 3
        assert arm["config"] == "openai/gpt-6-luna|3x|research=no_research"
    # The run cap sees both arms' spending, not just the submitted one.
    assert abs(run_total - (main_arm["cost_usd"] + shadow_arm["cost_usd"])) < 1e-9
    assert shadow_arm["cost_usd"] > 0


def test_log_keeps_full_distributions() -> None:
    from forecasting_tools import (
        MultipleChoiceQuestion,
        NumericDistribution,
        NumericQuestion,
        Percentile,
        PredictedOptionList,
    )
    from forecasting_tools.data_models.multiple_choice_report import PredictedOption

    mc_q = MultipleChoiceQuestion(question_text="Which?", options=["a", "b"])
    mc = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="a", probability=0.7),
            PredictedOption(option_name="b", probability=0.3),
        ]
    )
    assert main.prediction_to_json(mc, mc_q) == {"a": 0.7, "b": 0.3}

    num_q = NumericQuestion(
        question_text="How many?",
        lower_bound=0,
        upper_bound=100,
        open_lower_bound=False,
        open_upper_bound=True,
    )
    dist = NumericDistribution.from_question(
        [Percentile(percentile=p / 10, value=p * 10) for p in range(1, 10)], num_q
    )
    cdf = main.prediction_to_json(dist, num_q)["cdf"]
    assert len(cdf) == num_q.cdf_size == 201
    assert cdf == sorted(cdf) and cdf[0] == 0.0  # closed lower bound
    assert main.prediction_to_json(0.25, num_q) == 0.25


def test_shadow_setting_rejects_typos() -> None:
    try:
        main.budget_llms("nonee")
    except ValueError:
        return
    raise AssertionError("an unknown research setting should fail at startup")


if __name__ == "__main__":
    test_no_paid_research_without_asknews()
    test_cost_is_tracked_and_cap_stops_calls()
    test_requests_are_spaced_under_the_rate_limit()
    test_shadow_arm_is_logged_and_shares_the_run_cap()
    test_log_keeps_full_distributions()
    test_shadow_setting_rejects_typos()
    print("ok")
