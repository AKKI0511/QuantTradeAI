from quanttradeai.agents.rule import RuleAgentStrategy


def _context(features: dict) -> dict:
    return {"features": features}


def test_sma_crossover_rule_buys_when_fast_is_above_slow():
    strategy = RuleAgentStrategy(
        agent_config={
            "name": "sma_trend",
            "rule": {
                "preset": "sma_crossover",
                "fast_feature": "sma_20",
                "slow_feature": "sma_50",
            },
        }
    )

    decision = strategy.decide(
        agent_name="sma_trend",
        symbol="AAPL",
        timestamp="2026-01-01",
        context=_context({"sma_20": {"sma_20": 105.0}, "sma_50": {"sma_50": 100.0}}),
        tools=[],
    )

    assert decision.action == "buy"
    assert "sma_crossover" in decision.reason


def test_sma_crossover_rule_sells_when_fast_is_below_slow():
    strategy = RuleAgentStrategy(
        agent_config={
            "name": "sma_trend",
            "rule": {
                "preset": "sma_crossover",
                "fast_feature": "sma_20",
                "slow_feature": "sma_50",
            },
        }
    )

    decision = strategy.decide(
        agent_name="sma_trend",
        symbol="AAPL",
        timestamp="2026-01-01",
        context=_context({"sma_20": {"sma_20": 99.0}, "sma_50": {"sma_50": 100.0}}),
        tools=[],
    )

    assert decision.action == "sell"


def test_sma_crossover_rule_holds_when_values_are_equal_or_missing():
    strategy = RuleAgentStrategy(
        agent_config={
            "name": "sma_trend",
            "rule": {
                "preset": "sma_crossover",
                "fast_feature": "sma_20",
                "slow_feature": "sma_50",
            },
        }
    )

    equal_decision = strategy.decide(
        agent_name="sma_trend",
        symbol="AAPL",
        timestamp="2026-01-01",
        context=_context({"sma_20": {"sma_20": 100.0}, "sma_50": {"sma_50": 100.0}}),
        tools=[],
    )
    missing_decision = strategy.decide(
        agent_name="sma_trend",
        symbol="AAPL",
        timestamp="2026-01-02",
        context=_context({"sma_20": {"sma_20": 100.0}}),
        tools=[],
    )

    assert equal_decision.action == "hold"
    assert missing_decision.action == "hold"
    assert "missing scalar feature value" in missing_decision.reason


def test_rsi_threshold_rule_behavior_is_unchanged():
    strategy = RuleAgentStrategy(
        agent_config={
            "name": "rsi_reversion",
            "rule": {
                "preset": "rsi_threshold",
                "feature": "rsi_14",
                "buy_below": 30.0,
                "sell_above": 70.0,
            },
        }
    )

    decision = strategy.decide(
        agent_name="rsi_reversion",
        symbol="AAPL",
        timestamp="2026-01-01",
        context=_context({"rsi_14": {"rsi": 25.0}}),
        tools=[],
    )

    assert decision.action == "buy"
