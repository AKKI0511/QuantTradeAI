"""Deterministic rule-based agent strategies."""

from __future__ import annotations

from typing import Any

from .base import AgentDecision, BaseStrategy


class RuleAgentStrategy(BaseStrategy):
    """Execute a small built-in rule set from canonical agent config."""

    def __init__(self, *, agent_config: dict[str, Any]) -> None:
        rule_cfg = dict(agent_config.get("rule") or {})
        self.agent_name = str(agent_config.get("name") or "rule_agent")
        self.preset = str(rule_cfg.get("preset") or "").strip()
        self.feature_name = str(rule_cfg.get("feature") or "").strip()
        self.buy_below = float(rule_cfg.get("buy_below"))
        self.sell_above = float(rule_cfg.get("sell_above"))

        if self.preset != "rsi_threshold":
            raise ValueError(
                f"Agent '{self.agent_name}' uses unsupported rule preset: {self.preset}"
            )
        if not self.feature_name:
            raise ValueError(
                f"Agent '{self.agent_name}' must define rule.feature for kind=rule."
            )

    def _resolve_feature_value(self, context: dict[str, Any]) -> tuple[str, float]:
        feature_payload = dict(
            (context.get("features") or {}).get(self.feature_name) or {}
        )
        if not feature_payload:
            raise ValueError(
                f"Agent '{self.agent_name}' could not resolve rule feature '{self.feature_name}' from the runtime context."
            )

        scalar_items: list[tuple[str, float]] = []
        for column_name, raw_value in feature_payload.items():
            if raw_value is None or isinstance(raw_value, (dict, list, tuple, set)):
                continue
            try:
                scalar_items.append((str(column_name), float(raw_value)))
            except (TypeError, ValueError):
                continue

        if len(scalar_items) != 1:
            raise ValueError(
                f"Agent '{self.agent_name}' expected exactly one scalar value for rule feature "
                f"'{self.feature_name}', found {len(scalar_items)}."
            )

        return scalar_items[0]

    def decide(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> AgentDecision:
        column_name, feature_value = self._resolve_feature_value(context)

        if feature_value <= self.buy_below:
            action = "buy"
            reason = (
                f"{self.preset}: {column_name}={feature_value:.4f} is at or below "
                f"buy_below={self.buy_below:.4f}"
            )
        elif feature_value >= self.sell_above:
            action = "sell"
            reason = (
                f"{self.preset}: {column_name}={feature_value:.4f} is at or above "
                f"sell_above={self.sell_above:.4f}"
            )
        else:
            action = "hold"
            reason = (
                f"{self.preset}: {column_name}={feature_value:.4f} is between "
                f"buy_below={self.buy_below:.4f} and sell_above={self.sell_above:.4f}"
            )

        return AgentDecision(action=action, reason=reason)
