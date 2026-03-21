"""LiteLLM-backed agent decision strategy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from litellm import completion

from quanttradeai.utils.project_paths import resolve_project_path

from .base import AgentDecision, BaseStrategy


DEFAULT_API_KEY_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
}


class LLMAgentStrategy(BaseStrategy):
    """Backtest-time LLM strategy with a fixed JSON contract."""

    def __init__(
        self,
        *,
        project_config_path: str | Path,
        llm_config: dict[str, Any],
    ) -> None:
        self.project_config_path = Path(project_config_path)
        self.provider = str(llm_config.get("provider") or "").strip()
        self.model = str(llm_config.get("model") or "").strip()
        self.prompt_path = resolve_project_path(
            self.project_config_path,
            str(llm_config.get("prompt_file") or ""),
        )
        self.extra = dict(llm_config.get("extra") or {})
        self.api_key_env_var = llm_config.get(
            "api_key_env_var"
        ) or DEFAULT_API_KEY_ENV_VARS.get(self.provider)
        if not self.provider or not self.model:
            raise ValueError("Agent llm.provider and llm.model are required.")
        if not self.prompt_path.is_file():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_path}")
        self.prompt_template = self.prompt_path.read_text(encoding="utf-8").strip()
        self.api_key = self._load_api_key()

    def _load_api_key(self) -> str | None:
        if not self.api_key_env_var:
            return None
        api_key = os.environ.get(self.api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable '{self.api_key_env_var}' is not set."
            )
        return api_key

    @staticmethod
    def _extract_json_payload(content: str) -> dict[str, Any]:
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 3:
                cleaned = "\n".join(lines[1:-1]).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1:
            cleaned = cleaned[start : end + 1]
        payload = json.loads(cleaned)
        action = payload.get("action")
        if action not in {"buy", "sell", "hold"}:
            raise ValueError("LLM response action must be one of buy, sell, or hold.")
        reason = payload.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("LLM response must include a non-empty reason.")
        return {
            "action": action,
            "reason": reason.strip(),
        }

    def _build_messages(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> list[dict[str, str]]:
        prompt_payload = {
            "agent_name": agent_name,
            "symbol": symbol,
            "timestamp": str(timestamp),
            "tools": tools,
            "context": context,
        }
        contract = (
            "Return JSON only with this exact shape: "
            '{"action":"buy|sell|hold","reason":"brief explanation"}'
        )
        user_prompt = (
            f"{self.prompt_template}\n\n"
            f"{contract}\n\n"
            "Trading context:\n"
            f"{json.dumps(prompt_payload, indent=2, default=str)}"
        )
        return [
            {
                "role": "system",
                "content": "You are a disciplined trading agent. Respond with JSON only.",
            },
            {"role": "user", "content": user_prompt},
        ]

    def decide(
        self,
        *,
        agent_name: str,
        symbol: str,
        timestamp: Any,
        context: dict[str, Any],
        tools: list[str],
    ) -> AgentDecision:
        messages = self._build_messages(
            agent_name=agent_name,
            symbol=symbol,
            timestamp=timestamp,
            context=context,
            tools=tools,
        )
        model_name = (
            f"{self.provider}/{self.model}" if "/" not in self.model else self.model
        )
        response = completion(
            model=model_name,
            messages=messages,
            api_key=self.api_key,
            **self.extra,
        )
        raw_content = response["choices"][0]["message"]["content"]
        parsed = self._extract_json_payload(raw_content)
        return AgentDecision(
            action=parsed["action"],
            reason=parsed["reason"],
            prompt_payload={"messages": messages},
            response_payload=parsed,
            raw_response=raw_content,
        )
