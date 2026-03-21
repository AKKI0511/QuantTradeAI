import pytest

from quanttradeai.agents.llm import LLMAgentStrategy


def test_extract_json_payload_accepts_fenced_json():
    payload = LLMAgentStrategy._extract_json_payload(
        """```json
{"action":"buy","reason":"Momentum confirmed"}
```"""
    )

    assert payload == {
        "action": "buy",
        "reason": "Momentum confirmed",
    }


def test_extract_json_payload_rejects_invalid_action():
    with pytest.raises(ValueError, match="must be one of buy, sell, or hold"):
        LLMAgentStrategy._extract_json_payload(
            '{"action":"close","reason":"Not allowed"}'
        )
