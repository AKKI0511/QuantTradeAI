"""LLM-powered sentiment analysis using LiteLLM.

This module provides :class:`SentimentAnalyzer` which scores free-form text
and returns a sentiment value in the range [-1, 1]. LiteLLM is used as the
sole interface to call different model providers.  Provider, model and API key
are supplied at runtime through configuration and environment variables.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict

from litellm import completion

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Compute sentiment scores via LiteLLM.

    Parameters
    ----------
    provider:
        Name of the LiteLLM provider (e.g. ``openai``, ``anthropic``).
    model:
        Model name for the provider.
    api_key_env_var:
        Environment variable containing the API key.
    extra:
        Optional dictionary of extra parameters passed directly to
        :func:`litellm.completion` (e.g. ``{"base_url": "..."}``).
    """

    def __init__(
        self,
        provider: str,
        model: str,
        api_key_env_var: str,
        extra: Dict[str, Any] | None = None,
    ) -> None:
        if not provider or not model:
            raise ValueError(
                "Both provider and model must be specified for sentiment analysis."
            )

        self.provider = provider
        self.model = model
        self.api_key_env_var = api_key_env_var
        self.extra = extra or {}

        api_key = os.environ.get(api_key_env_var)
        if not api_key:
            raise ValueError(
                f"API key environment variable '{api_key_env_var}' is not set."
            )
        self.api_key = api_key

    def score(self, text: str) -> float:
        """Return a sentiment score for ``text``.

        The prompt asks the model to output only a numeric value between -1 and 1.
        """

        prompt = (
            "Provide a sentiment score from -1 (negative) to 1 (positive) for the "
            "following text. Respond with only the numeric score.\n" + text
        )

        messages = [
            {"role": "user", "content": prompt},
        ]

        model_name = (
            f"{self.provider}/{self.model}" if "/" not in self.model else self.model
        )
        try:
            response = completion(
                model=model_name,
                messages=messages,
                api_key=self.api_key,
                **self.extra,
            )
        except Exception as exc:  # pragma: no cover - network errors are mocked
            logger.error("LiteLLM completion failed: %s", exc)
            raise

        try:
            content = response["choices"][0]["message"]["content"].strip()
            return float(content)
        except (KeyError, ValueError, TypeError) as exc:
            logger.error("Invalid sentiment response: %s", response)
            raise ValueError("Unable to parse sentiment score") from exc
