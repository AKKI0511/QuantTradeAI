from __future__ import annotations

from pathlib import Path
from typing import Mapping, MutableMapping, Set

import pytest
import yaml
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from quanttradeai.streaming.adapters.example_provider import ExampleStreamingProvider
from quanttradeai.streaming.providers.config import (
    ProviderConfigValidator,
    ProviderConfigurationError,
)
from quanttradeai.streaming.providers.base import ProviderCapabilities


def _write_config(path: Path, data: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle)


asset_strategy = st.sets(st.sampled_from(["stocks", "crypto"]), min_size=1, max_size=2)
data_strategy = st.sets(
    st.sampled_from(["trades", "quotes", "order_book"]), min_size=1, max_size=3
)
rate_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=1200))
subscription_strategy = st.one_of(st.none(), st.integers(min_value=1, max_value=500))


@given(
    assets=asset_strategy,
    data_types=data_strategy,
    rate_limit=rate_strategy,
    max_subscriptions=subscription_strategy,
)
@settings(
    max_examples=25,
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
    ],
)
def test_provider_config_validator_property(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    assets: Set[str],
    data_types: Set[str],
    rate_limit: int | None,
    max_subscriptions: int | None,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_data = {
        "provider": "example",
        "environment": "dev",
        "environments": {
            "dev": {
                "asset_types": sorted(assets),
                "data_types": sorted(data_types),
                "rate_limit_per_minute": rate_limit,
                "max_subscriptions": max_subscriptions,
                "credentials": {"token": "${API_TOKEN}"},
                "options": {"mode": "paper"},
            }
        },
    }
    _write_config(config_path, config_data)
    monkeypatch.setenv("API_TOKEN", "secret-token")

    validator = ProviderConfigValidator()
    model = validator.load_from_path(config_path, environment="dev")
    adapter = ExampleStreamingProvider()
    runtime = validator.validate(adapter, model, environment="dev")

    assert runtime.provider == "example"
    assert runtime.asset_types == set(assets)
    assert runtime.data_types == set(data_types)
    assert runtime.credentials["token"] == "secret-token"
    if rate_limit is not None:
        assert runtime.rate_limit_per_minute == rate_limit
    if max_subscriptions is not None:
        assert runtime.max_subscriptions == max_subscriptions


def test_provider_config_validator_rejects_unsupported_assets(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_data = {
        "provider": "example",
        "environments": {
            "dev": {
                "asset_types": ["forex"],
                "data_types": ["trades"],
            }
        },
    }
    _write_config(config_path, config_data)

    validator = ProviderConfigValidator()
    model = validator.load_from_path(config_path, environment="dev")
    adapter = ExampleStreamingProvider()
    with pytest.raises(ProviderConfigurationError):
        validator.validate(adapter, model, environment="dev")


class AuthRequiredStreamingProvider(ExampleStreamingProvider):
    """Adapter that advertises authentication as a hard requirement."""

    def get_capabilities(self) -> ProviderCapabilities:
        base = super().get_capabilities()
        return ProviderCapabilities(
            asset_types=base.asset_types,
            data_types=base.data_types,
            max_subscriptions=base.max_subscriptions,
            rate_limit_per_minute=base.rate_limit_per_minute,
            requires_authentication=True,
            supports_order_book=base.supports_order_book,
            metadata=base.metadata,
        )


def test_validator_rejects_disabling_required_auth_via_env(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_data = {
        "provider": "example",
        "environment": "dev",
        "environments": {
            "dev": {
                "asset_types": ["stocks"],
                "data_types": ["trades"],
                "requires_authentication": False,
                "credentials": {},
            }
        },
    }
    _write_config(config_path, config_data)

    validator = ProviderConfigValidator()
    model = validator.load_from_path(config_path, environment="dev")
    adapter = AuthRequiredStreamingProvider()

    with pytest.raises(ProviderConfigurationError):
        validator.validate(adapter, model, environment="dev")


class AuthOverrideStreamingProvider(AuthRequiredStreamingProvider):
    """Adapter that attempts to disable authentication via normalization."""

    def validate_config(self, config: Mapping[str, object]) -> MutableMapping[str, object]:
        normalized = super().validate_config(config)
        normalized["requires_authentication"] = False
        return normalized


def test_validator_rejects_disabling_required_auth_via_normalization(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_data = {
        "provider": "example",
        "environment": "dev",
        "environments": {
            "dev": {
                "asset_types": ["stocks"],
                "data_types": ["trades"],
                "credentials": {"token": "abc"},
            }
        },
    }
    _write_config(config_path, config_data)

    validator = ProviderConfigValidator()
    model = validator.load_from_path(config_path, environment="dev")
    adapter = AuthOverrideStreamingProvider()

    with pytest.raises(ProviderConfigurationError):
        validator.validate(adapter, model, environment="dev")
