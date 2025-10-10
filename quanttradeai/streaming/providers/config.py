"""Configuration validation and capability negotiation helpers."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Set

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .base import (
    ProviderCapabilities,
    ProviderConfigurationError,
    StreamingProviderAdapter,
)


@dataclass(frozen=True)
class ProviderRuntimeConfiguration:
    """Runtime configuration delivered to provider adapters."""

    provider: str
    environment: str
    credentials: Dict[str, str]
    options: Dict[str, Any]
    asset_types: Set[str]
    data_types: Set[str]
    rate_limit_per_minute: Optional[int]
    max_subscriptions: Optional[int]
    requires_authentication: bool


class EnvironmentConfig(BaseModel):
    """Environment scoped configuration segment."""

    asset_types: Set[str] = Field(default_factory=set)
    data_types: Set[str] = Field(default_factory=set)
    rate_limit_per_minute: Optional[int] = Field(default=None, ge=1)
    max_subscriptions: Optional[int] = Field(default=None, ge=1)
    requires_authentication: Optional[bool] = None
    credentials: Dict[str, str] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @field_validator("credentials", mode="before")
    @classmethod
    def ensure_credentials_dict(cls, value: Any) -> Dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("credentials must be a mapping")
        return dict(value)


class ProviderConfigModel(BaseModel):
    """Top-level provider configuration schema."""

    provider: str
    environment: Optional[str] = Field(default=None)
    environments: Dict[str, EnvironmentConfig]

    @field_validator("environments")
    @classmethod
    def validate_environments(
        cls, value: Mapping[str, EnvironmentConfig]
    ) -> Mapping[str, EnvironmentConfig]:
        if not value:
            raise ValueError("at least one environment must be defined")
        return value


class ProviderConfigValidator:
    """Validate provider configuration files and negotiate capabilities."""

    def __init__(self, *, env: Optional[Mapping[str, str]] = None) -> None:
        self._env = env or os.environ

    # ------------------------------------------------------------------
    def load_from_path(
        self, path: Path | str, *, environment: Optional[str] = None
    ) -> ProviderConfigModel:
        raw = self._load_file(Path(path))
        try:
            model = ProviderConfigModel.model_validate(raw)
        except ValidationError as exc:  # pragma: no cover - delegated to tests
            raise ProviderConfigurationError(str(exc)) from exc
        if environment is not None:
            return self._select_environment(model, environment)
        return model

    def validate(
        self,
        adapter: StreamingProviderAdapter,
        config: ProviderConfigModel,
        *,
        environment: Optional[str] = None,
    ) -> ProviderRuntimeConfiguration:
        env_name = environment or config.environment or "dev"
        if env_name not in config.environments:
            raise ProviderConfigurationError(
                f"Environment '{env_name}' not available for provider '{config.provider}'"
            )
        env_config = config.environments[env_name]
        if not env_config.enabled:
            raise ProviderConfigurationError(
                f"Environment '{env_name}' is disabled for provider '{config.provider}'"
            )
        capabilities = adapter.get_capabilities()
        resolved_credentials = self._resolve_env(env_config.credentials)
        resolved_options = self._resolve_env(env_config.options)
        base_payload: Dict[str, Any] = {
            "credentials": resolved_credentials,
            "options": resolved_options,
            "asset_types": set(env_config.asset_types),
            "data_types": set(env_config.data_types),
            "rate_limit_per_minute": env_config.rate_limit_per_minute,
            "max_subscriptions": env_config.max_subscriptions,
        }

        normalized = adapter.validate_config(base_payload)
        if not isinstance(normalized, Mapping):
            raise ProviderConfigurationError(
                "validate_config must return a mapping of normalized configuration"
            )

        credentials_value = normalized.get("credentials", resolved_credentials)
        if not isinstance(credentials_value, Mapping):
            raise ProviderConfigurationError("'credentials' must be a mapping")
        credentials = dict(credentials_value)

        options_value = normalized.get("options", resolved_options)
        if not isinstance(options_value, Mapping):
            raise ProviderConfigurationError("'options' must be a mapping")
        options = dict(options_value)

        asset_types = set(normalized.get("asset_types", base_payload["asset_types"]))
        data_types = set(normalized.get("data_types", base_payload["data_types"]))
        rate_limit = normalized.get(
            "rate_limit_per_minute", env_config.rate_limit_per_minute
        )
        if rate_limit is not None and not isinstance(rate_limit, int):
            raise ProviderConfigurationError(
                "'rate_limit_per_minute' must be an integer"
            )
        max_subscriptions = normalized.get(
            "max_subscriptions", env_config.max_subscriptions
        )
        if max_subscriptions is not None and not isinstance(max_subscriptions, int):
            raise ProviderConfigurationError("'max_subscriptions' must be an integer")

        runtime = ProviderRuntimeConfiguration(
            provider=config.provider,
            environment=env_name,
            credentials=credentials,
            options=options,
            asset_types=asset_types,
            data_types=data_types,
            rate_limit_per_minute=rate_limit,
            max_subscriptions=max_subscriptions,
            requires_authentication=(
                env_config.requires_authentication
                if env_config.requires_authentication is not None
                else capabilities.requires_authentication
            ),
        )

        self._validate_capabilities(capabilities, runtime)
        if runtime.requires_authentication and not runtime.credentials:
            raise ProviderConfigurationError(
                f"Provider '{runtime.provider}' requires authentication but no credentials provided"
            )
        return runtime

    # ------------------------------------------------------------------
    def _select_environment(
        self, config: ProviderConfigModel, environment: str
    ) -> ProviderConfigModel:
        if environment not in config.environments:
            raise ProviderConfigurationError(
                f"Environment '{environment}' not found for provider '{config.provider}'"
            )
        config.environment = environment
        return config

    def _load_file(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise ProviderConfigurationError(
                f"Configuration file '{path}' does not exist"
            )
        if path.suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(path.read_text())
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
        else:
            raise ProviderConfigurationError(
                f"Unsupported configuration format: {path.suffix or 'unknown'}"
            )
        if data is None:
            raise ProviderConfigurationError("Configuration file is empty")
        if not isinstance(data, Mapping):
            raise ProviderConfigurationError("Configuration root must be a mapping")
        return dict(data)

    def _resolve_env(self, value: Any) -> Any:
        if isinstance(value, Mapping):
            return {k: self._resolve_env(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self._resolve_env(item) for item in value]
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            key = value[2:-1]
            if key not in self._env:
                raise ProviderConfigurationError(
                    f"Environment variable '{key}' not set"
                )
            return self._env[key]
        return value

    def _validate_capabilities(
        self, capabilities: ProviderCapabilities, runtime: ProviderRuntimeConfiguration
    ) -> None:
        if capabilities.asset_types and not runtime.asset_types.issubset(
            capabilities.asset_types
        ):
            missing = runtime.asset_types.difference(capabilities.asset_types)
            raise ProviderConfigurationError(
                f"Provider '{runtime.provider}' does not support asset types: {sorted(missing)}"
            )
        if capabilities.data_types and not runtime.data_types.issubset(
            capabilities.data_types
        ):
            missing = runtime.data_types.difference(capabilities.data_types)
            raise ProviderConfigurationError(
                f"Provider '{runtime.provider}' does not support data types: {sorted(missing)}"
            )
        if (
            capabilities.max_subscriptions is not None
            and runtime.max_subscriptions is not None
            and runtime.max_subscriptions > capabilities.max_subscriptions
        ):
            raise ProviderConfigurationError(
                f"Requested subscriptions ({runtime.max_subscriptions}) exceed provider limit "
                f"({capabilities.max_subscriptions})"
            )
        if (
            capabilities.rate_limit_per_minute is not None
            and runtime.rate_limit_per_minute is not None
            and runtime.rate_limit_per_minute > capabilities.rate_limit_per_minute
        ):
            raise ProviderConfigurationError(
                "Requested rate limit exceeds provider capability"
            )


__all__ = [
    "ProviderConfigValidator",
    "ProviderRuntimeConfiguration",
    "ProviderConfigModel",
]
