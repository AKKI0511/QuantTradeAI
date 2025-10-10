"""Provider registry management utilities."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, Optional, Sequence, Tuple, Type

from .base import (
    ProviderCapabilities,
    ProviderConfigurationError,
    ProviderDiscoveryError,
    StreamingProviderAdapter,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderMetadata:
    """Metadata describing a discovered provider implementation."""

    name: str
    version: str
    adapter_cls: Type[StreamingProviderAdapter]
    module_name: str
    capabilities: ProviderCapabilities
    description: str = ""
    dependencies: Sequence[str] = ()
    module_path: Optional[Path] = None

    def requirement_key(self) -> Tuple[str, str]:
        return self.name, self.version


class ProviderRegistry:
    """Registry responsible for managing discovered provider adapters."""

    def __init__(self) -> None:
        self._providers: Dict[str, ProviderMetadata] = {}
        self._modules: Dict[str, ModuleType] = {}

    # ------------------------------------------------------------------
    @staticmethod
    def _is_dependency_available(dependency: str) -> bool:
        return importlib.util.find_spec(dependency) is not None

    @staticmethod
    def _compare_versions(version_a: str, version_b: str) -> int:
        try:
            from packaging.version import Version

            a_val = Version(version_a)
            b_val = Version(version_b)
            if a_val == b_val:
                return 0
            return -1 if a_val < b_val else 1
        except Exception:  # pragma: no cover - fallback when packaging missing

            def normalize(version: str) -> Tuple[int, ...]:
                parts = []
                for token in version.replace("-", ".").split("."):
                    try:
                        parts.append(int(token))
                    except ValueError:
                        continue
                return tuple(parts)

            a = normalize(version_a)
            b = normalize(version_b)
            if a == b:
                return 0
            return -1 if a < b else 1

    # ------------------------------------------------------------------
    def register(
        self,
        metadata: ProviderMetadata,
        *,
        module: Optional[ModuleType] = None,
    ) -> None:
        """Register provider metadata, performing dependency validation."""

        missing = [
            dep
            for dep in metadata.dependencies
            if not self._is_dependency_available(dep)
        ]
        if missing:
            raise ProviderConfigurationError(
                f"Provider '{metadata.name}' missing dependencies: {', '.join(missing)}"
            )

        existing = self._providers.get(metadata.name)
        if existing:
            comparison = self._compare_versions(metadata.version, existing.version)
            if comparison < 0:
                logger.warning(
                    "provider_version_ignored",
                    extra={
                        "provider": metadata.name,
                        "version": metadata.version,
                        "active_version": existing.version,
                    },
                )
                return
            if comparison == 0 and metadata.module_name != existing.module_name:
                logger.warning(
                    "provider_conflict",
                    extra={
                        "provider": metadata.name,
                        "module_name": metadata.module_name,
                        "existing_module_name": existing.module_name,
                    },
                )
                return

        self._providers[metadata.name] = metadata
        if module is not None:
            self._modules[metadata.module_name] = module
        logger.info(
            "provider_registered",
            extra={
                "provider": metadata.name,
                "version": metadata.version,
                "module_name": metadata.module_name,
            },
        )

    # ------------------------------------------------------------------
    def unregister(self, provider_name: str) -> None:
        self._providers.pop(provider_name, None)

    def clear(self) -> None:
        self._providers.clear()
        self._modules.clear()

    # ------------------------------------------------------------------
    def get(self, provider_name: str) -> ProviderMetadata:
        try:
            return self._providers[provider_name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise ProviderDiscoveryError(
                f"Provider '{provider_name}' not registered"
            ) from exc

    def list(self) -> Iterable[ProviderMetadata]:
        return tuple(self._providers.values())

    # ------------------------------------------------------------------
    def create_instance(
        self, provider_name: str, *, config: Optional[dict] = None
    ) -> StreamingProviderAdapter:
        metadata = self.get(provider_name)
        adapter = metadata.adapter_cls(config=config)
        return adapter

    # ------------------------------------------------------------------
    def module_for(self, provider_name: str) -> Optional[ModuleType]:
        metadata = self._providers.get(provider_name)
        if metadata is None:
            return None
        return self._modules.get(metadata.module_name)
