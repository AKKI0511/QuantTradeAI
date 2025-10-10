"""Dynamic discovery of streaming provider adapters."""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterator, List, Optional, Sequence

from .base import (
    ProviderDiscoveryError,
    ProviderError,
    StreamingProviderAdapter,
)
from .registry import ProviderMetadata, ProviderRegistry

logger = logging.getLogger(__name__)


@dataclass
class _ModuleSpec:
    name: str
    path: Path


class ProviderDiscovery:
    """Discover and register streaming provider adapters at runtime."""

    def __init__(
        self,
        *,
        registry: Optional[ProviderRegistry] = None,
        search_paths: Optional[Sequence[Path]] = None,
        package_prefix: Optional[str] = "quanttradeai.streaming.adapters",
    ) -> None:
        self.registry = registry or ProviderRegistry()
        if search_paths is None:
            package = (
                importlib.import_module(package_prefix) if package_prefix else None
            )
            if package is None or not getattr(package, "__path__", None):
                raise ProviderDiscoveryError("Unable to resolve adapter package path")
            package_paths = [Path(p) for p in package.__path__]
        else:
            package_paths = [Path(p) for p in search_paths]
        self.search_paths = package_paths
        self.package_prefix = package_prefix
        self._module_cache: Dict[str, ModuleType] = {}
        self._module_mtimes: Dict[str, float] = {}

    # ------------------------------------------------------------------
    def discover(self) -> ProviderRegistry:
        """Discover providers and populate the registry."""

        for module_spec in self._iter_modules():
            try:
                module = self._load_module(module_spec)
                self._register_module_adapters(module_spec, module)
            except ProviderError as exc:
                logger.error(
                    "provider_discovery_failed",
                    extra={"module_name": module_spec.name, "error": str(exc)},
                )
            except Exception as exc:  # pragma: no cover - defensive logging
                logger.exception(
                    "provider_discovery_unexpected",
                    extra={"module_name": module_spec.name, "error": str(exc)},
                )
        return self.registry

    def refresh(self) -> ProviderRegistry:
        """Re-run discovery to pick up newly added or modified providers."""

        self._module_cache.clear()
        self._module_mtimes.clear()
        return self.discover()

    # ------------------------------------------------------------------
    def _iter_modules(self) -> Iterator[_ModuleSpec]:
        for path in self.search_paths:
            if not path.exists():
                continue
            for file_path in path.iterdir():
                if file_path.name.startswith("_"):
                    continue
                if file_path.suffix == ".py":
                    module_name = file_path.stem
                    yield _ModuleSpec(
                        self._qualified_module_name(module_name), file_path
                    )
                elif file_path.is_dir() and (file_path / "__init__.py").exists():
                    module_name = file_path.name
                    yield _ModuleSpec(
                        self._qualified_module_name(module_name), file_path
                    )

    def _qualified_module_name(self, module_name: str) -> str:
        if self.package_prefix:
            return f"{self.package_prefix}.{module_name}"
        return module_name

    # ------------------------------------------------------------------
    def _load_module(self, module_spec: _ModuleSpec) -> ModuleType:
        existing = sys.modules.get(module_spec.name)
        module_path = module_spec.path
        mtime = module_path.stat().st_mtime
        if existing is not None:
            if module_spec.name in self._module_mtimes:
                cached_mtime = self._module_mtimes[module_spec.name]
                if mtime > cached_mtime:
                    module = importlib.reload(existing)
                else:
                    module = existing
            else:
                module = importlib.reload(existing)
        else:
            module = importlib.import_module(module_spec.name)
        self._module_cache[module_spec.name] = module
        self._module_mtimes[module_spec.name] = mtime
        return module

    def _register_module_adapters(
        self, module_spec: _ModuleSpec, module: ModuleType
    ) -> None:
        adapter_classes = self._extract_adapter_classes(module)
        if not adapter_classes:
            return
        for adapter_cls in adapter_classes:
            adapter = self._instantiate_adapter(adapter_cls)
            if (
                not adapter.provider_name
                or adapter.provider_name == StreamingProviderAdapter.provider_name
            ):
                raise ProviderDiscoveryError(
                    f"Adapter '{adapter_cls.__name__}' must define a unique provider_name"
                )
            metadata = ProviderMetadata(
                name=adapter.provider_name,
                version=adapter.provider_version,
                adapter_cls=adapter_cls,
                module_name=module_spec.name,
                capabilities=adapter.get_capabilities(),
                description=adapter.provider_description,
                dependencies=tuple(adapter.provider_dependencies),
                module_path=(
                    Path(getattr(module, "__file__", "")).resolve()
                    if getattr(module, "__file__", None)
                    else None
                ),
            )
            self.registry.register(metadata, module=module)

    @staticmethod
    def _extract_adapter_classes(
        module: ModuleType,
    ) -> List[type[StreamingProviderAdapter]]:
        adapters: List[type[StreamingProviderAdapter]] = []
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if not issubclass(obj, StreamingProviderAdapter):
                continue
            if obj is StreamingProviderAdapter:
                continue
            if getattr(obj, "__abstractmethods__", False):
                continue
            adapters.append(obj)
        return adapters

    @staticmethod
    def _instantiate_adapter(
        adapter_cls: type[StreamingProviderAdapter],
    ) -> StreamingProviderAdapter:
        try:
            adapter = adapter_cls()
        except TypeError:
            adapter = adapter_cls(config={})
        except Exception as exc:
            raise ProviderDiscoveryError(
                f"Unable to instantiate provider '{adapter_cls.__name__}': {exc}"
            ) from exc
        return adapter
