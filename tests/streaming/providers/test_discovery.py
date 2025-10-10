from __future__ import annotations

import importlib
import sys
import time
from pathlib import Path

import pytest

from quanttradeai.streaming.providers import (
    ProviderDiscovery,
    ProviderDiscoveryError,
    ProviderRegistry,
)


def test_provider_discovery_finds_example_adapter() -> None:
    discovery = ProviderDiscovery()
    registry = discovery.discover()
    metadata = registry.get("example")
    assert metadata.name == "example"
    assert metadata.capabilities.supports_order_book is True
    assert "quotes" in metadata.capabilities.data_types


def _write_temp_adapter(path: Path, version: str, dependencies: tuple[str, ...] = ()) -> str:
    module_name = "temp_adapter"
    (path / "__init__.py").write_text("")
    adapter_path = path / f"{module_name}.py"
    adapter_path.write_text(
        """
from quanttradeai.streaming.providers.base import ProviderCapabilities, StreamingProviderAdapter

class TempAdapter(StreamingProviderAdapter):
    provider_name = "temp"
    provider_version = "{version}"
    provider_dependencies = {dependencies!r}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def subscribe(self, symbols):
        return None

    async def unsubscribe(self, symbols):
        return None

    def get_capabilities(self):
        return ProviderCapabilities(asset_types={{"stocks"}}, data_types={{"trades"}})
""".format(
            version=version, dependencies=dependencies
        )
    )
    return module_name


def test_provider_discovery_skips_modules_with_missing_dependencies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "temp_missing"
    package_root.mkdir()
    module_name = _write_temp_adapter(package_root, "1.0.0", ("nonexistent_dependency_xyz",))
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        discovery = ProviderDiscovery(
            registry=ProviderRegistry(),
            search_paths=[package_root],
            package_prefix="temp_missing",
        )
        registry = discovery.discover()
        with pytest.raises(ProviderDiscoveryError):
            registry.get("temp")
    finally:
        sys.modules.pop(f"temp_missing.{module_name}", None)
        sys.modules.pop("temp_missing", None)


def test_provider_discovery_supports_hot_reload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package_root = tmp_path / "temp_hot"
    package_root.mkdir()
    module_name = _write_temp_adapter(package_root, "1.0.0")
    monkeypatch.syspath_prepend(str(tmp_path))
    discovery = ProviderDiscovery(
        registry=ProviderRegistry(),
        search_paths=[package_root],
        package_prefix="temp_hot",
    )
    registry = discovery.discover()
    assert registry.get("temp").version == "1.0.0"

    time.sleep(1.1)
    _write_temp_adapter(package_root, "2.0.0")
    importlib.invalidate_caches()
    registry = discovery.refresh()
    assert registry.get("temp").version == "2.0.0"

    sys.modules.pop(f"temp_hot.{module_name}", None)
    sys.modules.pop("temp_hot", None)
