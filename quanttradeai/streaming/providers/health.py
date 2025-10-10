"""Provider aware health monitoring utilities."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Awaitable, Callable, Deque, Dict, Optional, TypeVar

from pybreaker import CircuitBreaker, CircuitBreakerError

from ..monitoring.health_monitor import StreamingHealthMonitor
from ..monitoring.recovery_manager import RecoveryManager
from .base import ProviderConnectionError, ProviderHealthStatus

logger = logging.getLogger(__name__)


T = TypeVar("T")


class ProviderHealthMonitor:
    """Extend streaming health monitoring with provider specific metrics."""

    def __init__(
        self,
        *,
        streaming_monitor: Optional[StreamingHealthMonitor] = None,
        error_window: float = 60.0,
        error_threshold: int = 3,
        circuit_breaker_fail_max: int = 5,
        circuit_breaker_reset_timeout: float = 60.0,
    ) -> None:
        self.streaming_monitor = streaming_monitor or StreamingHealthMonitor()
        self.recovery_manager: RecoveryManager = self.streaming_monitor.recovery_manager
        self._statuses: Dict[str, ProviderHealthStatus] = {}
        self._error_events: Dict[str, Deque[float]] = defaultdict(deque)
        self._failover_handlers: Dict[str, Callable[[], Awaitable[None]]] = {}
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._status_sources: Dict[str, Callable[[], ProviderHealthStatus]] = {}
        self._lock = asyncio.Lock()
        self.error_window = error_window
        self.error_threshold = error_threshold
        self.circuit_breaker_fail_max = circuit_breaker_fail_max
        self.circuit_breaker_reset_timeout = circuit_breaker_reset_timeout

    # ------------------------------------------------------------------
    def register_provider(
        self,
        provider_name: str,
        *,
        failover_handler: Optional[Callable[[], Awaitable[None]]] = None,
        status_provider: Optional[Callable[[], ProviderHealthStatus]] = None,
    ) -> None:
        self.streaming_monitor.register_connection(provider_name)
        status = ProviderHealthStatus(provider=provider_name)
        self._statuses[provider_name] = status
        if failover_handler is not None:
            self._failover_handlers[provider_name] = failover_handler
        if status_provider is not None:
            self._status_sources[provider_name] = status_provider

    async def record_success(
        self,
        provider_name: str,
        latency_ms: float,
        *,
        bytes_received: int = 0,
    ) -> None:
        async with self._lock:
            status = self._statuses.setdefault(
                provider_name, ProviderHealthStatus(provider=provider_name)
            )
            status.status = "connected"
            status.latency_ms = latency_ms
            status.error_rate = self._calculate_error_rate(provider_name)
            status.last_error = None
            status.circuit_open = False
            status.last_check = datetime.now(timezone.utc)
            status.reconnect_attempts = 0
            self._update_metrics_from_source(provider_name, status)
        self.streaming_monitor.record_latency(provider_name, latency_ms)
        self.streaming_monitor.record_message(provider_name, size_bytes=bytes_received)

    async def record_failure(self, provider_name: str, error: Exception) -> None:
        now = time.time()
        history = self._error_events.setdefault(provider_name, deque())
        history.append(now)
        self._trim_history(provider_name, now)
        async with self._lock:
            status = self._statuses.setdefault(
                provider_name, ProviderHealthStatus(provider=provider_name)
            )
            status.status = "error"
            status.last_error = str(error)
            status.error_rate = self._calculate_error_rate(provider_name)
            status.last_check = datetime.now(timezone.utc)
            self._update_metrics_from_source(provider_name, status)
        logger.error(
            "provider_failure",
            extra={"provider": provider_name, "error": str(error)},
        )
        if len(history) >= self.error_threshold:
            await self._trigger_failover(provider_name)

    async def _trigger_failover(self, provider_name: str) -> None:
        handler = self._failover_handlers.get(provider_name)
        if handler is None:
            return
        try:
            async with self._lock:
                status = self._statuses.setdefault(
                    provider_name, ProviderHealthStatus(provider=provider_name)
                )
                status.reconnect_attempts += 1
            success = await self.recovery_manager.reconnect(provider_name, handler)
            if success:
                await self.record_success(provider_name, latency_ms=0.0)
        except Exception as exc:  # pragma: no cover - logging path
            logger.exception(
                "provider_failover_failed",
                extra={"provider": provider_name, "error": str(exc)},
            )

    # ------------------------------------------------------------------
    async def execute_with_health(
        self,
        provider_name: str,
        operation: Callable[[], Awaitable[T]],
        *,
        fallback: Optional[Callable[[], Awaitable[T]]] = None,
    ) -> T:
        breaker = self._get_circuit_breaker(provider_name)
        start = time.perf_counter()
        try:
            result = await breaker.call_async(operation)
        except CircuitBreakerError as exc:
            await self._mark_circuit_open(provider_name, exc)
            if fallback is not None:
                return await fallback()
            raise ProviderConnectionError(str(exc)) from exc
        except Exception as exc:
            await self.record_failure(provider_name, exc)
            if fallback is not None:
                return await fallback()
            raise
        else:
            latency_ms = (time.perf_counter() - start) * 1000.0
            await self.record_success(provider_name, latency_ms)
            return result

    # ------------------------------------------------------------------
    async def _mark_circuit_open(self, provider_name: str, error: Exception) -> None:
        async with self._lock:
            status = self._statuses.setdefault(
                provider_name, ProviderHealthStatus(provider=provider_name)
            )
            status.status = "circuit_open"
            status.circuit_open = True
            status.last_error = str(error)
            status.last_check = datetime.now(timezone.utc)
            status.error_rate = self._calculate_error_rate(provider_name)
            self._update_metrics_from_source(provider_name, status)
        logger.warning(
            "provider_circuit_open",
            extra={"provider": provider_name, "error": str(error)},
        )

    # ------------------------------------------------------------------
    def _get_circuit_breaker(self, provider_name: str) -> CircuitBreaker:
        breaker = self._circuit_breakers.get(provider_name)
        if breaker is None:
            breaker = CircuitBreaker(
                fail_max=self.circuit_breaker_fail_max,
                reset_timeout=self.circuit_breaker_reset_timeout,
            )
            self._circuit_breakers[provider_name] = breaker
        return breaker

    def _trim_history(self, provider_name: str, current_ts: float) -> None:
        history = self._error_events.get(provider_name)
        if not history:
            return
        threshold = current_ts - self.error_window
        while history and history[0] < threshold:
            history.popleft()

    def _calculate_error_rate(self, provider_name: str) -> float:
        history = self._error_events.get(provider_name)
        if not history:
            return 0.0
        return len(history) / max(self.error_window, 1.0)

    # ------------------------------------------------------------------
    def get_status(self, provider_name: str) -> ProviderHealthStatus:
        return self._statuses.setdefault(
            provider_name, ProviderHealthStatus(provider=provider_name)
        )

    def health_report(self) -> Dict[str, ProviderHealthStatus]:
        return {name: status for name, status in self._statuses.items()}

    def _update_metrics_from_source(
        self, provider_name: str, status: ProviderHealthStatus
    ) -> None:
        source = self._status_sources.get(provider_name)
        if source is None:
            return
        try:
            provider_status = source()
        except Exception:  # pragma: no cover - defensive logging skipped
            return
        status.metrics.update(provider_status.metrics)


__all__ = ["ProviderHealthMonitor"]
