from __future__ import annotations

"""REST API exposing streaming health metrics."""

from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from .health_monitor import StreamingHealthMonitor


def create_health_app(monitor: StreamingHealthMonitor) -> FastAPI:
    """Create a FastAPI app exposing health endpoints."""

    app = FastAPI()

    @app.get("/health")
    @app.get("/status")
    async def health() -> dict:  # pragma: no cover - simple return
        return monitor.generate_health_report()

    @app.get("/metrics")
    def metrics() -> Response:  # pragma: no cover - simple return
        data = generate_latest()
        return Response(content=data, media_type=CONTENT_TYPE_LATEST)

    return app
