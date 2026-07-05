"""
Monitoring setup extracted from app.py.

Provides the ``_setup_monitoring`` function that optionally registers a
``/metrics`` endpoint with Prometheus exposition format and JSON fallback.
"""

import json
import os
from pathlib import Path
from typing import Any, List

import torch
from fastapi import FastAPI, Request
from starlette.responses import Response as StarletteResponse

from ..utils.logging import setup_logger
from .middleware import _GpuMetrics, _RUNTIME_METRICS


def _setup_health_endpoint(app: FastAPI) -> None:
    """Add a detailed health check endpoint that probes component readiness."""

    @app.get("/health/detailed", tags=["monitoring"])
    async def health_detailed():
        """Detailed health check with component-level status."""
        from ..routes.state import STATE, VARIANT_WORKFLOW_AVAILABLE, SIGNALING_NETWORK_AVAILABLE

        components = {
            "model_loaded": STATE.model is not None,
            "variant_workflow": STATE.variant_workflow is not None,
            "pathway_mapper": STATE.pathway_mapper is not None,
        }

        overall = "healthy" if all(components.values()) else "degraded"

        return {
            "status": overall,
            "components": components,
            "variant_workflow_available": VARIANT_WORKFLOW_AVAILABLE,
            "signaling_network_available": SIGNALING_NETWORK_AVAILABLE,
        }

logger = setup_logger(__name__)


def _setup_monitoring(app: FastAPI) -> None:
    """Optionally register a /metrics endpoint from production.yaml monitoring config."""
    config_path = os.environ.get("PTM2CELLNET_CONFIG", "configs/production.yaml")
    monitoring_config: dict = {}
    try:
        import yaml

        if Path(config_path).exists():
            cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
            monitoring_config = cfg.get("monitoring", {})
    except (OSError, ValueError) as exc:  # pragma: no cover - config loading is best-effort
        logger.debug("Could not load monitoring config: %s", exc)

    metrics_enabled = monitoring_config.get("metrics_enabled", False)
    health_check_interval = monitoring_config.get("health_check_interval", 30)

    if metrics_enabled:
        logger.info(
            "Monitoring enabled (metrics_port=%s, health_check_interval=%s)",
            monitoring_config.get("metrics_port", 9090),
            health_check_interval,
        )

        def _format_prometheus() -> str:
            """Render runtime metrics in Prometheus text exposition format.

            Prometheus scrapes ``/metrics`` and expects a flat text body with
            ``# HELP`` / ``# TYPE`` headers followed by ``name{labels} value``
            lines. Returning JSON there silently breaks scrape pipelines, so
            this emits real Prometheus text by default.
            """
            snap = _RUNTIME_METRICS.snapshot()
            lines: List[str] = []
            lines.append("# HELP ptm2cellnet_requests_total Total HTTP requests processed.")
            lines.append("# TYPE ptm2cellnet_requests_total counter")
            lines.append(f"ptm2cellnet_requests_total {snap['request_count']}")
            lines.append("# HELP ptm2cellnet_errors_total Total HTTP responses with status>=400.")
            lines.append("# TYPE ptm2cellnet_errors_total counter")
            lines.append(f"ptm2cellnet_errors_total {snap['error_count']}")
            lines.append("# HELP ptm2cellnet_error_rate Rolling error rate [0,1].")
            lines.append("# TYPE ptm2cellnet_error_rate gauge")
            lines.append(f"ptm2cellnet_error_rate {snap['error_rate']}")
            latency = snap.get("latency", {})
            lines.append("# HELP ptm2cellnet_latency_ms Request latency in milliseconds.")
            lines.append("# TYPE ptm2cellnet_latency_ms summary")
            for quantile in ("avg_ms", "p50_ms", "p95_ms", "p99_ms"):
                if quantile in latency:
                    label = quantile.replace("_ms", "")
                    lines.append(
                        f'ptm2cellnet_latency_ms{{quantile="{label}"}} {latency[quantile]}'
                    )
            # Histogram buckets as cumulative counters (Prometheus convention).
            lines.append("# HELP ptm2cellnet_latency_bucket_ms Latency histogram buckets (cumulative).")
            lines.append("# TYPE ptm2cellnet_latency_bucket_ms counter")
            buckets = snap.get("latency_buckets", {})
            for name, value in buckets.items():
                # name looks like "le_50ms" or "le_+Inf".
                le = name.split("_", 1)[1] if "_" in name else name
                lines.append(f'ptm2cellnet_latency_bucket_ms{{le="{le}"}} {value}')
            # GPU metrics when available.
            try:
                if torch.cuda.is_available():
                    lines.append("# HELP ptm2cellnet_gpu_utilization_pct GPU utilization percent.")
                    lines.append("# TYPE ptm2cellnet_gpu_utilization_pct gauge")
                    lines.append(
                        f"ptm2cellnet_gpu_utilization_pct {round(float(torch.cuda.utilization()), 2)}"
                    )
                    lines.append("# HELP ptm2cellnet_gpu_memory_allocated_mb GPU memory allocated (MB).")
                    lines.append("# TYPE ptm2cellnet_gpu_memory_allocated_mb gauge")
                    allocated = torch.cuda.memory_allocated()
                    lines.append(
                        f"ptm2cellnet_gpu_memory_allocated_mb {round(allocated / (1024 * 1024), 2)}"
                    )
                    lines.append("# HELP ptm2cellnet_gpu_memory_reserved_mb GPU memory reserved (MB).")
                    lines.append("# TYPE ptm2cellnet_gpu_memory_reserved_mb gauge")
                    reserved = torch.cuda.memory_reserved()
                    lines.append(
                        f"ptm2cellnet_gpu_memory_reserved_mb {round(reserved / (1024 * 1024), 2)}"
                    )
            except (RuntimeError, OSError, ModuleNotFoundError) as exc:  # GPU metrics are best-effort.
                logger.debug("GPU metrics unavailable: %s", exc)
            return "\n".join(lines) + "\n"

        @app.get("/metrics")
        async def metrics(request: Request):
            """Service metrics endpoint.

            By default returns Prometheus text exposition format
            (``text/plain; version=0.0.4``) so a Prometheus scraper can ingest
            it directly. Pass ``Accept: application/json`` (or
            ``?format=json``) to receive the legacy JSON payload — useful for
            ad-hoc inspection or dashboards that already consume the JSON.
            """
            accept = request.headers.get("accept", "")
            query_format = request.query_params.get("format", "").lower()
            want_json = query_format == "json" or "application/json" in accept

            runtime = _RUNTIME_METRICS.snapshot()

            gpu_metrics: _GpuMetrics = {}
            try:
                if torch.cuda.is_available():
                    gpu_metrics["gpu_utilization_pct"] = round(
                        float(torch.cuda.utilization()), 2
                    )
                    allocated = torch.cuda.memory_allocated()
                    reserved = torch.cuda.memory_reserved()
                    gpu_metrics["gpu_memory_allocated_mb"] = round(
                        allocated / (1024 * 1024), 2
                    )
                    gpu_metrics["gpu_memory_reserved_mb"] = round(
                        reserved / (1024 * 1024), 2
                    )
            except (RuntimeError, OSError, ModuleNotFoundError) as exc:  # GPU metrics are best-effort.
                logger.debug("GPU metrics unavailable: %s", exc)

            if not want_json:
                # Prometheus exposition format.
                return StarletteResponse(
                    content=_format_prometheus(),
                    media_type="text/plain; version=0.0.4; charset=utf-8",
                )

            return {
                "service": "PTM2CellNet",
                "version": app.version,
                "health_check_interval_seconds": health_check_interval,
                "runtime": runtime,
                "gpu": gpu_metrics if gpu_metrics else None,
            }
    else:
        logger.info("Monitoring metrics endpoint disabled (metrics_enabled=false)")
    _setup_health_endpoint(app)
