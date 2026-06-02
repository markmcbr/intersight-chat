"""GPU + model VRAM telemetry for the sidebar panel.

Combines three data sources, all best-effort:

  1. Ollama `/api/ps`  - currently-loaded models and their `size_vram`.
  2. Ollama `/api/tags` - all pulled models with their on-disk `size`.
  3. `nvidia-smi`       - real GPU model name + total/used VRAM, IF the
                          host happens to expose it to the app container.

When nvidia-smi isn't reachable (the default for this app's compose layout
- only the ollama container has GPU passthrough), we fall back to the
operator-set `GPU_LABEL` and `TOTAL_VRAM_GB` environment variables. That
keeps the panel useful without privileged docker exec calls.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any

import requests


# Bytes per power-of-two unit. Ollama reports `size` and `size_vram` in bytes.
_BYTES_PER_GB = 1024 ** 3


@dataclass
class LoadedModel:
    """A model currently pinned in VRAM (from Ollama /api/ps)."""

    name: str
    size_bytes: int           # total model size as reported by Ollama
    vram_bytes: int           # VRAM footprint (usually ~= size for fully-GPU models)
    context_length: int | None = None  # max context this load supports

    @property
    def vram_gb(self) -> float:
        return self.vram_bytes / _BYTES_PER_GB


@dataclass
class PulledModel:
    """A model present on disk in the ollama_models volume."""

    name: str
    size_bytes: int

    @property
    def size_gb(self) -> float:
        return self.size_bytes / _BYTES_PER_GB


@dataclass
class GpuSnapshot:
    """One-shot view of GPU + Ollama model state."""

    gpu_label: str                     # e.g. "NVIDIA L40S 48GB" (best-effort)
    total_vram_gb: float | None        # None if unknown
    used_vram_gb: float                # sum of loaded model VRAM footprints
    free_vram_gb: float | None         # total - used, when total is known
    loaded: list[LoadedModel] = field(default_factory=list)
    pulled: list[PulledModel] = field(default_factory=list)
    source_notes: list[str] = field(default_factory=list)  # provenance log

    @property
    def utilization_pct(self) -> float | None:
        if self.total_vram_gb is None or self.total_vram_gb <= 0:
            return None
        return min(100.0, 100.0 * self.used_vram_gb / self.total_vram_gb)


def _ollama_native_base(api_base_v1: str) -> str:
    """Strip a trailing `/v1` off OLLAMA_BASE_URL so we can hit native paths."""
    base = api_base_v1.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _fetch_loaded_models(ollama_base_v1: str) -> tuple[list[LoadedModel], str | None]:
    """Hit Ollama's /api/ps. Returns (loaded, error_note)."""
    url = f"{_ollama_native_base(ollama_base_v1)}/api/ps"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as exc:
        return [], f"Could not query Ollama /api/ps: {exc}"

    loaded: list[LoadedModel] = []
    for m in data.get("models") or []:
        ctx_len = None
        details = m.get("details") or {}
        # Recent Ollama versions surface context_length at the top level;
        # older versions hide it under details.
        for key in ("context_length", "ctx_size"):
            if isinstance(m.get(key), int):
                ctx_len = m[key]
                break
        if ctx_len is None:
            for key in ("context_length", "ctx_size"):
                if isinstance(details.get(key), int):
                    ctx_len = details[key]
                    break
        loaded.append(
            LoadedModel(
                name=str(m.get("name") or m.get("model") or ""),
                size_bytes=int(m.get("size") or 0),
                vram_bytes=int(m.get("size_vram") or m.get("size") or 0),
                context_length=ctx_len,
            )
        )
    return loaded, None


def _fetch_pulled_models(ollama_base_v1: str) -> tuple[list[PulledModel], str | None]:
    """Hit Ollama's /api/tags. Returns (pulled_models, error_note)."""
    url = f"{_ollama_native_base(ollama_base_v1)}/api/tags"
    try:
        r = requests.get(url, timeout=3)
        r.raise_for_status()
        data = r.json() or {}
    except Exception as exc:
        return [], f"Could not query Ollama /api/tags: {exc}"

    pulled: list[PulledModel] = []
    for m in data.get("models") or []:
        pulled.append(
            PulledModel(
                name=str(m.get("name") or ""),
                size_bytes=int(m.get("size") or 0),
            )
        )
    pulled.sort(key=lambda p: p.name)
    return pulled, None


def _query_nvidia_smi() -> tuple[str | None, float | None, float | None, str | None]:
    """Best-effort nvidia-smi probe.

    Returns (gpu_name, total_gb, used_gb, error_note). All values may be
    None when nvidia-smi isn't available — which is the expected case in
    this app's compose layout (the GPU is passed only to the ollama
    container, not to the app container). We still try because some
    deployments run the app on the host directly.
    """
    if shutil.which("nvidia-smi") is None:
        return None, None, None, "nvidia-smi not on PATH in this container"
    try:
        # Single-GPU query is enough for a demo; multi-GPU support would
        # need iteration. The query format pins units so we don't have to
        # parse "MiB" strings.
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        ).stdout.strip()
    except Exception as exc:
        return None, None, None, f"nvidia-smi failed: {exc}"

    first_line = out.splitlines()[0] if out else ""
    parts = [p.strip() for p in first_line.split(",")]
    if len(parts) < 3:
        return None, None, None, f"unexpected nvidia-smi output: {first_line!r}"
    name = parts[0]
    try:
        total_mib = float(parts[1])
        used_mib = float(parts[2])
    except ValueError:
        return None, None, None, f"could not parse nvidia-smi memory: {first_line!r}"
    # nvidia-smi reports MiB; convert to GiB.
    return name, total_mib / 1024.0, used_mib / 1024.0, None


def collect_snapshot(
    ollama_base_v1: str,
    *,
    env_gpu_label: str | None = None,
    env_total_vram_gb: float | None = None,
) -> GpuSnapshot:
    """Build a single snapshot of GPU + Ollama state.

    `env_gpu_label` and `env_total_vram_gb` come from the GPU_LABEL and
    TOTAL_VRAM_GB env vars set by the operator in docker-compose.yml.
    They're used as the fallback when nvidia-smi isn't reachable (i.e.
    the common case for this app, where the GPU is only passed through
    to the ollama container).
    """
    notes: list[str] = []

    gpu_name, total_gb, used_gb_smi, smi_err = _query_nvidia_smi()
    if smi_err:
        notes.append(smi_err)
    gpu_label = gpu_name or env_gpu_label or "GPU (not auto-detected)"
    if total_gb is None and env_total_vram_gb is not None:
        total_gb = float(env_total_vram_gb)
        notes.append(f"Using operator-set TOTAL_VRAM_GB={env_total_vram_gb}")

    loaded, loaded_err = _fetch_loaded_models(ollama_base_v1)
    if loaded_err:
        notes.append(loaded_err)
    pulled, pulled_err = _fetch_pulled_models(ollama_base_v1)
    if pulled_err:
        notes.append(pulled_err)

    used_gb_models = sum(m.vram_gb for m in loaded)
    # Prefer nvidia-smi's "used" when we have it (more truthful — accounts
    # for KV cache, CUDA context, other tenants); fall back to the sum of
    # loaded model VRAM footprints from Ollama.
    used_gb = used_gb_smi if used_gb_smi is not None else used_gb_models

    free_gb = None
    if total_gb is not None:
        free_gb = max(0.0, total_gb - used_gb)

    return GpuSnapshot(
        gpu_label=gpu_label,
        total_vram_gb=total_gb,
        used_vram_gb=used_gb,
        free_vram_gb=free_gb,
        loaded=loaded,
        pulled=pulled,
        source_notes=notes,
    )


def would_fit(model: PulledModel, snapshot: GpuSnapshot) -> tuple[bool | None, str]:
    """Heuristic: would this pulled model fit alongside what's already loaded?

    The check is intentionally conservative — Ollama needs headroom beyond
    raw model size for KV cache and CUDA context. We treat anything within
    90 percent of free VRAM as a 'tight fit' rather than a clean 'fits'.
    Returns (fits, label). `fits` is None when total VRAM is unknown.
    """
    if snapshot.total_vram_gb is None:
        return None, "unknown"
    # If the model is already loaded, no extra cost to keep it.
    if any(L.name == model.name for L in snapshot.loaded):
        return True, "loaded"
    free = snapshot.free_vram_gb or 0.0
    needed = model.size_gb
    if needed <= free * 0.9:
        return True, "fits"
    if needed <= free:
        return True, "tight fit"
    return False, f"needs {needed:.1f} GB / {free:.1f} GB free"
