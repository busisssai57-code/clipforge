"""T6 — make pip-installed NVIDIA CUDA DLLs loadable on native Windows.

faster-whisper/CTranslate2 dlopen cuBLAS + cuDNN by bare DLL name. The pip
packages ``nvidia-cublas-cu12`` / ``nvidia-cudnn-cu12`` ship those DLLs under
``site-packages/nvidia/<lib>/bin``, which is NOT on the default Windows DLL
search path — the classic "Could not locate cudnn_ops64_9.dll" failure.

:func:`ensure_nvidia_dll_dirs` registers every such directory via
``os.add_dll_directory``. Called from CLI boot and from the doctor's DLL
probe, so the doctor tests the same search path the pipeline runs with.
Idempotent; a no-op on non-Windows and when no nvidia packages exist.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

_added: set[Path] = set()


def _candidate_roots() -> list[Path]:
    """site-packages roots to scan, deduplicated, order-stable."""
    roots: list[Path] = []
    for getter in (site.getsitepackages, lambda: [site.getusersitepackages()]):
        try:
            for p in getter():
                path = Path(p)
                if path not in roots:
                    roots.append(path)
        except (AttributeError, OSError):  # some embedded interpreters
            continue
    return roots


def ensure_nvidia_dll_dirs() -> list[Path]:
    """Register ``site-packages/nvidia/*/bin`` dirs on the DLL search path.

    Returns the directories registered by this call (empty on repeat calls,
    non-Windows, or when nothing is installed) — callers may log them.
    """
    if sys.platform != "win32":
        return []
    registered: list[Path] = []
    for root in _candidate_roots():
        nvidia = root / "nvidia"
        if not nvidia.is_dir():
            continue
        for bin_dir in sorted(nvidia.glob("*/bin")):
            if bin_dir in _added or not bin_dir.is_dir():
                continue
            try:
                os.add_dll_directory(str(bin_dir))
                _added.add(bin_dir)
                registered.append(bin_dir)
            except OSError:
                continue  # directory vanished between glob and call
    return registered
