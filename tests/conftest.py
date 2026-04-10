"""Test-session compatibility shims."""

from __future__ import annotations

import os
import platform
import sys


if sys.platform == "win32":
    _SAFE_UNAME = platform.uname_result(
        "Windows",
        os.environ.get("COMPUTERNAME", "localhost"),
        os.environ.get("OS", "Windows_NT"),
        os.environ.get("PROCESSOR_REVISION", ""),
        os.environ.get("PROCESSOR_ARCHITECTURE", "AMD64"),
    )

    platform.uname = lambda: _SAFE_UNAME  # type: ignore[assignment]
    platform.system = lambda: _SAFE_UNAME.system  # type: ignore[assignment]
    platform.machine = lambda: _SAFE_UNAME.machine  # type: ignore[assignment]
    platform.processor = lambda: _SAFE_UNAME.processor  # type: ignore[assignment]
