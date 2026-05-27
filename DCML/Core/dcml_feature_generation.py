"""
@file dcml_feature_generation.py
@author Mohamed EL BOUKHIARI
@brief Optional wrappers for DCML feature-generation scripts.

The training and evaluation GUI only needs feature.zip + label.npy. This file is
kept intentionally thin because full DCML feature generation depends on external
chemistry tools such as Open Babel and PDB2PQR.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

from DCML.Core.common import PathLike, resolve_path


class DCMLFeatureGenerationError(RuntimeError):
    """Raised when an external DCML feature-generation script fails."""


def run_python_script(script_path: PathLike, args: Sequence[str | PathLike]) -> dict[str, Any]:
    """Run an external Python feature-generation script safely.

    Use this only if you decide to integrate the heavy DCML feature-generation
    phase into the GUI. For ordinary train/test/HPO, the GUI should consume
    already generated ``feature.zip`` and ``label.npy`` files.
    """
    script = resolve_path(script_path)
    if not script.is_file():
        raise FileNotFoundError(f"Script not found: {script}")

    cmd = [sys.executable, str(script), *[str(value) for value in args]]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise DCMLFeatureGenerationError(
            "DCML feature-generation script failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )
    return {"status": "success", "command": cmd, "stdout": proc.stdout, "stderr": proc.stderr}
