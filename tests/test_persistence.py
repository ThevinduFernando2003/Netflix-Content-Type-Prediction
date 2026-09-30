"""Artifacts round-trip through a fresh interpreter."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from src.features.inference import build_inference_frame
from src.models.registry import load_runtime
from tests.conftest import sample_payload


def test_subprocess_reload_matches(model_dir: Path) -> None:
    runtime = load_runtime(model_dir)
    assert runtime is not None
    frame = build_inference_frame(sample_payload(), runtime.metadata)
    local = runtime.calibrator.predict_proba(frame)[0].tolist()
    script = f"""
import sys
sys.path.insert(0, r"{Path(__file__).resolve().parents[1]}")
from src.features.inference import build_inference_frame
from src.models.registry import load_runtime
from tests.conftest import sample_payload
runtime = load_runtime(r"{model_dir}")
frame = build_inference_frame(sample_payload(), runtime.metadata)
print(",".join(str(value) for value in runtime.calibrator.predict_proba(frame)[0].tolist()))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
    reloaded = [float(part) for part in completed.stdout.strip().split(",")]
    assert len(reloaded) == len(local)
    assert all(abs(left - right) < 1e-6 for left, right in zip(reloaded, local, strict=True))
