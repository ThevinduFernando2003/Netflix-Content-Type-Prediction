"""Score one JSON object from the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.inference import build_inference_frame
from src.logging_conf import configure_logging
from src.models.registry import load_runtime, probability_tv


def main() -> None:
    configure_logging()
    if len(sys.argv) != 2:
        print("Usage: python scripts/predict_cli.py payload.json")
        raise SystemExit(2)
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    if "duration" in payload:
        print("duration is excluded by design")
        raise SystemExit(2)
    runtime = load_runtime()
    if runtime is None:
        print("Model is not loaded. Run make train.")
        raise SystemExit(1)
    frame = build_inference_frame(payload, runtime.metadata)
    probability = float(probability_tv(runtime, frame)[0])
    label = "TV Show" if probability >= runtime.threshold else "Movie"
    print(
        json.dumps(
            {
                "prediction": label,
                "probability_tv_show": probability,
                "threshold_used": runtime.threshold,
            }
        )
    )


if __name__ == "__main__":
    main()
