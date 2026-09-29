#!/usr/bin/env python3
"""Run the full single-product identification procedure on one image folder.

Steps, in order:

1. Laura YOLOv5 barcode boxes.
   conf 0.25, imgsz 640, long edge 640, GPU, boxes only (no SAM).
2. Three PaddleOCR reads of the same images, each on GPU:
   original, crop/CLAHE/rotation, and boost (tiles + Korean/English + second pass).
3. Decision in run_db_input_matching.py:
   text DB is 정면 + 멀티뷰, coverage is the max over views (views are not unioned),
   OCR score is the max over the three reads, product threshold is 0.60.
   Two or more barcode products -> REINSERT.
   Otherwise OCR must name exactly one product, and that product must agree
   with the barcode when the barcode also names one.
   Then |measured - nominal| <= 1% of nominal -> CONFIRMED, else REINSERT.

The measured weight is still the simulated value inside run_db_input_matching.py:
manifest expected product, nominal + 25% of its tolerance. It is not a live scale.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LAURA_WEIGHTS = ROOT / "models" / "laura_yolov5_barcode" / "barcode_model.pt"
FRONT_TEXT = ROOT / "DB" / "DB_정면TEXT"
MULTIVIEW_TEXT = ROOT / "DB" / "DB_멀티뷰TEXT"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full single-product identification")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "Kiosk_experiment" / "정면INPUT이미지",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "full_identification",
    )
    return parser.parse_args()


def run(args: list[str]) -> None:
    print("+ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()
    if not LAURA_WEIGHTS.is_file():
        raise FileNotFoundError(f"Laura weights not found: {LAURA_WEIGHTS}")

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    laura_dir = output_dir / "laura"
    paddle_dirs = {
        "paddleocr": output_dir / "paddleocr",
        "paddleocr_preproc": output_dir / "paddleocr_preproc",
        "paddleocr_boost": output_dir / "paddleocr_boost",
    }
    py = sys.executable

    run(
        [
            py,
            "src/inspect_barcode_instances.py",
            "--dataset-root",
            str(input_dir),
            "--barcode-model-family",
            "laura_yolov5",
            "--barcode-model",
            str(LAURA_WEIGHTS),
            "--yolo-only",
            "--conf",
            "0.25",
            "--yolo-imgsz",
            "640",
            "--max-long-edge",
            "640",
            "--device",
            "cuda",
            "--output-dir",
            str(laura_dir),
        ]
    )
    for name, script in (
        ("paddleocr", "src/run_db_input_paddleocr.py"),
        ("paddleocr_preproc", "src/run_db_input_paddleocr_preproc.py"),
        ("paddleocr_boost", "src/run_db_input_paddleocr_boost.py"),
    ):
        run(
            [
                py,
                script,
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(paddle_dirs[name]),
            ]
        )
    run(
        [
            py,
            "src/run_db_input_matching.py",
            "--input-dir",
            str(input_dir),
            "--laura-results",
            str(laura_dir / "results.jsonl"),
            "--paddle-dir",
            str(paddle_dirs["paddleocr"]),
            "--paddle-dir",
            str(paddle_dirs["paddleocr_preproc"]),
            "--paddle-dir",
            str(paddle_dirs["paddleocr_boost"]),
            "--db-text-dir",
            str(FRONT_TEXT),
            "--db-text-dir",
            str(MULTIVIEW_TEXT),
            "--output-dir",
            str(output_dir),
        ]
    )
    print(f"Wrote {output_dir / 'matching_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
