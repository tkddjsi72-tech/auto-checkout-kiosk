#!/usr/bin/env python3
"""Run the single-product identification procedure on one image folder.

NVIDIA GPU is required. The three OCR scripts stop if Paddle is CPU-only,
and barcode detection in this entry point uses CUDA.

1. Laura YOLOv5 barcode boxes.
   conf 0.25, imgsz 640, long edge 640, boxes only.
2. Three PaddleOCR reads: original, crop/CLAHE/rotation, and boost.
3. Decision in src/run_db_input_matching.py.
   Text DB is db/text_front and db/text_multiview.
   Coverage is the max over views. OCR score is the max over the three reads.
   A product matches at coverage >= 0.60.
   Two or more barcode products -> REINSERT.
   Otherwise OCR must name exactly one product, and that product must agree
   with the barcode when the barcode also names one.
   If the manifest item has measured_weight_g, |measured - nominal| must be
   within 1% of nominal. If that field is absent, weight is skipped and
   barcode+OCR confirmation stands.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LAURA_WEIGHTS = ROOT / "models" / "laura_yolov5_barcode" / "barcode_model.pt"
FRONT_TEXT = ROOT / "db" / "text_front"
MULTIVIEW_TEXT = ROOT / "db" / "text_multiview"
PRODUCT_DB = ROOT / "db" / "products.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-product identification")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "dataset" / "single_front",
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
            "--product-db",
            str(PRODUCT_DB),
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
