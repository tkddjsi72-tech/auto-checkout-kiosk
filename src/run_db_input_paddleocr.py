#!/usr/bin/env python3
"""Run PaddleOCR on DB_정면INPUT이미지 and save raw/overlay results."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "dataset" / "single_front"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "db_input_matching" / "paddleocr"
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR on DB input images")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def unwrap_page(result: Any) -> dict[str, Any]:
    if not result:
        return {}
    page = result[0]
    raw_json = getattr(page, "json", None)
    if callable(raw_json):
        page = raw_json()
    elif isinstance(raw_json, dict):
        page = raw_json
    if isinstance(page, dict) and isinstance(page.get("res"), dict):
        page = page["res"]
    if isinstance(page, dict):
        return jsonable(page)
    try:
        return jsonable(dict(page))
    except (TypeError, ValueError):
        return {}


def build_text_set(texts: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for token in text.split():
            token = token.strip()
            if not token or SYMBOL_ONLY_RE.fullmatch(token):
                continue
            key = token.casefold()
            if key not in seen:
                seen.add(key)
                ordered.append(token)
    return ordered


def read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size, index=0)
            except OSError:
                pass
    return ImageFont.load_default()


def save_overlay(
    image_path: Path,
    output_path: Path,
    texts: list[str],
    polys: list[list[list[float]]],
    scores: list[float],
) -> None:
    bgr = read_bgr(image_path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(24, round(min(canvas.size) * 0.012)))

    for index, (text, poly) in enumerate(zip(texts, polys), start=1):
        if not text or len(poly) < 3:
            continue
        points = [(int(round(point[0])), int(round(point[1]))) for point in poly]
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        draw.line(points + [points[0]], fill=(20, 90, 220), width=5)
        label = f"{index}. {text} ({score:.2f})"
        x = max(0, min(point[0] for point in points))
        y = max(0, min(point[1] for point in points))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width = right - left
        height = bottom - top
        pad = 6
        label_x = min(x, max(0, canvas.width - width - pad * 2))
        label_y = max(0, y - height - pad * 2)
        draw.rectangle(
            (label_x, label_y, label_x + width + pad * 2, label_y + height + pad * 2),
            fill=(20, 90, 220),
        )
        draw.text((label_x + pad, label_y + pad), label, font=font, fill=(255, 255, 255))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def main() -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = input_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        raise ValueError(f"No items in manifest: {manifest_path}")

    from paddleocr import PaddleOCR
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            "PaddleOCR must run on GPU. Install paddlepaddle-gpu, not CPU paddlepaddle."
        )

    print("Loading PaddleOCR models...")
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False,
        device="gpu",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        image_path = input_dir / item["input_image"]
        if not image_path.exists() and item.get("source"):
            image_path = (ROOT / item["source"]).resolve()
        started = time.perf_counter()
        page = unwrap_page(ocr.predict(str(image_path)))
        elapsed_s = time.perf_counter() - started

        output_stem = Path(item["input_image"]).stem
        texts = [str(text).strip() for text in (page.get("rec_texts") or [])]
        polys = jsonable(page.get("rec_polys") or page.get("dt_polys") or [])
        scores = [float(score) for score in (page.get("rec_scores") or [])]
        result = {
            "input_image": item["input_image"],
            "source_image": str(image_path),
            "expected_product": item.get("expected_product")
            or ", ".join(item.get("expected_products") or []),
            "model": {
                "detection": "PP-OCRv6_medium_det",
                "recognition": "korean_PP-OCRv5_mobile_rec",
            },
            "elapsed_s": round(elapsed_s, 4),
            "rec_texts": texts,
            "rec_scores": scores,
            "rec_polys": polys,
            "text_set": build_text_set(texts),
        }
        result_path = output_dir / f"{output_stem}_res.json"
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        save_overlay(
            image_path,
            output_dir / "overlays" / f"{output_stem}.jpg",
            texts,
            polys,
            scores,
        )
        aggregate.append(result)
        print(
            f"[{index}/{len(items)}] {output_stem}: "
            f"texts={len(texts)} elapsed={elapsed_s:.2f}s"
        )

    (output_dir / "results.json").write_text(
        json.dumps({"items": aggregate}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote PaddleOCR results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
