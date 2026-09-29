#!/usr/bin/env python3
"""Boosted PaddleOCR: multi-view(1) + dual-rec union(2) + 2-pass crop re-OCR(3)."""

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

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "dataset" / "single_front"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "db_input_matching" / "paddleocr_boost"
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
ROTATIONS = (
    ("r0", None),
    ("r90", cv2.ROTATE_90_CLOCKWISE),
    ("r180", cv2.ROTATE_180),
    ("r270", cv2.ROTATE_90_COUNTERCLOCKWISE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Boosted multi-view dual-rec 2-pass OCR")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-side", type=int, default=1400)
    parser.add_argument("--pass2-min-side", type=int, default=220)
    parser.add_argument("--pass2-max-boxes", type=int, default=24)
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


def normalize_keep(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text or "")


def score_texts(texts: list[str]) -> int:
    hangul = sum(("가" <= ch <= "힣") for text in texts for ch in text)
    useful = sum(1 for text in texts if normalize_keep(text))
    return hangul + useful * 2


def read_bgr(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def write_bgr(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, enc = cv2.imencode(path.suffix or ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError(f"Failed to encode {path}")
    enc.tofile(str(path))


def product_crop_box(bgr: np.ndarray, pad_ratio: float = 0.35) -> tuple[int, int, int, int]:
    height, width = bgr.shape[:2]

    def center_box(frac: float = 0.42) -> tuple[int, int, int, int]:
        side = int(min(height, width) * frac)
        cx, cy = width // 2, height // 2
        return cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = ((sat > 35) | (val < 140)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    ys, xs = np.where(mask > 0)
    if len(xs) < 500:
        return center_box()
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    box_w, box_h = max(1, x1 - x0), max(1, y1 - y0)
    if (box_w * box_h) / float(height * width) > 0.55:
        return center_box()
    side = int(max(box_w, box_h) * (1.0 + pad_ratio * 2))
    side = max(side, int(min(height, width) * 0.25))
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    left = max(0, cx - side // 2)
    top = max(0, cy - side // 2)
    right = min(width, left + side)
    bottom = min(height, top + side)
    left = max(0, right - side)
    top = max(0, bottom - side)
    return left, top, right, bottom


def apply_clahe(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    luminance, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(luminance)
    return cv2.cvtColor(cv2.merge((enhanced, a_ch, b_ch)), cv2.COLOR_LAB2BGR)


def upscale_min_side(bgr: np.ndarray, min_side: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    short = min(height, width)
    if short >= min_side:
        return bgr
    scale = min_side / float(short)
    return cv2.resize(
        bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_CUBIC,
    )


def downscale_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    height, width = bgr.shape[:2]
    long_side = max(height, width)
    if long_side <= max_side:
        return bgr
    scale = max_side / float(long_side)
    return cv2.resize(
        bgr,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )


def make_tiles(bgr: np.ndarray, grid: int = 2, overlap: float = 0.12) -> list[np.ndarray]:
    height, width = bgr.shape[:2]
    tiles: list[np.ndarray] = []
    step_y = height / grid
    step_x = width / grid
    pad_y = int(step_y * overlap)
    pad_x = int(step_x * overlap)
    for row in range(grid):
        for col in range(grid):
            y0 = max(0, int(row * step_y) - pad_y)
            x0 = max(0, int(col * step_x) - pad_x)
            y1 = min(height, int((row + 1) * step_y) + pad_y)
            x1 = min(width, int((col + 1) * step_x) + pad_x)
            tile = bgr[y0:y1, x0:x1]
            if tile.size:
                tiles.append(tile)
    return tiles


def predict_page(ocr: Any, image: np.ndarray, tmp_path: Path) -> dict[str, Any]:
    write_bgr(tmp_path, image)
    return unwrap_page(ocr.predict(str(tmp_path)))


def page_texts(page: dict[str, Any]) -> list[str]:
    return [str(text).strip() for text in (page.get("rec_texts") or []) if str(text).strip()]


def page_boxes(page: dict[str, Any]) -> list[tuple[str, float, list[list[float]]]]:
    texts = page.get("rec_texts") or []
    scores = page.get("rec_scores") or []
    polys = page.get("rec_polys") or page.get("dt_polys") or []
    boxes: list[tuple[str, float, list[list[float]]]] = []
    for index, text in enumerate(texts):
        clean = str(text).strip()
        if not clean or not normalize_keep(clean):
            continue
        score = float(scores[index]) if index < len(scores) else 0.0
        poly = polys[index] if index < len(polys) else []
        if len(poly) < 3:
            continue
        boxes.append((clean, score, poly))
    return boxes


def poly_bbox(poly: list[list[float]], pad: float, width: int, height: int) -> tuple[int, int, int, int]:
    xs = [float(point[0]) for point in poly]
    ys = [float(point[1]) for point in poly]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    x0 = max(0, int(x0 - bw * pad))
    y0 = max(0, int(y0 - bh * pad))
    x1 = min(width, int(x1 + bw * pad))
    y1 = min(height, int(y1 + bh * pad))
    return x0, y0, x1, y1


def select_best_rotation(ocr_ko: Any, image: np.ndarray, tmp_dir: Path, tag: str) -> tuple[str, np.ndarray, dict[str, Any]]:
    best_name = "r0"
    best_img = image
    best_page: dict[str, Any] = {}
    best_score = -1
    for rot_name, rot_code in ROTATIONS:
        rotated = image if rot_code is None else cv2.rotate(image, rot_code)
        page = predict_page(ocr_ko, rotated, tmp_dir / f"{tag}_{rot_name}.jpg")
        score = score_texts(page_texts(page))
        if score > best_score:
            best_score = score
            best_name = rot_name
            best_img = rotated
            best_page = page
    return best_name, best_img, best_page


def make_ocr(rec_model: str) -> Any:
    from paddleocr import PaddleOCR
    import paddle

    if not paddle.is_compiled_with_cuda():
        raise RuntimeError(
            "PaddleOCR must run on GPU. Install paddlepaddle-gpu, not CPU paddlepaddle."
        )

    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name=rec_model,
        use_textline_orientation=True,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        enable_mkldnn=False,
        device="gpu",
    )


def run_one(
    ocr_ko: Any,
    ocr_en: Any,
    image_path: Path,
    expected_product: str,
    output_dir: Path,
    min_side: int,
    pass2_min_side: int,
    pass2_max_boxes: int,
    output_stem: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = read_bgr(image_path)
    stem = output_stem or image_path.stem
    tmp_dir = output_dir / "tmp" / stem
    tmp_dir.mkdir(parents=True, exist_ok=True)

    loose = product_crop_box(original, pad_ratio=0.55)
    tight = product_crop_box(original, pad_ratio=0.12)
    lx0, ly0, lx1, ly1 = loose
    tx0, ty0, tx1, ty1 = tight
    loose_crop = original[ly0:ly1, lx0:lx1].copy()
    tight_crop = original[ty0:ty1, tx0:tx1].copy()

    views: list[tuple[str, np.ndarray, dict[str, Any] | None]] = []

    # 1) original (downscaled) — no forced rotation
    views.append(("original", downscale_max_side(original, 2000), None))

    # 1) loose / tight with CLAHE+upscale+best rotation (selected by Korean model)
    for name, crop in (("loose", loose_crop), ("tight", tight_crop)):
        enhanced = upscale_min_side(apply_clahe(crop), min_side)
        rot_name, rot_img, ko_page = select_best_rotation(ocr_ko, enhanced, tmp_dir, name)
        views.append((f"{name}_{rot_name}", rot_img, ko_page))

    # 1) tiles on loose CLAHE crop (r0 only)
    loose_enhanced = upscale_min_side(apply_clahe(loose_crop), min_side)
    for index, tile in enumerate(make_tiles(loose_enhanced, grid=2, overlap=0.12)):
        views.append((f"tile{index}", upscale_min_side(tile, 900), None))

    all_texts: list[str] = []
    source_runs: list[dict[str, Any]] = []
    pass2_candidates: list[tuple[np.ndarray, str, float]] = []

    for view_name, view_img, cached_ko in views:
        # Korean
        if cached_ko is None:
            ko_page = predict_page(ocr_ko, view_img, tmp_dir / f"{view_name}_ko.jpg")
        else:
            ko_page = cached_ko
        # English/multilingual v6 rec
        en_page = predict_page(ocr_en, view_img, tmp_dir / f"{view_name}_en.jpg")

        ko_texts = page_texts(ko_page)
        en_texts = page_texts(en_page)
        all_texts.extend(ko_texts)
        all_texts.extend(en_texts)
        source_runs.append(
            {
                "view": view_name,
                "ko_texts": ko_texts,
                "en_texts": en_texts,
                "ko_count": len(ko_texts),
                "en_count": len(en_texts),
            }
        )

        height, width = view_img.shape[:2]
        for text, score, poly in page_boxes(ko_page) + page_boxes(en_page):
            if score < 0.45 and len(normalize_keep(text)) < 4:
                continue
            x0, y0, x1, y1 = poly_bbox(poly, pad=0.35, width=width, height=height)
            crop = view_img[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            area = (x1 - x0) * (y1 - y0)
            pass2_candidates.append((crop, text, float(area)))

    # 3) 2-pass: re-OCR top boxes with both models
    pass2_candidates.sort(key=lambda item: item[2], reverse=True)
    seen_pass2: set[str] = set()
    pass2_texts: list[str] = []
    used = 0
    for crop, seed_text, _area in pass2_candidates:
        key = normalize_keep(seed_text).casefold()
        if key in seen_pass2:
            continue
        seen_pass2.add(key)
        scaled = upscale_min_side(crop, pass2_min_side)
        # also try a mild CLAHE variant
        for tag, img in (("raw", scaled), ("clahe", apply_clahe(scaled))):
            ko_page = predict_page(ocr_ko, img, tmp_dir / f"pass2_{used}_{tag}_ko.jpg")
            en_page = predict_page(ocr_en, img, tmp_dir / f"pass2_{used}_{tag}_en.jpg")
            texts = page_texts(ko_page) + page_texts(en_page)
            pass2_texts.extend(texts)
            all_texts.extend(texts)
        used += 1
        if used >= pass2_max_boxes:
            break

    text_set = build_text_set(all_texts)
    elapsed_s = time.perf_counter() - started
    result = {
        "input_image": f"{stem}{image_path.suffix}",
        "source_image": str(image_path),
        "expected_product": expected_product,
        "model": {
            "detection": "PP-OCRv6_medium_det",
            "recognition": [
                "korean_PP-OCRv5_mobile_rec",
                "PP-OCRv6_medium_rec",
            ],
            "boost": {
                "multi_view": True,
                "dual_rec_union": True,
                "two_pass": True,
                "views": [run["view"] for run in source_runs],
                "pass2_boxes": used,
            },
        },
        "elapsed_s": round(elapsed_s, 4),
        "rec_texts": all_texts,
        "rec_scores": [],
        "rec_polys": [],
        "text_set": text_set,
        "source_runs": source_runs,
        "pass2_texts": pass2_texts,
        "preprocess": {
            "loose_crop_xyxy": [lx0, ly0, lx1, ly1],
            "tight_crop_xyxy": [tx0, ty0, tx1, ty1],
        },
    }
    (output_dir / f"{stem}_res.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> int:
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")
    os.environ.setdefault("PYTHONNOUSERSITE", "1")

    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = json.loads((input_dir / "manifest.json").read_text(encoding="utf-8"))
    items = manifest.get("items", [])
    if not items:
        raise ValueError("manifest has no items")

    print("Loading dual PaddleOCR models...")
    ocr_ko = make_ocr("korean_PP-OCRv5_mobile_rec")
    ocr_en = make_ocr("PP-OCRv6_medium_rec")

    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        image_path = input_dir / item["input_image"]
        if not image_path.exists() and item.get("source"):
            image_path = (ROOT / item["source"]).resolve()
        output_stem = Path(item["input_image"]).stem
        result = run_one(
            ocr_ko,
            ocr_en,
            image_path,
            item.get("expected_product")
            or ", ".join(item.get("expected_products") or []),
            output_dir,
            args.min_side,
            args.pass2_min_side,
            args.pass2_max_boxes,
            output_stem=output_stem,
        )
        aggregate.append(result)
        print(
            f"[{index}/{len(items)}] {output_stem}: "
            f"texts={len(result['rec_texts'])} tokens={len(result['text_set'])} "
            f"pass2={result['model']['boost']['pass2_boxes']} "
            f"elapsed={result['elapsed_s']:.1f}s"
        )

    (output_dir / "results.json").write_text(
        json.dumps({"items": aggregate}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote boosted PaddleOCR results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
