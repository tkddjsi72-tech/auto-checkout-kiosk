#!/usr/bin/env python3
"""PaddleOCR with product-crop / rotation / upscale / CLAHE preprocessing."""

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
DEFAULT_OUTPUT_DIR = ROOT / "output" / "db_input_matching" / "paddleocr_preproc"
SYMBOL_ONLY_RE = re.compile(r"^[\W_]+$", re.UNICODE)
FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
)
ROTATIONS = (
    ("r0", None),
    ("r90", cv2.ROTATE_90_CLOCKWISE),
    ("r180", cv2.ROTATE_180),
    ("r270", cv2.ROTATE_90_COUNTERCLOCKWISE),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PaddleOCR with preprocessing")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-side", type=int, default=1400, help="Upscale short side to this")
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


def upscale_min_side(bgr: np.ndarray, min_side: int) -> tuple[np.ndarray, float]:
    height, width = bgr.shape[:2]
    short = min(height, width)
    if short >= min_side:
        return bgr, 1.0
    scale = min_side / float(short)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC), scale


def map_point_from_rotated(
    x: float,
    y: float,
    rot_name: str,
    rot_h: int,
    rot_w: int,
) -> tuple[float, float]:
    """Map a point in rotated image coords back to pre-rotation crop coords."""
    if rot_name == "r0":
        return x, y
    if rot_name == "r90":
        # rotated = ROTATE_90_CLOCKWISE
        return y, rot_w - 1 - x
    if rot_name == "r180":
        return rot_w - 1 - x, rot_h - 1 - y
    if rot_name == "r270":
        return rot_h - 1 - y, x
    return x, y


def score_texts(texts: list[str]) -> tuple[int, int]:
    hangul = sum(("가" <= ch <= "힣") for text in texts for ch in text)
    useful = sum(1 for text in texts if normalize_keep(text))
    return hangul + useful * 2, useful


def normalize_keep(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", text or "")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size, index=0)
            except OSError:
                pass
    return ImageFont.load_default()


def save_overlay(
    bgr: np.ndarray,
    output_path: Path,
    texts: list[str],
    polys: list[list[list[float]]],
    scores: list[float],
) -> None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    canvas = Image.fromarray(rgb)
    draw = ImageDraw.Draw(canvas)
    font = load_font(max(22, round(min(canvas.size) * 0.012)))
    for index, (text, poly) in enumerate(zip(texts, polys), start=1):
        if not text or len(poly) < 3:
            continue
        points = [(int(round(p[0])), int(round(p[1]))) for p in poly]
        score = scores[index - 1] if index - 1 < len(scores) else 0.0
        draw.line(points + [points[0]], fill=(20, 90, 220), width=4)
        label = f"{index}. {text} ({score:.2f})"
        x = max(0, min(p[0] for p in points))
        y = max(0, min(p[1] for p in points))
        left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
        width = right - left
        height = bottom - top
        pad = 5
        lx = min(x, max(0, canvas.width - width - pad * 2))
        ly = max(0, y - height - pad * 2)
        draw.rectangle(
            (lx, ly, lx + width + pad * 2, ly + height + pad * 2),
            fill=(20, 90, 220),
        )
        draw.text((lx + pad, ly + pad), label, font=font, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="JPEG", quality=92)


def run_one(
    ocr: Any,
    image_path: Path,
    expected_product: str,
    output_dir: Path,
    min_side: int,
    output_stem: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    original = read_bgr(image_path)
    stem = output_stem or image_path.stem
    x0, y0, x1, y1 = product_crop_box(original)
    crop = original[y0:y1, x0:x1].copy()
    enhanced = apply_clahe(crop)
    scaled, scale = upscale_min_side(enhanced, min_side)

    best: dict[str, Any] | None = None
    trial_dir = output_dir / "preproc_trials" / stem
    trial_dir.mkdir(parents=True, exist_ok=True)
    write_bgr(trial_dir / "crop_clahe_scaled.jpg", scaled)

    for rot_name, rot_code in ROTATIONS:
        rotated = scaled if rot_code is None else cv2.rotate(scaled, rot_code)
        tmp = trial_dir / f"{rot_name}.jpg"
        write_bgr(tmp, rotated)
        page = unwrap_page(ocr.predict(str(tmp)))
        texts = [str(text).strip() for text in (page.get("rec_texts") or []) if str(text).strip()]
        scores = [float(score) for score in (page.get("rec_scores") or [])]
        polys_rot = page.get("rec_polys") or page.get("dt_polys") or []
        hangul_score, useful = score_texts(texts)

        # map polys back to original image coordinates
        rot_h, rot_w = rotated.shape[:2]
        mapped_polys: list[list[list[float]]] = []
        for poly in polys_rot:
            mapped = []
            for point in poly:
                px, py = float(point[0]), float(point[1])
                cx, cy = map_point_from_rotated(px, py, rot_name, rot_h, rot_w)
                ox = cx / scale + x0
                oy = cy / scale + y0
                mapped.append([ox, oy])
            mapped_polys.append(mapped)

        candidate = {
            "rotation": rot_name,
            "score": hangul_score,
            "useful_text_count": useful,
            "rec_texts": texts,
            "rec_scores": scores,
            "rec_polys": mapped_polys,
        }
        if best is None or candidate["score"] > best["score"] or (
            candidate["score"] == best["score"]
            and candidate["useful_text_count"] > best["useful_text_count"]
        ):
            best = candidate

    assert best is not None
    elapsed_s = time.perf_counter() - started
    result = {
        "input_image": f"{stem}{image_path.suffix}",
        "source_image": str(image_path),
        "expected_product": expected_product,
        "model": {
            "detection": "PP-OCRv6_medium_det",
            "recognition": "korean_PP-OCRv5_mobile_rec",
        },
        "preprocess": {
            "crop_xyxy": [x0, y0, x1, y1],
            "clahe": True,
            "upscale_min_side": min_side,
            "upscale_factor": round(scale, 4),
            "selected_rotation": best["rotation"],
            "selection_score": best["score"],
        },
        "elapsed_s": round(elapsed_s, 4),
        "rec_texts": best["rec_texts"],
        "rec_scores": best["rec_scores"],
        "rec_polys": best["rec_polys"],
        "text_set": build_text_set(best["rec_texts"]),
    }
    (output_dir / f"{stem}_res.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    save_overlay(
        original,
        output_dir / "overlays" / f"{stem}.jpg",
        result["rec_texts"],
        result["rec_polys"],
        result["rec_scores"],
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
        output_stem = Path(item["input_image"]).stem
        result = run_one(
            ocr,
            image_path,
            item.get("expected_product")
            or ", ".join(item.get("expected_products") or []),
            output_dir,
            args.min_side,
            output_stem=output_stem,
        )
        aggregate.append(result)
        print(
            f"[{index}/{len(items)}] {output_stem}: "
            f"rot={result['preprocess']['selected_rotation']} "
            f"texts={len(result['rec_texts'])} "
            f"score={result['preprocess']['selection_score']} "
            f"elapsed={result['elapsed_s']:.1f}s"
        )

    (output_dir / "results.json").write_text(
        json.dumps({"items": aggregate}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote preprocessed PaddleOCR results: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
