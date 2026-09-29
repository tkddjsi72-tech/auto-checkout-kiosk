#!/usr/bin/env python3
"""Combine Laura barcode, PaddleOCR, and simulated weight DB matching.

Decision policy:
  1. Laura barcode detection, PaddleOCR, and weight acquisition are independent cues.
  2. Multiple barcode DB matches -> REINSERT. Single/no barcode match -> OCR.
  3. OCR must match exactly one DB product. If barcode matched, OCR must agree.
  4. The measured weight must be inside that OCR product's tolerance.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "dataset" / "single_front"
DEFAULT_DB_TEXT_DIR = ROOT / "db" / "text_front"
DEFAULT_PRODUCT_DB = ROOT / "db" / "products.json"
DEFAULT_LAURA_RESULTS = ROOT / "output" / "db_input_matching" / "laura" / "results.jsonl"
DEFAULT_PADDLE_DIR = ROOT / "output" / "db_input_matching" / "paddleocr"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "db_input_matching"

ELEMENT_FUZZY_THRESHOLD = 0.70
PRODUCT_COVERAGE_THRESHOLD = 0.60
# Containment contributes its length ratio only when ratio >= this value.
# Example: "자유" ⊂ "자유시간" = 0.5 → +0.5; "IN" ⊂ "PRINGLES" = 0.25 → 0.
CONTAINMENT_MIN_RATIO = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DB matching decision policy")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--db-text-dir",
        type=Path,
        action="append",
        default=None,
        help="DB text dir. Pass multiple times to merge (max coverage across views).",
    )
    parser.add_argument("--product-db", type=Path, default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--laura-results", type=Path, default=DEFAULT_LAURA_RESULTS)
    parser.add_argument(
        "--paddle-dir",
        type=Path,
        action="append",
        default=None,
        help="PaddleOCR result dir. Pass multiple times to ensemble by max coverage.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--element-threshold", type=float, default=ELEMENT_FUZZY_THRESHOLD)
    parser.add_argument("--coverage-threshold", type=float, default=PRODUCT_COVERAGE_THRESHOLD)
    args = parser.parse_args()
    if not args.paddle_dir:
        args.paddle_dir = [DEFAULT_PADDLE_DIR]
    if not args.db_text_dir:
        args.db_text_dir = [DEFAULT_DB_TEXT_DIR]
    return args


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_text(text: str) -> str:
    """Strip symbols/spaces for comparison-only matching (does not rewrite DB files)."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[^0-9a-z가-힣]+", "", text)


def is_scorable_token(norm: str) -> bool:
    """Drop empty/noise tokens from the coverage denominator."""
    if not norm:
        return False
    # Single latin letter or lone 'x' is usually OCR/DB split noise.
    if len(norm) == 1 and (norm.isascii() and not norm.isdigit()):
        return False
    return True


def token_match_score(db_token: str, query_token: str) -> tuple[float, str]:
    """Return (score, method) using exact / containment / fuzzy rules."""
    db_norm = normalize_text(db_token)
    query_norm = normalize_text(query_token)
    if not db_norm or not query_norm:
        return 0.0, "none"

    if db_norm == query_norm:
        return 1.0, "exact"

    # Prefer DB token contained in OCR token: "중량"/"100g" in "중량100g".
    # Raw length ratio; soft credit is applied later if ratio >= CONTAINMENT_MIN_RATIO.
    # Require length >= 2 to avoid "t" ⊂ "cantata" false positives.
    if len(db_norm) >= 2 and db_norm in query_norm:
        return len(db_norm) / len(query_norm), "contain_db_in_query"

    # Reverse containment: "자유" ⊂ "자유시간" = 0.5.
    if len(query_norm) >= 2 and query_norm in db_norm:
        return len(query_norm) / len(db_norm), "contain_query_in_db"

    fuzzy = SequenceMatcher(None, db_norm, query_norm).ratio()
    return fuzzy, "fuzzy"


def accumulate_query_in_db_coverage(
    db_norm: str,
    query_norms: list[str],
) -> tuple[float, list[str]]:
    """Cover DB chars by OCR substrings; e.g. 자유+시간 → 자유시간 = 1.0."""
    if len(db_norm) < 2:
        return 0.0, []
    covered = [False] * len(db_norm)
    used_queries: list[str] = []
    # Longer queries first so bigger chunks claim spans earlier.
    candidates = sorted(
        {norm for norm in query_norms if len(norm) >= 2},
        key=len,
        reverse=True,
    )
    for query_norm in candidates:
        start = db_norm.find(query_norm)
        found = False
        while start >= 0:
            for index in range(start, start + len(query_norm)):
                covered[index] = True
            found = True
            start = db_norm.find(query_norm, start + 1)
        if found:
            used_queries.append(query_norm)
    ratio = sum(covered) / len(db_norm)
    return ratio, used_queries


def load_text_db(text_dir: Path) -> dict[str, dict[str, Any]]:
    products: dict[str, dict[str, Any]] = {}
    for path in sorted(text_dir.rglob("*_text_set.json")):
        payload = read_json(path)
        product = str(payload.get("product", "")).strip()
        if not product:
            continue
        view = str(payload.get("view") or path.stem.replace("_text_set", ""))
        tokens = [str(token) for token in payload.get("text_set", []) if str(token).strip()]
        bucket = products.setdefault(product, {"views": []})
        bucket["views"].append({"view": view, "tokens": tokens, "path": str(path)})
    if not products:
        raise ValueError(f"No *_text_set.json found under: {text_dir}")
    for info in products.values():
        info["tokens"] = info["views"][0]["tokens"]
        info["path"] = info["views"][0]["path"]
    return products


def merge_text_dbs(text_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for text_dir in text_dirs:
        for product, info in load_text_db(text_dir).items():
            bucket = merged.setdefault(product, {"views": []})
            bucket["views"].extend(info["views"])
    for info in merged.values():
        info["tokens"] = info["views"][0]["tokens"]
        info["path"] = info["views"][0]["path"]
    return merged


def token_credit(
    score: float,
    method: str,
    element_threshold: float,
    containment_min_ratio: float = CONTAINMENT_MIN_RATIO,
) -> float:
    """Containment: proportional credit if ratio >= min. Exact/fuzzy: binary at threshold."""
    if method in {
        "contain_db_in_query",
        "contain_query_in_db",
        "contain_query_spans",
        "contain_joined",
    }:
        return score if score >= containment_min_ratio else 0.0
    if score >= element_threshold:
        return 1.0
    return 0.0


def score_ocr_candidate(
    query_tokens: list[str],
    db_tokens: list[str],
    element_threshold: float,
) -> dict[str, Any]:
    """Match with normalize + soft containment + fuzzy (rule-based, no LLM)."""
    query_norms = [normalize_text(token) for token in query_tokens]
    query_compact = "".join(norm for norm in query_norms if norm)
    matched_pairs: list[dict[str, Any]] = []
    unmatched_db_tokens: list[str] = []
    skipped_db_tokens: list[str] = []
    credit_sum = 0.0

    for db_token in db_tokens:
        db_norm = normalize_text(db_token)
        if not is_scorable_token(db_norm):
            skipped_db_tokens.append(db_token)
            continue

        best_token = ""
        best_score = 0.0
        best_method = "none"
        for query_token in query_tokens:
            score, method = token_match_score(db_token, query_token)
            if score > best_score:
                best_score = score
                best_token = query_token
                best_method = method

        # Multi-OCR containment into one DB token:
        # "자유" + "시간" cover "자유시간" → 0.5 + 0.5 = 1.0 (non-overlapping char spans).
        span_score, span_queries = accumulate_query_in_db_coverage(db_norm, query_norms)
        if span_score > best_score:
            best_score = span_score
            best_method = "contain_query_spans"
            best_token = "+".join(span_queries) if span_queries else best_token

        # Joined OCR fallback: "뒷면"+"확인" style splits → "뒷면확인".
        # Prefer this over weak single-token containment when full token is present.
        if (
            best_method.startswith("contain")
            and best_score < 0.9
            and len(db_norm) >= 2
            and db_norm in query_compact
        ) or (
            best_score < element_threshold
            and not best_method.startswith("contain")
            and len(db_norm) >= 2
            and db_norm in query_compact
        ):
            best_score = 0.9
            best_method = "contain_joined"
            best_token = "[joined_ocr]"

        credit = token_credit(best_score, best_method, element_threshold)
        if credit > 0:
            credit_sum += credit
            matched_pairs.append(
                {
                    "db_token": db_token,
                    "db_token_norm": db_norm,
                    "query_token": best_token,
                    "similarity": round(best_score, 4),
                    "credit": round(credit, 4),
                    "method": best_method,
                }
            )
        else:
            unmatched_db_tokens.append(db_token)

    scored_count = len(matched_pairs) + len(unmatched_db_tokens)
    coverage = credit_sum / scored_count if scored_count else 0.0
    return {
        "db_token_count": len(db_tokens),
        "scored_db_token_count": scored_count,
        "skipped_db_tokens": skipped_db_tokens,
        "matched_count": len(matched_pairs),
        "credit_sum": round(credit_sum, 4),
        "coverage": round(coverage, 4),
        "matched_pairs": matched_pairs,
        "unmatched_db_tokens": unmatched_db_tokens,
        "match_mode": "normalize+soft_contain+fuzzy",
        "containment_min_ratio": CONTAINMENT_MIN_RATIO,
    }


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def padded_crop(image: np.ndarray, xyxy: list[float], pad_ratio: float = 0.18) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = xyxy
    box_width = max(1.0, x2 - x1)
    box_height = max(1.0, y2 - y1)
    pad_x = box_width * pad_ratio
    pad_y = box_height * pad_ratio
    left = max(0, int(round(x1 - pad_x)))
    top = max(0, int(round(y1 - pad_y)))
    right = min(width, int(round(x2 + pad_x)))
    bottom = min(height, int(round(y2 + pad_y)))
    return image[top:bottom, left:right]


def normalize_barcode_text(text: str) -> str:
    return re.sub(r"[\s-]+", "", text or "")


def is_valid_ean13(code: str) -> bool:
    if not code.isdigit() or len(code) != 13:
        return False
    digits = [int(char) for char in code]
    checksum = sum(digits[index] * (1 if index % 2 == 0 else 3) for index in range(12))
    return (10 - checksum % 10) % 10 == digits[12]


def is_plausible_barcode(code: str) -> bool:
    """Accept EAN-13 with checksum, or other long numeric retail codes."""
    if is_valid_ean13(code):
        return True
    # Avoid short false positives like random EAN-8 from blur.
    return code.isdigit() and 12 <= len(code) <= 14


def _barcode_image_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        color = image
    else:
        gray = image
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    sharpen = cv2.filter2D(
        gray,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    )
    blur = cv2.GaussianBlur(gray, (0, 0), 1.2)
    unsharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    return [
        ("color", color),
        ("gray", gray),
        ("clahe", clahe),
        ("sharpen", sharpen),
        ("unsharp", unsharp),
    ]


def _decode_opencv_once(image: np.ndarray) -> list[tuple[str, str]]:
    if image.size == 0 or not hasattr(cv2, "barcode_BarcodeDetector"):
        return []
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    detector = cv2.barcode_BarcodeDetector()
    try:
        ok, infos, types, _points = detector.detectAndDecodeWithType(image)
    except cv2.error:
        return []
    if not ok:
        return []
    results: list[tuple[str, str]] = []
    for info, barcode_type in zip(infos, types):
        code = normalize_barcode_text(str(info))
        if code:
            results.append((code, str(barcode_type)))
    return results


def _decode_pyzbar_once(image: np.ndarray) -> list[tuple[str, str]]:
    try:
        from pyzbar import pyzbar
        from pyzbar.pyzbar import ZBarSymbol
    except ImportError:
        return []
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    results: list[tuple[str, str]] = []
    for obj in pyzbar.decode(
        image,
        symbols=[
            ZBarSymbol.EAN13,
            ZBarSymbol.EAN8,
            ZBarSymbol.UPCA,
            ZBarSymbol.UPCE,
            ZBarSymbol.CODE128,
            ZBarSymbol.CODE39,
        ],
    ):
        code = normalize_barcode_text(obj.data.decode("utf-8", errors="ignore"))
        if code:
            results.append((code, str(obj.type)))
    return results


def decode_with_opencv(image: np.ndarray) -> list[dict[str, str]]:
    """Backward-compatible name: boosted OpenCV + pyzbar decode."""
    return decode_barcode_boosted(image)


def decode_barcode_boosted(image: np.ndarray) -> list[dict[str, str]]:
    """Decode with OpenCV + pyzbar over preprocess / scale / rotation variants."""
    if image.size == 0:
        return []

    decoded: list[dict[str, str]] = []
    seen: set[str] = set()
    rotations = (
        ("r0", None),
        ("r90", cv2.ROTATE_90_CLOCKWISE),
        ("r180", cv2.ROTATE_180),
        ("r270", cv2.ROTATE_90_COUNTERCLOCKWISE),
    )
    scales = (1.0, 1.5, 2.0, 2.5, 3.0)

    for variant_name, variant in _barcode_image_variants(image):
        for scale in scales:
            if scale == 1.0:
                scaled = variant
                scale_name = "s1"
            else:
                height, width = variant.shape[:2]
                if max(height, width) * scale > 2400:
                    continue
                scaled = cv2.resize(
                    variant,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_CUBIC,
                )
                scale_name = f"s{scale:g}"
            for rotation_name, rotation in rotations:
                rotated = scaled if rotation is None else cv2.rotate(scaled, rotation)
                for engine_name, engine in (
                    ("opencv", _decode_opencv_once),
                    ("pyzbar", _decode_pyzbar_once),
                ):
                    for code, barcode_type in engine(rotated):
                        if not is_plausible_barcode(code) or code in seen:
                            continue
                        seen.add(code)
                        decoded.append(
                            {
                                "code": code,
                                "type": barcode_type,
                                "engine": engine_name,
                                "preprocess": f"{variant_name}/{scale_name}/{rotation_name}",
                            }
                        )
                if decoded:
                    return decoded
    return decoded


def decode_laura_detections(
    image_path: Path,
    detections: list[dict[str, Any]],
    crop_output_dir: Path,
) -> list[dict[str, Any]]:
    image = load_image(image_path)
    decoded_detections: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        xyxy = [float(value) for value in detection.get("xyxy", [])]
        if len(xyxy) != 4:
            continue
        crop = padded_crop(image, xyxy, pad_ratio=0.18)
        decoded = decode_barcode_boosted(crop)
        # Wider crop retry when the tight box is too blurry / clipped.
        if not decoded:
            wide = padded_crop(image, xyxy, pad_ratio=0.45)
            decoded = decode_barcode_boosted(wide)
            if decoded:
                crop = wide
        crop_output_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_output_dir / f"{image_path.stem}_barcode_{index}.jpg"
        cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 94])[1].tofile(
            str(crop_path)
        )
        decoded_detections.append(
            {
                **detection,
                "crop_path": str(crop_path),
                "decoded": decoded,
            }
        )
    return decoded_detections


def poly_to_xyxy(poly: list[Any]) -> list[float] | None:
    points: list[tuple[float, float]] = []
    for point in poly or []:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            points.append((float(point[0]), float(point[1])))
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_intersection_area(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def bboxes_overlap(a: list[float], b: list[float]) -> bool:
    return bbox_intersection_area(a, b) > 0.0


def collect_ocr_text_boxes(paddle_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect OCR texts that have aligned polygons (skip boost union without polys)."""
    boxes: list[dict[str, Any]] = []
    for payload in paddle_payloads:
        texts = payload.get("rec_texts") or []
        polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        if not texts or not polys:
            continue
        source = str(payload.get("_source_name") or payload.get("_source_dir") or "")
        for index, text in enumerate(texts):
            if index >= len(polys):
                break
            xyxy = poly_to_xyxy(polys[index])
            if xyxy is None:
                continue
            boxes.append(
                {
                    "text": str(text),
                    "xyxy": xyxy,
                    "source": source,
                }
            )
    return boxes


def recover_known_barcodes_from_ocr_text(
    text: str,
    known_barcodes: set[str],
) -> list[str]:
    """Recover DB barcodes from OCR digit strings (exact / substring, min 12 digits)."""
    if not known_barcodes:
        return []
    compact = re.sub(r"[\s-]+", "", str(text))
    digits = re.sub(r"\D", "", compact)
    found: list[str] = []
    seen: set[str] = set()

    def add(code: str) -> None:
        if code in known_barcodes and code not in seen:
            seen.add(code)
            found.append(code)

    for match in re.findall(r"\d{8,14}", compact):
        add(normalize_barcode_text(match))
    for code in known_barcodes:
        if code and code in digits:
            add(code)
    # OCR often drops a leading digit near the bars (e.g. 801019206818 ⊂ 8801019206818).
    if len(digits) >= 12:
        for code in known_barcodes:
            if digits in code:
                add(code)
        for length in range(12, min(15, len(digits) + 1)):
            for start in range(0, len(digits) - length + 1):
                run = digits[start : start + length]
                for code in known_barcodes:
                    if run == code or run in code:
                        add(code)
    return found


def apply_ocr_digit_barcode_fallback(
    detections: list[dict[str, Any]],
    ocr_boxes: list[dict[str, Any]],
    known_barcodes: set[str],
) -> list[dict[str, str]]:
    """Fill undecoded Laura boxes using OCR digits whose bbox overlaps the barcode box.

    Gate:
      1) at least one Laura barcode detection exists (caller responsibility)
      2) OCR text bbox overlaps that detection's xyxy
      3) recovered digits match a known DB barcode
    """
    if not detections or not ocr_boxes or not known_barcodes:
        return []

    recovered: list[dict[str, str]] = []
    for detection in detections:
        if detection.get("decoded"):
            continue
        xyxy = detection.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        barcode_xyxy = [float(v) for v in xyxy]
        hit_codes: list[str] = []
        evidence: list[dict[str, Any]] = []
        for box in ocr_boxes:
            if not bboxes_overlap(barcode_xyxy, box["xyxy"]):
                continue
            codes = recover_known_barcodes_from_ocr_text(box["text"], known_barcodes)
            if not codes:
                continue
            for code in codes:
                if code not in hit_codes:
                    hit_codes.append(code)
            evidence.append(
                {
                    "text": box["text"],
                    "source": box["source"],
                    "xyxy": box["xyxy"],
                    "codes": codes,
                }
            )
        if not hit_codes:
            continue
        decoded = [
            {
                "code": code,
                "type": "OCR_DIGITS",
                "engine": "ocr_digits_overlap",
                "preprocess": "bbox_overlap",
            }
            for code in hit_codes
        ]
        detection["decoded"] = decoded
        detection["ocr_digit_fallback"] = evidence
        recovered.extend(decoded)
    return recovered


def unique_in_order(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def make_simulated_measurement(product: dict[str, Any]) -> float:
    """Deterministic value at +25% of tolerance, guaranteed in-range."""
    return round(float(product["weight_g"]) + float(product["tolerance_g"]) * 0.25, 2)


def status_for_count(count: int) -> str:
    if count == 0:
        return "NO_MATCH"
    if count == 1:
        return "SINGLE"
    return "MULTIPLE"


def load_paddle_result(paddle_dir: Path, image_stem: str) -> dict[str, Any]:
    path = paddle_dir / f"{image_stem}_res.json"
    payload = read_json(path)
    payload["_source_dir"] = str(paddle_dir)
    payload["_source_name"] = paddle_dir.name
    return payload


def score_products_for_ocr(
    query_tokens: list[str],
    text_db: dict[str, dict[str, Any]],
    element_threshold: float,
    source_name: str,
) -> list[dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for product_name, text_info in text_db.items():
        views = text_info.get("views") or [{"view": "front", "tokens": text_info["tokens"]}]
        best: dict[str, Any] | None = None
        for view in views:
            score = score_ocr_candidate(
                query_tokens,
                view["tokens"],
                element_threshold,
            )
            candidate = {**score, "db_view": view.get("view", "")}
            if best is None or (
                candidate["coverage"],
                candidate["matched_count"],
            ) > (best["coverage"], best["matched_count"]):
                best = candidate
        scores.append({"product": product_name, "ocr_source": source_name, **best})
    return scores


def ensemble_candidate_scores(
    per_source_scores: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """For each product, keep the OCR source with the higher coverage."""
    best_by_product: dict[str, dict[str, Any]] = {}
    for source_scores in per_source_scores:
        for candidate in source_scores:
            product = candidate["product"]
            current = best_by_product.get(product)
            if current is None or (
                candidate["coverage"],
                candidate["matched_count"],
            ) > (current["coverage"], current["matched_count"]):
                best_by_product[product] = candidate
    ensembled = list(best_by_product.values())
    ensembled.sort(
        key=lambda candidate: (
            candidate["coverage"],
            candidate["matched_count"],
            candidate["product"],
        ),
        reverse=True,
    )
    return ensembled


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    manifest = read_json(input_dir / "manifest.json")
    products_payload = read_json(args.product_db.resolve())
    products = products_payload.get("items", [])
    product_by_name = {item["product"]: item for item in products}
    barcode_to_products: dict[str, list[str]] = {}
    for product in products:
        barcode_to_products.setdefault(str(product["barcode"]), []).append(product["product"])

    text_db = merge_text_dbs([path.resolve() for path in args.db_text_dir])
    laura_path = args.laura_results.resolve()
    laura_rows = read_jsonl(laura_path) if laura_path.exists() else []
    laura_by_file = {
        row.get("record", {}).get("file_name"): row
        for row in laura_rows
        if row.get("record", {}).get("file_name")
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for item in manifest.get("items", []):
        image_name = item["input_image"]
        image_path = input_dir / image_name
        if not image_path.exists() and item.get("source"):
            image_path = (ROOT / item["source"]).resolve()
        expected_product = item["expected_product"]
        expected_db = product_by_name[expected_product]
        measured_weight = make_simulated_measurement(expected_db)

        laura_row = laura_by_file.get(image_name, {})
        raw_detections = laura_row.get("barcodes", [])
        decoded_detections = decode_laura_detections(
            image_path,
            raw_detections,
            output_dir / "barcode_crops",
        )
        decoded_codes = unique_in_order(
            [
                decoded["code"]
                for detection in decoded_detections
                for decoded in detection.get("decoded", [])
            ]
        )
        barcode_matches = unique_in_order(
            [
                product
                for code in decoded_codes
                for product in barcode_to_products.get(code, [])
            ]
        )
        barcode_status = status_for_count(len(barcode_matches))

        image_stem = Path(image_name).stem
        paddle_sources = [path.resolve() for path in args.paddle_dir]
        paddle_payloads = [
            load_paddle_result(paddle_dir, image_stem) for paddle_dir in paddle_sources
        ]
        per_source_scores: list[list[dict[str, Any]]] = []
        source_summaries: list[dict[str, Any]] = []
        for paddle in paddle_payloads:
            query_tokens = [str(token) for token in paddle.get("text_set", [])]
            source_name = str(paddle["_source_name"])
            source_scores = score_products_for_ocr(
                query_tokens,
                text_db,
                args.element_threshold,
                source_name,
            )
            per_source_scores.append(source_scores)
            top_source = max(
                source_scores,
                key=lambda candidate: (
                    candidate["coverage"],
                    candidate["matched_count"],
                    candidate["product"],
                ),
            )
            source_summaries.append(
                {
                    "source": source_name,
                    "dir": paddle["_source_dir"],
                    "rec_texts": paddle.get("rec_texts", []),
                    "rec_scores": paddle.get("rec_scores", []),
                    "text_set": query_tokens,
                    "top_product": top_source["product"],
                    "top_coverage": top_source["coverage"],
                }
            )

        candidate_scores = ensemble_candidate_scores(per_source_scores)
        selected_source = (
            candidate_scores[0]["ocr_source"] if candidate_scores else source_summaries[0]["source"]
        )
        paddle = next(
            payload
            for payload in paddle_payloads
            if payload["_source_name"] == selected_source
        )
        query_tokens = [str(token) for token in paddle.get("text_set", [])]
        ocr_matches = [
            candidate["product"]
            for candidate in candidate_scores
            if candidate["coverage"] >= args.coverage_threshold
        ]
        ocr_status = status_for_count(len(ocr_matches))
        source_detail = ", ".join(
            f"{summary['source']} top={summary['top_product']}({summary['top_coverage']:.3f})"
            for summary in source_summaries
        )

        flow = [
            {
                "stage": 1,
                "status": "COMPLETED",
                "detail": (
                    f"Laura detection={len(raw_detections)}, "
                    f"PaddleOCR sources={len(paddle_sources)} [{source_detail}], "
                    f"selected={selected_source}, "
                    f"simulated weight={measured_weight:.2f}g"
                ),
            }
        ]
        final_action = "REINSERT"
        final_status = ""
        final_product: str | None = None
        weight_result: dict[str, Any] = {
            "simulated": True,
            "measured_g": measured_weight,
            "expected_product_for_simulation": expected_product,
            "evaluated": False,
        }

        if barcode_status == "MULTIPLE":
            final_status = "BARCODE_MULTIPLE"
            flow.append(
                {
                    "stage": 2,
                    "status": "MULTIPLE",
                    "detail": f"Barcode DB matches={barcode_matches}; 재투입 요구",
                }
            )
        else:
            flow.append(
                {
                    "stage": 2,
                    "status": barcode_status,
                    "detail": (
                        f"Barcode DB matches={barcode_matches or '없음'}; "
                        "OCR matching으로 진행"
                    ),
                }
            )

            if ocr_status == "MULTIPLE":
                final_status = "OCR_MULTIPLE"
                flow.append(
                    {
                        "stage": 3,
                        "status": "MULTIPLE",
                        "detail": f"OCR DB matches={ocr_matches}; 재투입 요구",
                    }
                )
            elif ocr_status == "NO_MATCH":
                final_status = "OCR_NO_MATCH"
                flow.append(
                    {
                        "stage": 3,
                        "status": "NO_MATCH",
                        "detail": "OCR DB match 없음; 재투입 요구",
                    }
                )
            else:
                ocr_product = ocr_matches[0]
                if barcode_status == "SINGLE" and barcode_matches[0] != ocr_product:
                    final_status = "BARCODE_OCR_CONFLICT"
                    flow.append(
                        {
                            "stage": 3,
                            "status": "CONFLICT",
                            "detail": (
                                f"Barcode={barcode_matches[0]}, OCR={ocr_product}; "
                                "재투입 요구"
                            ),
                        }
                    )
                else:
                    flow.append(
                        {
                            "stage": 3,
                            "status": "SINGLE",
                            "detail": f"OCR DB match={ocr_product}; weight 비교로 진행",
                        }
                    )
                    matched_product = product_by_name.get(ocr_product)
                    if matched_product is None:
                        final_status = "WEIGHT_DB_MISSING"
                        flow.append(
                            {
                                "stage": 4,
                                "status": "NO_MATCH",
                                "detail": f"{ocr_product} weight DB 없음; 재투입 요구",
                            }
                        )
                    else:
                        nominal = float(matched_product["weight_g"])
                        tolerance = float(matched_product["tolerance_g"])
                        difference = abs(measured_weight - nominal)
                        passed = difference <= tolerance + 1e-9
                        weight_result.update(
                            {
                                "evaluated": True,
                                "matched_product": ocr_product,
                                "nominal_g": nominal,
                                "tolerance_g": tolerance,
                                "difference_g": round(difference, 4),
                                "passed": passed,
                            }
                        )
                        if passed:
                            final_action = "CONFIRMED"
                            final_status = "CONFIRMED"
                            final_product = ocr_product
                            detail = (
                                f"|{measured_weight:.2f}-{nominal:g}|="
                                f"{difference:.2f}g <= {tolerance:g}g; {ocr_product} 확정"
                            )
                        else:
                            final_status = "WEIGHT_OUT_OF_TOLERANCE"
                            detail = (
                                f"|{measured_weight:.2f}-{nominal:g}|="
                                f"{difference:.2f}g > {tolerance:g}g; 재투입 요구"
                            )
                        flow.append(
                            {
                                "stage": 4,
                                "status": "PASS" if passed else "FAIL",
                                "detail": detail,
                            }
                        )

        result = {
            "input_image": image_name,
            "input_path": str(image_path),
            "expected_product": expected_product,
            "barcode": {
                "detector": "Laura YOLOv5 barcodeDetector",
                "detector_confidence_threshold": laura_row.get("thresholds", {}).get(
                    "barcode_confidence"
                ),
                "detections": decoded_detections,
                "decoded_codes": decoded_codes,
                "db_matches": barcode_matches,
                "status": barcode_status,
            },
            "ocr": {
                "engine": paddle.get("model", {}),
                "ensemble": {
                    "mode": "max_coverage_per_product",
                    "sources": source_summaries,
                    "selected_source_for_texts": selected_source,
                },
                "rec_texts": paddle.get("rec_texts", []),
                "rec_scores": paddle.get("rec_scores", []),
                "text_set": query_tokens,
                "element_fuzzy_threshold": args.element_threshold,
                "product_coverage_threshold": args.coverage_threshold,
                "candidate_scores": candidate_scores,
                "db_matches": ocr_matches,
                "status": ocr_status,
            },
            "weight": weight_result,
            "flow": flow,
            "final": {
                "action": final_action,
                "status": final_status,
                "product": final_product,
                "correct_vs_manifest": final_product == expected_product,
            },
        }
        results.append(result)
        print(
            f"{image_name}: barcode={barcode_status} {barcode_matches} "
            f"ocr={ocr_status} {ocr_matches} -> {final_status}"
        )

    report = {
        "algorithm": {
            "barcode_detector": "Laura YOLOv5 barcodeDetector",
            "barcode_decoder": "OpenCV BarcodeDetector",
            "ocr": "PP-OCRv6_medium_det + korean_PP-OCRv5_mobile_rec",
            "ocr_match_mode": "normalize+soft_contain+fuzzy",
            "ocr_containment_min_ratio": CONTAINMENT_MIN_RATIO,
            "ocr_ensemble_mode": "max_coverage_per_product",
            "ocr_sources": [str(path.resolve()) for path in args.paddle_dir],
            "ocr_element_fuzzy_threshold": args.element_threshold,
            "ocr_product_coverage_threshold": args.coverage_threshold,
            "weight_source": "simulated at nominal + 25% of tolerance",
        },
        "summary": {
            "image_count": len(results),
            "confirmed_count": sum(
                result["final"]["action"] == "CONFIRMED" for result in results
            ),
            "reinsert_count": sum(
                result["final"]["action"] == "REINSERT" for result in results
            ),
        },
        "items": results,
    }
    (output_dir / "matching_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    with (output_dir / "matching_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fieldnames = [
            "input_image",
            "expected_product",
            "measured_weight_g",
            "barcode_detection_count",
            "decoded_barcodes",
            "barcode_matches",
            "ocr_text",
            "ocr_matches",
            "ocr_top_product",
            "ocr_top_coverage",
            "ocr_top_view",
            "ocr_source",
            "weight_passed",
            "final_status",
            "final_product",
            "action",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            top = result["ocr"]["candidate_scores"][0]
            writer.writerow(
                {
                    "input_image": result["input_image"],
                    "expected_product": result["expected_product"],
                    "measured_weight_g": result["weight"]["measured_g"],
                    "barcode_detection_count": len(result["barcode"]["detections"]),
                    "decoded_barcodes": "|".join(result["barcode"]["decoded_codes"]),
                    "barcode_matches": "|".join(result["barcode"]["db_matches"]),
                    "ocr_text": " | ".join(result["ocr"]["rec_texts"]),
                    "ocr_matches": "|".join(result["ocr"]["db_matches"]),
                    "ocr_top_product": top["product"],
                    "ocr_top_coverage": top["coverage"],
                    "ocr_top_view": top.get("db_view", ""),
                    "ocr_source": top.get("ocr_source", ""),
                    "weight_passed": result["weight"].get("passed", ""),
                    "final_status": result["final"]["status"],
                    "final_product": result["final"]["product"] or "",
                    "action": result["final"]["action"],
                }
            )

    print(f"Wrote matching results: {output_dir / 'matching_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
