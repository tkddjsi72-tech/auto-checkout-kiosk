#!/usr/bin/env python3
"""Multi-item cue matching for dataset/multiple images.

Same decision algorithm as single-item kiosk policy:

  1. top-view input
  2. barcode detection + OCR + weight measurement (independent)
  3. barcode matching
     - multiple products → MULTIPLE → REINSERT
     - single product → weight vs that product
         match → CONFIRMED / mismatch → MULTIPLE → REINSERT
     - no match → OCR
  4. OCR matching
     - multiple products → MULTIPLE → REINSERT
     - single product → weight vs that product
         match → CONFIRMED / mismatch → MULTIPLE → REINSERT
     - no match → REINSERT

Measured weight for multi-item trays = sum of all expected item weights.
So multi scenes almost always REINSERT (barcode/OCR multiple, or single+weight mismatch).
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from run_db_input_matching import (
    CONTAINMENT_MIN_RATIO,
    ELEMENT_FUZZY_THRESHOLD,
    PRODUCT_COVERAGE_THRESHOLD,
    ROOT,
    apply_ocr_digit_barcode_fallback,
    collect_ocr_text_boxes,
    decode_laura_detections,
    ensemble_candidate_scores,
    load_paddle_result,
    load_text_db,
    read_json,
    read_jsonl,
    score_products_for_ocr,
    status_for_count,
    unique_in_order,
)

DEFAULT_INPUT_DIR = ROOT / "dataset" / "multiple"
DEFAULT_OUTPUT_DIR = ROOT / "output" / "multiple_matching"
DEFAULT_PRODUCT_DB = ROOT / "db" / "products.json"
DEFAULT_DB_TEXT_DIR = ROOT / "db" / "text_front"
DEFAULT_LAURA_RESULTS = (
    ROOT / "output" / "multiple_matching" / "laura" / "results.jsonl"
)
NAME_MAP = {
    "paddleocr": "원본",
    "paddleocr_preproc": "전처리",
    "paddleocr_boost": "boost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-item DB cue matching")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-text-dir", type=Path, default=DEFAULT_DB_TEXT_DIR)
    parser.add_argument("--product-db", type=Path, default=DEFAULT_PRODUCT_DB)
    parser.add_argument("--laura-results", type=Path, default=DEFAULT_LAURA_RESULTS)
    parser.add_argument(
        "--paddle-dir",
        type=Path,
        action="append",
        default=None,
        help="PaddleOCR result dir. Pass multiple times for ensemble.",
    )
    parser.add_argument("--element-threshold", type=float, default=ELEMENT_FUZZY_THRESHOLD)
    parser.add_argument(
        "--coverage-threshold", type=float, default=PRODUCT_COVERAGE_THRESHOLD
    )
    args = parser.parse_args()
    if not args.paddle_dir:
        base = ROOT / "output" / "multiple_matching"
        args.paddle_dir = [
            base / "paddleocr",
            base / "paddleocr_preproc",
            base / "paddleocr_boost",
        ]
    return args


def load_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def draw_barcode_overlay(
    image_path: Path,
    detections: list[dict[str, Any]],
    output_path: Path,
) -> None:
    image = load_image(image_path)
    for index, detection in enumerate(detections):
        xyxy = detection.get("xyxy") or []
        if len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        conf = detection.get("confidence")
        decoded = detection.get("decoded") or []
        code = decoded[0]["code"] if decoded else "?"
        label = f"#{index} {code}"
        if conf is not None:
            label += f" ({float(conf):.2f})"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 180, 0), 4)
        cv2.putText(
            image,
            label,
            (x1, max(28, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])[1].tofile(
        str(output_path)
    )


def cue_barcode(barcode_matches: list[str], decoded_codes: list[str], det_count: int) -> str:
    if not barcode_matches:
        if det_count:
            return f"no matching (det={det_count}, decode={decoded_codes or '실패'})"
        return "no matching"
    if len(barcode_matches) == 1:
        return f"<{barcode_matches[0]}> matching"
    return "multiple matching(" + ", ".join(barcode_matches) + ")"


def cue_ocr_for_products(
    expected: list[str],
    candidate_by_product: dict[str, dict[str, Any]],
    threshold: float,
) -> tuple[str, list[str], list[str], list[str]]:
    parts: list[str] = []
    matched: list[str] = []
    missing: list[str] = []
    for product in expected:
        score = candidate_by_product.get(product, {})
        cov = float(score.get("coverage") or 0.0)
        pct = f"{cov * 100:.1f}%"
        if cov >= threshold:
            parts.append(f"<{product}> matching ({pct})")
            matched.append(product)
        else:
            parts.append(f"{product} no matching ({pct})")
            missing.append(product)
    extras = [
        product
        for product, score in candidate_by_product.items()
        if product not in expected and float(score.get("coverage") or 0.0) >= threshold
    ]
    ocr_cue = "; ".join(parts) if parts else "no matching"
    if extras:
        ocr_cue += " | false: " + ", ".join(
            f"<{p}> ({candidate_by_product[p]['coverage']*100:.1f}%)" for p in extras
        )
    return ocr_cue, matched, missing, extras


def evaluate_weight_for_product(
    measured: float,
    product_name: str,
    product_by_name: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    product = product_by_name.get(product_name)
    if product is None:
        return {
            "evaluated": True,
            "passed": False,
            "matched_product": product_name,
            "error": "WEIGHT_DB_MISSING",
        }
    nominal = float(product["weight_g"])
    tolerance = float(product["tolerance_g"])
    difference = abs(measured - nominal)
    passed = difference <= tolerance + 1e-9
    return {
        "evaluated": True,
        "passed": passed,
        "matched_product": product_name,
        "nominal_g": nominal,
        "tolerance_g": tolerance,
        "difference_g": round(difference, 4),
        "measured_g": measured,
    }


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    barcode_dir = output_dir / "barcode"
    barcode_dir.mkdir(parents=True, exist_ok=True)
    crops_dir = barcode_dir / "crops"
    overlays_dir = barcode_dir / "overlays"
    crops_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    manifest = read_json(input_dir / "manifest.json")
    products_payload = read_json(args.product_db.resolve())
    products = products_payload.get("items", [])
    product_by_name = {item["product"]: item for item in products}
    barcode_to_products: dict[str, list[str]] = {}
    for product in products:
        barcode_to_products.setdefault(str(product["barcode"]), []).append(product["product"])

    text_db = load_text_db(args.db_text_dir.resolve())
    laura_rows = read_jsonl(args.laura_results.resolve())
    laura_by_file = {
        row.get("record", {}).get("file_name"): row
        for row in laura_rows
        if row.get("record", {}).get("file_name")
    }

    results: list[dict[str, Any]] = []
    summary_lines: list[str] = []
    canvas_items: list[dict[str, Any]] = []

    for item in manifest.get("items", []):
        image_name = item["input_image"]
        image_path = input_dir / image_name
        expected = list(item.get("expected_products") or [])
        arrangement = item.get("arrangement", "")
        count = item.get("count", len(expected))

        expected_dbs = [product_by_name[name] for name in expected if name in product_by_name]
        nominal_sum = sum(float(p["weight_g"]) for p in expected_dbs)
        tol_sum = sum(float(p["tolerance_g"]) for p in expected_dbs)
        measured = round(nominal_sum + tol_sum * 0.25, 2) if expected_dbs else 0.0

        laura_row = laura_by_file.get(image_name, {})
        raw_detections = laura_row.get("barcodes", [])
        decoded_detections = decode_laura_detections(image_path, raw_detections, crops_dir)

        image_stem = Path(image_name).stem
        paddle_sources = [path.resolve() for path in args.paddle_dir]
        paddle_payloads = [
            load_paddle_result(paddle_dir, image_stem) for paddle_dir in paddle_sources
        ]

        # OCR digit fallback: only if Laura detected a barcode box AND an OCR
        # text bbox overlaps that box; accept only known DB barcodes.
        recovered_from_ocr: list[dict[str, str]] = []
        if raw_detections:
            ocr_boxes = collect_ocr_text_boxes(paddle_payloads)
            recovered_from_ocr = apply_ocr_digit_barcode_fallback(
                decoded_detections,
                ocr_boxes,
                set(barcode_to_products.keys()),
            )

        overlay_name = f"{Path(image_name).stem}_barcode.jpg"
        draw_barcode_overlay(image_path, decoded_detections, overlays_dir / overlay_name)

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
        barcode_cue = cue_barcode(barcode_matches, decoded_codes, len(raw_detections))
        if recovered_from_ocr:
            barcode_cue += (
                " +ocr_digits_overlap("
                + ", ".join(item["code"] for item in recovered_from_ocr)
                + ")"
            )
        per_source_scores: list[list[dict[str, Any]]] = []
        source_summaries: list[dict[str, Any]] = []
        source_coverages: dict[str, dict[str, float]] = {}
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
            cov_map = {s["product"]: float(s["coverage"]) for s in source_scores}
            source_coverages[NAME_MAP.get(source_name, source_name)] = {
                product: cov_map.get(product, 0.0) for product in expected
            }
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
                    "text_set": query_tokens,
                    "rec_texts": paddle.get("rec_texts", []),
                    "top_product": top_source["product"],
                    "top_coverage": top_source["coverage"],
                }
            )

        candidate_scores = ensemble_candidate_scores(per_source_scores)
        candidate_by_product = {c["product"]: c for c in candidate_scores}
        ocr_cue, ocr_matched_expected, ocr_missing, ocr_extras = cue_ocr_for_products(
            expected, candidate_by_product, args.coverage_threshold
        )
        ocr_all_matches = [
            candidate["product"]
            for candidate in candidate_scores
            if float(candidate["coverage"]) >= args.coverage_threshold
        ]
        barcode_status = status_for_count(len(barcode_matches))
        ocr_status = status_for_count(len(ocr_all_matches))

        weight_result: dict[str, Any] = {
            "simulated": True,
            "measured_g": measured,
            "measured_note": "tray total = sum of all expected (multiple) items",
            "nominal_sum_g": nominal_sum,
            "tolerance_sum_g": tol_sum,
            "expected_products": expected,
            "evaluated": False,
            "passed": False,
        }
        flow: list[dict[str, Any]] = [
            {
                "stage": 1,
                "status": "COMPLETED",
                "detail": (
                    f"barcode det={len(decoded_detections)}, "
                    f"OCR sources={len(paddle_sources)}, "
                    f"measured tray weight={measured:g}g "
                    f"(sum of {len(expected)} items)"
                ),
            }
        ]

        final_action = "REINSERT"
        final_status = ""
        final_product: str | None = None
        weight_cue = "not evaluated"
        decision_path = ""

        # --- 3. barcode matching ---
        if barcode_status == "MULTIPLE":
            final_status = "BARCODE_MULTIPLE"
            decision_path = "barcode:MULTIPLE"
            weight_cue = "not evaluated (barcode multiple → 재투입)"
            flow.append(
                {
                    "stage": 2,
                    "status": "MULTIPLE",
                    "detail": (
                        f"Barcode DB matches={barcode_matches}; "
                        "여러 종류 matching → multiple 판단 후 재투입 요구"
                    ),
                }
            )
        elif barcode_status == "SINGLE":
            product = barcode_matches[0]
            decision_path = f"barcode:SINGLE({product})→weight"
            w = evaluate_weight_for_product(measured, product, product_by_name)
            weight_result.update(w)
            flow.append(
                {
                    "stage": 2,
                    "status": "SINGLE",
                    "detail": f"Barcode DB match={product}; weight 비교로 진행",
                }
            )
            if w.get("error") == "WEIGHT_DB_MISSING":
                final_status = "WEIGHT_DB_MISSING"
                weight_cue = f"no matching ({product} weight DB 없음)"
                flow.append(
                    {
                        "stage": 3,
                        "status": "NO_MATCH",
                        "detail": f"{product} weight DB 없음; 재투입 요구",
                    }
                )
            elif w["passed"]:
                final_action = "CONFIRMED"
                final_status = "CONFIRMED"
                final_product = product
                weight_cue = (
                    f"<{product}> matching "
                    f"({measured:g}g / {w['nominal_g']:g}±{w['tolerance_g']:g}g)"
                )
                flow.append(
                    {
                        "stage": 3,
                        "status": "PASS",
                        "detail": (
                            f"|{measured:g}-{w['nominal_g']:g}|="
                            f"{w['difference_g']}g <= {w['tolerance_g']:g}g; "
                            f"{product} 확정"
                        ),
                    }
                )
            else:
                final_status = "BARCODE_WEIGHT_MISMATCH_MULTIPLE"
                weight_cue = (
                    f"no matching — multiple 판단 후 재투입 요구 "
                    f"(측정 {measured:g}g vs {product} {w['nominal_g']:g}"
                    f"±{w['tolerance_g']:g}g)"
                )
                flow.append(
                    {
                        "stage": 3,
                        "status": "FAIL",
                        "detail": (
                            f"|{measured:g}-{w['nominal_g']:g}|="
                            f"{w['difference_g']}g > {w['tolerance_g']:g}g; "
                            "weight 불일치 → multiple 판단 후 재투입 요구"
                        ),
                    }
                )
        else:
            # barcode NO_MATCH → OCR
            flow.append(
                {
                    "stage": 2,
                    "status": "NO_MATCH",
                    "detail": "Barcode no matching; OCR matching으로 진행",
                }
            )
            decision_path = "barcode:NO_MATCH→ocr"

            # --- 4. OCR matching ---
            if ocr_status == "MULTIPLE":
                final_status = "OCR_MULTIPLE"
                decision_path = "ocr:MULTIPLE"
                weight_cue = "not evaluated (OCR multiple → 재투입)"
                flow.append(
                    {
                        "stage": 3,
                        "status": "MULTIPLE",
                        "detail": (
                            f"OCR DB matches={ocr_all_matches}; "
                            "여러 종류 matching → multiple 판단 후 재투입 요구"
                        ),
                    }
                )
            elif ocr_status == "NO_MATCH":
                final_status = "OCR_NO_MATCH"
                decision_path = "ocr:NO_MATCH"
                weight_cue = "not evaluated (OCR no matching → 재투입)"
                flow.append(
                    {
                        "stage": 3,
                        "status": "NO_MATCH",
                        "detail": "OCR no matching; 재투입 요구",
                    }
                )
            else:
                product = ocr_all_matches[0]
                decision_path = f"ocr:SINGLE({product})→weight"
                w = evaluate_weight_for_product(measured, product, product_by_name)
                weight_result.update(w)
                flow.append(
                    {
                        "stage": 3,
                        "status": "SINGLE",
                        "detail": f"OCR DB match={product}; weight 비교로 진행",
                    }
                )
                if w.get("error") == "WEIGHT_DB_MISSING":
                    final_status = "WEIGHT_DB_MISSING"
                    weight_cue = f"no matching ({product} weight DB 없음)"
                    flow.append(
                        {
                            "stage": 4,
                            "status": "NO_MATCH",
                            "detail": f"{product} weight DB 없음; 재투입 요구",
                        }
                    )
                elif w["passed"]:
                    final_action = "CONFIRMED"
                    final_status = "CONFIRMED"
                    final_product = product
                    weight_cue = (
                        f"<{product}> matching "
                        f"({measured:g}g / {w['nominal_g']:g}±{w['tolerance_g']:g}g)"
                    )
                    flow.append(
                        {
                            "stage": 4,
                            "status": "PASS",
                            "detail": (
                                f"|{measured:g}-{w['nominal_g']:g}|="
                                f"{w['difference_g']}g <= {w['tolerance_g']:g}g; "
                                f"{product} 확정"
                            ),
                        }
                    )
                else:
                    final_status = "OCR_WEIGHT_MISMATCH_MULTIPLE"
                    weight_cue = (
                        f"no matching — multiple 판단 후 재투입 요구 "
                        f"(측정 {measured:g}g vs {product} {w['nominal_g']:g}"
                        f"±{w['tolerance_g']:g}g)"
                    )
                    flow.append(
                        {
                            "stage": 4,
                            "status": "FAIL",
                            "detail": (
                                f"|{measured:g}-{w['nominal_g']:g}|="
                                f"{w['difference_g']}g > {w['tolerance_g']:g}g; "
                                "weight 불일치 → multiple 판단 후 재투입 요구"
                            ),
                        }
                    )

        ocr_status_cue = (
            f"multiple matching({', '.join(ocr_all_matches)})"
            if ocr_status == "MULTIPLE"
            else (
                f"<{ocr_all_matches[0]}> matching"
                if ocr_status == "SINGLE"
                else "no matching"
            )
        )
        # Keep detailed expected-product OCR breakdown for analysis.
        ocr_detail_cue = ocr_cue

        per_product_lines = []
        for product in expected:
            score = candidate_by_product.get(product, {})
            cov = float(score.get("coverage") or 0.0)
            src = NAME_MAP.get(str(score.get("ocr_source") or ""), str(score.get("ocr_source") or "-"))
            in_barcode = product in barcode_matches
            b = f"<{product}> matching" if in_barcode else "no matching"
            if cov >= args.coverage_threshold:
                o = f"<{product}> matching ({cov*100:.1f}%)"
            else:
                o = f"no matching ({cov*100:.1f}%)"
            if final_action == "CONFIRMED" and final_product == product:
                w = f"<{product}> matching"
            elif weight_result.get("evaluated") and weight_result.get("matched_product") == product:
                w = "no matching (weight 불일치→multiple/재투입)"
            else:
                w = "not evaluated"
            line = f"{product}: barcode: {b}, ocr: {o}, weight: {w}"
            per_product_lines.append(
                {
                    "product": product,
                    "line": line,
                    "barcode": b,
                    "ocr": o,
                    "weight": w,
                    "coverage": cov,
                    "ocr_source": src,
                }
            )

        image_line = (
            f"{image_name} [{arrangement} · {count}품목: {', '.join(expected)}] "
            f"barcode: {barcode_cue} [{barcode_status}], "
            f"ocr: {ocr_status_cue} [{ocr_status}], "
            f"weight: {weight_cue} → {final_action} ({final_status})"
        )
        summary_lines.append(image_line)

        # copy overlay also to flat barcode/ for quick browse
        shutil.copy2(overlays_dir / overlay_name, barcode_dir / overlay_name)

        result = {
            "input_image": image_name,
            "input_path": str(image_path),
            "arrangement": arrangement,
            "count": count,
            "expected_products": expected,
            "decision_path": decision_path,
            "flow": flow,
            "barcode": {
                "detector": "Laura YOLOv5 barcodeDetector",
                "detections": decoded_detections,
                "decoded_codes": decoded_codes,
                "db_matches": barcode_matches,
                "status": barcode_status,
                "cue": barcode_cue,
                "ocr_digit_fallback": recovered_from_ocr,
                "overlay": str((barcode_dir / overlay_name).relative_to(output_dir)),
            },
            "ocr": {
                "ensemble": {
                    "mode": "max_coverage_per_product",
                    "sources": source_summaries,
                },
                "element_fuzzy_threshold": args.element_threshold,
                "product_coverage_threshold": args.coverage_threshold,
                "candidate_scores": candidate_scores,
                "db_matches": ocr_all_matches,
                "status": ocr_status,
                "matched_expected": ocr_matched_expected,
                "missing_expected": ocr_missing,
                "false_positives": ocr_extras,
                "cue": ocr_status_cue,
                "detail_cue": ocr_detail_cue,
                "source_coverages": source_coverages,
            },
            "weight": {**weight_result, "cue": weight_cue},
            "per_product": per_product_lines,
            "summary_line": image_line,
            "final": {
                "action": final_action,
                "status": final_status,
                "product": final_product,
                "correct_vs_manifest": (
                    final_action == "CONFIRMED"
                    and final_product in expected
                    and len(expected) == 1
                ),
            },
        }
        results.append(result)

        canvas_items.append(
            {
                "image": image_name,
                "arrangement": arrangement,
                "count": count,
                "expected": expected,
                "barcode": barcode_cue,
                "barcodeStatus": barcode_status,
                "ocr": ocr_status_cue,
                "ocrStatus": ocr_status,
                "ocrDetail": ocr_detail_cue,
                "weight": weight_cue,
                "line": image_line,
                "action": final_action,
                "status": final_status,
                "decisionPath": decision_path,
                "finalProduct": final_product or "",
                "perProduct": per_product_lines,
                "found": len(ocr_matched_expected),
                "expectedCount": len(expected),
                "measured_g": measured,
                "overlay": f"barcode/{overlay_name}",
            }
        )
        print(image_line)

    report = {
        "algorithm": {
            "barcode_detector": "Laura YOLOv5 barcodeDetector",
            "ocr_ensemble_mode": "max_coverage_per_product",
            "ocr_sources": [str(path.resolve()) for path in args.paddle_dir],
            "ocr_element_fuzzy_threshold": args.element_threshold,
            "ocr_product_coverage_threshold": args.coverage_threshold,
            "ocr_containment_min_ratio": CONTAINMENT_MIN_RATIO,
            "weight_source": (
                "simulated tray total = sum(all expected item nominal) "
                "+ 0.25*sum(all expected tolerances)"
            ),
            "decision_policy": (
                "barcode: MULTIPLE→REINSERT; SINGLE→weight "
                "(match→CONFIRMED, mismatch→MULTIPLE/REINSERT); "
                "NO_MATCH→OCR. "
                "OCR: MULTIPLE→REINSERT; SINGLE→weight "
                "(match→CONFIRMED, mismatch→MULTIPLE/REINSERT); "
                "NO_MATCH→REINSERT. "
                "Barcode OCR-digit fallback: only when Laura detection exists "
                "AND OCR text bbox overlaps barcode bbox; DB barcodes only."
            ),
        },
        "summary": {
            "image_count": len(results),
            "confirmed_count": sum(r["final"]["action"] == "CONFIRMED" for r in results),
            "reinsert_count": sum(r["final"]["action"] == "REINSERT" for r in results),
            "expected_product_hits": sum(len(r["ocr"]["matched_expected"]) for r in results),
            "expected_product_total": sum(len(r["expected_products"]) for r in results),
        },
        "items": results,
    }

    (output_dir / "matching_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "canvas_items.json").write_text(
        json.dumps(canvas_items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "product_summary_lines.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )

    with (output_dir / "matching_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        fieldnames = [
            "input_image",
            "arrangement",
            "expected_products",
            "decision_path",
            "barcode_status",
            "barcode_cue",
            "decoded_barcodes",
            "barcode_matches",
            "ocr_status",
            "ocr_cue",
            "ocr_matches",
            "weight_cue",
            "measured_weight_g",
            "final_status",
            "final_product",
            "action",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "input_image": result["input_image"],
                    "arrangement": result["arrangement"],
                    "expected_products": "|".join(result["expected_products"]),
                    "decision_path": result["decision_path"],
                    "barcode_status": result["barcode"]["status"],
                    "barcode_cue": result["barcode"]["cue"],
                    "decoded_barcodes": "|".join(result["barcode"]["decoded_codes"]),
                    "barcode_matches": "|".join(result["barcode"]["db_matches"]),
                    "ocr_status": result["ocr"]["status"],
                    "ocr_cue": result["ocr"]["cue"],
                    "ocr_matches": "|".join(result["ocr"]["db_matches"]),
                    "weight_cue": result["weight"]["cue"],
                    "measured_weight_g": result["weight"]["measured_g"],
                    "final_status": result["final"]["status"],
                    "final_product": result["final"].get("product") or "",
                    "action": result["final"]["action"],
                }
            )

    # HTML report
    html = build_html(report, canvas_items)
    (output_dir / "multiple_matching_summary.html").write_text(html, encoding="utf-8")

    print(
        f"\nDone: CONFIRMED {report['summary']['confirmed_count']}/"
        f"{report['summary']['image_count']} → {output_dir}"
    )
    return 0


def build_html(report: dict[str, Any], canvas_items: list[dict[str, Any]]) -> str:
    cards = []
    for item in canvas_items:
        tone = "ok" if item["action"] == "CONFIRMED" else "warn"
        product_rows = "".join(
            f"<li><code>{p['line']}</code></li>" for p in item["perProduct"]
        )
        cards.append(
            f"""
    <div class="card {tone}">
      <div class="head">
        <div>
          <strong>{item['image']}</strong>
          <span class="meta">{item['arrangement']} · {item['count']}품목 · {', '.join(item['expected'])}</span>
        </div>
        <span class="pill {tone}">{item['action']}</span>
      </div>
      <div class="line">{item['line']}</div>
      <div class="pills">
        <span class="pill">barcode [{item.get('barcodeStatus','')}] {item['barcode']}</span>
        <span class="pill">ocr [{item.get('ocrStatus','')}] {item['ocr']}</span>
        <span class="pill">path: {item.get('decisionPath','')}</span>
        <span class="pill">weight: {item['weight']}</span>
        <span class="pill">{item.get('status','')}</span>
      </div>
      <div class="split">
        <div>
          <h3>상품별 cue</h3>
          <ul>{product_rows}</ul>
        </div>
        <div>
          <h3>barcode overlay</h3>
          <img src="{item['overlay']}" alt="barcode overlay" />
        </div>
      </div>
    </div>"""
        )
    summary = report["summary"]
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Multiple Matching 요약</title>
  <style>
    :root {{
      --bg:#f7f7f5; --card:#fff; --text:#1a1a1a; --muted:#5c5c5c; --stroke:#e5e5e2;
      --ok:#1f6f4a; --ok-bg:#e8f5ee; --warn:#8a5a00; --warn-bg:#fff4e0;
    }}
    body {{ margin:0; font-family:"Noto Sans KR","Apple SD Gothic Neo",sans-serif; background:var(--bg); color:var(--text); }}
    .wrap {{ max-width:1100px; margin:0 auto; padding:28px 20px 48px; }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    .sub {{ color:var(--muted); font-size:13px; margin-bottom:18px; }}
    .stats {{ display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:18px; }}
    .stat {{ background:var(--card); border:1px solid var(--stroke); border-radius:10px; padding:14px 16px; }}
    .stat .v {{ font-size:26px; font-weight:700; }}
    .stat.ok .v {{ color:var(--ok); }} .stat.warn .v {{ color:var(--warn); }}
    .card {{ background:var(--card); border:1px solid var(--stroke); border-radius:10px; padding:14px 16px; margin-bottom:12px; }}
    .card.ok {{ background:#f4fbf7; }} .card.warn {{ background:#fffaf0; }}
    .head {{ display:flex; justify-content:space-between; gap:10px; align-items:center; }}
    .meta {{ color:var(--muted); margin-left:8px; font-size:13px; }}
    .line {{ margin:10px 0; font-weight:550; font-size:14px; }}
    .pills {{ display:flex; flex-wrap:wrap; gap:8px; margin-bottom:10px; }}
    .pill {{ border-radius:999px; padding:4px 10px; font-size:12px; font-weight:600; background:#f0f0ee; }}
    .pill.ok {{ background:var(--ok-bg); color:var(--ok); }}
    .pill.warn {{ background:var(--warn-bg); color:var(--warn); }}
    .split {{ display:grid; grid-template-columns:1.2fr 1fr; gap:14px; }}
    img {{ width:100%; border:1px solid var(--stroke); border-radius:8px; background:#111; }}
    ul {{ margin:0; padding-left:18px; }} li {{ margin:4px 0; }}
    code {{ font-size:12.5px; }}
    @media (max-width:820px) {{ .stats,.split {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Multiple Matching 요약</h1>
    <p class="sub">Kiosk_experiment/multiple input · barcode→weight / OCR→weight · coverage ≥ {report['algorithm']['ocr_product_coverage_threshold']}</p>
    <div class="stats">
      <div class="stat"><div class="v">{summary['image_count']}</div><div>Images</div></div>
      <div class="stat ok"><div class="v">{summary['confirmed_count']} / {summary['image_count']}</div><div>CONFIRMED</div></div>
      <div class="stat warn"><div class="v">{summary['expected_product_hits']} / {summary['expected_product_total']}</div><div>Expected product OCR hits</div></div>
    </div>
    {''.join(cards)}
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
