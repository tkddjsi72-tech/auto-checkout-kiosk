#!/usr/bin/env python3
"""Check whether detected barcodes fall inside SAM1 instance masks.

The script treats each leaf directory containing images as one 6-view sample set.
It can also run in --dry-run mode to validate dataset discovery without loading
SAM or YOLO weights.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from dataset_index import find_index_json

# Suppress noisy upstream FutureWarnings we can't fix here (YOLOv5 hub, transformers).
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.cuda\.amp\.autocast.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*key `labels`.*post_process_grounded_object_detection.*Use `text_labels` instead.*",
    category=FutureWarning,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT_DEFAULT = _REPO_ROOT / "dataset" / "single_front"
BARCODE_MODEL_DEFAULT = _REPO_ROOT / "YOLOV8s_Barcode_Detection.pt"
# Daudmax/barcode-decoder: 학습 산출 best.pt 를 아래에 두거나 --barcode-model 로 지정
# https://github.com/Daudmax/barcode-decoder
DAUDMAX_BARCODE_MODEL_DEFAULT = (
    _REPO_ROOT / "models" / "daudmax_barcode_decoder" / "best.pt"
)
LAURA_YOLOV5_BARCODE_MODEL_DEFAULT = (
    _REPO_ROOT / "models" / "laura_yolov5_barcode" / "barcode_model.pt"
)
SAM_CHECKPOINT_DEFAULT = _REPO_ROOT / "sam_vit_h_4b8939.pth"
VIEW_NAMES = ("앞", "뒤", "좌", "우", "위", "아래")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class ImageRecord:
    set_id: str
    set_path: str
    image_path: str
    file_name: str
    dataset_kind: str
    view: str | None
    view_index: int
    item_name: str | None = None
    barcode_pose: str | None = None
    item_count: str | None = None
    item_combo: str | None = None
    layout: str | None = None


@dataclass(frozen=True)
class BarcodeBox:
    box_id: int
    class_id: int | None
    class_name: str | None
    confidence: float
    xyxy: list[float]


@dataclass(frozen=True)
class BarcodeModelConfig:
    default_path: Path
    backend: str
    description: str
    recommended_imgsz: int | None = None


BARCODE_MODEL_REGISTRY = {
    "huggingface": BarcodeModelConfig(
        default_path=BARCODE_MODEL_DEFAULT,
        backend="ultralytics",
        description="Piero2411 YOLOV8s-Barcode-Detection",
    ),
    "daudmax": BarcodeModelConfig(
        default_path=DAUDMAX_BARCODE_MODEL_DEFAULT,
        backend="ultralytics",
        description="Daudmax/barcode-decoder 학습 산출 best.pt",
        recommended_imgsz=640,
    ),
    "laura_yolov5": BarcodeModelConfig(
        default_path=LAURA_YOLOV5_BARCODE_MODEL_DEFAULT,
        backend="yolov5_hub",
        description="lauraAriasFdez/barcodeDetector YOLOv5 barcode_model.pt",
        recommended_imgsz=640,
    ),
}


@dataclass(frozen=True)
class MaskInfo:
    mask_id: int
    area: int
    bbox_xywh: list[float]
    predicted_iou: float | None
    stability_score: float | None


@dataclass(frozen=True)
class InstanceBox:
    instance_id: int
    label: str
    score: float
    xyxy: list[float]


@dataclass(frozen=True)
class BarcodeMaskMatch:
    box_id: int
    matched: bool
    matched_mask_id: int | None
    center_inside: bool
    box_mask_coverage: float
    iou: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SAM1 automatic masks and YOLO barcode detection on kiosk dataset images.",
    )
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT_DEFAULT)
    parser.add_argument("--output-dir", type=Path, default=Path("barcode_instance_results"))
    parser.add_argument(
        "--barcode-model-family",
        choices=tuple(BARCODE_MODEL_REGISTRY.keys()),
        default="huggingface",
        help=(
            "바코드 탐지 모델 선택. 기본 경로는 레지스트리에서 결정 "
            f"({', '.join(BARCODE_MODEL_REGISTRY.keys())})."
        ),
    )
    parser.add_argument(
        "--barcode-model",
        type=Path,
        default=None,
        help="YOLO .pt 파일. 지정 시 --barcode-model-family 보다 우선.",
    )
    parser.add_argument(
        "--yolo-imgsz",
        type=int,
        default=None,
        metavar="N",
        help="YOLO predict imgsz(미지정=모델 기본). daudmax/laura_yolov5 권장값은 640.",
    )
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=SAM_CHECKPOINT_DEFAULT,
        help=f"SAM1 checkpoint .pth (default: {SAM_CHECKPOINT_DEFAULT}; --sam-model-type과 맞출 것).",
    )
    parser.add_argument("--sam-model-type", default="vit_h", choices=("vit_h", "vit_l", "vit_b"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--conf", type=float, default=0.40, help="YOLO confidence threshold.")
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument(
        "--instance-detector",
        choices=("sam", "grounding_dino"),
        default="sam",
        help="상품 instance 후보 생성기. sam=기존 마스크, grounding_dino=bbox detection.",
    )
    # SAM 자동 마스크 과다 생성/오버세그 방지용 기본값(빡빡하게 제한)
    parser.add_argument("--sam-points-per-side", type=int, default=8)
    parser.add_argument("--sam-pred-iou-thresh", type=float, default=0.93)
    parser.add_argument("--sam-stability-score-thresh", type=float, default=0.98)
    parser.add_argument("--sam-crop-n-layers", type=int, default=0)
    parser.add_argument("--sam-min-mask-region-area", type=int, default=800)
    parser.add_argument("--grounding-dino-model", default="IDEA-Research/grounding-dino-tiny")
    parser.add_argument(
        "--grounding-dino-prompt",
        default="product. package. box. bottle. can. snack. object.",
        help="Grounding DINO text prompt. 클래스/개념은 마침표로 구분.",
    )
    parser.add_argument("--grounding-dino-box-threshold", type=float, default=0.35)
    parser.add_argument("--grounding-dino-text-threshold", type=float, default=0.25)
    parser.add_argument(
        "--min-item-count",
        type=int,
        default=None,
        help="품목 수가 이 값 이상인 세트만 추론 (index.json item_count 기준).",
    )
    parser.add_argument(
        "--max-item-count",
        type=int,
        default=None,
        help="품목 수가 이 값 이하인 세트만 추론 (index.json item_count 기준).",
    )
    parser.add_argument(
        "--limit-images",
        type=int,
        help="이미지 N장만 무작위 표본(비복원). 후보는 --limit-sets 적용 뒤 목록. 최종 추론 순서는 --sample-seed로 셔플.",
    )
    parser.add_argument(
        "--limit-sets",
        type=int,
        help="리프 세트 N개만 무작위 표본(비복원). 세트 내·간 순서는 --sample-seed로 셔플된 뒤 추론.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=0,
        help="--limit-sets / --limit-images 사용 시 표본 추출 + 최종 이미지 목록 셔플 시드(동일 시드·동일 제한이면 같은 결과).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only discover dataset records and write summary.")
    parser.add_argument("--no-overlays", action="store_true", help="Skip overlay image generation.")
    parser.add_argument(
        "--max-long-edge",
        type=int,
        default=None,
        metavar="PX",
        help="If set, longest side is scaled down to this many pixels (YOLO+SAM share this size; overlay on full-res).",
    )
    parser.add_argument(
        "--yolo-only",
        action="store_true",
        help="Skip SAM; run barcode detection only and draw detection boxes on overlays.",
    )
    return parser.parse_args()


def iter_image_files(directory: Path) -> list[Path]:
    return sorted(
        (
            child
            for child in directory.iterdir()
            if child.is_file() and child.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name,
    )


def discover_leaf_sets(dataset_root: Path) -> list[tuple[Path, list[Path]]]:
    leaf_sets: list[tuple[Path, list[Path]]] = []
    root_images = iter_image_files(dataset_root)
    if root_images:
        leaf_sets.append((dataset_root, root_images))
    for directory in sorted((p for p in dataset_root.rglob("*") if p.is_dir()), key=lambda p: str(p)):
        image_files = iter_image_files(directory)
        if image_files:
            leaf_sets.append((directory, image_files))
    return leaf_sets


def build_records_from_index_json(index_path: Path) -> list[ImageRecord]:
    """Build ImageRecord list from our v1 dataset index.json schema."""
    import json

    base_dir = index_path.parent
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise ValueError(f"Unsupported index.json version: {version} ({index_path})")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid index.json: 'entries' must be a list ({index_path})")

    records: list[ImageRecord] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("id", "")).strip()
        group = str(entry.get("group", "")).strip() or "unknown"
        image_files = entry.get("image_files", [])
        if not entry_id or not isinstance(image_files, list) or not image_files:
            continue

        # Keep grouping stable: <group>/<id>
        set_id = f"{group}/{entry_id}"

        items = entry.get("items", [])
        item_name = str(items[0]) if isinstance(items, list) and len(items) == 1 else None
        item_count = entry.get("item_count", None)
        item_count_str = None if item_count is None else str(item_count)
        item_combo = None
        if isinstance(items, list) and items:
            item_combo = "+".join(str(x) for x in items)

        state = entry.get("state", None)
        barcode_pose = None if state is None else str(state)

        views = entry.get("views", {}) or {}
        view_by_relpath: dict[str, str] = {}
        if isinstance(views, dict):
            for view_name, rel_path in views.items():
                if rel_path is None:
                    continue
                view_by_relpath[str(rel_path)] = str(view_name)

        for index, rel_image_path in enumerate(image_files):
            rel_str = str(rel_image_path)
            image_path = (base_dir / rel_str).resolve()
            if not image_path.exists():
                # Keep going; missing files will be surfaced via dataset warnings/errors.
                continue
            view = view_by_relpath.get(rel_str)
            records.append(
                ImageRecord(
                    set_id=set_id,
                    set_path=str(base_dir),
                    image_path=str(image_path),
                    file_name=image_path.name,
                    dataset_kind=group,
                    view=view,
                    view_index=index,
                    item_name=item_name,
                    barcode_pose=barcode_pose,
                    item_count=item_count_str,
                    item_combo=item_combo,
                    layout=None,
                )
            )
    return records


def parse_metadata(dataset_root: Path, set_path: Path) -> dict[str, str | None]:
    parts = set_path.relative_to(dataset_root).parts
    metadata: dict[str, str | None] = {
        "dataset_kind": parts[0] if parts else "unknown",
        "item_name": None,
        "barcode_pose": None,
        "item_count": None,
        "item_combo": None,
        "layout": None,
    }

    if len(parts) >= 3 and parts[0] == "단일품목":
        metadata["item_name"] = parts[1]
        metadata["barcode_pose"] = parts[2]
    elif len(parts) >= 4 and parts[0] == "다중품목":
        metadata["item_count"] = parts[1]
        metadata["item_combo"] = parts[2]
        metadata["layout"] = parts[3]
    return metadata


def build_records(
    dataset_root: Path, leaf_sets: list[tuple[Path, list[Path]]]
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for set_path, image_files in leaf_sets:
        metadata = parse_metadata(dataset_root, set_path)
        rel_set_path = set_path.relative_to(dataset_root)
        set_id = dataset_root.name if rel_set_path == Path(".") else str(rel_set_path)
        for index, image_path in enumerate(image_files):
            stem = image_path.stem
            view = stem if stem in VIEW_NAMES else None
            records.append(
                ImageRecord(
                    set_id=set_id,
                    set_path=str(set_path),
                    image_path=str(image_path),
                    file_name=image_path.name,
                    dataset_kind=metadata["dataset_kind"] or "unknown",
                    view=view,
                    view_index=index,
                    item_name=metadata["item_name"],
                    barcode_pose=metadata["barcode_pose"],
                    item_count=metadata["item_count"],
                    item_combo=metadata["item_combo"],
                    layout=metadata["layout"],
                )
            )
    return records


def parse_item_count(value: str | None) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def item_count_by_set(records: list[ImageRecord]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for record in records:
        if record.set_id not in counts:
            counts[record.set_id] = parse_item_count(record.item_count)
    return counts


def filter_records_by_item_count(
    records: list[ImageRecord],
    min_item_count: int | None,
    max_item_count: int | None,
) -> list[ImageRecord]:
    if min_item_count is None and max_item_count is None:
        return records

    min_value = min_item_count if min_item_count is not None else 0
    max_value = max_item_count if max_item_count is not None else 10**9
    if min_value > max_value:
        raise ValueError(
            f"--min-item-count ({min_value}) must be <= --max-item-count ({max_value})"
        )

    set_counts = item_count_by_set(records)
    keep_sets = {
        set_id
        for set_id, count in set_counts.items()
        if count is not None and min_value <= count <= max_value
    }
    return [record for record in records if record.set_id in keep_sets]


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class YoloV5HubDetector:
    """Normalize legacy YOLOv5 hub output to the same path as Ultralytics models."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.names = getattr(model, "names", {}) or {}


class GroundingDinoDetector:
    def __init__(
        self,
        processor: Any,
        model: Any,
        prompt: str,
        box_threshold: float,
        text_threshold: float,
    ) -> None:
        self.processor = processor
        self.model = model
        self.prompt = prompt
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold


def torch_hub_device(device: str) -> str:
    """YOLOv5 hub expects '0'/'0,1'/cpu instead of torch-style 'cuda'/'cuda:0'."""
    if device == "cuda":
        return "0"
    if device.startswith("cuda:"):
        return device.split(":", 1)[1]
    return device


def load_barcode_detector(args: argparse.Namespace, device: str) -> Any:
    config = BARCODE_MODEL_REGISTRY[args.barcode_model_family]

    if config.backend == "ultralytics":
        from ultralytics import YOLO

        return YOLO(str(args.barcode_model))

    if config.backend == "yolov5_hub":
        try:
            from ultralytics import YOLO

            return YOLO(str(args.barcode_model))
        except Exception as ultralytics_exc:  # noqa: BLE001 - legacy YOLOv5 weights may not load in ultralytics.
            import os
            import torch

            try:
                # PyTorch 2.6+ flips torch.load default to weights_only=True which breaks
                # some YOLOv5 checkpoints loaded via torch.hub. If user didn't explicitly
                # set torch.load(weights_only=...), allow opt-out via env var.
                os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
                model = torch.hub.load(
                    "ultralytics/yolov5:v6.2",
                    "custom",
                    path=str(args.barcode_model),
                    device=torch_hub_device(device),
                    trust_repo=True,
                    verbose=False,
                )
            except Exception as hub_exc:  # noqa: BLE001 - include both loader errors for actionable setup logs.
                raise RuntimeError(
                    "Failed to load YOLOv5 barcode weights with both ultralytics.YOLO "
                    f"and torch.hub. ultralytics error: {ultralytics_exc!r}; "
                    f"torch.hub error: {hub_exc!r}"
                ) from hub_exc
            return YoloV5HubDetector(model)

    raise ValueError(f"Unsupported barcode model backend: {config.backend}")


def load_grounding_dino_detector(args: argparse.Namespace, device: str) -> GroundingDinoDetector:
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.grounding_dino_model)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(args.grounding_dino_model)
    model.to(device)
    model.eval()
    return GroundingDinoDetector(
        processor=processor,
        model=model,
        prompt=args.grounding_dino_prompt,
        box_threshold=args.grounding_dino_box_threshold,
        text_threshold=args.grounding_dino_text_threshold,
    )


def load_models(args: argparse.Namespace) -> tuple[Any | None, Any | None, Any, str]:
    if not args.barcode_model:
        raise ValueError("--barcode-model path is empty unless --dry-run is used.")
    if not args.barcode_model.exists():
        raise FileNotFoundError(f"Barcode model not found: {args.barcode_model}")
    if not getattr(args, "yolo_only", False) and args.instance_detector == "sam":
        if not args.sam_checkpoint:
            raise ValueError("--sam-checkpoint path is empty unless --dry-run or --yolo-only is used.")
        if not args.sam_checkpoint.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {args.sam_checkpoint}")

    import torch
    device = resolve_device(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device}, but CUDA is not available.")
    yolo = load_barcode_detector(args, device)

    if getattr(args, "yolo_only", False):
        return None, None, yolo, device

    if args.instance_detector == "grounding_dino":
        grounding_dino = load_grounding_dino_detector(args, device)
        return None, grounding_dino, yolo, device

    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    sam = sam_model_registry[args.sam_model_type](checkpoint=str(args.sam_checkpoint))
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=args.sam_points_per_side,
        pred_iou_thresh=args.sam_pred_iou_thresh,
        stability_score_thresh=args.sam_stability_score_thresh,
        crop_n_layers=args.sam_crop_n_layers,
        min_mask_region_area=args.sam_min_mask_region_area,
    )
    return mask_generator, None, yolo, device


def load_image_rgb(image_path: Path) -> Any:
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def compute_resize_hw(
    height: int, width: int, max_long_edge: int | None
) -> tuple[int, int, float]:
    """Return (new_height, new_width, scale) where scale maps original -> resized."""
    if not max_long_edge or max(height, width) <= max_long_edge:
        return height, width, 1.0
    scale = max_long_edge / max(height, width)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    return new_h, new_w, scale


def resize_rgb(image_rgb: Any, new_h: int, new_w: int) -> Any:
    import cv2

    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def scale_masks_to_full_res(
    masks: list[dict[str, Any]], orig_h: int, orig_w: int, work_h: int, work_w: int
) -> list[dict[str, Any]]:
    if work_h == orig_h and work_w == orig_w:
        return masks
    import cv2
    import numpy as np

    out: list[dict[str, Any]] = []
    for mask in masks:
        seg = mask["segmentation"].astype(np.uint8)
        seg_up = cv2.resize(seg, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        new_mask = dict(mask)
        new_mask["segmentation"] = seg_up
        out.append(new_mask)
    return out


def scale_xyxy_to_full_res(
    xyxy: list[float], orig_h: int, orig_w: int, work_h: int, work_w: int
) -> list[float]:
    if work_h == orig_h and work_w == orig_w:
        return list(xyxy)
    sx = orig_w / work_w
    sy = orig_h / work_h
    return [xyxy[0] * sx, xyxy[1] * sy, xyxy[2] * sx, xyxy[3] * sy]


def scale_barcodes_to_full_res(
    barcodes: list[BarcodeBox], orig_h: int, orig_w: int, work_h: int, work_w: int
) -> list[BarcodeBox]:
    return [
        BarcodeBox(
            box_id=b.box_id,
            class_id=b.class_id,
            class_name=b.class_name,
            confidence=b.confidence,
            xyxy=scale_xyxy_to_full_res(b.xyxy, orig_h, orig_w, work_h, work_w),
        )
        for b in barcodes
    ]


def scale_instances_to_full_res(
    instances: list[InstanceBox], orig_h: int, orig_w: int, work_h: int, work_w: int
) -> list[InstanceBox]:
    return [
        InstanceBox(
            instance_id=inst.instance_id,
            label=inst.label,
            score=inst.score,
            xyxy=scale_xyxy_to_full_res(inst.xyxy, orig_h, orig_w, work_h, work_w),
        )
        for inst in instances
    ]


def scale_mask_infos_to_full_res(
    mask_infos: list[MaskInfo], orig_h: int, orig_w: int, work_h: int, work_w: int
) -> list[MaskInfo]:
    if work_h == orig_h and work_w == orig_w:
        return mask_infos
    sx = orig_w / work_w
    sy = orig_h / work_h
    scaled: list[MaskInfo] = []
    for mi in mask_infos:
        x, y, w, h = mi.bbox_xywh
        scaled.append(
            MaskInfo(
                mask_id=mi.mask_id,
                area=int(round(mi.area * sx * sy)),
                bbox_xywh=[x * sx, y * sy, w * sx, h * sy],
                predicted_iou=mi.predicted_iou,
                stability_score=mi.stability_score,
            )
        )
    return scaled


def detect_barcodes_on_rgb(
    yolo: Any, image_rgb: Any, conf: float, device: str, imgsz: int | None = None
) -> list[BarcodeBox]:
    if isinstance(yolo, YoloV5HubDetector):
        return detect_barcodes_yolov5_hub(yolo, image_rgb, conf, imgsz=imgsz)

    import cv2

    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    predict_kw: dict[str, Any] = {"conf": conf, "device": device, "verbose": False}
    if imgsz is not None:
        predict_kw["imgsz"] = imgsz
    result = yolo.predict(source=bgr, **predict_kw)[0]
    names = getattr(result, "names", {}) or {}
    detections: list[BarcodeBox] = []
    if result.boxes is None:
        return detections

    for box_id, box in enumerate(result.boxes):
        cls_value = int(box.cls.item()) if box.cls is not None else None
        class_name = names.get(cls_value) if cls_value is not None else None
        detections.append(
            BarcodeBox(
                box_id=box_id,
                class_id=cls_value,
                class_name=class_name,
                confidence=float(box.conf.item()) if box.conf is not None else 0.0,
                xyxy=[float(v) for v in box.xyxy[0].tolist()],
            )
        )
    return detections


def detect_barcodes_yolov5_hub(
    detector: YoloV5HubDetector, image_rgb: Any, conf: float, imgsz: int | None = None
) -> list[BarcodeBox]:
    model = detector.model
    model.conf = conf
    kwargs: dict[str, Any] = {}
    if imgsz is not None:
        kwargs["size"] = imgsz

    results = model(image_rgb, **kwargs)
    xyxy = results.xyxy[0] if getattr(results, "xyxy", None) else []
    detections: list[BarcodeBox] = []
    for box_id, row in enumerate(xyxy):
        values = row.detach().cpu().tolist() if hasattr(row, "detach") else list(row)
        if len(values) < 6:
            continue
        x1, y1, x2, y2, confidence, cls_value = values[:6]
        cls_id = int(cls_value)
        class_name = detector.names.get(cls_id) if isinstance(detector.names, dict) else None
        detections.append(
            BarcodeBox(
                box_id=box_id,
                class_id=cls_id,
                class_name=class_name,
                confidence=float(confidence),
                xyxy=[float(x1), float(y1), float(x2), float(y2)],
            )
        )
    return detections


def detect_barcodes(
    yolo: Any, image_path: Path, conf: float, device: str, imgsz: int | None = None
) -> list[BarcodeBox]:
    """Run YOLO on image file at full resolution (no resize)."""
    return detect_barcodes_on_rgb(yolo, load_image_rgb(image_path), conf, device, imgsz=imgsz)


def detect_instances_grounding_dino(detector: GroundingDinoDetector, image_rgb: Any, device: str) -> list[InstanceBox]:
    import torch
    from PIL import Image

    image = Image.fromarray(image_rgb)
    inputs = detector.processor(images=image, text=detector.prompt, return_tensors="pt")
    input_ids = inputs.get("input_ids")
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = detector.model(**model_inputs)

    results = detector.processor.post_process_grounded_object_detection(
        outputs,
        input_ids=input_ids,
        threshold=detector.box_threshold,
        text_threshold=detector.text_threshold,
        target_sizes=[image_rgb.shape[:2]],
    )[0]

    boxes = results.get("boxes", [])
    scores = results.get("scores", [])
    # transformers v4.51+ warns that `labels` may become integer ids; prefer `text_labels` when present.
    labels = results.get("text_labels", results.get("labels", []))
    raw_instances: list[tuple[str, float, list[float]]] = []
    for box, score, label in zip(boxes, scores, labels):
        xyxy = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)
        score_f = float(score.detach().cpu().item()) if hasattr(score, "detach") else float(score)
        raw_instances.append((str(label), score_f, [float(v) for v in xyxy]))

    raw_instances.sort(key=lambda item: item[1], reverse=True)
    return [
        InstanceBox(instance_id=instance_id, label=label, score=score, xyxy=xyxy)
        for instance_id, (label, score, xyxy) in enumerate(raw_instances)
    ]


def generate_masks(mask_generator: Any, image_rgb: Any) -> tuple[list[dict[str, Any]], list[MaskInfo]]:
    def sort_key(mask: dict[str, Any]) -> tuple[float, float, int]:
        # 점수 높은 마스크가 먼저 오도록(=낮은 mask_id)
        pred_iou = _optional_float(mask.get("predicted_iou")) or 0.0
        stability = _optional_float(mask.get("stability_score")) or 0.0
        area = int(mask.get("area", 0))
        return (pred_iou, stability, area)

    masks = sorted(mask_generator.generate(image_rgb), key=sort_key, reverse=True)
    mask_infos = [
        MaskInfo(
            mask_id=mask_id,
            area=int(mask.get("area", 0)),
            bbox_xywh=[float(v) for v in mask.get("bbox", [])],
            predicted_iou=_optional_float(mask.get("predicted_iou")),
            stability_score=_optional_float(mask.get("stability_score")),
        )
        for mask_id, mask in enumerate(masks)
    ]
    return masks, mask_infos


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def box_to_int_bounds(xyxy: list[float], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = image_shape
    x1 = max(0, min(width, int(round(xyxy[0]))))
    y1 = max(0, min(height, int(round(xyxy[1]))))
    x2 = max(0, min(width, int(round(xyxy[2]))))
    y2 = max(0, min(height, int(round(xyxy[3]))))
    return x1, y1, x2, y2


def calculate_match_for_box(
    barcode: BarcodeBox,
    masks: list[dict[str, Any]],
    image_shape: tuple[int, int],
    coverage_threshold: float,
) -> BarcodeMaskMatch:
    import numpy as np

    x1, y1, x2, y2 = box_to_int_bounds(barcode.xyxy, image_shape)
    box_area = max(0, x2 - x1) * max(0, y2 - y1)
    if box_area == 0:
        return BarcodeMaskMatch(barcode.box_id, False, None, False, 0.0, 0.0)

    cx = max(0, min(image_shape[1] - 1, int(round((barcode.xyxy[0] + barcode.xyxy[2]) / 2))))
    cy = max(0, min(image_shape[0] - 1, int(round((barcode.xyxy[1] + barcode.xyxy[3]) / 2))))
    box_region = np.zeros(image_shape, dtype=bool)
    box_region[y1:y2, x1:x2] = True

    best_mask_id: int | None = None
    best_center_inside = False
    best_coverage = 0.0
    best_iou = 0.0

    for mask_id, mask in enumerate(masks):
        segmentation = mask["segmentation"].astype(bool)
        intersection = int(np.logical_and(box_region, segmentation).sum())
        mask_area = int(segmentation.sum())
        union = box_area + mask_area - intersection
        coverage = intersection / box_area
        iou = intersection / union if union > 0 else 0.0
        center_inside = bool(segmentation[cy, cx])

        if coverage > best_coverage or (coverage == best_coverage and center_inside and not best_center_inside):
            best_mask_id = mask_id
            best_center_inside = center_inside
            best_coverage = coverage
            best_iou = iou

    matched = best_center_inside or best_coverage >= coverage_threshold
    return BarcodeMaskMatch(
        box_id=barcode.box_id,
        matched=matched,
        matched_mask_id=best_mask_id,
        center_inside=best_center_inside,
        box_mask_coverage=round(best_coverage, 6),
        iou=round(best_iou, 6),
    )


def calculate_match_for_instance_box(
    barcode: BarcodeBox,
    instances: list[InstanceBox],
    image_shape: tuple[int, int],
    coverage_threshold: float,
) -> BarcodeMaskMatch:
    bx1, by1, bx2, by2 = box_to_int_bounds(barcode.xyxy, image_shape)
    box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    if box_area == 0:
        return BarcodeMaskMatch(barcode.box_id, False, None, False, 0.0, 0.0)

    cx = max(0, min(image_shape[1] - 1, int(round((barcode.xyxy[0] + barcode.xyxy[2]) / 2))))
    cy = max(0, min(image_shape[0] - 1, int(round((barcode.xyxy[1] + barcode.xyxy[3]) / 2))))

    best_instance_id: int | None = None
    best_center_inside = False
    best_coverage = 0.0
    best_iou = 0.0
    for instance in instances:
        ix1, iy1, ix2, iy2 = box_to_int_bounds(instance.xyxy, image_shape)
        inter_w = max(0, min(bx2, ix2) - max(bx1, ix1))
        inter_h = max(0, min(by2, iy2) - max(by1, iy1))
        intersection = inter_w * inter_h
        instance_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = box_area + instance_area - intersection
        coverage = intersection / box_area
        iou = intersection / union if union > 0 else 0.0
        center_inside = ix1 <= cx <= ix2 and iy1 <= cy <= iy2
        if coverage > best_coverage or (coverage == best_coverage and center_inside and not best_center_inside):
            best_instance_id = instance.instance_id
            best_center_inside = center_inside
            best_coverage = coverage
            best_iou = iou

    matched = best_center_inside or best_coverage >= coverage_threshold
    return BarcodeMaskMatch(
        box_id=barcode.box_id,
        matched=matched,
        matched_mask_id=best_instance_id,
        center_inside=best_center_inside,
        box_mask_coverage=round(best_coverage, 6),
        iou=round(best_iou, 6),
    )


def match_barcodes_to_masks(
    barcodes: list[BarcodeBox],
    masks: list[dict[str, Any]],
    image_shape: tuple[int, int],
    coverage_threshold: float,
) -> list[BarcodeMaskMatch]:
    return [
        calculate_match_for_box(barcode, masks, image_shape, coverage_threshold)
        for barcode in barcodes
    ]


def match_barcodes_to_instances(
    barcodes: list[BarcodeBox],
    instances: list[InstanceBox],
    image_shape: tuple[int, int],
    coverage_threshold: float,
) -> list[BarcodeMaskMatch]:
    return [
        calculate_match_for_instance_box(barcode, instances, image_shape, coverage_threshold)
        for barcode in barcodes
    ]


def safe_output_stem(record: ImageRecord) -> str:
    clean_set = record.set_id.replace("/", "__").replace("\\", "__").replace(" ", "_")
    return f"{clean_set}__{Path(record.file_name).stem}"


def save_overlay(
    output_path: Path,
    image_rgb: Any,
    barcodes: list[BarcodeBox],
    masks: list[dict[str, Any]] | None,
    matches: list[BarcodeMaskMatch] | None,
    instances: list[InstanceBox] | None = None,
    *,
    draw_instance_masks: bool,
) -> None:
    import cv2
    import numpy as np

    # RGB — YOLO 바코드 박스 가시성 (굵은 빨간색)
    barcode_box_color = (255, 0, 0)
    barcode_box_thickness = 8  # ~1.5× 이전 5px
    # 라벨 가독성(약 +30%)
    label_font_scale = 1.625
    label_thickness = 4
    label_bg_color = (0, 0, 0)  # RGB black
    label_text_color = (255, 255, 255)  # RGB white

    overlay = image_rgb.copy()

    def draw_text_box(text: str, x: int, y: int, *, text_color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, label_font_scale, label_thickness)
        pad = 6
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(overlay.shape[1] - 1, x1 + tw + 2 * pad)
        y2 = min(overlay.shape[0] - 1, y1 + th + 2 * pad + baseline)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), label_bg_color, thickness=-1)
        cv2.putText(
            overlay,
            text,
            (x1 + pad, y1 + pad + th),
            font,
            label_font_scale,
            text_color,
            label_thickness,
            lineType=cv2.LINE_AA,
        )

    def draw_text_box_centered(text: str, cx: int, cy: int, *, text_color: tuple[int, int, int]) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, label_font_scale, label_thickness)
        pad = 6
        bw = tw + 2 * pad
        bh = th + 2 * pad + baseline
        x = int(round(cx - bw / 2))
        y = int(round(cy - bh / 2))
        x = max(0, min(overlay.shape[1] - 1, x))
        y = max(0, min(overlay.shape[0] - 1, y))
        draw_text_box(text, x, y, text_color=text_color)

    if draw_instance_masks:
        mask_canvas = overlay.copy()
        match_by_box = {match.box_id: match for match in (matches or [])}

        for mask_id, mask in enumerate(masks or []):
            segmentation = mask["segmentation"].astype(bool)
            color = np.array(
                [
                    (37 * (mask_id + 3)) % 255,
                    (83 * (mask_id + 5)) % 255,
                    (127 * (mask_id + 7)) % 255,
                ],
                dtype=np.uint8,
            )
            mask_canvas[segmentation] = (0.65 * mask_canvas[segmentation] + 0.35 * color).astype(np.uint8)

        overlay = mask_canvas

        # 마스크 윤곽선(마스크 색 기반으로 조금 더 진하게 + 굵게) + mask_id 표기
        contour_thickness = 3
        for mask_id, mask in enumerate(masks or []):
            seg_u8 = mask["segmentation"].astype(np.uint8) * 255
            contours, _hier = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            base_color = np.array(
                [
                    (37 * (mask_id + 3)) % 255,
                    (83 * (mask_id + 5)) % 255,
                    (127 * (mask_id + 7)) % 255,
                ],
                dtype=np.uint8,
            )
            contour_color = tuple(int(c) for c in (base_color * 0.7).tolist())
            mask_text_color = tuple(int(c) for c in base_color.tolist())

            if contours:
                cv2.drawContours(
                    overlay,
                    contours,
                    contourIdx=-1,
                    color=contour_color,
                    thickness=contour_thickness,
                    lineType=cv2.LINE_AA,
                )

            # 마스크 무게중심(centroid)에 mask_id + score 표시 (검정 박스 안)
            m = cv2.moments(seg_u8, binaryImage=True)
            if m.get("m00", 0) > 0:
                cx = int(round(m["m10"] / m["m00"]))
                cy = int(round(m["m01"] / m["m00"]))
                pred_iou = mask.get("predicted_iou")
                stab = mask.get("stability_score")
                pred_iou_s = f"{float(pred_iou):.2f}" if pred_iou is not None else "na"
                stab_s = f"{float(stab):.2f}" if stab is not None else "na"
                draw_text_box_centered(
                    f"mask_id:{mask_id} iou:{pred_iou_s} stab:{stab_s}",
                    cx,
                    cy,
                    text_color=mask_text_color,
                )
    else:
        match_by_box = {}

    if instances:
        for inst in instances:
            x1, y1, x2, y2 = [int(round(value)) for value in inst.xyxy]
            inst_color = (
                (37 * (inst.instance_id + 3)) % 255,
                (83 * (inst.instance_id + 5)) % 255,
                (127 * (inst.instance_id + 7)) % 255,
            )
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                inst_color,
                thickness=5,
                lineType=cv2.LINE_AA,
            )
            draw_text_box(
                f"inst_id:{inst.instance_id} {inst.label} score:{inst.score:.2f}",
                x1,
                max(0, y1 + 4),
                text_color=inst_color,
            )

    for barcode in barcodes:
        if draw_instance_masks and matches is not None:
            match = match_by_box.get(barcode.box_id)
        else:
            match = None
        x1, y1, x2, y2 = [int(round(value)) for value in barcode.xyxy]
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            barcode_box_color,
            thickness=barcode_box_thickness,
            lineType=cv2.LINE_AA,
        )
        label = f"box:{barcode.box_id} conf:{barcode.confidence:.2f}"
        if barcode.class_name:
            label += f" {barcode.class_name}"
        # bbox 라벨도 검정 박스 안에 표시
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, label_font_scale, label_thickness)
        pad = 6
        lx = max(0, x1)
        ly = max(0, y1 - (th + 2 * pad + baseline + 4))
        cv2.rectangle(
            overlay,
            (lx, ly),
            (min(overlay.shape[1] - 1, lx + tw + 2 * pad), min(overlay.shape[0] - 1, ly + th + 2 * pad + baseline)),
            label_bg_color,
            thickness=-1,
        )
        cv2.putText(
            overlay,
            label,
            (lx + pad, ly + pad + th),
            font,
            label_font_scale,
            barcode_box_color,
            label_thickness,
            lineType=cv2.LINE_AA,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset_kind",
        "item_name",
        "barcode_pose",
        "item_count",
        "item_combo",
        "layout",
        "set_id",
        "set_path",
        "image_path",
        "view",
        "view_index",
        "file_name",
        "barcode_count",
        "matched_count",
        "unmatched_count",
        "mask_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def create_dry_run_rows(records: list[ImageRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    detail_rows = [
        {
            "record": asdict(record),
            "dry_run": True,
            "barcode_count": None,
            "mask_count": None,
            "matches": [],
        }
        for record in records
    ]
    summary_rows = [
        {
            **asdict(record),
            "barcode_count": "",
            "matched_count": "",
            "unmatched_count": "",
            "mask_count": "",
        }
        for record in records
    ]
    return detail_rows, summary_rows


def validate_dataset(records: list[ImageRecord]) -> list[dict[str, Any]]:
    by_set: dict[str, list[ImageRecord]] = {}
    for record in records:
        by_set.setdefault(record.set_id, []).append(record)

    warnings: list[dict[str, Any]] = []
    expected_views = set(VIEW_NAMES)
    for set_id, set_records in by_set.items():
        views = {record.view for record in set_records if record.view}
        if len(set_records) != 6:
            warnings.append({"set_id": set_id, "type": "unexpected_image_count", "count": len(set_records)})
        elif views and views != expected_views:
            warnings.append(
                {
                    "set_id": set_id,
                    "type": "unexpected_view_names",
                    "views": sorted(views),
                }
            )
    return warnings


def run_inference(args: argparse.Namespace, records: list[ImageRecord]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mask_generator, grounding_dino, yolo, device = load_models(args)
    detail_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    errors_path = args.output_dir / "errors.jsonl"
    yolo_only: bool = bool(getattr(args, "yolo_only", False))
    instance_detector: str = "none" if yolo_only else args.instance_detector

    for index, record in enumerate(tqdm(records, desc="inference", unit="img"), start=1):
        image_path = Path(record.image_path)
        try:
            image_rgb_orig = load_image_rgb(image_path)
            orig_h, orig_w = image_rgb_orig.shape[:2]
            work_h, work_w, resize_scale = compute_resize_hw(orig_h, orig_w, args.max_long_edge)
            if (work_h, work_w) == (orig_h, orig_w):
                image_rgb_work = image_rgb_orig
            else:
                image_rgb_work = resize_rgb(image_rgb_orig, work_h, work_w)

            barcodes_work = detect_barcodes_on_rgb(
                yolo, image_rgb_work, args.conf, device, imgsz=args.yolo_imgsz
            )

            if yolo_only:
                masks_orig: list[dict[str, Any]] = []
                mask_infos_orig: list[MaskInfo] = []
                instances_orig: list[InstanceBox] = []
                matches: list[BarcodeMaskMatch] = []
                barcodes_orig = scale_barcodes_to_full_res(
                    barcodes_work, orig_h, orig_w, work_h, work_w
                )
            elif args.instance_detector == "sam":
                assert mask_generator is not None
                masks_work, mask_infos_work = generate_masks(mask_generator, image_rgb_work)
                matches = match_barcodes_to_masks(
                    barcodes_work, masks_work, (work_h, work_w), args.coverage_threshold
                )
                masks_orig = scale_masks_to_full_res(masks_work, orig_h, orig_w, work_h, work_w)
                mask_infos_orig = scale_mask_infos_to_full_res(
                    mask_infos_work, orig_h, orig_w, work_h, work_w
                )
                barcodes_orig = scale_barcodes_to_full_res(
                    barcodes_work, orig_h, orig_w, work_h, work_w
                )
                instances_orig = []
            else:
                assert grounding_dino is not None
                instances_work = detect_instances_grounding_dino(grounding_dino, image_rgb_work, device)
                matches = match_barcodes_to_instances(
                    barcodes_work, instances_work, (work_h, work_w), args.coverage_threshold
                )
                masks_orig = []
                mask_infos_orig = []
                instances_orig = scale_instances_to_full_res(
                    instances_work, orig_h, orig_w, work_h, work_w
                )
                barcodes_orig = scale_barcodes_to_full_res(
                    barcodes_work, orig_h, orig_w, work_h, work_w
                )

            matched_count = sum(1 for match in matches if match.matched) if matches else 0

            detail_rows.append(
                {
                    "record": asdict(record),
                    "yolo_only": yolo_only,
                    "image_size": {"width": orig_w, "height": orig_h},
                    "inference_size": {"width": work_w, "height": work_h},
                    "preprocess": {
                        "max_long_edge": args.max_long_edge,
                        "resize_scale": resize_scale,
                    },
                    "barcode_model_family": args.barcode_model_family,
                    "barcode_model_backend": BARCODE_MODEL_REGISTRY[args.barcode_model_family].backend,
                    "barcode_model_path": str(args.barcode_model),
                    "yolo_imgsz": args.yolo_imgsz,
                    "instance_detector": instance_detector,
                    "grounding_dino_model": args.grounding_dino_model if instance_detector == "grounding_dino" else "",
                    "grounding_dino_prompt": args.grounding_dino_prompt if instance_detector == "grounding_dino" else "",
                    "barcodes": [asdict(barcode) for barcode in barcodes_orig],
                    "masks": [asdict(mask_info) for mask_info in mask_infos_orig],
                    "instances": [asdict(instance) for instance in instances_orig],
                    "matches": [asdict(match) for match in matches],
                    "thresholds": {
                        "barcode_confidence": args.conf,
                        "coverage_threshold": args.coverage_threshold,
                        "grounding_dino_box_threshold": (
                            args.grounding_dino_box_threshold if instance_detector == "grounding_dino" else ""
                        ),
                        "grounding_dino_text_threshold": (
                            args.grounding_dino_text_threshold if instance_detector == "grounding_dino" else ""
                        ),
                    },
                }
            )
            summary_rows.append(
                {
                    **asdict(record),
                    "barcode_count": len(barcodes_orig),
                    "matched_count": ("" if yolo_only else matched_count),
                    "unmatched_count": ("" if yolo_only else len(barcodes_orig) - matched_count),
                    "mask_count": len(instances_orig) if instance_detector == "grounding_dino" else (0 if yolo_only else len(masks_orig)),
                }
            )

            if not args.no_overlays:
                overlay_path = args.output_dir / "overlays" / f"{safe_output_stem(record)}.jpg"
                save_overlay(
                    overlay_path,
                    image_rgb_orig,
                    barcodes_orig,
                    masks_orig,
                    matches,
                    instances=instances_orig,
                    draw_instance_masks=not yolo_only,
                )

        except Exception as exc:  # noqa: BLE001 - keep batch processing alive and record failures.
            error_row = {"record": asdict(record), "error": repr(exc)}
            append_jsonl(errors_path, error_row)
            tqdm.write(f"[{index}/{len(records)}] error: {record.image_path}: {exc}", file=sys.stderr)

    return detail_rows, summary_rows


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    args.dataset_root = dataset_root
    args.output_dir = output_dir
    barcode_model_config = BARCODE_MODEL_REGISTRY[args.barcode_model_family]
    if args.barcode_model is None:
        args.barcode_model = barcode_model_config.default_path
    args.barcode_model = args.barcode_model.expanduser().resolve()
    if args.yolo_imgsz is None and barcode_model_config.recommended_imgsz is not None:
        args.yolo_imgsz = barcode_model_config.recommended_imgsz
    if args.sam_checkpoint is not None:
        args.sam_checkpoint = args.sam_checkpoint.expanduser().resolve()

    if not dataset_root.exists():
        print(f"Dataset root does not exist: {dataset_root}", file=sys.stderr)
        return 2

    if args.max_long_edge is not None and args.max_long_edge < 1:
        print("error: --max-long-edge must be >= 1 when set", file=sys.stderr)
        return 2

    if args.yolo_imgsz is not None and args.yolo_imgsz < 32:
        print("error: --yolo-imgsz must be >= 32 when set", file=sys.stderr)
        return 2

    if (
        args.min_item_count is not None
        and args.max_item_count is not None
        and args.min_item_count > args.max_item_count
    ):
        print(
            "error: --min-item-count must be <= --max-item-count when both are set",
            file=sys.stderr,
        )
        return 2

    rng = random.Random(args.sample_seed)

    index_path = find_index_json(dataset_root)
    if index_path is not None:
        records = build_records_from_index_json(index_path)
        if args.limit_sets is not None:
            if args.limit_sets < 1:
                print("error: --limit-sets must be >= 1 when set", file=sys.stderr)
                return 2
            unique_set_ids = sorted({r.set_id for r in records})
            if unique_set_ids:
                k = min(args.limit_sets, len(unique_set_ids))
                keep = set(rng.sample(unique_set_ids, k=k))
                records = [r for r in records if r.set_id in keep]
    else:
        leaf_sets = discover_leaf_sets(dataset_root)
        if args.limit_sets is not None:
            if args.limit_sets < 1:
                print("error: --limit-sets must be >= 1 when set", file=sys.stderr)
                return 2
            k = min(args.limit_sets, len(leaf_sets))
            leaf_sets = rng.sample(leaf_sets, k=k)
        records = build_records(dataset_root, leaf_sets)

    if args.min_item_count is not None or args.max_item_count is not None:
        before_sets = len({record.set_id for record in records})
        before_images = len(records)
        records = filter_records_by_item_count(
            records,
            args.min_item_count,
            args.max_item_count,
        )
        after_sets = len({record.set_id for record in records})
        min_text = args.min_item_count if args.min_item_count is not None else "-inf"
        max_text = args.max_item_count if args.max_item_count is not None else "+inf"
        print(
            f"Item count filter [{min_text}, {max_text}]: "
            f"{before_sets} -> {after_sets} sets, {before_images} -> {len(records)} images"
        )

    if args.limit_images is not None:
        if args.limit_images < 1:
            print("error: --limit-images must be >= 1 when set", file=sys.stderr)
            return 2
        k = min(args.limit_images, len(records))
        records = rng.sample(records, k=k)

    if args.limit_sets is not None or args.limit_images is not None:
        rng.shuffle(records)

    if not records:
        print(f"No images found under: {dataset_root}", file=sys.stderr)
        return 2

    warnings = validate_dataset(records)
    write_jsonl(output_dir / "dataset_warnings.jsonl", warnings)

    if args.dry_run:
        detail_rows, summary_rows = create_dry_run_rows(records)
    else:
        detail_rows, summary_rows = run_inference(args, records)

    write_jsonl(output_dir / "results.jsonl", detail_rows)
    write_summary_csv(output_dir / "summary.csv", summary_rows)
    print(f"Wrote results to: {output_dir}")
    print(f"Images listed: {len(records)}")
    print(f"Dataset warnings: {len(warnings)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
