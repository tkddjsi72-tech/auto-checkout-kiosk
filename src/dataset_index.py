#!/usr/bin/env python3
"""Shared dataset index.json utilities for kiosk pipelines."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

VIEW_ORDER_EN = ("top", "front", "left", "back", "right", "bottom")


def parse_view_list(spec: str) -> set[str]:
    """Parse comma-separated view names (e.g. 'bottom' or 'bottom,top')."""
    names = {part.strip().lower() for part in spec.split(",") if part.strip()}
    unknown = names - set(VIEW_ORDER_EN)
    if unknown:
        raise ValueError(
            f"Unknown view(s): {sorted(unknown)}. "
            f"Valid: {', '.join(VIEW_ORDER_EN)}"
        )
    return names


def active_view_order(exclude_views: set[str] | None = None) -> tuple[str, ...]:
    """Return VIEW_ORDER_EN minus excluded views."""
    if not exclude_views:
        return VIEW_ORDER_EN
    return tuple(v for v in VIEW_ORDER_EN if v not in exclude_views)


def find_index_json(dataset_root: Path) -> Path | None:
    """Find an index.json describing the dataset structure.

    Supports either:
    - <dataset_root>/index.json
    - <dataset_root>/dataset/index.json  (common when dataset_root is a wrapper folder)
    """
    direct = dataset_root / "index.json"
    if direct.exists() and direct.is_file():
        return direct
    nested = dataset_root / "dataset" / "index.json"
    if nested.exists() and nested.is_file():
        return nested
    return None


def load_index_entries(index_path: Path) -> tuple[list[dict[str, Any]], Path]:
    """Load raw entry dicts and base_dir from v1 index.json."""
    base_dir = index_path.parent
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    version = int(payload.get("version", 0) or 0)
    if version != 1:
        raise ValueError(f"Unsupported index.json version: {version} ({index_path})")
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid index.json: 'entries' must be a list ({index_path})")
    return [e for e in entries if isinstance(e, dict)], base_dir


def entry_set_id(entry: dict[str, Any]) -> str:
    entry_id = str(entry.get("id", "")).strip()
    group = str(entry.get("group", "")).strip() or "unknown"
    return f"{group}/{entry_id}"


def resolve_set_image_paths(
    entry: dict[str, Any],
    base_dir: Path,
    *,
    require_all_views: bool = False,
    exclude_views: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Resolve multi-view image paths in VIEW_ORDER_EN order.

    Returns:
        (absolute_paths, view_names) — only views that exist on disk unless
        require_all_views=True (then raises ValueError on any missing view).
    """
    views = entry.get("views", {}) or {}
    if not isinstance(views, dict):
        views = {}

    paths: list[str] = []
    view_names: list[str] = []
    missing: list[str] = []

    for view in active_view_order(exclude_views):
        rel = views.get(view)
        if rel is None:
            missing.append(view)
            continue
        abs_path = (base_dir / str(rel)).resolve()
        if not abs_path.exists():
            missing.append(view)
            continue
        paths.append(str(abs_path))
        view_names.append(view)

    if require_all_views and missing:
        entry_id = entry.get("id", "?")
        raise ValueError(f"Entry {entry_id}: missing views {missing}")

    return paths, view_names


def filter_entries(
    entries: list[dict[str, Any]],
    min_item_count: int | None,
    max_item_count: int | None,
) -> list[dict[str, Any]]:
    """Filter entries by item_count (inclusive range)."""
    if min_item_count is None and max_item_count is None:
        return entries

    min_value = min_item_count if min_item_count is not None else 0
    max_value = max_item_count if max_item_count is not None else 10**9
    if min_value > max_value:
        raise ValueError(
            f"min_item_count ({min_value}) must be <= max_item_count ({max_value})"
        )

    kept: list[dict[str, Any]] = []
    for entry in entries:
        raw = entry.get("item_count")
        if raw is None:
            continue
        try:
            count = int(raw)
        except (TypeError, ValueError):
            continue
        if min_value <= count <= max_value:
            kept.append(entry)
    return kept


def sample_entries(
    entries: list[dict[str, Any]],
    *,
    limit_sets: int | None,
    sample_seed: int | None,
) -> list[dict[str, Any]]:
    """Randomly sample up to limit_sets entries (non-replacement)."""
    if limit_sets is None:
        return entries
    if limit_sets < 1:
        raise ValueError("limit_sets must be >= 1 when set")

    rng = random.Random(sample_seed)
    if len(entries) <= limit_sets:
        result = list(entries)
    else:
        result = rng.sample(entries, k=limit_sets)
    rng.shuffle(result)
    return result
