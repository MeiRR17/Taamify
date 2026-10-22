#!/usr/bin/env python3
"""Taamify Research OS planning CLI. No training, network, or live endpoint access."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PIPELINES = ROOT / "config" / "pipelines.json"
AUGMENTATIONS = ROOT / "config" / "augmentations.json"


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def calculate(config: dict, measured: dict | None = None) -> dict:
    measured = measured or {}
    rows: list[dict] = []
    ids: set[str] = set()
    for item in config["pipelines"]:
        key = item["id"]
        if key in ids:
            raise ValueError("Duplicate pipeline id: " + key)
        ids.add(key)
        fits, epochs = int(item["fits"]), int(item["epochs"])
        if fits < 1 or epochs < 1:
            raise ValueError("Invalid fit count or epochs: " + key)
        row = {
            "id": key,
            "name": item["name"],
            "hypothesis": item["hyp"],
            "status": item["status"],
            "dependencies": item["needs"],
            "fits": fits,
            "epochs_per_fit": epochs,
            "fit_epochs": fits * epochs,
            "hours_measured_inputs": None,
        }
        if key in measured:
            m = measured[key]
            minutes = float(m["minutes_per_epoch"])
            overhead = float(m.get("setup_hours", 0))
            if not (math.isfinite(minutes) and minutes > 0 and math.isfinite(overhead) and overhead >= 0):
                raise ValueError("Invalid benchmark: " + key)
            row["hours_measured_inputs"] = round(fits * epochs * minutes / 60 + overhead, 3)
        rows.append(row)
    unknown = set(measured) - ids
    if unknown:
        raise ValueError("Unknown pipeline measurements: " + ", ".join(sorted(unknown)))
    groups: dict[str, dict[str, int]] = {}
    for record in config["humanHours"]:
        for k in ("academic", "engineering", "product"):
            slot = groups.setdefault(k, {"minimum": 0, "maximum": 0})
            low, high = record[k]
            slot["minimum"] += int(low)
            slot["maximum"] += int(high)
    return {
        "status": "planning_only",
        "measurement_basis": "user_provided_benchmark_not_verified",
        "pipeline_count": len(rows),
        "planned_fits": sum(p["fits"] for p in rows),
        "planned_fit_epochs": sum(p["fit_epochs"] for p in rows),
        "hours_from_measured_inputs_only": round(sum(p["hours_measured_inputs"] or 0 for p in rows), 3),
        "pipelines_with_benchmarks": sum(p["hours_measured_inputs"] is not None for p in rows),
        "human_hours_planned": groups,
        "pipelines": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan Taamify academic/product experiments; no jobs are launched.")
    parser.add_argument("--measurements", type=Path, help="Local JSON with measured minutes_per_epoch per pipeline")
    parser.add_argument("--json", action="store_true", help="Machine-readable JSON output (no private data)")
    args = parser.parse_args()
    config = load_json(PIPELINES)
    aug = load_json(AUGMENTATIONS)
    measurement = load_json(args.measurements) if args.measurements else {}
    result = calculate(config, measurement)
    result["augmentation_modules"] = len(aug["modules"])
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return
    print("Taamify Research OS | plan only; no live jobs or server connection")
    print(f"  {result['pipeline_count']} pipelines, {result['augmentation_modules']} augmentations")
    print(f"  {result['planned_fits']} proposed fits; {result['planned_fit_epochs']} planned fit-epochs")
    print("  Hours of training are NOT known until a benchmark file has measured minutes/epoch.")
    for item in result["pipelines"]:
        hours = item["hours_measured_inputs"]
        time = f"{hours:.2f} accelerator-h (provided input)" if hours is not None else "unmeasured"
        print(f"  {item['id']}  {item['status']:<24} {item['fits']}x{item['epochs']:<3}  {time}  {item['name']}")
    print("  Human academic/product/engineering hours are estimates, never computed GPU time.")
    for name, estimate in result["human_hours_planned"].items():
        print(f"    {name}: {estimate['minimum']}–{estimate['maximum']} planned person-hours")


if __name__ == "__main__":
    main()
