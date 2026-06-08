#!/usr/bin/env python3
"""Streaming comparison of two matched NISAR L2 GCOV products."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.transform import from_origin


ROOT = "/science/LSAR/GCOV/grids/frequencyA"
POLS = {"HH": "HHHH", "HV": "HVHV"}
FLOOR = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("earlier", type=Path)
    parser.add_argument("later", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("gcov_change_results"))
    parser.add_argument("--block-rows", type=int, default=512)
    parser.add_argument("--sample-step", type=int, default=32)
    return parser.parse_args()


def metadata(path: Path) -> dict:
    with h5py.File(path, "r") as src:
        group = src[ROOT]
        x = np.asarray(group["xCoordinates"][:], dtype=np.float64)
        y = np.asarray(group["yCoordinates"][:], dtype=np.float64)
        return {
            "path": path,
            "x": x,
            "y": y,
            "shape": tuple(group["HHHH"].shape),
            "epsg": int(group["projection"].attrs["epsg_code"]),
        }


def validate(a: dict, b: dict) -> None:
    if a["shape"] != b["shape"]:
        raise ValueError(f"Shape mismatch: {a['shape']} vs {b['shape']}")
    if a["epsg"] != b["epsg"]:
        raise ValueError(f"CRS mismatch: EPSG:{a['epsg']} vs EPSG:{b['epsg']}")
    if not np.array_equal(a["x"], b["x"]) or not np.array_equal(a["y"], b["y"]):
        raise ValueError("Coordinate grids differ; reprojection is required first.")


def to_db(power: np.ndarray) -> np.ndarray:
    out = np.full(power.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(power) & (power > FLOOR)
    out[valid] = 10.0 * np.log10(power[valid])
    return out


def transform(x: np.ndarray, y: np.ndarray):
    xres = float(np.median(np.diff(x)))
    yres = float(np.median(np.abs(np.diff(y))))
    return from_origin(float(x.min() - abs(xres) / 2), float(y.max() + yres / 2), abs(xres), yres)


def profile(meta: dict) -> dict:
    rows, cols = meta["shape"]
    return {
        "driver": "GTiff",
        "height": rows,
        "width": cols,
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{meta['epsg']}",
        "transform": transform(meta["x"], meta["y"]),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": np.nan,
        "BIGTIFF": "YES",
    }


def write_window(dataset, y_ascending: bool, row0: int, row1: int, array: np.ndarray) -> None:
    if y_ascending:
        height = dataset.height
        write_row = height - row1
        array = np.flipud(array)
    else:
        write_row = row0
    dataset.write(array.astype(np.float32), 1, window=((write_row, write_row + array.shape[0]), (0, array.shape[1])))


def stat_init() -> dict:
    return {"n": 0, "sum": 0.0, "sum2": 0.0, "abs_sum": 0.0, "samples": []}


def update_stats(stats: dict, change: np.ndarray, sample_step: int) -> None:
    finite = change[np.isfinite(change)]
    if finite.size:
        stats["n"] += int(finite.size)
        stats["sum"] += float(np.sum(finite, dtype=np.float64))
        stats["sum2"] += float(np.sum(finite.astype(np.float64) ** 2))
        stats["abs_sum"] += float(np.sum(np.abs(finite), dtype=np.float64))
    sample = change[::sample_step, ::sample_step]
    sample = sample[np.isfinite(sample)]
    if sample.size:
        stats["samples"].append(sample.astype(np.float32))


def finalize_stats(stats: dict) -> dict:
    samples = np.concatenate(stats["samples"]) if stats["samples"] else np.array([], dtype=np.float32)
    mean = stats["sum"] / stats["n"]
    var = max(stats["sum2"] / stats["n"] - mean * mean, 0.0)
    if samples.size:
        percentiles = [float(v) for v in np.percentile(samples, (2, 25, 50, 75, 98))]
        threshold = float(np.percentile(np.abs(samples), 95))
        frac_inc = float(np.mean(samples >= threshold))
        frac_dec = float(np.mean(samples <= -threshold))
    else:
        percentiles = [float("nan")] * 5
        threshold = frac_inc = frac_dec = float("nan")
    return {
        "valid_pixel_count": stats["n"],
        "mean_change_db": mean,
        "standard_deviation_change_db": float(np.sqrt(var)),
        "mean_absolute_change_db": stats["abs_sum"] / stats["n"],
        "sampled_change_db_percentiles_2_25_50_75_98": percentiles,
        "sampled_large_change_threshold_abs_db_p95": threshold,
        "sampled_fraction_large_increase": frac_inc,
        "sampled_fraction_large_decrease": frac_dec,
        "sample_count_for_percentiles": int(samples.size),
    }


def quicklook_init(rows: int, cols: int, max_side: int = 1800) -> tuple[int, int, int]:
    step = max(1, int(np.ceil(max(rows, cols) / max_side)))
    return step, int(np.ceil(rows / step)), int(np.ceil(cols / step))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    earlier_meta = metadata(args.earlier)
    later_meta = metadata(args.later)
    validate(earlier_meta, later_meta)

    rows, cols = earlier_meta["shape"]
    y_ascending = earlier_meta["y"][0] < earlier_meta["y"][-1]
    prof = profile(earlier_meta)
    qstep, qrows, qcols = quicklook_init(rows, cols)
    quicklooks = {}
    summaries = {}

    with h5py.File(args.earlier, "r") as earlier, h5py.File(args.later, "r") as later:
        eg = earlier[ROOT]
        lg = later[ROOT]
        for label, dataset_name in POLS.items():
            stats = stat_init()
            q_earlier = np.full((qrows, qcols), np.nan, dtype=np.float32)
            q_later = np.full((qrows, qcols), np.nan, dtype=np.float32)
            q_change = np.full((qrows, qcols), np.nan, dtype=np.float32)
            outputs = {
                "earlier": rasterio.open(args.output_dir / f"{label}_earlier_db.tif", "w", **prof),
                "later": rasterio.open(args.output_dir / f"{label}_later_db.tif", "w", **prof),
                "change": rasterio.open(args.output_dir / f"{label}_change_db.tif", "w", **prof),
            }
            try:
                for row0 in range(0, rows, args.block_rows):
                    row1 = min(rows, row0 + args.block_rows)
                    e_db = to_db(np.asarray(eg[dataset_name][row0:row1, :], dtype=np.float32))
                    l_db = to_db(np.asarray(lg[dataset_name][row0:row1, :], dtype=np.float32))
                    change = l_db - e_db
                    update_stats(stats, change, args.sample_step)
                    write_window(outputs["earlier"], y_ascending, row0, row1, e_db)
                    write_window(outputs["later"], y_ascending, row0, row1, l_db)
                    write_window(outputs["change"], y_ascending, row0, row1, change)

                    first_sample_row = ((row0 + qstep - 1) // qstep) * qstep
                    source_rows = np.arange(first_sample_row, row1, qstep)
                    local_rows = source_rows - row0
                    target_rows = source_rows // qstep
                    q_earlier[target_rows, :] = e_db[local_rows, ::qstep]
                    q_later[target_rows, :] = l_db[local_rows, ::qstep]
                    q_change[target_rows, :] = change[local_rows, ::qstep]
            finally:
                for out in outputs.values():
                    out.close()
            summaries[label] = finalize_stats(stats)
            quicklooks[label] = (q_earlier, q_later, q_change)

    write_reports(args, earlier_meta, later_meta, summaries)
    write_overview(args.output_dir / "gcov_change_overview.png", quicklooks)
    print(f"Wrote analysis products to {args.output_dir.resolve()}")


def write_reports(args: argparse.Namespace, earlier_meta: dict, later_meta: dict, summaries: dict) -> None:
    report = {
        "earlier_file": earlier_meta["path"].name,
        "later_file": later_meta["path"].name,
        "epsg": earlier_meta["epsg"],
        "shape": list(earlier_meta["shape"]),
        "polarization_summaries": summaries,
        "note": "Percentiles and large-change fractions use deterministic spatial sampling; mean and standard deviation use all valid pixels.",
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with (args.output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("polarization", "metric", "value"))
        writer.writeheader()
        for pol, metrics in summaries.items():
            for key, value in metrics.items():
                writer.writerow({"polarization": pol, "metric": key, "value": json.dumps(value) if isinstance(value, list) else value})


def write_overview(path: Path, quicklooks: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for row, pol in enumerate(("HH", "HV")):
        earlier, later, change = quicklooks[pol]
        db_vals = np.concatenate((earlier[np.isfinite(earlier)], later[np.isfinite(later)]))
        low, high = np.percentile(db_vals, (2, 98))
        limit = max(1.0, float(np.percentile(np.abs(change[np.isfinite(change)]), 98)))
        axes[row, 0].imshow(earlier, cmap="gray", vmin=low, vmax=high)
        axes[row, 0].set_title(f"{pol} Nov 18 (dB)")
        axes[row, 1].imshow(later, cmap="gray", vmin=low, vmax=high)
        axes[row, 1].set_title(f"{pol} Nov 30 (dB)")
        im = axes[row, 2].imshow(change, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit))
        axes[row, 2].set_title(f"{pol} change (dB)")
        fig.colorbar(im, ax=axes[row, 2], shrink=0.75)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("NISAR GCOV 12-day change: Nov 30 minus Nov 18, 2025")
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
