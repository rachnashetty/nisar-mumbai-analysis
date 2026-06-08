#!/usr/bin/env python3
"""Compare two matched NISAR L2 GCOV products."""

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
POLARIZATIONS = ("HHHH", "HVHV")
FLOOR = 1e-10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create HH/HV backscatter and change products from two NISAR GCOV files."
    )
    parser.add_argument("earlier", type=Path, help="Earlier GCOV .h5 file")
    parser.add_argument("later", type=Path, help="Later GCOV .h5 file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("gcov_change_results")
    )
    return parser.parse_args()


def read_scalar(dataset: h5py.Dataset) -> float:
    value = np.asarray(dataset[()]).squeeze()
    return float(value)


def read_product(path: Path) -> dict:
    with h5py.File(path, "r") as src:
        group = src[ROOT]
        missing = [name for name in POLARIZATIONS if name not in group]
        if missing:
            raise KeyError(f"{path.name} is missing: {', '.join(missing)}")

        x = np.asarray(group["xCoordinates"][:], dtype=np.float64)
        y = np.asarray(group["yCoordinates"][:], dtype=np.float64)
        projection = group["projection"]
        epsg = int(
            projection.attrs.get(
                "epsg_code",
                projection.attrs.get("spatial_ref", 0),
            )
        )

        arrays = {
            pol: np.asarray(group[pol][:], dtype=np.float32)
            for pol in POLARIZATIONS
        }

    return {
        "path": path,
        "x": x,
        "y": y,
        "epsg": epsg,
        "arrays": arrays,
    }


def validate_grids(earlier: dict, later: dict) -> None:
    for axis in ("x", "y"):
        if earlier[axis].shape != later[axis].shape or not np.allclose(
            earlier[axis], later[axis], rtol=0, atol=0.01
        ):
            raise ValueError(
                f"The products do not share the same {axis}-coordinate grid. "
                "Reprojection is required before comparison."
            )
    if earlier["epsg"] != later["epsg"]:
        raise ValueError(
            f"CRS mismatch: EPSG:{earlier['epsg']} versus EPSG:{later['epsg']}"
        )


def to_db(power: np.ndarray) -> np.ndarray:
    result = np.full(power.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(power) & (power > FLOOR)
    result[valid] = 10.0 * np.log10(power[valid])
    return result


def raster_transform(x: np.ndarray, y: np.ndarray):
    xres = float(np.median(np.diff(x)))
    yres = float(np.median(np.abs(np.diff(y))))
    return from_origin(
        float(x.min() - abs(xres) / 2),
        float(y.max() + yres / 2),
        abs(xres),
        yres,
    )


def orient_north_up(array: np.ndarray, y: np.ndarray) -> np.ndarray:
    return array if y[0] > y[-1] else np.flipud(array)


def write_tif(
    path: Path,
    array: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epsg: int,
) -> None:
    output = orient_north_up(array, y)
    profile = {
        "driver": "GTiff",
        "height": output.shape[0],
        "width": output.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": f"EPSG:{epsg}",
        "transform": raster_transform(x, y),
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "nodata": np.nan,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(output.astype(np.float32), 1)


def finite_percentiles(array: np.ndarray, values: tuple[int, ...]) -> list[float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return [float("nan")] * len(values)
    return [float(value) for value in np.percentile(finite, values)]


def summarize(
    earlier_db: np.ndarray, later_db: np.ndarray, change_db: np.ndarray
) -> dict:
    valid = np.isfinite(earlier_db) & np.isfinite(later_db)
    change = change_db[valid]
    if change.size == 0:
        raise ValueError("No valid overlapping pixels were found.")

    threshold = float(np.percentile(np.abs(change), 95))
    return {
        "valid_pixel_count": int(change.size),
        "earlier_db_percentiles_2_50_98": finite_percentiles(
            earlier_db[valid], (2, 50, 98)
        ),
        "later_db_percentiles_2_50_98": finite_percentiles(
            later_db[valid], (2, 50, 98)
        ),
        "change_db_percentiles_2_25_50_75_98": finite_percentiles(
            change, (2, 25, 50, 75, 98)
        ),
        "mean_change_db": float(np.mean(change)),
        "standard_deviation_change_db": float(np.std(change)),
        "mean_absolute_change_db": float(np.mean(np.abs(change))),
        "large_change_threshold_abs_db_p95": threshold,
        "fraction_large_increase": float(np.mean(change >= threshold)),
        "fraction_large_decrease": float(np.mean(change <= -threshold)),
    }


def plot_summary(
    output_path: Path,
    earlier_db: dict[str, np.ndarray],
    later_db: dict[str, np.ndarray],
    change_db: dict[str, np.ndarray],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    labels = {"HHHH": "HH", "HVHV": "HV"}

    for row, pol in enumerate(POLARIZATIONS):
        combined = np.concatenate(
            (
                earlier_db[pol][np.isfinite(earlier_db[pol])],
                later_db[pol][np.isfinite(later_db[pol])],
            )
        )
        low, high = np.percentile(combined, (2, 98))
        change_limit = max(
            1.0,
            float(
                np.percentile(
                    np.abs(change_db[pol][np.isfinite(change_db[pol])]), 98
                )
            ),
        )

        axes[row, 0].imshow(earlier_db[pol], cmap="gray", vmin=low, vmax=high)
        axes[row, 0].set_title(f"{labels[pol]} earlier (dB)")
        axes[row, 1].imshow(later_db[pol], cmap="gray", vmin=low, vmax=high)
        axes[row, 1].set_title(f"{labels[pol]} later (dB)")
        image = axes[row, 2].imshow(
            change_db[pol],
            cmap="RdBu_r",
            norm=TwoSlopeNorm(vmin=-change_limit, vcenter=0, vmax=change_limit),
        )
        axes[row, 2].set_title(f"{labels[pol]} change: later - earlier (dB)")
        fig.colorbar(image, ax=axes[row, 2], shrink=0.75, label="dB")

    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle("NISAR GCOV 12-day backscatter comparison", fontsize=16)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary_csv(path: Path, summaries: dict[str, dict]) -> None:
    rows = []
    for polarization, summary in summaries.items():
        for metric, value in summary.items():
            rows.append(
                {
                    "polarization": polarization,
                    "metric": metric,
                    "value": json.dumps(value) if isinstance(value, list) else value,
                }
            )
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target, fieldnames=("polarization", "metric", "value")
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    earlier = read_product(args.earlier)
    later = read_product(args.later)
    validate_grids(earlier, later)

    earlier_db = {pol: to_db(earlier["arrays"][pol]) for pol in POLARIZATIONS}
    later_db = {pol: to_db(later["arrays"][pol]) for pol in POLARIZATIONS}
    change_db = {
        pol: later_db[pol] - earlier_db[pol] for pol in POLARIZATIONS
    }

    summaries = {}
    for pol in POLARIZATIONS:
        short = pol[:2]
        summaries[short] = summarize(
            earlier_db[pol], later_db[pol], change_db[pol]
        )
        for label, array in (
            ("earlier_db", earlier_db[pol]),
            ("later_db", later_db[pol]),
            ("change_db", change_db[pol]),
        ):
            write_tif(
                args.output_dir / f"{short}_{label}.tif",
                array,
                earlier["x"],
                earlier["y"],
                earlier["epsg"],
            )

    ratio_earlier = earlier_db["HHHH"] - earlier_db["HVHV"]
    ratio_later = later_db["HHHH"] - later_db["HVHV"]
    ratio_change = ratio_later - ratio_earlier
    for label, array in (
        ("earlier_db", ratio_earlier),
        ("later_db", ratio_later),
        ("change_db", ratio_change),
    ):
        write_tif(
            args.output_dir / f"HH_minus_HV_{label}.tif",
            array,
            earlier["x"],
            earlier["y"],
            earlier["epsg"],
        )

    report = {
        "earlier_file": earlier["path"].name,
        "later_file": later["path"].name,
        "epsg": earlier["epsg"],
        "shape": list(earlier_db["HHHH"].shape),
        "polarization_summaries": summaries,
        "interpretation_note": (
            "Large dB changes are radar-observed surface changes, not definitive "
            "land-cover labels. Review rainfall, irrigation, harvest, water, RFI, "
            "edge artifacts, and the X05009/X05010 processing-version difference."
        ),
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    write_summary_csv(args.output_dir / "summary.csv", summaries)
    plot_summary(
        args.output_dir / "gcov_change_overview.png",
        earlier_db,
        later_db,
        change_db,
    )
    print(f"Wrote analysis products to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
