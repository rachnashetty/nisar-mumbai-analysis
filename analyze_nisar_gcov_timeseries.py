#!/usr/bin/env python3
"""Summarize and visualize a four-date NISAR GCOV time series."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import rasterio
from PIL import Image, ImageDraw, ImageFont
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import transform_geom


ROOT = "/science/LSAR/GCOV/grids/frequencyA"
POLARIZATIONS = {"HH": "HHHH", "HV": "HVHV"}
DATES = [
    (
        "Nov 18",
        Path("NISAR_L2_PR_GCOV_005_113_A_011_4005_DHDH_A_20251118T003447_20251118T003522_X05009_N_F_J_001.h5"),
    ),
    (
        "Nov 30",
        Path("NISAR_L2_PR_GCOV_006_113_A_011_4005_DHDH_A_20251130T003448_20251130T003522_X05010_F_F_J_001.h5"),
    ),
    (
        "Dec 24",
        Path("NISAR_L2_PR_GCOV_008_113_A_011_4005_DHDH_A_20251224T003449_20251224T003523_X05009_N_F_J_001.h5"),
    ),
    (
        "Jan 5",
        Path("NISAR_L2_PR_GCOV_009_113_A_011_4005_DHDH_A_20260105T003449_20260105T003524_X05009_N_F_J_001.h5"),
    ),
]
AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            (70.8877, 18.5631),
            (73.819, 18.5631),
            (73.819, 20.1037),
            (70.8877, 20.1037),
            (70.8877, 18.5631),
        ]
    ],
}
FLOOR = 1e-10
SAMPLE_STEP = 20


def to_db(power: np.ndarray) -> np.ndarray:
    result = np.full(power.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(power) & (power > FLOOR)
    result[valid] = 10.0 * np.log10(power[valid])
    return result


def get_grid(path: Path) -> dict:
    with h5py.File(path, "r") as src:
        group = src[ROOT]
        x = np.asarray(group["xCoordinates"][:], dtype=np.float64)
        y = np.asarray(group["yCoordinates"][:], dtype=np.float64)
        epsg = int(group["projection"].attrs["epsg_code"])
        shape = tuple(group["HHHH"].shape)
    xres = float(abs(np.median(np.diff(x))))
    yres = float(abs(np.median(np.diff(y))))
    transform = from_origin(x.min() - xres / 2, y.max() + yres / 2, xres, yres)
    return {"x": x, "y": y, "epsg": epsg, "shape": shape, "transform": transform}


def aoi_window(grid: dict) -> tuple[slice, slice, np.ndarray]:
    geom = transform_geom("EPSG:4326", f"EPSG:{grid['epsg']}", AOI, precision=3)
    coords = geom["coordinates"][0]
    xs = [point[0] for point in coords]
    ys = [point[1] for point in coords]
    inverse = ~grid["transform"]
    col0, row1 = inverse * (min(xs), min(ys))
    col1, row0 = inverse * (max(xs), max(ys))
    row0 = max(0, int(np.floor(row0)))
    row1 = min(grid["shape"][0], int(np.ceil(row1)))
    col0 = max(0, int(np.floor(col0)))
    col1 = min(grid["shape"][1], int(np.ceil(col1)))
    window_transform = grid["transform"] * rasterio.Affine.translation(col0, row0)
    mask = geometry_mask(
        [geom],
        out_shape=(row1 - row0, col1 - col0),
        transform=window_transform,
        invert=True,
    )
    return slice(row0, row1), slice(col0, col1), mask


def summarize(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    return {
        "valid_sample_count": int(finite.size),
        "mean_db": float(np.mean(finite)),
        "median_db": float(np.median(finite)),
        "standard_deviation_db": float(np.std(finite)),
        "percentiles_2_25_50_75_98_db": [
            float(value) for value in np.percentile(finite, (2, 25, 50, 75, 98))
        ],
    }


def summarize_change(values: np.ndarray) -> dict:
    result = summarize(values)
    finite = values[np.isfinite(values)]
    threshold = float(np.percentile(np.abs(finite), 95))
    result.update(
        {
            "mean_absolute_change_db": float(np.mean(np.abs(finite))),
            "absolute_change_p95_db": threshold,
            "fraction_large_increase": float(np.mean(finite >= threshold)),
            "fraction_large_decrease": float(np.mean(finite <= -threshold)),
        }
    )
    return result


def grayscale(array: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.nan_to_num(np.clip((array - low) / (high - low), 0, 1))
    return Image.fromarray((scaled * 255).astype(np.uint8), "L").convert("RGB")


def diverging(array: np.ndarray, limit: float) -> Image.Image:
    normalized = np.clip(array / limit, -1, 1)
    magnitude = np.maximum(normalized, -normalized)
    base = 235 * (1 - magnitude)
    rgb = np.empty((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(base + 220 * np.clip(normalized, 0, 1), 0, 255)
    rgb[..., 1] = np.clip(base + 70 * (1 - magnitude), 0, 255)
    rgb[..., 2] = np.clip(base + 220 * np.clip(-normalized, 0, 1), 0, 255)
    rgb[~np.isfinite(array)] = 0
    return Image.fromarray(rgb, "RGB")


def resize_for_panel(image: Image.Image, width: int = 470) -> Image.Image:
    height = int(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.BILINEAR)


def make_overview(
    output: Path,
    date_arrays: dict[str, dict[str, np.ndarray]],
    changes: dict[str, dict[str, np.ndarray]],
) -> None:
    font = ImageFont.load_default()
    panels: list[tuple[str, Image.Image]] = []
    for pol in POLARIZATIONS:
        all_date_values = np.concatenate(
            [
                array[np.isfinite(array)]
                for array in date_arrays[pol].values()
            ]
        )
        low, high = np.percentile(all_date_values, (2, 98))
        for date_label, array in date_arrays[pol].items():
            panels.append(
                (f"{pol} {date_label} dB", resize_for_panel(grayscale(array, low, high)))
            )
        change_values = np.concatenate(
            [array[np.isfinite(array)] for array in changes[pol].values()]
        )
        limit = max(1.0, float(np.percentile(np.abs(change_values), 98)))
        for interval, array in changes[pol].items():
            panels.append(
                (f"{pol} change {interval}", resize_for_panel(diverging(array, limit)))
            )

    columns = 7
    rows = 2
    panel_width = panels[0][1].width
    panel_height = panels[0][1].height
    margin = 10
    header = 26
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns + 1) * margin,
            rows * (panel_height + header) + (rows + 1) * margin,
        ),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (label, image) in enumerate(panels):
        row, column = divmod(index, columns)
        x = margin + column * (panel_width + margin)
        y = margin + row * (panel_height + header + margin)
        draw.text((x, y), label, fill=(0, 0, 0), font=font)
        canvas.paste(image, (x, y + header))
    canvas.save(output)


def main() -> None:
    output_dir = Path("gcov_timeseries_results")
    output_dir.mkdir(exist_ok=True)
    grid = get_grid(DATES[0][1])
    row_slice, col_slice, mask = aoi_window(grid)

    date_arrays: dict[str, dict[str, np.ndarray]] = {pol: {} for pol in POLARIZATIONS}
    report = {
        "aoi_wgs84": AOI,
        "dates": [label for label, _ in DATES],
        "polarizations": {},
    }
    for pol, dataset_name in POLARIZATIONS.items():
        report["polarizations"][pol] = {"dates": {}, "changes": {}}
        for date_label, path in DATES:
            with h5py.File(path, "r") as src:
                power = np.asarray(
                    src[ROOT][dataset_name][row_slice, col_slice],
                    dtype=np.float32,
                )
            db = to_db(power)
            db[~mask] = np.nan
            sampled = db[::SAMPLE_STEP, ::SAMPLE_STEP]
            date_arrays[pol][date_label] = sampled
            report["polarizations"][pol]["dates"][date_label] = summarize(sampled)

    changes: dict[str, dict[str, np.ndarray]] = {pol: {} for pol in POLARIZATIONS}
    intervals = list(zip(DATES[:-1], DATES[1:]))
    for pol in POLARIZATIONS:
        for (earlier_label, _), (later_label, _) in intervals:
            interval = f"{earlier_label} to {later_label}"
            change = date_arrays[pol][later_label] - date_arrays[pol][earlier_label]
            changes[pol][interval] = change
            report["polarizations"][pol]["changes"][interval] = summarize_change(change)
        full_change = date_arrays[pol]["Jan 5"] - date_arrays[pol]["Nov 18"]
        report["polarizations"][pol]["changes"]["Nov 18 to Jan 5"] = summarize_change(
            full_change
        )

    with (output_dir / "timeseries_summary.json").open("w", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
    make_overview(output_dir / "gcov_four_date_overview.png", date_arrays, changes)
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
