#!/usr/bin/env python3
"""Create animated GIFs for the four-date NISAR GCOV AOI series."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_nisar_gcov_timeseries import (
    DATES,
    FLOOR,
    POLARIZATIONS,
    ROOT,
    aoi_window,
    get_grid,
)


OUTPUT_DIR = Path("gcov_timeseries_results")
FRAME_WIDTH = 720
HEADER_HEIGHT = 55
FOOTER_HEIGHT = 42


def to_db(power: np.ndarray) -> np.ndarray:
    result = np.full(power.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(power) & (power > FLOOR)
    result[valid] = 10.0 * np.log10(power[valid])
    return result


def load_aoi_arrays() -> dict[str, dict[str, np.ndarray]]:
    grid = get_grid(DATES[0][1])
    row_slice, col_slice, mask = aoi_window(grid)
    stride = max(1, int(np.ceil((col_slice.stop - col_slice.start) / FRAME_WIDTH)))
    sampled_mask = mask[::stride, ::stride]
    arrays = {pol: {} for pol in POLARIZATIONS}
    for pol, dataset_name in POLARIZATIONS.items():
        for date_label, path in DATES:
            with h5py.File(path, "r") as source:
                power = np.asarray(
                    source[ROOT][dataset_name][
                        row_slice.start : row_slice.stop : stride,
                        col_slice.start : col_slice.stop : stride,
                    ],
                    dtype=np.float32,
                )
            db = to_db(power)
            db[~sampled_mask] = np.nan
            arrays[pol][date_label] = db
    return arrays


def grayscale(array: np.ndarray, low: float, high: float) -> Image.Image:
    scaled = np.nan_to_num(np.clip((array - low) / (high - low), 0, 1))
    pixels = (scaled * 255).astype(np.uint8)
    rgb = np.repeat(pixels[..., None], 3, axis=2)
    rgb[~np.isfinite(array)] = 0
    return Image.fromarray(rgb, "RGB")


def diverging(array: np.ndarray, limit: float) -> Image.Image:
    valid = np.isfinite(array)
    normalized = np.nan_to_num(np.clip(array / limit, -1, 1))
    magnitude = np.abs(normalized)
    base = 240 * (1 - magnitude)
    rgb = np.empty((*array.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(base + 235 * np.clip(normalized, 0, 1), 0, 255)
    rgb[..., 1] = np.clip(base + 70 * (1 - magnitude), 0, 255)
    rgb[..., 2] = np.clip(base + 235 * np.clip(-normalized, 0, 1), 0, 255)
    rgb[~valid] = 0
    return Image.fromarray(rgb, "RGB")


def add_text_panel(
    image: Image.Image,
    title: str,
    subtitle: str,
    footer: str,
) -> Image.Image:
    font = ImageFont.load_default()
    canvas = Image.new(
        "RGB",
        (image.width, image.height + HEADER_HEIGHT + FOOTER_HEIGHT),
        "white",
    )
    canvas.paste(image, (0, HEADER_HEIGHT))
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill=(0, 0, 0), font=font)
    draw.text((12, 30), subtitle, fill=(45, 45, 45), font=font)
    draw.text(
        (12, HEADER_HEIGHT + image.height + 12),
        footer,
        fill=(25, 25, 25),
        font=font,
    )
    return canvas


def combine(left: Image.Image, right: Image.Image) -> Image.Image:
    gap = 8
    canvas = Image.new(
        "RGB",
        (left.width + right.width + gap, max(left.height, right.height)),
        "white",
    )
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width + gap, 0))
    return canvas


def save_gif(path: Path, frames: list[Image.Image], duration: int) -> None:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True,
        disposal=2,
    )


def make_backscatter_gif(arrays: dict[str, dict[str, np.ndarray]]) -> Path:
    limits = {}
    for pol in POLARIZATIONS:
        values = np.concatenate(
            [array[np.isfinite(array)] for array in arrays[pol].values()]
        )
        limits[pol] = tuple(np.percentile(values, (2, 98)))

    frames = []
    for date_label, _ in DATES:
        panels = []
        for pol in ("HH", "HV"):
            low, high = limits[pol]
            image = grayscale(arrays[pol][date_label], low, high)
            panels.append(
                add_text_panel(
                    image,
                    f"{pol} backscatter: {date_label}",
                    f"Fixed display range: {low:.1f} to {high:.1f} dB",
                    "Brighter = stronger radar return; black = no valid data",
                )
            )
        frames.append(combine(*panels))

    output = OUTPUT_DIR / "gcov_hh_hv_timeseries.gif"
    save_gif(output, frames, 1100)
    return output


def make_delta_gif(arrays: dict[str, dict[str, np.ndarray]]) -> Path:
    intervals = []
    all_changes = []
    for (earlier, _), (later, _) in zip(DATES[:-1], DATES[1:]):
        changes = {
            pol: arrays[pol][later] - arrays[pol][earlier]
            for pol in POLARIZATIONS
        }
        intervals.append((earlier, later, changes))
        all_changes.extend(
            change[np.isfinite(change)] for change in changes.values()
        )
    limit = max(1.0, float(np.percentile(np.abs(np.concatenate(all_changes)), 98)))

    frames = []
    for earlier, later, changes in intervals:
        panels = []
        for pol in ("HH", "HV"):
            image = diverging(changes[pol], limit)
            panels.append(
                add_text_panel(
                    image,
                    f"{pol} delta: {earlier} to {later}",
                    f"Fixed symmetric scale: -{limit:.1f} to +{limit:.1f} dB",
                    "Red = increase; blue = decrease; pale = little change",
                )
            )
        frames.append(combine(*panels))

    output = OUTPUT_DIR / "gcov_hh_hv_deltas.gif"
    save_gif(output, frames, 1300)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    arrays = load_aoi_arrays()
    print(make_backscatter_gif(arrays))
    print(make_delta_gif(arrays))


if __name__ == "__main__":
    main()
