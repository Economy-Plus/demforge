#!/usr/bin/env python3
"""Build paired DEM training tiles from GeoTIFF rasters."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from demforge.terrain_ops import make_model_sample, robust_normalize


def valid_tile(tile: np.ndarray, nodata: float | None, min_valid_fraction: float) -> bool:
    """Check whether a raster tile contains enough valid data.

    Args:
        tile: Raster tile as a 2D array.
        nodata: Raster nodata value, if present.
        min_valid_fraction: Minimum fraction of finite/non-nodata pixels required.

    Returns:
        True if the tile is useful for training, otherwise False.
    """

    if nodata is None:
        mask = np.isfinite(tile)
    else:
        mask = np.isfinite(tile) & (tile != nodata)

    if float(mask.mean()) < min_valid_fraction:
        return False

    clean = tile[mask]
    if clean.size == 0:
        return False

    relief = float(np.nanpercentile(clean, 98) - np.nanpercentile(clean, 2))
    return relief > 2.0


def fill_invalid(tile: np.ndarray, nodata: float | None) -> np.ndarray:
    """Fill invalid cells with the tile median.

    Args:
        tile: Raster tile as a 2D array.
        nodata: Raster nodata value, if present.

    Returns:
        Float32 tile with invalid values replaced.
    """

    arr = tile.astype(np.float32)
    if nodata is None:
        mask = np.isfinite(arr)
    else:
        mask = np.isfinite(arr) & (arr != nodata)

    median = float(np.nanmedian(arr[mask])) if mask.any() else 0.0
    arr[~mask] = median
    return arr


def split_for_file(path: Path, val_sources: set[str]) -> str:
    """Assign split at source-file granularity.

    Args:
        path: Source raster path.
        val_sources: Source stems assigned to validation.

    Returns:
        "val" if the source file belongs to validation, otherwise "train".
    """

    if path.stem in val_sources:
        return "val"
    return "train"


def tile_output_path(out: Path, split: str, raster_path: Path, x0: int, y0: int) -> Path:
    """Build the output path for one training tile.

    Args:
        out: Output dataset root.
        split: Dataset split, usually "train" or "val".
        raster_path: Source raster path.
        x0: Tile window x offset.
        y0: Tile window y offset.

    Returns:
        Destination `.npz` path.
    """

    name = f"{raster_path.stem}_x{x0:06d}_y{y0:06d}.npz"
    return out / split / name


def is_existing_nonempty(path: Path) -> bool:
    """Check whether a destination file already exists and is non-empty.

    Args:
        path: Destination file path.

    Returns:
        True if the file exists and has non-zero size.
    """

    return path.exists() and path.is_file() and path.stat().st_size > 0


def save_tile_atomic(
    destination: Path,
    x: np.ndarray,
    y: np.ndarray,
    target: np.ndarray,
    coarse: np.ndarray,
    meta: dict,
) -> None:
    """Save a tile atomically to avoid half-written destination files.

    Args:
        destination: Final `.npz` destination path.
        x: Model input channels.
        y: Residual target.
        target: Normalized high-res target height.
        coarse: Normalized coarse/upscaled height.
        meta: JSON-serializable metadata.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_name(destination.name + ".tmp")

    if tmp_path.exists():
        tmp_path.unlink()

    np.savez_compressed(
        tmp_path,
        x=x,
        y=y,
        target=target,
        coarse=coarse,
        meta=json.dumps(meta),
    )

    tmp_npz_path = tmp_path.with_suffix(tmp_path.suffix + ".npz")
    if tmp_npz_path.exists():
        tmp_npz_path.replace(destination)
    else:
        tmp_path.replace(destination)


def build_positions(width: int, height: int, tile_size: int, stride: int) -> list[tuple[int, int]]:
    """Build top-left raster positions for fixed-size tiles.

    Args:
        width: Raster width in pixels.
        height: Raster height in pixels.
        tile_size: Square tile size in pixels.
        stride: Tile stride in pixels.

    Returns:
        List of `(x0, y0)` window origins.
    """

    return [
        (x, y)
        for y in range(0, max(1, height - tile_size + 1), stride)
        for x in range(0, max(1, width - tile_size + 1), stride)
    ]


def build_tiles(
    src: Path,
    out: Path,
    tile_size: int,
    stride: int,
    downscale: int,
    val_fraction: float,
    min_valid_fraction: float,
    overwrite: bool,
) -> None:
    """Build DEMForge training tiles.

    Args:
        src: Source directory containing GeoTIFF rasters.
        out: Output directory for train/val `.npz` tiles.
        tile_size: Square tile size in pixels.
        stride: Tile stride in pixels.
        downscale: Downscale factor used to create coarse inputs.
        val_fraction: Fraction of source rasters assigned to validation.
        min_valid_fraction: Minimum valid pixel fraction per tile.
        overwrite: Whether to rebuild existing non-empty destination files.
    """

    try:
        import rasterio
        from rasterio.windows import Window
    except Exception as exc:
        raise SystemExit("rasterio is required for real DEM tile building. Install requirements.txt.") from exc

    files = sorted([*src.rglob("*.tif"), *src.rglob("*.tiff")])
    if not files:
        raise SystemExit(f"No GeoTIFF files found under {src}")

    random.seed(1337)
    stems = [path.stem for path in files]
    random.shuffle(stems)
    val_count = max(1, int(len(stems) * val_fraction))
    val_sources = set(stems[:val_count])

    for split in ("train", "val"):
        (out / split).mkdir(parents=True, exist_ok=True)

    stats = {
        "written": {"train": 0, "val": 0},
        "skipped_existing": {"train": 0, "val": 0},
        "skipped_invalid": {"train": 0, "val": 0},
        "skipped_bounds": {"train": 0, "val": 0},
    }

    for raster_path in tqdm(files, desc="rasters"):
        split = split_for_file(raster_path, val_sources)

        with rasterio.open(raster_path) as ds:
            nodata = ds.nodata
            width = ds.width
            height = ds.height
            positions = build_positions(width, height, tile_size, stride)

            progress = tqdm(positions, desc=raster_path.name, leave=False)
            for x0, y0 in progress:
                destination = tile_output_path(out, split, raster_path, x0, y0)

                if is_existing_nonempty(destination) and not overwrite:
                    stats["skipped_existing"][split] += 1
                    continue

                if x0 + tile_size > width or y0 + tile_size > height:
                    stats["skipped_bounds"][split] += 1
                    continue

                window = Window(x0, y0, tile_size, tile_size)
                tile = ds.read(1, window=window).astype(np.float32)

                if not valid_tile(tile, nodata, min_valid_fraction):
                    stats["skipped_invalid"][split] += 1
                    continue

                tile = fill_invalid(tile, nodata)
                normalized, norm_meta = robust_normalize(tile)
                x, y, target, coarse = make_model_sample(normalized, downscale=downscale)

                meta = {
                    "source_file": str(raster_path),
                    "source_stem": raster_path.stem,
                    "split": split,
                    "x0": x0,
                    "y0": y0,
                    "tile_size": tile_size,
                    "stride": stride,
                    "downscale": downscale,
                    "crs": str(ds.crs),
                    "transform": list(ds.window_transform(window))[:6],
                    "normalization": norm_meta,
                }

                save_tile_atomic(destination, x=x, y=y, target=target, coarse=coarse, meta=meta)
                stats["written"][split] += 1

                progress.set_postfix(
                    written=stats["written"][split],
                    existing=stats["skipped_existing"][split],
                    invalid=stats["skipped_invalid"][split],
                )

    print(json.dumps(stats, indent=2))


def main() -> int:
    """Run the tile builder CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw")
    parser.add_argument("--out", default="data/tiles")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    parser.add_argument("--downscale", type=int, default=8)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.98)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rebuild tiles even when the destination .npz already exists and is non-empty.",
    )
    args = parser.parse_args()

    build_tiles(
        src=Path(args.src),
        out=Path(args.out),
        tile_size=args.tile_size,
        stride=args.stride,
        downscale=args.downscale,
        val_fraction=args.val_fraction,
        min_valid_fraction=args.min_valid_fraction,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
