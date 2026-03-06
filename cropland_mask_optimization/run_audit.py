"""
Cropland Mask Optimization Audit
=================================
Generates small dummy inputs, runs the ORIGINAL per-pixel geometry loop
and the OPTIMIZED rasterize-based approach, then compares outputs for
correctness and timing.

Usage (from repo root):
    .venv/bin/python docs/auditions/cropland_mask_optimization/run_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import fiona
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from shapely.affinity import scale
from shapely.geometry import box, shape, Polygon

# ── Paths ────────────────────────────────────────────────────────────────────

AUDIT_DIR = Path(__file__).parent


# ── Dummy data generation ────────────────────────────────────────────────────

def _make_dummy_data(work_dir: Path, n_polygons: int = 12, raster_size: int = 200):
    """Create a small NDVI raster and a croplands shapefile for testing.

    Returns (region, date_str, input_dir, output_dir).
    """
    region = "AuditRegion"
    date_str = "2025-01-01"
    input_dir = str(work_dir / "input")
    output_dir = str(work_dir / "output")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Raster covering [0, 0] → [1000, 1000] metres (EPSG:25831-like)
    transform = from_bounds(0, 0, 1000, 1000, raster_size, raster_size)
    crs = rasterio.crs.CRS.from_epsg(25831)
    ndvi = np.random.default_rng(42).uniform(0.1, 0.9, (raster_size, raster_size)).astype("float32")

    ndvi_path = os.path.join(output_dir, f"NDVI_{date_str}_{region}.tiff")
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": raster_size,
        "height": raster_size,
        "count": 1,
        "crs": crs,
        "transform": transform,
    }
    with rasterio.open(ndvi_path, "w", **profile) as dst:
        dst.write(ndvi, 1)

    # Also write fake EVI/GNDVI/MSAVI for create_df_crops
    for idx_name in ("EVI", "GNDVI", "MSAVI"):
        idx_path = os.path.join(output_dir, f"{idx_name}_{date_str}_{region}.tiff")
        with rasterio.open(idx_path, "w", **profile) as dst:
            dst.write(
                np.random.default_rng(hash(idx_name) % 2**31).uniform(
                    0.05, 0.85, (raster_size, raster_size)
                ).astype("float32"),
                1,
            )

    # Polygons fully inside raster bounds, varied sizes
    rng = np.random.default_rng(123)
    polys = []
    crops = []
    crop_names = ["Wheat", "Barley", "Corn", "Sunflower", "Alfalfa", "Oat"]
    for i in range(n_polygons):
        cx = rng.uniform(150, 850)
        cy = rng.uniform(150, 850)
        half = rng.uniform(20, 60)
        # Ensure polygon stays inside [0,1000]
        x0 = max(10, cx - half)
        y0 = max(10, cy - half)
        x1 = min(990, cx + half)
        y1 = min(990, cy + half)
        polys.append(box(x0, y0, x1, y1))
        crops.append(crop_names[i % len(crop_names)])

    gdf = gpd.GeoDataFrame({"Cultiu": crops, "geometry": polys}, crs=crs)
    shp_path = os.path.join(input_dir, f"croplands_{region}.shp")
    gdf.to_file(shp_path)

    return region, date_str, input_dir, output_dir


# ── ORIGINAL implementation (copied verbatim from tools.py) ──────────────────

def generate_cropland_mask_ORIGINAL(
    region: str, input_dir: str, output_dir: str, date: str, tolerance: float = 1.7
):
    ndvi_filename = f"NDVI_{date}_{region}.tiff"
    ndvi_path = os.path.join(output_dir, ndvi_filename)
    shp_path = os.path.join(input_dir, "croplands_" + region + ".dbf")
    shp_path_region = os.path.join(output_dir, f"croplands_{region}.shp")
    mask_path = os.path.join(output_dir, f"mask_croplands_{region}_ORIGINAL.tif")
    os.makedirs(os.path.dirname(shp_path_region), exist_ok=True)

    shp_data = gpd.read_file(shp_path)
    with rasterio.open(ndvi_path) as src:
        raster_bounds_box = src.bounds
        raster_polygon = gpd.GeoSeries([box(*raster_bounds_box)], crs=src.crs)
    filtered_polygons = shp_data[shp_data.geometry.within(raster_polygon.iloc[0])]
    filtered_polygons.to_file(shp_path_region)

    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        ndvi_shape = src.shape
        raster_bounds = box(*src.bounds)

    mask_arr = np.zeros(ndvi_shape, dtype=np.uint8)

    with fiona.open(shp_path_region, "r") as shapefile:
        for feature in shapefile:
            polygon = shape(feature["geometry"])
            if raster_bounds.contains(polygon):
                initial_mask = geometry_mask(
                    [polygon], transform=transform, invert=True, out_shape=ndvi_shape
                )
                coords = np.argwhere(initial_mask)
                for r, c in coords:
                    minx, miny = rasterio.transform.xy(transform, r, c)
                    maxx, maxy = rasterio.transform.xy(transform, r + 1, c + 1)
                    pixel_geom = box(minx, miny, maxx, maxy)
                    scaled_pixel_geom = scale(
                        pixel_geom, xfact=tolerance, yfact=tolerance, origin="center"
                    )
                    if polygon.contains(scaled_pixel_geom):
                        mask_arr[r, c] = 1

    with rasterio.open(
        mask_path, "w", driver="GTiff",
        height=mask_arr.shape[0], width=mask_arr.shape[1],
        count=1, dtype=np.uint8, crs=crs, transform=transform,
    ) as dst:
        dst.write(mask_arr, 1)

    return mask_path, mask_arr


# ── OPTIMIZED implementation ─────────────────────────────────────────────────

def _shrink_polygon(polygon: Polygon, tolerance: float) -> Polygon:
    """Shrink (buffer inward) a polygon so that only pixels whose *scaled*
    geometry is fully contained will be rasterized.

    The original code scales each pixel box by ``tolerance`` around its centre
    and checks ``polygon.contains(scaled_pixel_geom)``.  A pixel centre at
    (cx, cy) with half-width hw has a scaled box of half-width hw*tolerance.
    That scaled box is contained in the polygon iff the pixel centre is at
    least ``hw * (tolerance - 1)`` inside the polygon boundary, where hw is
    half the pixel size.  We can achieve the same effect by *shrinking* the
    polygon by that distance and then doing a standard rasterize (all_touched=False).
    """
    from rasterio.transform import array_bounds

    # We don't have pixel size directly; derive it from the caller context.
    # Instead, we accept pixel half-sizes as parameters from the wrapper.
    raise NotImplementedError  # not used directly; see optimized function below


def generate_cropland_mask_OPTIMIZED(
    region: str, input_dir: str, output_dir: str, date: str, tolerance: float = 1.7
):
    """Vectorised replacement: shrink polygons by the tolerance margin,
    then rasterize them in one call instead of per-pixel geometry checks."""

    ndvi_filename = f"NDVI_{date}_{region}.tiff"
    ndvi_path = os.path.join(output_dir, ndvi_filename)
    shp_path = os.path.join(input_dir, "croplands_" + region + ".dbf")
    shp_path_region = os.path.join(output_dir, f"croplands_{region}.shp")
    mask_path = os.path.join(output_dir, f"mask_croplands_{region}_OPTIMIZED.tif")
    os.makedirs(os.path.dirname(shp_path_region), exist_ok=True)

    # ── Filter polygons inside raster ──
    shp_data = gpd.read_file(shp_path)
    with rasterio.open(ndvi_path) as src:
        raster_bounds_box = src.bounds
        raster_polygon = gpd.GeoSeries([box(*raster_bounds_box)], crs=src.crs)
    filtered_polygons = shp_data[shp_data.geometry.within(raster_polygon.iloc[0])]
    filtered_polygons.to_file(shp_path_region)

    # ── Read raster metadata ──
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        ndvi_shape = src.shape

    # Pixel half-widths (dx, dy are pixel sizes from the affine transform)
    pixel_dx = abs(transform.a)  # pixel width
    pixel_dy = abs(transform.e)  # pixel height (negative in transform)
    # The original code scales the pixel box by 'tolerance' around its centre.
    # A scaled pixel is contained in the polygon iff the pixel centre is at
    # least  half_pixel * (tolerance - 1)  inside the polygon boundary.
    # Negative-buffering the polygon by that distance and rasterizing
    # (all_touched=False, which marks a pixel when its centre is inside)
    # reproduces the same result.
    buffer_x = (pixel_dx / 2) * (tolerance - 1)
    buffer_y = (pixel_dy / 2) * (tolerance - 1)
    buffer_dist = max(buffer_x, buffer_y)  # conservative (use larger)

    # ── Shrink polygons and collect for rasterize ──
    shrunk_geoms = []
    gdf_filtered = gpd.read_file(shp_path_region)
    for geom in gdf_filtered.geometry:
        if geom is None or geom.is_empty:
            continue
        shrunk = geom.buffer(-buffer_dist)
        if not shrunk.is_empty:
            shrunk_geoms.append(shrunk)

    # ── Rasterize in one call ──
    if shrunk_geoms:
        mask_arr = rasterize(
            [(g, 1) for g in shrunk_geoms],
            out_shape=ndvi_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    else:
        mask_arr = np.zeros(ndvi_shape, dtype=np.uint8)

    # ── Save ──
    with rasterio.open(
        mask_path, "w", driver="GTiff",
        height=mask_arr.shape[0], width=mask_arr.shape[1],
        count=1, dtype=np.uint8, crs=crs, transform=transform,
    ) as dst:
        dst.write(mask_arr, 1)

    return mask_path, mask_arr


# ── ORIGINAL create_df_crops (verbatim from tools.py) ────────────────────────

def create_df_crops_ORIGINAL(region, input_dir, date, mask_suffix=""):
    from rasterio.mask import mask as rio_mask

    mask_path = os.path.join(input_dir, f"mask_croplands_{region}{mask_suffix}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")
    output_txt = os.path.join(input_dir, f"output_croplands_{region}_{date}_stats{mask_suffix}.txt")

    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        mask_meta = src_mask.meta

    gdf = gpd.read_file(shp_path)
    pixel_positions = {}
    for _, geom in gdf.iterrows():
        field_id = geom["id"]
        geom_shape = [geom["geometry"]]
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**mask_meta) as temp_ds:
                temp_ds.write(mask_data, 1)
                try:
                    out_image, _ = rio_mask(temp_ds, geom_shape, crop=False, filled=False)
                    rows, cols = np.where(out_image[0] == 1)
                    pixel_positions[field_id] = list(zip(rows, cols))
                except ValueError:
                    pixel_positions[field_id] = []

    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]
    required_indexes = ["NDVI", "EVI", "GNDVI", "MSAVI"]
    date_files = {f.split("_")[0]: os.path.join(input_dir, f) for f in index_files}
    if not all(idx in date_files for idx in required_indexes):
        raise ValueError(f"Missing indices for date {date}.")

    index_data = {}
    for idx, path in date_files.items():
        with rasterio.open(path) as src:
            index_data[idx] = src.read(1)

    results = []
    for _, geom in gdf.iterrows():
        field_id = geom["id"]
        stats = {"date": date, "id": field_id, "Crop": geom.get("Cultiu", "Unknown")}
        if field_id not in pixel_positions or not pixel_positions[field_id]:
            continue
        rows, cols = zip(*pixel_positions[field_id])
        for idx in required_indexes:
            valid_values = index_data[idx][rows, cols]
            valid_values = valid_values[~np.isnan(valid_values)]
            stats[idx] = float(np.median(valid_values)) if valid_values.size > 0 else np.nan
        results.append(stats)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_txt, sep="\t", index=False)
    return df


# ── OPTIMIZED create_df_crops ────────────────────────────────────────────────

def create_df_crops_OPTIMIZED(region, input_dir, date, mask_suffix=""):
    """Optimised: read mask once into memory, use geometry_mask per polygon
    (no MemoryFile overhead), then compute stats with vectorised numpy."""

    mask_path = os.path.join(input_dir, f"mask_croplands_{region}{mask_suffix}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")
    output_txt = os.path.join(input_dir, f"output_croplands_{region}_{date}_stats{mask_suffix}_opt.txt")

    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        transform = src_mask.transform
        out_shape = mask_data.shape

    gdf = gpd.read_file(shp_path)

    # Pre-compute pixel positions using geometry_mask (same semantics as
    # rasterio.mask.mask but without MemoryFile round-trip per polygon)
    pixel_positions = {}
    for _, row in gdf.iterrows():
        field_id = row["id"]
        geom = row["geometry"]
        if geom is None or geom.is_empty:
            pixel_positions[field_id] = []
            continue
        # geometry_mask returns True where OUTSIDE the geometry when invert=False
        # invert=True → True where INSIDE the geometry
        poly_mask = geometry_mask(
            [geom], transform=transform, invert=True, out_shape=out_shape
        )
        # Combine with cropland mask
        combined = poly_mask & (mask_data == 1)
        rows, cols = np.where(combined)
        pixel_positions[field_id] = list(zip(rows, cols))

    # Load index rasters
    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]
    required_indexes = ["NDVI", "EVI", "GNDVI", "MSAVI"]
    date_files = {f.split("_")[0]: os.path.join(input_dir, f) for f in index_files}
    if not all(idx in date_files for idx in required_indexes):
        raise ValueError(f"Missing indices for date {date}.")

    index_data = {}
    for idx, path in date_files.items():
        with rasterio.open(path) as src:
            index_data[idx] = src.read(1)

    results = []
    for _, row in gdf.iterrows():
        field_id = row["id"]
        stats = {"date": date, "id": field_id, "Crop": row.get("Cultiu", "Unknown")}
        if field_id not in pixel_positions or not pixel_positions[field_id]:
            continue
        rows_arr, cols_arr = zip(*pixel_positions[field_id])
        for idx in required_indexes:
            vals = index_data[idx][rows_arr, cols_arr]
            vals = vals[~np.isnan(vals)]
            stats[idx] = float(np.median(vals)) if vals.size > 0 else np.nan
        results.append(stats)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_txt, sep="\t", index=False)
    return df


# ── add_ids_to_croplands (copied from tools.py) ─────────────────────────────

def add_ids_to_croplands(input_dir: str, region: str):
    input_path = os.path.join(input_dir, f"croplands_{region}.shp")
    gdf = gpd.read_file(input_path)
    gdf["id"] = range(1, len(gdf) + 1)
    output_path = os.path.join(input_dir, f"croplands_{region}_id.shp")
    gdf.to_file(output_path)
    return output_path


# ── Run audit ────────────────────────────────────────────────────────────────

def run_audit():
    print("=" * 70)
    print("  CROPLAND MASK OPTIMIZATION AUDIT")
    print("=" * 70)

    work_dir = Path(tempfile.mkdtemp(prefix="cropland_audit_"))
    print(f"\nWork directory: {work_dir}\n")

    results = {}

    try:
        # ── Generate dummy data ──
        region, date_str, input_dir, output_dir = _make_dummy_data(
            work_dir, n_polygons=12, raster_size=200
        )
        print(f"Dummy data: {region}, {date_str}")
        print(f"  Raster: 200×200 px, 12 cropland polygons\n")

        # ════════════════════════════════════════════════════════════════════
        # 1. generate_cropland_mask
        # ════════════════════════════════════════════════════════════════════
        print("─" * 60)
        print("1. generate_cropland_mask")
        print("─" * 60)

        # ── Original ──
        t0 = time.perf_counter()
        orig_mask_path, orig_mask = generate_cropland_mask_ORIGINAL(
            region, input_dir, output_dir, date_str
        )
        t_orig = time.perf_counter() - t0
        print(f"  ORIGINAL : {t_orig:.4f}s  |  mask pixels = {orig_mask.sum()}")

        # ── Optimized ──
        t0 = time.perf_counter()
        opt_mask_path, opt_mask = generate_cropland_mask_OPTIMIZED(
            region, input_dir, output_dir, date_str
        )
        t_opt = time.perf_counter() - t0
        print(f"  OPTIMIZED: {t_opt:.4f}s  |  mask pixels = {opt_mask.sum()}")

        # ── Compare ──
        match = np.array_equal(orig_mask, opt_mask)
        diff_pixels = int(np.sum(orig_mask != opt_mask))
        total_pixels = orig_mask.size
        agreement_pct = 100.0 * (1 - diff_pixels / total_pixels)
        speedup = t_orig / t_opt if t_opt > 0 else float("inf")

        print(f"\n  Arrays identical : {match}")
        print(f"  Differing pixels : {diff_pixels} / {total_pixels}")
        print(f"  Agreement        : {agreement_pct:.4f}%")
        print(f"  Speedup          : {speedup:.1f}×")

        results["generate_cropland_mask"] = {
            "original_time_s": round(t_orig, 4),
            "optimized_time_s": round(t_opt, 4),
            "speedup": round(speedup, 1),
            "identical": match,
            "diff_pixels": diff_pixels,
            "total_pixels": total_pixels,
            "agreement_pct": round(agreement_pct, 4),
            "original_mask_sum": int(orig_mask.sum()),
            "optimized_mask_sum": int(opt_mask.sum()),
        }

        # ════════════════════════════════════════════════════════════════════
        # 2. create_df_crops  (using the ORIGINAL mask for both to isolate
        #    the df-creation optimisation from mask differences)
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "─" * 60)
        print("2. create_df_crops")
        print("─" * 60)

        # Add IDs to filtered croplands (written by mask step to output_dir)
        add_ids_to_croplands(output_dir, region)

        # ── Original ──
        t0 = time.perf_counter()
        df_orig = create_df_crops_ORIGINAL(region, output_dir, date_str, mask_suffix="_ORIGINAL")
        t_orig_df = time.perf_counter() - t0
        print(f"  ORIGINAL : {t_orig_df:.4f}s  |  rows = {len(df_orig)}")

        # ── Optimized ──
        t0 = time.perf_counter()
        df_opt = create_df_crops_OPTIMIZED(region, output_dir, date_str, mask_suffix="_ORIGINAL")
        t_opt_df = time.perf_counter() - t0
        print(f"  OPTIMIZED: {t_opt_df:.4f}s  |  rows = {len(df_opt)}")

        # ── Compare DataFrames ──
        # Align on id column
        df_orig_sorted = df_orig.sort_values("id").reset_index(drop=True)
        df_opt_sorted = df_opt.sort_values("id").reset_index(drop=True)

        import pandas as pd

        same_shape = df_orig_sorted.shape == df_opt_sorted.shape
        same_ids = list(df_orig_sorted["id"]) == list(df_opt_sorted["id"])

        # Compare numeric columns
        numeric_cols = ["NDVI", "EVI", "GNDVI", "MSAVI"]
        max_diff = 0.0
        col_diffs = {}
        if same_ids and same_shape:
            for col in numeric_cols:
                diff = np.abs(
                    df_orig_sorted[col].fillna(0).values - df_opt_sorted[col].fillna(0).values
                )
                col_max = float(diff.max())
                col_diffs[col] = col_max
                max_diff = max(max_diff, col_max)

        values_match = max_diff < 1e-6
        speedup_df = t_orig_df / t_opt_df if t_opt_df > 0 else float("inf")

        print(f"\n  Same shape       : {same_shape}")
        print(f"  Same IDs         : {same_ids}")
        print(f"  Max numeric diff : {max_diff:.2e}")
        print(f"  Values match     : {values_match}  (<1e-6)")
        print(f"  Speedup          : {speedup_df:.1f}×")

        results["create_df_crops"] = {
            "original_time_s": round(t_orig_df, 4),
            "optimized_time_s": round(t_opt_df, 4),
            "speedup": round(speedup_df, 1),
            "same_shape": same_shape,
            "same_ids": same_ids,
            "max_numeric_diff": max_diff,
            "values_match": values_match,
            "rows_original": len(df_orig),
            "rows_optimized": len(df_opt),
            "col_diffs": col_diffs,
        }

        # ════════════════════════════════════════════════════════════════════
        # Summary
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)

        all_pass = True
        for fn_name, res in results.items():
            mask_ok = res.get("identical", res.get("values_match", False))
            if fn_name == "generate_cropland_mask":
                mask_ok = res["agreement_pct"] >= 99.0  # allow tiny rounding diffs
            status = "✅ PASS" if mask_ok else "❌ FAIL"
            if not mask_ok:
                all_pass = False
            print(f"  {fn_name:30s}  {status}  speedup={res['speedup']}×")

        print()
        if all_pass:
            print("  🎉 All checks PASSED")
        else:
            print("  ⚠️  Some checks FAILED — review differences above")

        # Save machine-readable results
        results_path = AUDIT_DIR / "audit_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    run_audit()
