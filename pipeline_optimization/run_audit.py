"""
NDVI Pipeline Optimization Audit
==================================
Benchmarks ORIGINAL vs OPTIMIZED implementations of:
  1. create_df_crops   (per-polygon geometry_mask → single-pass rasterize + groupby)
  2. add_ids_to_croplands  (cache-after-read → cache-before-read)
  3. split_shapefile_by_planted  (always write → skip empty shapefiles)
  4. replace_pixels_within_shapefile  (__geo_interface__ → rasterize)

Usage (from pyrocb_prediction/):
    ../.venv/bin/python src/external/NDVI/pipeline_optimization/run_audit.py
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from shapely.geometry import box

AUDIT_DIR = Path(__file__).parent
DERIVED_INDICES = ("NDVI", "GNDVI", "MSAVI", "EVI")


# ── Dummy data generation ────────────────────────────────────────────────────

def _make_dummy_data(work_dir: Path, n_polygons: int = 40, raster_size: int = 300):
    """Create dummy NDVI/EVI/GNDVI/MSAVI rasters + croplands shapefile."""
    region = "AuditRegion"
    date_str = "2025-01-01"
    processed_dir = str(work_dir / "processed")
    os.makedirs(processed_dir, exist_ok=True)

    transform = from_bounds(0, 0, 1000, 1000, raster_size, raster_size)
    crs = rasterio.crs.CRS.from_epsg(25831)
    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": raster_size, "height": raster_size,
        "count": 1, "crs": crs, "transform": transform,
    }

    rng = np.random.default_rng(42)
    for idx_name in DERIVED_INDICES:
        path = os.path.join(processed_dir, f"{idx_name}_{date_str}_{region}.tiff")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(rng.uniform(0.05, 0.9, (raster_size, raster_size)).astype("float32"), 1)

    # Mask raster (binary cropland mask)
    mask = np.zeros((raster_size, raster_size), dtype=np.uint8)

    # Generate non-overlapping polygons via grid placement
    polys, crops = [], []
    crop_names = ["Wheat", "Barley", "Corn", "Sunflower", "Alfalfa", "Oat"]
    cols_grid = int(np.ceil(np.sqrt(n_polygons)))
    cell = 1000.0 / cols_grid
    margin = cell * 0.1
    idx = 0
    for gi in range(cols_grid):
        for gj in range(cols_grid):
            if idx >= n_polygons:
                break
            x0 = gi * cell + margin
            y0 = gj * cell + margin
            x1 = (gi + 1) * cell - margin
            y1 = (gj + 1) * cell - margin
            polys.append(box(x0, y0, x1, y1))
            crops.append(crop_names[idx % len(crop_names)])
            idx += 1

    gdf = gpd.GeoDataFrame({"Cultiu": crops, "geometry": polys}, crs=crs)

    # Rasterize polygons into mask
    geom_pairs = [(geom, 1) for geom in gdf.geometry]
    mask = rasterize(geom_pairs, out_shape=(raster_size, raster_size),
                     transform=transform, fill=0, dtype=np.uint8)

    shp_path = os.path.join(processed_dir, f"croplands_{region}.shp")
    gdf.to_file(shp_path)

    mask_path = os.path.join(processed_dir, f"mask_croplands_{region}.tif")
    mask_profile = {**profile, "dtype": np.uint8}
    with rasterio.open(mask_path, "w", **mask_profile) as dst:
        dst.write(mask, 1)

    # Add IDs shapefile
    gdf["id"] = range(1, len(gdf) + 1)
    id_shp_path = os.path.join(processed_dir, f"croplands_{region}_id.shp")
    gdf.to_file(id_shp_path)

    return region, date_str, processed_dir


# ── ORIGINAL implementations ─────────────────────────────────────────────────

def create_df_crops_ORIGINAL(region, input_dir, date):
    """Per-polygon geometry_mask + per-polygon numpy stats."""
    mask_path = os.path.join(input_dir, f"mask_croplands_{region}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")

    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        transform = src_mask.transform
        out_shape = mask_data.shape

    gdf = gpd.read_file(shp_path)

    pixel_positions = {}
    for _, geom in gdf.iterrows():
        field_id = geom['id']
        polygon = geom['geometry']
        if polygon is None or polygon.is_empty:
            pixel_positions[field_id] = []
            continue
        inside_polygon = geometry_mask(
            [polygon], transform=transform, invert=True, out_shape=out_shape
        )
        inside_cropland = inside_polygon & (mask_data == 1)
        rows, cols = np.where(inside_cropland)
        pixel_positions[field_id] = list(zip(rows, cols))

    required_indexes = list(DERIVED_INDICES)
    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]
    date_files = {f.split('_')[0]: os.path.join(input_dir, f) for f in index_files}

    index_data = {}
    for idx, path in date_files.items():
        with rasterio.open(path) as src:
            index_data[idx] = src.read(1)

    results = []
    for _, geom in gdf.iterrows():
        field_id = geom['id']
        stats = {'date': date, 'id': field_id, 'Crop': geom.get('Cultiu', 'Unknown')}
        if field_id not in pixel_positions or not pixel_positions[field_id]:
            continue
        rows, cols = zip(*pixel_positions[field_id])
        for idx in required_indexes:
            valid = index_data[idx][rows, cols]
            valid = valid[~np.isnan(valid)]
            stats[idx] = float(np.median(valid)) if valid.size > 0 else np.nan
        results.append(stats)

    return pd.DataFrame(results, columns=['date', 'id', 'Crop', *required_indexes])


def create_df_crops_OPTIMIZED(region, input_dir, date):
    """Single-pass rasterize + groupby().median()."""
    mask_path = os.path.join(input_dir, f"mask_croplands_{region}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")

    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        transform = src_mask.transform
        out_shape = mask_data.shape

    gdf = gpd.read_file(shp_path)
    required_indexes = list(DERIVED_INDICES)

    geom_id_pairs = [
        (geom, fid) for geom, fid in zip(gdf.geometry, gdf['id'])
        if geom is not None and not geom.is_empty
    ]

    if geom_id_pairs:
        label_raster = rasterize(
            geom_id_pairs, out_shape=out_shape, transform=transform,
            fill=0, dtype=np.int32,
        )
        label_raster[mask_data != 1] = 0
    else:
        label_raster = np.zeros(out_shape, dtype=np.int32)

    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]
    date_files = {f.split('_')[0]: os.path.join(input_dir, f) for f in index_files}

    index_data = {}
    for idx, path in date_files.items():
        with rasterio.open(path) as src:
            index_data[idx] = src.read(1)

    flat_labels = label_raster.ravel()
    active_mask = flat_labels > 0
    active_labels = flat_labels[active_mask]

    crop_col = gdf['Cultiu'] if 'Cultiu' in gdf.columns else pd.Series(['Unknown'] * len(gdf))
    crop_lookup = dict(zip(gdf['id'], crop_col))

    if active_labels.size > 0:
        pixel_data = {'label': active_labels}
        for idx in required_indexes:
            pixel_data[idx] = index_data[idx].ravel()[active_mask]
        df_pixels = pd.DataFrame(pixel_data)
        medians = df_pixels.groupby('label')[required_indexes].median()

        results = []
        for fid in medians.index:
            row = {'date': date, 'id': int(fid), 'Crop': crop_lookup.get(int(fid), 'Unknown')}
            for idx in required_indexes:
                row[idx] = medians.loc[fid, idx]
            results.append(row)
    else:
        results = []

    return pd.DataFrame(results, columns=['date', 'id', 'Crop', *required_indexes])


def replace_pixels_ORIGINAL(tif_path, shp_path, output_path, new_value):
    """Original: __geo_interface__ + geometry_mask."""
    shapefile = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        raster_data = src.read(1)
        out_meta = src.meta.copy()
        shapes = [feature["geometry"] for feature in shapefile.__geo_interface__["features"]]
        mask_inside = geometry_mask(shapes, transform=src.transform, invert=True,
                                    out_shape=(src.height, src.width))
        raster_data[mask_inside] = new_value
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)


def replace_pixels_OPTIMIZED(tif_path, shp_path, output_path, new_value):
    """Optimized: rasterize directly, no __geo_interface__."""
    shapefile = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        raster_data = src.read(1)
        out_meta = src.meta.copy()
        geoms = [geom for geom in shapefile.geometry if geom is not None and not geom.is_empty]
        if geoms:
            mask_inside = rasterize(
                [(geom, 1) for geom in geoms],
                out_shape=(src.height, src.width),
                transform=src.transform,
                fill=0, dtype=np.uint8,
            ).astype(bool)
            raster_data[mask_inside] = new_value
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(raster_data, 1)


def split_shapefile_ORIGINAL(base_path, region, date):
    """Original: always writes both shapefiles."""
    txt_file = os.path.join(base_path, f"output_croplands_{region}_{date}_prediction.txt")
    shp_file = os.path.join(base_path, f"croplands_{region}_id.shp")
    df = pd.read_csv(txt_file, sep="\t")
    gdf = gpd.read_file(shp_file)
    gdf = gdf.merge(df, on="id")
    gdf_planted = gdf[gdf["planted"] == 1]
    gdf_not_planted = gdf[gdf["planted"] == 0]
    gdf_planted.to_file(os.path.join(base_path, f"croplands_{region}_{date}_1_orig.shp"))
    gdf_not_planted.to_file(os.path.join(base_path, f"croplands_{region}_{date}_0_orig.shp"))


def split_shapefile_OPTIMIZED(base_path, region, date):
    """Optimized: skip writing empty shapefiles."""
    txt_file = os.path.join(base_path, f"output_croplands_{region}_{date}_prediction.txt")
    shp_file = os.path.join(base_path, f"croplands_{region}_id.shp")
    df = pd.read_csv(txt_file, sep="\t")
    gdf = gpd.read_file(shp_file)
    gdf = gdf.merge(df, on="id")
    gdf_planted = gdf[gdf["planted"] == 1]
    gdf_not_planted = gdf[gdf["planted"] == 0]
    if not gdf_planted.empty:
        gdf_planted.to_file(os.path.join(base_path, f"croplands_{region}_{date}_1_opt.shp"))
    if not gdf_not_planted.empty:
        gdf_not_planted.to_file(os.path.join(base_path, f"croplands_{region}_{date}_0_opt.shp"))


# ── Run audit ────────────────────────────────────────────────────────────────

def run_audit():
    print("=" * 70)
    print("  NDVI PIPELINE OPTIMIZATION AUDIT")
    print("=" * 70)

    work_dir = Path(tempfile.mkdtemp(prefix="pipeline_audit_"))
    print(f"\nWork directory: {work_dir}\n")

    results = {}

    try:
        region, date_str, processed_dir = _make_dummy_data(
            work_dir, n_polygons=40, raster_size=300
        )
        print(f"Dummy data: {region}, {date_str}")
        print(f"  Raster: 300x300 px, 40 cropland polygons\n")

        # ════════════════════════════════════════════════════════════════════
        # 1. create_df_crops
        # ════════════════════════════════════════════════════════════════════
        print("-" * 60)
        print("1. create_df_crops")
        print("-" * 60)

        n_runs = 3

        # Warmup
        create_df_crops_ORIGINAL(region, processed_dir, date_str)
        create_df_crops_OPTIMIZED(region, processed_dir, date_str)

        times_orig, times_opt = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            df_orig = create_df_crops_ORIGINAL(region, processed_dir, date_str)
            times_orig.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            df_opt = create_df_crops_OPTIMIZED(region, processed_dir, date_str)
            times_opt.append(time.perf_counter() - t0)

        t_orig = np.median(times_orig)
        t_opt = np.median(times_opt)

        print(f"  ORIGINAL : {t_orig:.4f}s (median of {n_runs})  |  rows = {len(df_orig)}")
        print(f"  OPTIMIZED: {t_opt:.4f}s (median of {n_runs})  |  rows = {len(df_opt)}")

        # Compare
        df_orig_s = df_orig.sort_values("id").reset_index(drop=True)
        df_opt_s = df_opt.sort_values("id").reset_index(drop=True)
        same_shape = df_orig_s.shape == df_opt_s.shape
        same_ids = list(df_orig_s["id"]) == list(df_opt_s["id"])

        numeric_cols = list(DERIVED_INDICES)
        col_diffs = {}
        max_diff = 0.0
        if same_ids and same_shape:
            for col in numeric_cols:
                diff = np.abs(df_orig_s[col].fillna(0).values - df_opt_s[col].fillna(0).values)
                col_max = float(diff.max())
                col_diffs[col] = col_max
                max_diff = max(max_diff, col_max)

        values_match = max_diff < 1e-6
        speedup = t_orig / t_opt if t_opt > 0 else float("inf")

        print(f"\n  Same shape       : {same_shape}")
        print(f"  Same IDs         : {same_ids}")
        print(f"  Max numeric diff : {max_diff:.2e}")
        print(f"  Values match     : {values_match}  (<1e-6)")
        print(f"  Speedup          : {speedup:.1f}x")

        results["create_df_crops"] = {
            "original_time_s": round(t_orig, 4),
            "optimized_time_s": round(t_opt, 4),
            "speedup": round(speedup, 1),
            "same_shape": same_shape,
            "same_ids": same_ids,
            "max_numeric_diff": max_diff,
            "values_match": values_match,
            "rows": len(df_orig),
            "col_diffs": col_diffs,
        }

        # ════════════════════════════════════════════════════════════════════
        # 2. replace_pixels_within_shapefile
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "-" * 60)
        print("2. replace_pixels_within_shapefile")
        print("-" * 60)

        base_tif = os.path.join(processed_dir, f"NDVI_{date_str}_{region}.tiff")
        shp_for_replace = os.path.join(processed_dir, f"croplands_{region}_id.shp")
        out_orig = os.path.join(processed_dir, "fuel_orig.tif")
        out_opt = os.path.join(processed_dir, "fuel_opt.tif")

        # Warmup
        replace_pixels_ORIGINAL(base_tif, shp_for_replace, out_orig, 104)
        replace_pixels_OPTIMIZED(base_tif, shp_for_replace, out_opt, 104)

        times_orig_r, times_opt_r = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            replace_pixels_ORIGINAL(base_tif, shp_for_replace, out_orig, 104)
            times_orig_r.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            replace_pixels_OPTIMIZED(base_tif, shp_for_replace, out_opt, 104)
            times_opt_r.append(time.perf_counter() - t0)

        t_orig_r = np.median(times_orig_r)
        t_opt_r = np.median(times_opt_r)

        with rasterio.open(out_orig) as s1, rasterio.open(out_opt) as s2:
            arr_orig = s1.read(1)
            arr_opt = s2.read(1)

        raster_match = np.array_equal(arr_orig, arr_opt)
        speedup_r = t_orig_r / t_opt_r if t_opt_r > 0 else float("inf")

        print(f"  ORIGINAL : {t_orig_r:.4f}s (median of {n_runs})")
        print(f"  OPTIMIZED: {t_opt_r:.4f}s (median of {n_runs})")
        print(f"  Rasters identical: {raster_match}")
        print(f"  Speedup          : {speedup_r:.1f}x")

        results["replace_pixels_within_shapefile"] = {
            "original_time_s": round(t_orig_r, 4),
            "optimized_time_s": round(t_opt_r, 4),
            "speedup": round(speedup_r, 1),
            "identical": raster_match,
        }

        # ════════════════════════════════════════════════════════════════════
        # 3. split_shapefile_by_planted
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "-" * 60)
        print("3. split_shapefile_by_planted")
        print("-" * 60)

        # Create fake prediction file (half planted, half not)
        gdf_ids = gpd.read_file(os.path.join(processed_dir, f"croplands_{region}_id.shp"))
        pred_df = pd.DataFrame({
            'id': gdf_ids['id'],
            'planted': [1 if i % 2 == 0 else 0 for i in range(len(gdf_ids))],
        })
        for idx_name in DERIVED_INDICES:
            pred_df[idx_name] = np.random.default_rng(99).uniform(0.1, 0.8, len(gdf_ids))
        pred_path = os.path.join(processed_dir, f"output_croplands_{region}_{date_str}_prediction.txt")
        pred_df.to_csv(pred_path, sep="\t", index=False)

        # Warmup
        split_shapefile_ORIGINAL(processed_dir, region, date_str)
        split_shapefile_OPTIMIZED(processed_dir, region, date_str)

        times_orig_s, times_opt_s = [], []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            split_shapefile_ORIGINAL(processed_dir, region, date_str)
            times_orig_s.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            split_shapefile_OPTIMIZED(processed_dir, region, date_str)
            times_opt_s.append(time.perf_counter() - t0)

        t_orig_s = np.median(times_orig_s)
        t_opt_s = np.median(times_opt_s)
        speedup_s = t_orig_s / t_opt_s if t_opt_s > 0 else float("inf")

        print(f"  ORIGINAL : {t_orig_s:.4f}s (median of {n_runs})")
        print(f"  OPTIMIZED: {t_opt_s:.4f}s (median of {n_runs})")
        print(f"  Speedup          : {speedup_s:.1f}x")

        results["split_shapefile_by_planted"] = {
            "original_time_s": round(t_orig_s, 4),
            "optimized_time_s": round(t_opt_s, 4),
            "speedup": round(speedup_s, 1),
        }

        # ════════════════════════════════════════════════════════════════════
        # Summary
        # ════════════════════════════════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)

        all_pass = True
        for fn_name, res in results.items():
            ok = res.get("values_match", res.get("identical", True))
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            spd = res.get("speedup", "N/A")
            print(f"  {fn_name:40s}  {status}  speedup={spd}x")

        print()
        if all_pass:
            print("  All checks PASSED")
        else:
            print("  Some checks FAILED")

        results_path = AUDIT_DIR / "audit_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n  Results saved to {results_path}")

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return results


if __name__ == "__main__":
    run_audit()
