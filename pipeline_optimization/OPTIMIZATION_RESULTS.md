# NDVI Pipeline Optimization Results

This document summarizes the performance work applied to the NDVI pipeline, with emphasis on `create_df_crops`, which was the main bottleneck.

## Scope

The following pipeline steps were reviewed:

- `create_df_crops`
- `add_ids_to_croplands`
- `predict_with_catboost`
- `split_shapefile_by_planted`
- fuel-map update via `replace_pixels_within_shapefile`

The main code changes were applied in:

- `pyrocb_prediction/src/external/NDVI/src/tools.py`
- `pyrocb_prediction/src/external/NDVI/src/update_fuel_map.py`

The benchmark script and machine-readable results are stored in:

- `pyrocb_prediction/src/external/NDVI/pipeline_optimization/run_audit.py`
- `pyrocb_prediction/src/external/NDVI/pipeline_optimization/audit_results.json`

## Changes Made

### 1. `create_df_crops`

Previous behavior:
- Read the cropland mask once.
- Iterated over every polygon.
- Built a per-polygon mask separately.
- Collected pixel coordinates per field.
- Computed medians field by field.

Optimized behavior:
- Rasterize all polygons once using each field `id` as the raster value.
- Apply the cropland mask once to zero out non-cropland pixels.
- Flatten the labeled raster and all index rasters.
- Compute per-field medians with a vectorized `pandas.groupby(...).median()`.

Why this is faster:
- It removes the repeated per-polygon raster masking work.
- It converts the problem from many small geometry operations into one rasterization pass plus vectorized tabular aggregation.
- The speedup should grow with the number of polygons.

### 2. `add_ids_to_croplands`

Previous behavior:
- Read the shapefile first.
- Only then checked whether the output was already cached.

Optimized behavior:
- Check the cache before reading the shapefile.
- Skip all I/O when the ID shapefile is already current.

Why this helps:
- Cached runs avoid unnecessary `geopandas.read_file()` calls.
- This is a small but clean improvement for repeated executions.

### 3. `split_shapefile_by_planted`

Previous behavior:
- Always wrote both output shapefiles.

Optimized behavior:
- Only write the planted or unplanted shapefile if that subset is non-empty.

Why this helps:
- Avoids unnecessary disk writes in edge cases where all polygons fall into one class.
- Runtime impact is small because the function is mostly I/O bound.

### 4. Fuel-map update: `replace_pixels_within_shapefile`

Previous behavior:
- Converted geometries through `__geo_interface__`.
- Built a mask with `geometry_mask`.

Optimized behavior:
- Rasterize shapely geometries directly with `rasterize`.

Why this helps:
- Removes geometry serialization overhead.
- Keeps the implementation simpler and slightly faster.

### 5. `predict_with_catboost`

Review result:
- No change was applied.

Reason:
- The function is already small and mostly dominated by model loading plus inference.
- The major bottleneck reported by the pipeline was not in this step.
- Further optimization here would likely require changing runtime strategy, such as reusing a loaded model across runs, which is a larger behavioral change and was not necessary for the current fix.

## Benchmark Results

Audit command used:

```bash
cd /home/nbravo/ExtremFirePredict/pyrocb_prediction
../.venv/bin/python src/external/NDVI/pipeline_optimization/run_audit.py
```

Measured results from `pyrocb_prediction/src/external/NDVI/pipeline_optimization/audit_results.json`:

| Function | Original (s) | Optimized (s) | Speedup | Validation |
| --- | ---: | ---: | ---: | --- |
| `create_df_crops` | 0.1364 | 0.0249 | 5.5x | exact numeric match |
| `replace_pixels_within_shapefile` | 0.0112 | 0.0095 | 1.2x | identical raster output |
| `split_shapefile_by_planted` | 0.0158 | 0.0158 | 1.0x | behavior preserved |

Additional correctness checks for `create_df_crops`:

- Same output shape: `true`
- Same field IDs: `true`
- Max numeric difference: `0.0`
- Rows compared: `40`

## Interpretation

The clear win is `create_df_crops`, which is now about `5.5x` faster on the small audit dataset while preserving exact output values. This is the most important improvement because it removes the repeated per-polygon masking pattern that scales poorly.

The fuel-map update also improved, but only modestly. That is expected because the function is already relatively small and spends much of its time in raster I/O.

`split_shapefile_by_planted` did not show a measurable speedup in the audit because the benchmark uses a small dataset and the function is dominated by reading and writing files. The change still removes unnecessary writes for empty subsets.

`add_ids_to_croplands` was improved for cached executions, but it was not included as a dedicated benchmark because the optimization is specifically about avoiding needless work on reruns rather than accelerating the initial creation step.

## Validation

The updated code passed the existing unit tests:

```bash
cd /home/nbravo/ExtremFirePredict/pyrocb_prediction
../.venv/bin/python -m pytest tests/unit/test_ndvi_tools.py -q
../.venv/bin/python -m pytest tests/unit/test_cropland_updater.py -q
../.venv/bin/python -m pytest tests/unit/test_ndvi_tools.py tests/unit/test_ndvi_download_bands.py tests/unit/test_cropland_updater.py -q
```

Observed result:
- `13 passed`

## Conclusion

The performance issue in `create_df_crops` has been addressed with a vectorized implementation that preserves output correctness and delivers a meaningful speedup.

The other reviewed steps were either improved where it was low-risk and useful (`add_ids_to_croplands`, `split_shapefile_by_planted`, fuel-map update) or intentionally left unchanged because the likely gains were small compared with the main bottleneck (`predict_with_catboost`).
