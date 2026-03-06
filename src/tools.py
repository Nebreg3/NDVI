import re
import glob
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import box
import os
import rasterio
import geopandas as gpd
import numpy as np
import pandas as pd


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")
DERIVED_INDICES = ("NDVI", "GNDVI", "MSAVI", "EVI")
SHAPEFILE_REQUIRED_EXTENSIONS = (".shp", ".shx", ".dbf")

# Background polygon detection thresholds
_MAX_INTERIOR_HOLES = 5
_MAX_FIELD_AREA_HA = 50.0


def _is_valid_cache_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    result = np.full(numerator.shape, np.nan, dtype=np.float32)
    valid = np.abs(denominator) > eps
    np.divide(numerator, denominator, out=result, where=valid)
    return result


def _build_index_output_paths(output_dir: str, date_str: str, region: str) -> dict[str, str]:
    return {
        index_name: os.path.join(output_dir, f"{index_name}_{date_str}_{region}.tiff")
        for index_name in DERIVED_INDICES
    }


def _shapefile_component_paths(path: str, require_all: bool = False) -> list[str]:
    root, ext = os.path.splitext(path)
    if ext:
        if require_all:
            return [root + suffix for suffix in SHAPEFILE_REQUIRED_EXTENSIONS]
        return sorted(glob.glob(root + ".*"))
    return sorted(glob.glob(path + ".*"))


def _cache_is_current(output_paths: list[str], dependency_paths: list[str]) -> bool:
    if not output_paths or not dependency_paths:
        return False

    if not all(_is_valid_cache_file(path) for path in output_paths):
        return False

    if not all(os.path.exists(path) for path in dependency_paths):
        return False

    newest_dependency = max(os.path.getmtime(path) for path in dependency_paths)
    oldest_output = min(os.path.getmtime(path) for path in output_paths)
    return oldest_output >= newest_dependency


def _count_interiors(geom) -> int:
    """Count interior rings (holes) in a polygon or multipolygon."""
    if geom is None or geom.is_empty:
        return 0
    if geom.geom_type == 'Polygon':
        return len(list(geom.interiors))
    if geom.geom_type == 'MultiPolygon':
        return sum(len(list(p.interiors)) for p in geom.geoms)
    return 0


def _is_real_crop_field(geometry_series) -> np.ndarray:
    """Return boolean mask identifying real crop field polygons.

    Removes background/inverted polygons that are artifacts from
    cadastral data (e.g. SIGPAC): large polygons with many interior
    holes representing the complement of crop fields.
    """
    keep = np.ones(len(geometry_series), dtype=bool)
    for i, geom in enumerate(geometry_series):
        if geom is None or geom.is_empty:
            keep[i] = False
            continue
        n_holes = _count_interiors(geom)
        area_ha = geom.area / 10000.0
        if n_holes > _MAX_INTERIOR_HOLES or area_ha > _MAX_FIELD_AREA_HA:
            keep[i] = False
    return keep


# ---------------------------------------------------------------------
# Compute NDVI, GNDVI, MSAVI, and EVI
# ---------------------------------------------------------------------
def process_indices(base_dir, date_str, output_dir, region):
    """
    Processes satellite bands for a specific date and saves NDVI, GNDVI, MSAVI, and EVI.

    Parameters:
    - base_dir (str): folder where the .tiff files are located
    - date_str (str): date in format "YYYY-MM-DD"
    - output_dir (str): folder where results will be saved
    - region (str): region name
    """

    # Compile regex pattern for filenames matching the given date
    pattern = re.compile(r"(B0[2348])_" + re.escape(date_str) + "_" + region + r"\.tiff")

    # Dictionary to store band file paths
    bands = {}

    # Search for files that match the date
    for filename in os.listdir(base_dir):
        match = pattern.match(filename)
        if match:
            band = match.group(1)
            bands[band] = os.path.join(base_dir, filename)

    # Check if all required bands are present
    if not all(b in bands for b in REQUIRED_BANDS):
        print(f"❌ Missing bands for date {date_str}")
        return

    # Create output directory if it does not exist
    os.makedirs(output_dir, exist_ok=True)
    output_paths = _build_index_output_paths(output_dir, date_str, region)

    if all(_is_valid_cache_file(path) for path in output_paths.values()):
        print(f"Using cached spectral indices for date {date_str}")
        return output_paths

    try:
        # Open each band file
        with rasterio.open(bands['B02']) as src_b2, \
             rasterio.open(bands['B03']) as src_b3, \
             rasterio.open(bands['B04']) as src_b4, \
             rasterio.open(bands['B08']) as src_b8:

            # Read band data as float32
            b2 = src_b2.read(1).astype("float32")
            b3 = src_b3.read(1).astype("float32")
            b4 = src_b4.read(1).astype("float32")
            b8 = src_b8.read(1).astype("float32")

            # Compute vegetation indices
            with np.errstate(invalid="ignore", divide="ignore"):
                ndvi = _safe_ratio(b8 - b4, b8 + b4)
                gndvi = _safe_ratio(b8 - b3, b8 + b3)

                msavi_radical = np.clip((2.0 * b8 + 1.0) ** 2 - 8.0 * (b8 - b4), a_min=0.0, a_max=None)
                msavi = ((2.0 * b8 + 1.0) - np.sqrt(msavi_radical)) / 2.0
                evi = _safe_ratio(2.5 * (b8 - b4), b8 + 6.0 * b4 - 7.5 * b2 + 1.0)

            ndvi = np.where(np.isfinite(ndvi), ndvi, np.nan).astype(np.float32)
            gndvi = np.where(np.isfinite(gndvi), gndvi, np.nan).astype(np.float32)
            msavi = np.where(np.isfinite(msavi), msavi, np.nan).astype(np.float32)
            evi = np.where(np.isfinite(evi), evi, np.nan).astype(np.float32)

            # Update raster profile to save results
            profile = src_b4.profile
            profile.update(dtype=rasterio.float32, count=1)

            # Save indices as GeoTIFFs
            indices = {'NDVI': ndvi, 'GNDVI': gndvi, 'MSAVI': msavi, 'EVI': evi}
            for index_name, index_data in indices.items():
                output_path = output_paths[index_name]
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(index_data.astype(rasterio.float32), 1)

            print(f"✅ Indices computed and saved for date {date_str}")
            return output_paths

    except rasterio.errors.RasterioIOError as e:
        print(f"⚠️ Error opening a file for date {date_str}: {e}")

# ---------------------------------------------------------------------
# Create pixels cropland mask
# ---------------------------------------------------------------------
def generate_cropland_mask(region: str, input_dir: str, output_dir: str, date: str, tolerance: float = 1.7):
    """
    Generate a cropland mask (TIFF) from a raster and a shapefile of crop fields.

    Args:
        region (str): Region name
        input_dir (str): Base directory containing the input data.
        output_dir (str): Directory where the results will be saved.
        date (str): Date of the NDVI file in format 'YYYY-MM-DD'.

    """

    # Input and output paths
    ndvi_filename = f"NDVI_{date}_{region}.tiff"
    ndvi_path = os.path.join(output_dir,ndvi_filename)
    shp_path = os.path.join(input_dir, "croplands_"+ region + ".dbf")

    # Output files
    shp_path_region = os.path.join(output_dir, f"croplands_{region}.shp")
    mask_path = os.path.join(output_dir, f"mask_croplands_{region}.tif")

    os.makedirs(os.path.dirname(shp_path_region), exist_ok=True)

    mask_outputs = [mask_path, *_shapefile_component_paths(shp_path_region, require_all=True)]
    mask_dependencies = [ndvi_path, *_shapefile_component_paths(shp_path)]
    if _cache_is_current(mask_outputs, mask_dependencies):
        print(f"Using cached cropland mask for {region} on {date}")
        return mask_path

    # ------------------------------------------------------------------------------------------
    # Filter croplands that are fully inside the raster bounds
    shp_data = gpd.read_file(shp_path)
    with rasterio.open(ndvi_path) as src:
        raster_bounds = src.bounds
        raster_polygon = gpd.GeoSeries([box(*raster_bounds)], crs=src.crs)

    filtered_polygons = shp_data[shp_data.geometry.within(raster_polygon.iloc[0])]

    # Remove background/inverted polygons (large polygons with many holes
    # that represent the complement of crop fields, not actual fields)
    n_before = len(filtered_polygons)
    keep_mask = _is_real_crop_field(filtered_polygons.geometry)
    filtered_polygons = filtered_polygons[keep_mask]
    n_removed = n_before - len(filtered_polygons)
    if n_removed > 0:
        print(f"Removed {n_removed} background polygons ({n_before} → {len(filtered_polygons)})")

    filtered_polygons.to_file(shp_path_region)

    # ------------------------------------------------------------------------------------------
    # Read raster metadata to build the mask
    with rasterio.open(ndvi_path) as src:
        transform = src.transform
        crs = src.crs
        ndvi_shape = src.shape
    # Convert per-pixel containment checks into a single rasterization pass
    # by shrinking polygons according to the tolerance margin.
    pixel_dx = abs(transform.a)
    pixel_dy = abs(transform.e)
    buffer_dist = max((pixel_dx / 2.0) * (tolerance - 1.0), (pixel_dy / 2.0) * (tolerance - 1.0))

    shrunk_geoms = []
    for geom in filtered_polygons.geometry:
        if geom is None or geom.is_empty:
            continue
        shrunk = geom.buffer(-buffer_dist)
        if not shrunk.is_empty:
            shrunk_geoms.append(shrunk)

    if shrunk_geoms:
        mask = rasterize(
            [(geom, 1) for geom in shrunk_geoms],
            out_shape=ndvi_shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=False,
        )
    else:
        mask = np.zeros(ndvi_shape, dtype=np.uint8)

    # ------------------------------------------------------------------------------------------
    # Save mask as GeoTIFF
    with rasterio.open(
        mask_path,
        "w",
        driver="GTiff",
        height=mask.shape[0],
        width=mask.shape[1],
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(mask, 1)

    print(f"Refined mask saved at {mask_path}")
    return mask_path

# ---------------------------------------------------------------------
# Index croplands
# ---------------------------------------------------------------------

def add_ids_to_croplands(input_dir: str, region: str):
    """
    Add a unique ID column to a croplands shapefile and save it as a new file.

    Parameters
    ----------
    input_dir : str
        Directory containing the croplands shapefile.
    region : str
        Name of the region.

    Returns
    -------
    str
        Path to the newly saved shapefile with IDs.
    """

    input_path = os.path.join(input_dir, f"croplands_{region}.shp")
    output_path = os.path.join(input_dir, f"croplands_{region}_id.shp")

    # Check cache BEFORE reading the shapefile to avoid wasted I/O
    output_components = _shapefile_component_paths(output_path, require_all=True)
    input_components = _shapefile_component_paths(input_path)
    if _cache_is_current(output_components, input_components):
        print(f"Using cached cropland IDs for {region}")
        return output_path

    gdf = gpd.read_file(input_path)
    gdf['id'] = range(1, len(gdf) + 1)
    gdf.to_file(output_path)

    return output_path


# ---------------------------------------------------------------------
# Create df
# ---------------------------------------------------------------------

def create_df_crops(region: str, input_dir: str, date: str):
    """
    Calculate crop-level statistics (median) of remote sensing indices
    for a single date using predefined masks for each field.

    Parameters
    ----------
    region : str
        Name of the region
    input_dir : str
        Input directory
    date : str
        Date of the index images to process
    """

    # Paths
    mask_path = os.path.join(input_dir, f"mask_croplands_{region}.tif")
    shp_path = os.path.join(input_dir, f"croplands_{region}_id.shp")
    output_txt = os.path.join(input_dir, f"output_croplands_{region}_{date}_stats.txt")

    index_paths = _build_index_output_paths(input_dir, date, region)
    stats_dependencies = [
        mask_path,
        *_shapefile_component_paths(shp_path),
        *index_paths.values(),
    ]
    if _cache_is_current([output_txt], stats_dependencies):
        try:
            print(f"Using cached cropland statistics for {region} on {date}")
            return pd.read_csv(output_txt, sep='\t')
        except pd.errors.EmptyDataError:
            pass

    # Read mask
    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        transform = src_mask.transform
        out_shape = mask_data.shape

    # Read shapefile
    gdf = gpd.read_file(shp_path)

    required_indexes = list(DERIVED_INDICES)

    # Single-pass rasterize: paint each polygon with its field ID
    # instead of calling geometry_mask N times (one per polygon).
    geom_id_pairs = [
        (geom, fid) for geom, fid in zip(gdf.geometry, gdf['id'])
        if geom is not None and not geom.is_empty
    ]

    if geom_id_pairs:
        label_raster = rasterize(
            geom_id_pairs,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype=np.int32,
        )
        # Zero out labels outside cropland mask
        label_raster[mask_data != 1] = 0
    else:
        label_raster = np.zeros(out_shape, dtype=np.int32)

    # Filter index files for the given date
    index_files = [f for f in os.listdir(input_dir) if f.endswith(".tiff") and f"_{date}_" in f]
    date_files = {f.split('_')[0]: os.path.join(input_dir, f) for f in index_files}

    if not all(idx in date_files for idx in required_indexes):
        raise ValueError(f"Missing indices for date {date}.")

    # Read index rasters
    index_data = {}
    for idx, path in date_files.items():
        try:
            with rasterio.open(path) as src:
                index_data[idx] = src.read(1)
        except Exception as e:
            raise IOError(f"Error reading file {path}: {e}")

    # Vectorized stats: extract all active pixels, compute medians via groupby
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

    # Save results
    df_results = pd.DataFrame(results, columns=['date', 'id', 'Crop', *required_indexes])
    df_results.to_csv(output_txt, sep='\t', index=False)

    return df_results


