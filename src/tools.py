import os
import re
from pathlib import Path
from typing import Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask
from shapely.geometry import box


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")
DERIVED_INDICES = ("NDVI", "EVI", "GNDVI", "MSAVI")


def _build_index_output_paths(output_dir: str, date_str: str, region: str) -> Dict[str, str]:
    return {
        index_name: os.path.join(output_dir, f"{index_name}_{date_str}_{region}.tiff")
        for index_name in DERIVED_INDICES
    }


def _resolve_cropland_source(input_dir: str, region: str) -> Path:
    base = Path(input_dir)
    shp_path = base / f"croplands_{region}.shp"
    if shp_path.exists():
        return shp_path

    dbf_path = base / f"croplands_{region}.dbf"
    if dbf_path.exists():
        return dbf_path

    raise FileNotFoundError(f"Cropland shapefile not found for region {region}")


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

    expected_paths = _build_index_output_paths(output_dir, date_str, region)
    if all(os.path.exists(path) for path in expected_paths.values()):
        return expected_paths

    pattern = re.compile(
        r"(B0[2348])_" + re.escape(date_str) + "_" + re.escape(region) + r"\.tiff"
    )

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
        return {}

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

            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi_den = b8 + b4
                gndvi_den = b8 + b3
                evi_den = b8 + 6 * b4 - 7.5 * b2 + 1
                msavi_radical = np.maximum((2 * b8 + 1) ** 2 - 8 * (b8 - b4), 0.0)

                ndvi = np.divide(
                    b8 - b4,
                    ndvi_den,
                    out=np.full_like(b8, np.nan, dtype=np.float32),
                    where=ndvi_den != 0,
                )
                gndvi = np.divide(
                    b8 - b3,
                    gndvi_den,
                    out=np.full_like(b8, np.nan, dtype=np.float32),
                    where=gndvi_den != 0,
                )
                msavi = (2 * b8 + 1 - np.sqrt(msavi_radical)) / 2
                evi = np.divide(
                    2.5 * (b8 - b4),
                    evi_den,
                    out=np.full_like(b8, np.nan, dtype=np.float32),
                    where=evi_den != 0,
                )

            # Update raster profile to save results
            profile = src_b4.profile
            profile.update(dtype=rasterio.float32, count=1)

            # Save indices as GeoTIFFs
            indices = {'NDVI': ndvi, 'GNDVI': gndvi, 'MSAVI': msavi, 'EVI': evi}
            for index_name, index_data in indices.items():
                output_path = expected_paths[index_name]
                with rasterio.open(output_path, 'w', **profile) as dst:
                    dst.write(index_data.astype(rasterio.float32), 1)

            print(f"✅ Indices computed and saved for date {date_str}")
            return expected_paths

    except rasterio.errors.RasterioIOError as e:
        print(f"⚠️ Error opening a file for date {date_str}: {e}")
        return {}

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
    ndvi_path = os.path.join(output_dir, ndvi_filename)
    shp_path = _resolve_cropland_source(input_dir, region)

    # Output files
    shp_path_region = os.path.join(output_dir, f"croplands_{region}.shp")
    mask_path = os.path.join(output_dir, f"mask_croplands_{region}.tif")

    os.makedirs(os.path.dirname(shp_path_region), exist_ok=True)

    if os.path.exists(mask_path) and os.path.exists(shp_path_region):
        return mask_path

    # ------------------------------------------------------------------------------------------
    # Filter croplands that are fully inside the raster bounds
    shp_data = gpd.read_file(shp_path)
    with rasterio.open(ndvi_path) as src:
        raster_bounds = src.bounds
        raster_polygon = gpd.GeoSeries([box(*raster_bounds)], crs=src.crs)
        transform = src.transform
        crs = src.crs
        ndvi_shape = src.shape

    if shp_data.crs is not None and crs is not None and shp_data.crs != crs:
        shp_data = shp_data.to_crs(crs)

    filtered_polygons = shp_data[shp_data.geometry.intersects(raster_polygon.iloc[0])]
    filtered_polygons.to_file(shp_path_region)

    geometries = [geom for geom in filtered_polygons.geometry if geom is not None and not geom.is_empty]
    if geometries:
        mask = geometry_mask(
            geometries,
            transform=transform,
            invert=True,
            out_shape=ndvi_shape,
        ).astype(np.uint8)
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

    output_path = os.path.join(input_dir, f"croplands_{region}_id.shp")

    if os.path.exists(output_path):
        return output_path

    gdf = gpd.read_file(input_path)
    gdf['id'] = range(1, len(gdf) + 1)

    # Save the new shapefile with IDs
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

    if os.path.exists(output_txt):
        return pd.read_csv(output_txt, sep='\t')

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

    required_indexes = list(DERIVED_INDICES)
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

    return pd.read_csv(output_txt, sep='\t')


