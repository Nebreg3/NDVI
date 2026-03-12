import pandas as pd
import os
import numpy as np
import rasterio
from rasterio.features import rasterize, geometry_mask
import geopandas as gpd

# FBFM40 non-burnable fuel types that should never be overwritten
_PROTECTED_FUEL_CODES = frozenset({91, 98, 99})  # NB1 Urban, NB8 Water, NB9 Bare

# ---------------------------------------------------------------------
# Split shapefile into planted vs not planted
# ---------------------------------------------------------------------
def split_shapefile_by_planted(base_path, region, date):
    """
    Splits a shapefile into two separate files based on the 'planted' column
    from a prediction text file.
    """
    txt_file = os.path.join(base_path, f"output_croplands_{region}_{date}_prediction.txt")
    shp_file = os.path.join(base_path, f"croplands_{region}_id.shp")

    df = pd.read_csv(txt_file, sep="\t")
    gdf = gpd.read_file(shp_file)

    gdf = gdf.merge(df, on="id")

    gdf_planted = gdf[gdf["planted"] == 1]
    gdf_not_planted = gdf[gdf["planted"] == 0]

    out_planted = os.path.join(base_path, f"croplands_{region}_{date}_1.shp")
    out_not_planted = os.path.join(base_path, f"croplands_{region}_{date}_0.shp")

    # Only write non-empty shapefiles
    if not gdf_planted.empty:
        gdf_planted.to_file(out_planted)
    if not gdf_not_planted.empty:
        gdf_not_planted.to_file(out_not_planted)

    print(f"✅ Saved: {out_planted} and {out_not_planted}")


# ---------------------------------------------------------------------
# Replace raster values inside shapefile polygons
# ---------------------------------------------------------------------
def replace_pixels_within_shapefile(tif_path, shp_path, output_path, new_value):
    """
    Replaces all pixels in a raster (TIFF) that fall inside a shapefile with a specified value.
    """
    shapefile = gpd.read_file(shp_path)
    with rasterio.open(tif_path) as src:
        raster_data = src.read()
        out_meta = src.meta.copy()
        shapes = [feature["geometry"] for feature in shapefile.__geo_interface__["features"]]
        mask_inside = geometry_mask(
            shapes,
            transform=src.transform,
            invert=True,
            out_shape=(src.height, src.width),
        )

        band_index = 3 if src.count >= 4 else 0
        raster_data[band_index][mask_inside] = new_value

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(raster_data)


