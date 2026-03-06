import os
from tools import process_indices, generate_cropland_mask, add_ids_to_croplands, create_df_crops
from update_fuel_map import split_shapefile_by_planted, replace_pixels_within_shapefile
from model import predict_with_catboost
from typing import Callable, Optional

def run_ndvi_pipeline(
    region: str,
    date: str,
    raw_dir: str,
    processed_dir: str,
    output_dir: str,
    model_dir: str,
    base_fuel_map: str,
    target_basename: Optional[str] = None,
    planted_value: int = 104,
    unplanted_value: int = 93,
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_requested_callback: Optional[Callable[[], bool]] = None,
):
    """
    End-to-End pipeline for cropland processing. 
    Accepts parameterized paths instead of hardcoded globals.
    """
    def _emit(message: str) -> None:
        print(message)
        if progress_callback is not None:
            progress_callback(message)

    def _check_stop(stage: str) -> None:
        if stop_requested_callback is not None and stop_requested_callback():
            raise InterruptedError(f"NDVI pipeline stopped by user during: {stage}")

    fuel_map_ext = os.path.splitext(base_fuel_map)[1] or ".tif"
    output_prefix = target_basename or region

    _emit(f"Starting pipeline for {region} on {date}...")

    # 1. Compute indexes
    _check_stop("index processing")
    _emit("Processing spectral indices (NDVI, EVI, GNDVI, MSAVI)")
    process_indices(base_dir=raw_dir, date_str=date, output_dir=processed_dir, region=region)

    # 2. Generate pixel mask for crop fields
    _check_stop("cropland mask generation")
    _emit("Generating cropland mask")
    generate_cropland_mask(region=region, input_dir=raw_dir, output_dir=processed_dir, date=date)

    # 3. Assign IDs
    _check_stop("cropland id assignment")
    _emit("Assigning polygon IDs to croplands")
    add_ids_to_croplands(input_dir=processed_dir, region=region)

    # 4. Create dataframe with stats
    _check_stop("stats dataframe creation")
    _emit("Computing per-cropland statistics")
    create_df_crops(region=region, input_dir=processed_dir, date=date)

    # 5. Predict with CatBoost
    _check_stop("catboost inference")
    _emit("Running CatBoost inference")
    predict_with_catboost(
        model_path=f"{model_dir}/catboost_model_trained.cbm",
        scaler_path=f"{model_dir}/scaler.pkl",
        input_data_path=f"{processed_dir}/output_croplands_{region}_{date}_stats.txt",
        output_data_path=f"{processed_dir}/output_croplands_{region}_{date}_prediction.txt"
    )

    # 6. Separate by planted
    _check_stop("planted/unplanted split")
    _emit("Splitting croplands by planted/unplanted prediction")
    split_shapefile_by_planted(processed_dir, region, date)

    # 7. Update fuel map (Planted)
    _check_stop("planted fuel-map update")
    planted_shp = f"{processed_dir}/croplands_{region}_{date}_1.shp"
    intermediate_tif = f"{processed_dir}/{output_prefix}_ndvi_planted{fuel_map_ext}"
    
    if os.path.exists(planted_shp):
        replace_pixels_within_shapefile(
            tif_path=base_fuel_map,
            shp_path=planted_shp,
            output_path=intermediate_tif,
            new_value=planted_value
        )
    else:
        intermediate_tif = base_fuel_map  # Fallback if no planted crops

    # 8. Update fuel map (Unplanted)
    _check_stop("unplanted fuel-map update")
    unplanted_shp = f"{processed_dir}/croplands_{region}_{date}_0.shp"
    final_tif = f"{output_dir}/{output_prefix}_ndvi_updated{fuel_map_ext}"
    
    if os.path.exists(unplanted_shp):
        replace_pixels_within_shapefile(
            tif_path=intermediate_tif,
            shp_path=unplanted_shp,
            output_path=final_tif,
            new_value=unplanted_value
        )
    else:
        if intermediate_tif != final_tif:
            # Just copy it over if there were planted but no unplanted
            import shutil
            shutil.copy(intermediate_tif, final_tif)

    _emit(f"Pipeline complete. Updated raster saved to {final_tif}")
    return final_tif
