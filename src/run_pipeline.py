import os
import shutil
from pathlib import Path
from typing import Callable, Optional

try:
    from .model import predict_with_catboost
    from .tools import (
        add_ids_to_croplands,
        create_df_crops,
        generate_cropland_mask,
        process_indices,
    )
    from .update_fuel_map import (
        replace_pixels_within_shapefile,
        split_shapefile_by_planted,
    )
except ImportError:
    from model import predict_with_catboost
    from tools import (
        add_ids_to_croplands,
        create_df_crops,
        generate_cropland_mask,
        process_indices,
    )
    from update_fuel_map import (
        replace_pixels_within_shapefile,
        split_shapefile_by_planted,
    )


def _emit_progress(
    progress_callback: Optional[Callable[[str], None]], message: str
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _check_stop(stop_requested_callback: Optional[Callable[[], bool]], stage: str) -> None:
    if stop_requested_callback is not None and stop_requested_callback():
        raise InterruptedError(f"NDVI pipeline stopped by user during: {stage}")


def _copy_or_preserve(source_path: str, output_path: str) -> str:
    if Path(source_path).resolve() != Path(output_path).resolve():
        shutil.copy2(source_path, output_path)
    return output_path


def run_ndvi_pipeline(
    *,
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
) -> str:
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    _emit_progress(progress_callback, "Computing vegetation indices")
    _check_stop(stop_requested_callback, "index processing")
    process_indices(base_dir=raw_dir, date_str=date, output_dir=processed_dir, region=region)

    _emit_progress(progress_callback, "Generating cropland mask")
    _check_stop(stop_requested_callback, "cropland masking")
    generate_cropland_mask(region=region, input_dir=raw_dir, output_dir=processed_dir, date=date)

    _emit_progress(progress_callback, "Indexing cropland polygons")
    _check_stop(stop_requested_callback, "cropland indexing")
    add_ids_to_croplands(input_dir=processed_dir, region=region)

    _emit_progress(progress_callback, "Building crop statistics table")
    _check_stop(stop_requested_callback, "crop statistics")
    create_df_crops(region=region, input_dir=processed_dir, date=date)

    _emit_progress(progress_callback, "Running CatBoost cropland classifier")
    _check_stop(stop_requested_callback, "crop prediction")
    predict_with_catboost(
        model_path=os.path.join(model_dir, "catboost_model_trained.cbm"),
        scaler_path=os.path.join(model_dir, "scaler.pkl"),
        input_data_path=os.path.join(
            processed_dir, f"output_croplands_{region}_{date}_stats.txt"
        ),
        output_data_path=os.path.join(
            processed_dir, f"output_croplands_{region}_{date}_prediction.txt"
        ),
    )

    _emit_progress(progress_callback, "Splitting planted and non-planted croplands")
    _check_stop(stop_requested_callback, "shapefile split")
    split_shapefile_by_planted(processed_dir, region, date)

    base_name = target_basename or region
    extension = Path(base_fuel_map).suffix or ".tif"
    planted_shape = os.path.join(processed_dir, f"croplands_{region}_{date}_1.shp")
    unplanted_shape = os.path.join(processed_dir, f"croplands_{region}_{date}_0.shp")
    planted_output = os.path.join(output_dir, f"{base_name}_ndvi_planted{extension}")
    final_output = os.path.join(output_dir, f"{base_name}_ndvi_updated{extension}")

    current_raster = base_fuel_map
    if os.path.exists(planted_shape):
        _emit_progress(progress_callback, "Applying planted cropland fuel update")
        _check_stop(stop_requested_callback, "planted fuel update")
        replace_pixels_within_shapefile(
            tif_path=current_raster,
            shp_path=planted_shape,
            output_path=planted_output,
            new_value=planted_value,
        )
        current_raster = planted_output
    else:
        planted_output = _copy_or_preserve(current_raster, planted_output)
        current_raster = planted_output

    if os.path.exists(unplanted_shape):
        _emit_progress(progress_callback, "Applying non-planted cropland fuel update")
        _check_stop(stop_requested_callback, "unplanted fuel update")
        replace_pixels_within_shapefile(
            tif_path=current_raster,
            shp_path=unplanted_shape,
            output_path=final_output,
            new_value=unplanted_value,
        )
    else:
        final_output = _copy_or_preserve(current_raster, final_output)

    return final_output