import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional

import openeo


REQUIRED_BANDS = ("B02", "B03", "B04", "B08")


def _emit_progress(
    progress_callback: Optional[Callable[[str], None]], message: str
) -> None:
    if progress_callback is not None:
        progress_callback(message)


def _check_stop(stop_requested_callback: Optional[Callable[[], bool]]) -> None:
    if stop_requested_callback is not None and stop_requested_callback():
        raise InterruptedError("NDVI download stopped by user")


def _build_expected_paths(date: datetime, output_dir: str, region_name: str) -> List[str]:
    date_str = date.strftime("%Y-%m-%d")
    return [
        os.path.join(output_dir, f"{band}_{date_str}_{region_name}.tiff")
        for band in REQUIRED_BANDS
    ]


def _copy_compatible_cache(date: datetime, output_dir: str, region_name: str) -> List[str]:
    date_str = date.strftime("%Y-%m-%d")
    output_path = Path(output_dir)
    expected_paths = _build_expected_paths(date, output_dir, region_name)

    for band, expected_path in zip(REQUIRED_BANDS, expected_paths):
        if os.path.exists(expected_path):
            continue

        pattern = f"{band}_{date_str}_*.tiff"
        for candidate in sorted(output_path.glob(pattern)):
            if candidate.name == Path(expected_path).name:
                continue
            shutil.copy2(candidate, expected_path)
            break

    return expected_paths


def download_sentinel2_bands_for_date(
    date: datetime,
    spatial_extent: dict,
    output_dir,
    region_name: str = "east",
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_requested_callback: Optional[Callable[[], bool]] = None,
):
    """Download Sentinel-2 bands or reuse a compatible on-disk cache."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    expected_paths = _build_expected_paths(date, output_dir, region_name)
    if all(os.path.exists(path) for path in expected_paths):
        _emit_progress(progress_callback, "Using cached Sentinel-2 bands")
        return expected_paths

    expected_paths = _copy_compatible_cache(date, output_dir, region_name)
    if all(os.path.exists(path) for path in expected_paths):
        _emit_progress(progress_callback, "Using compatible cached Sentinel-2 bands")
        return expected_paths

    _check_stop(stop_requested_callback)
    _emit_progress(progress_callback, "Connecting to Copernicus OpenEO")
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    
    # Try client credentials if available, otherwise fallback to device auth
    client_id = os.environ.get("OPENEO_AUTH_CLIENT_ID")
    client_secret = os.environ.get("OPENEO_AUTH_CLIENT_SECRET")
    
    if client_id and client_secret:
        _emit("Using OIDC Client Credentials for authentication...")
        con.authenticate_oidc_client_credentials(client_id=client_id, client_secret=client_secret)
    else:
        con.authenticate_oidc()

    end_date = date + timedelta(days=1)
    datacube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=spatial_extent,
        temporal_extent=[date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
        bands=list(REQUIRED_BANDS) + ["SCL"],
        max_cloud_cover=100,
    )

    scl_band = datacube.band("SCL")
    mask_valid = ~(
        (scl_band == 4)
        | (scl_band == 5)
        | (scl_band == 6)
        | (scl_band == 7)
    )

    for band, output_path in zip(REQUIRED_BANDS, expected_paths):
        _check_stop(stop_requested_callback)
        _emit_progress(progress_callback, f"Downloading {band}")
        datacube.band(band).mask(mask_valid).download(output_path)

    return expected_paths

