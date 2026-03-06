import openeo
from datetime import datetime, timedelta
import os
import time
import json
import glob
import shutil
from typing import Callable, Optional


def _is_valid_cached_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _band_output_path(output_dir: str, band: str, date: datetime, region_token: str) -> str:
    return os.path.join(output_dir, f"{band}_{date.strftime('%Y-%m-%d')}_{region_token}.tiff")


def _cached_band_candidates(output_dir: str, band: str, date: datetime) -> list[str]:
    pattern = os.path.join(output_dir, f"{band}_{date.strftime('%Y-%m-%d')}_*.tiff")
    return [
        candidate
        for candidate in sorted(glob.glob(pattern))
        if not candidate.endswith(".tmp.tiff") and _is_valid_cached_file(candidate)
    ]


def _resolve_cached_band_path(output_dir: str, band: str, date: datetime, region_token: str) -> Optional[str]:
    expected_path = _band_output_path(output_dir, band, date, region_token)

    if _is_valid_cached_file(expected_path):
        return expected_path

    for candidate in _cached_band_candidates(output_dir, band, date):
        if os.path.abspath(candidate) == os.path.abspath(expected_path):
            return expected_path
        shutil.copy2(candidate, expected_path)
        return expected_path

    return None

def download_sentinel2_bands_for_date(
    date: datetime,
    spatial_extent: dict,
    output_dir: str,
    region_name: Optional[str] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    stop_requested_callback: Optional[Callable[[], bool]] = None,
):
    """
    Downloads Sentinel-2 L2A bands B02, B03, B04, B08 for a single date and applies SCL masking.
    
    Parameters:
    - date: datetime object representing the date to download
    - spatial_extent: dict with keys 'west', 'south', 'east', 'north'
    - output_dir: directory to save files
    - region_name: case/region name used in output filenames
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    def _emit(message: str):
        print(message)
        if progress_callback is not None:
            progress_callback(message)

    def _check_stop(stage: str):
        if stop_requested_callback is not None and stop_requested_callback():
            raise InterruptedError(f"NDVI download stopped by user during: {stage}")

    def _checkpoint_path() -> str:
        return os.path.join(output_dir, ".ndvi_download_checkpoint.json")

    def _load_checkpoint() -> dict:
        path = _checkpoint_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_checkpoint(payload: dict) -> None:
        path = _checkpoint_path()
        temp_path = path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(temp_path, path)

    bands = ["B02", "B03", "B04", "B08"]
    region_token = region_name or "region"

    # Fast path: if all bands are already cached locally, avoid OpenEO auth/requests entirely.
    preflight_files = []
    for band in bands:
        cached_path = _resolve_cached_band_path(output_dir, band, date, region_token)
        if cached_path is not None:
            _emit(f"Using cached file for {band}: {os.path.basename(cached_path)}")
            preflight_files.append(cached_path)
            continue

        preflight_files = []
        break

    if len(preflight_files) == len(bands):
        _emit("Sentinel-2 band download stage completed (cache-only)")
        return preflight_files

    _emit("Connecting to Copernicus openeo backend...")
    con = openeo.connect("openeo.dataspace.copernicus.eu")
    
    # Try client credentials if available, otherwise fallback to device auth
    client_id = os.environ.get("OPENEO_AUTH_CLIENT_ID")
    client_secret = os.environ.get("OPENEO_AUTH_CLIENT_SECRET")
    
    if client_id and client_secret:
        _emit("Using OIDC Client Credentials for authentication...")
        con.authenticate_oidc_client_credentials(client_id=client_id, client_secret=client_secret)
    else:
        con.authenticate_oidc()

    current_date = date
    max_days_back = 30
    max_rate_limit_retries = 6

    def _is_no_data_error(exc: Exception) -> bool:
        text = str(exc)
        lowered = text.lower()
        return "NoDataAvailable" in text or "no data available" in lowered

    def _is_rate_limit_error(exc: Exception) -> bool:
        status_code = getattr(exc, "http_status_code", None)
        text = str(exc).lower()
        return status_code == 429 or "too many requests" in text or "[429]" in text
    
    for _ in range(max_days_back):
        _check_stop("date search")
        end_date = current_date + timedelta(days=1)
        rate_limit_retry = 0

        while True:
            try:
                _check_stop("collection loading")
                _emit(f"Loading Sentinel-2 collection for {current_date.strftime('%Y-%m-%d')}...")
                datacube = con.load_collection(
                    "SENTINEL2_L2A",
                    spatial_extent=spatial_extent,
                    temporal_extent=[current_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
                    bands=bands + ["SCL"]
                )

                scl_band = datacube.band("SCL")
                mask_valid = ~((scl_band == 4) | (scl_band == 5) | (scl_band == 6) | (scl_band == 7))

                downloaded_files = []
                checkpoint = _load_checkpoint()
                completed = set(checkpoint.get("completed_bands", []))
                source_date_str = current_date.strftime("%Y-%m-%d")

                for band in bands:
                    _check_stop(f"before downloading {band}")
                    output_path = _band_output_path(output_dir, band, date, region_token)

                    cached_path = _resolve_cached_band_path(output_dir, band, date, region_token)
                    if cached_path is not None:
                        completed.add(band)
                        _save_checkpoint(
                            {
                                "requested_date": date.strftime("%Y-%m-%d"),
                                "source_date": source_date_str,
                                "completed_bands": sorted(completed),
                            }
                        )
                        _emit(f"Using cached file for {band}: {os.path.basename(cached_path)}")
                        downloaded_files.append(cached_path)
                        continue

                    _emit(f"Downloading {band}...")
                    masked_band = datacube.band(band).mask(mask_valid)
                    temp_output = output_path + ".tmp.tiff"
                    masked_band.download(temp_output, format="GTiff")
                    os.replace(temp_output, output_path)

                    completed.add(band)
                    _save_checkpoint(
                        {
                            "requested_date": date.strftime("%Y-%m-%d"),
                            "source_date": source_date_str,
                            "completed_bands": sorted(completed),
                        }
                    )

                    _emit(f"Saved {band} successfully")
                    downloaded_files.append(output_path)

                _emit("Sentinel-2 band download stage completed")
                return downloaded_files

            except Exception as exc:
                if isinstance(exc, InterruptedError):
                    raise

                if _is_no_data_error(exc):
                    _emit(f"No data available for {current_date.strftime('%Y-%m-%d')}. Trying previous date...")
                    current_date = current_date - timedelta(days=1)
                    break

                if _is_rate_limit_error(exc):
                    if rate_limit_retry >= max_rate_limit_retries:
                        raise RuntimeError(
                            f"OpenEO rate limit persisted after {max_rate_limit_retries} retries for "
                            f"{current_date.strftime('%Y-%m-%d')}"
                        ) from exc

                    wait_seconds = min(60, 2 ** rate_limit_retry)
                    _emit(
                        f"OpenEO rate limit hit (429). Retrying in {wait_seconds}s "
                        f"(attempt {rate_limit_retry + 1}/{max_rate_limit_retries})..."
                    )
                    for _ in range(wait_seconds):
                        _check_stop("rate-limit backoff wait")
                        time.sleep(1)
                    rate_limit_retry += 1
                    continue

                raise
                
    raise RuntimeError(f"Could not find Sentinel-2 data within {max_days_back} days prior to {date.strftime('%Y-%m-%d')}")
