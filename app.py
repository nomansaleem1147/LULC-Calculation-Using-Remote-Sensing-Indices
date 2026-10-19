import base64
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Avoid inheriting an incompatible PostgreSQL/PostGIS PROJ database on Windows.
# Resolve this before importing GeoPandas/pyproj or running CRS transformations.
for _proj_var in ("PROJ_LIB", "PROJ_DATA"):
    _existing_proj_path = os.environ.get(_proj_var, "")
    if _existing_proj_path and (
        "postgres" in _existing_proj_path.lower()
        or "postgis" in _existing_proj_path.lower()
    ):
        os.environ.pop(_proj_var, None)

import rasterio

# Prefer the PROJ database shipped with the Rasterio wheel, if present.
_rasterio_proj_data = Path(rasterio.__file__).resolve().parent / "proj_data"
if (_rasterio_proj_data / "proj.db").is_file():
    os.environ["PROJ_DATA"] = str(_rasterio_proj_data)
    os.environ["PROJ_LIB"] = str(_rasterio_proj_data)

# Import CRS-dependent packages after the PROJ path is corrected.
import folium
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.features import geometry_mask
from affine import Affine
from shapely.geometry import mapping, shape
from folium.plugins import Draw
import streamlit as st
from streamlit_folium import st_folium


# ---------------------------------------------------------------------
# LULC WebGIS: local boundary + multiband raster processing.
# Satellite search/download needs a configured provider; this app does
# not pretend that an authenticated remote download occurred.
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="LULC WebGIS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { --ink:#14213d; --blue:#1769aa; --cyan:#00a6a6; --violet:#6c4ab6; --soft:#f3f7ff; }
.block-container {padding-top: 1.1rem; padding-bottom: 2rem; max-width: 1600px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#edf5ff 0%,#f6f3ff 100%); border-right:1px solid #dbe5f2;}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3 {color:#163a63;}
.hero-lulc {padding:1.35rem 1.6rem; border-radius:22px; color:white; background:linear-gradient(120deg,#123c69 0%,#1769aa 45%,#00a6a6 100%); box-shadow:0 12px 28px rgba(22,74,130,.20); margin-bottom:1rem;}
.hero-lulc h1 {color:white; margin:0; font-size:2.15rem; letter-spacing:.2px;}
.hero-lulc p {color:#eaf7ff; margin:.35rem 0 0 0;}
.section-card {border:1px solid #dbe7f5; border-radius:16px; padding:1rem 1.1rem; background:linear-gradient(145deg,#fff,#f6faff);}
div[data-testid="stMetric"] {background:linear-gradient(145deg,#ffffff,#f0f7ff); border:1px solid #d8e7f8; padding:.8rem; border-radius:14px; box-shadow:0 4px 12px rgba(36,76,125,.06);}
.stButton > button[kind="primary"] {background:linear-gradient(100deg,#1769aa,#00a6a6); color:white; border:0; border-radius:12px; font-weight:700; box-shadow:0 5px 14px rgba(0,130,170,.22);}
.stButton > button {border-radius:11px; border-color:#c8d9eb; transition:all .15s ease;}
.stButton > button:hover {border-color:#00a6a6; color:#135d8d; transform:translateY(-1px);}
[data-testid="stTabs"] button {font-weight:650;}
.small-note {color:#64748b; font-size:.88rem;}
</style>
""", unsafe_allow_html=True)

INDEX_NAMES = ["NDVI", "NDWI", "NDBI", "BSI", "EVI", "MNDWI", "SAVI", "NDMI", "NDSI", "AWEIsh", "AWEInsh"]
CLASS_NAMES = {
    0: "NoData",
    1: "Water",
    2: "Built-up",
    3: "Vegetation",
    4: "Cropland",
    5: "Rangeland",
    6: "Barren land",
    7: "Snow/Ice",
}
CLASS_COLORS = {
    0: (0, 0, 0, 0),          # NoData: transparent
    1: (0, 102, 255, 255),    # Water: blue
    2: (230, 45, 45, 255),    # Built-up: red
    3: (0, 150, 65, 255),     # Vegetation: green
    4: (255, 220, 0, 255),    # Cropland: yellow
    5: (190, 220, 120, 255),  # Rangeland: light yellow-green
    6: (145, 145, 145, 255),  # Barren land: gray
    7: (255, 255, 255, 255),  # Snow/Ice: white
}


def safe_ratio(a, b):
    den = a + b
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.divide(a - b, den, out=np.full_like(a, np.nan, dtype="float32"),
                        where=np.isfinite(den) & (np.abs(den) > 1e-8))
    return out.astype("float32")


def calculate_indices(bands):
    """Bands are float arrays in reflectance units (0-1 or scaled integer)."""
    blue, green, red, nir, swir1, swir2 = (
        bands["blue"], bands["green"], bands["red"],
        bands["nir"], bands["swir1"], bands["swir2"]
    )
    ndvi = safe_ratio(nir, red)
    ndwi = safe_ratio(green, nir)
    ndbi = safe_ratio(swir1, nir)
    bsi_num = (swir1 + red) - (nir + blue)
    bsi_den = (swir1 + red) + (nir + blue)
    with np.errstate(divide="ignore", invalid="ignore"):
        bsi = np.divide(bsi_num, bsi_den, out=np.full_like(red, np.nan, dtype="float32"),
                        where=np.abs(bsi_den) > 1e-8).astype("float32")
        evi_den = nir + 6 * red - 7.5 * blue + 1
        evi = np.divide(2.5 * (nir-red), evi_den,
                        out=np.full_like(red, np.nan, dtype="float32"),
                        where=np.abs(evi_den) > 1e-8).astype("float32")
    mndwi = safe_ratio(green, swir1)
    savi_den = nir + red + 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        savi = np.divide(1.5 * (nir-red), savi_den,
                         out=np.full_like(red, np.nan, dtype="float32"),
                         where=np.abs(savi_den) > 1e-8).astype("float32")
    ndmi = safe_ratio(nir, swir1)
    ndsi = safe_ratio(green, swir1)
    # Automated Water Extraction Indices (AWEI); use surface reflectance scaled to 0-1.
    # These complement normalized-difference indices and are especially useful for water/shadow separation.
    awei_sh = blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2
    awei_nsh = 4.0 * (green - swir1) - (0.25 * nir + 2.75 * swir2)
    return {
        "NDVI": ndvi, "NDWI": ndwi, "NDBI": ndbi, "BSI": bsi,
        "EVI": evi, "MNDWI": mndwi, "SAVI": savi, "NDMI": ndmi, "NDSI": ndsi,
        "AWEIsh": awei_sh.astype("float32"), "AWEInsh": awei_nsh.astype("float32"),
        "VIS_BRIGHTNESS": ((blue + green + red) / 3.0).astype("float32"),
    }


def load_boundary(uploaded_files):
    """Accept one ZIP, GeoJSON/JSON, KML, or the individual shapefile components."""
    if not uploaded_files:
        return None, "Upload a boundary to begin."

    names = [f.name.lower() for f in uploaded_files]
    with tempfile.TemporaryDirectory(prefix="lulc_boundary_") as td:
        paths = []
        for f in uploaded_files:
            p = Path(td) / Path(f.name).name
            p.write_bytes(f.getvalue())
            paths.append(p)

        shp_path = next((p for p in paths if p.suffix.lower() == ".shp"), None)
        zip_path = next((p for p in paths if p.suffix.lower() == ".zip"), None)
        geo_path = next((p for p in paths if p.suffix.lower() in (".geojson", ".json")), None)
        kml_path = next((p for p in paths if p.suffix.lower() == ".kml"), None)

        try:
            if zip_path:
                extract_dir = Path(td) / "unzipped"
                extract_dir.mkdir()
                with zipfile.ZipFile(zip_path) as zf:
                    # Reject path traversal and extract only safe archive members.
                    for member in zf.infolist():
                        target = (extract_dir / member.filename).resolve()
                        if not str(target).startswith(str(extract_dir.resolve()) + os.sep):
                            continue
                        if not member.is_dir():
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(zf.read(member))
                found = list(extract_dir.rglob("*.shp"))
                if not found:
                    found = list(extract_dir.rglob("*.geojson"))
                if not found:
                    found = list(extract_dir.rglob("*.json"))
                if not found:
                    return None, "ZIP must contain a .shp (with its sidecar files) or GeoJSON."
                gdf = gpd.read_file(found[0])
            elif shp_path:
                # All sidecar files must be uploaded together for a valid shapefile.
                stem = shp_path.stem.lower()
                suffixes = {p.suffix.lower() for p in paths if p.stem.lower() == stem}
                missing = [s for s in (".shx", ".dbf") if s not in suffixes]
                if missing:
                    return None, "Shapefile is incomplete. Upload the matching .shp, .shx and .dbf files; .prj is strongly recommended."
                gdf = gpd.read_file(shp_path)
            elif geo_path:
                gdf = gpd.read_file(geo_path)
            elif kml_path:
                try:
                    gdf = gpd.read_file(kml_path, driver="KML")
                except Exception as exc:
                    return None, f"Could not read KML. Check that the KML driver is available. Details: {exc}"
            else:
                return None, "Supported boundary inputs: .shp + .shx + .dbf (+ .prj), .zip, .geojson, .json, or .kml."

            if gdf.empty:
                return None, "The uploaded boundary contains no features."
            if gdf.crs is None:
                return None, "Boundary CRS is missing. Provide a .prj file or use a GeoJSON/KML with a known CRS."
            gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
            if gdf.empty:
                return None, "No valid geometries were found."
            gdf = gdf.to_crs("EPSG:4326")
            gdf["geometry"] = gdf.geometry.make_valid()
            return gdf, None
        except Exception as exc:
            return None, f"Could not read boundary: {exc}"


def normalize_reflectance(arr):
    arr = arr.astype("float32", copy=False)
    finite = arr[np.isfinite(arr)]
    if finite.size and np.nanpercentile(np.abs(finite), 99) > 2:
        arr = arr / 10000.0
    return arr


def classify_lulc(indices, valid_mask):
    """Rule-based Sentinel-2 classifier with multi-index, water-focused detection.

    The water decision combines independent NDWI/MNDWI/AWEI evidence. Thresholds are
    transparent starting points, not a substitute for scene-specific validation.
    Priority: snow/ice -> water -> built-up -> vegetation -> cropland -> rangeland -> barren.
    """
    ndvi = indices["NDVI"]
    ndwi = indices["NDWI"]
    ndbi = indices["NDBI"]
    bsi = indices["BSI"]
    ndsi = indices["NDSI"]
    mndwi = indices["MNDWI"]
    awei_sh = indices["AWEIsh"]
    awei_nsh = indices["AWEInsh"]
    visible_brightness = indices["VIS_BRIGHTNESS"]

    required = np.isfinite(ndvi) & np.isfinite(ndwi) & np.isfinite(ndbi) & np.isfinite(bsi) & np.isfinite(ndsi) & np.isfinite(mndwi) & np.isfinite(awei_sh) & np.isfinite(awei_nsh) & np.isfinite(visible_brightness)
    valid = valid_mask & required
    cls = np.zeros(ndvi.shape, dtype="uint8")

    # Snow: strong green-vs-SWIR response, with low vegetation response.
    # Brightness guard prevents dark water from being mistaken for snow merely because
    # both can have high NDSI when SWIR reflectance is very low.
    snow = valid & (ndsi >= 0.40) & (ndvi < 0.20) & (visible_brightness >= 0.25)

    # Multi-index water vote. AWEI thresholds are near zero for reflectance in 0-1 units.
    # Requiring 2 independent votes reduces isolated false positives; the MNDWI+NDWI
    # override helps retain clear and turbid water. A vegetation guard excludes most
    # healthy vegetation while allowing mixed/flooded edges at moderate NDVI.
    water_votes = (
        (mndwi >= 0.05).astype("uint8")
        + (ndwi >= 0.05).astype("uint8")
        + (awei_sh > 0.0).astype("uint8")
        + (awei_nsh > 0.0).astype("uint8")
    )
    water_candidate = (water_votes >= 2) | ((mndwi >= 0.12) & (ndwi >= 0.02)) | ((awei_sh > 0.05) & (mndwi > 0.0))
    water = valid & ~snow & water_candidate & (ndvi < 0.35)

    # Remaining classes: use NDVI plus built-up/bare-soil evidence.
    built = valid & ~snow & ~water & (ndbi >= 0.05) & (ndvi < 0.30) & (bsi > -0.10)
    vegetation = valid & ~snow & ~water & ~built & (ndvi >= 0.50)
    cropland = valid & ~snow & ~water & ~built & ~vegetation & (ndvi >= 0.30) & (ndvi < 0.50)
    rangeland = valid & ~snow & ~water & ~built & ~vegetation & ~cropland & (ndvi >= 0.15)
    barren = valid & ~snow & ~water & ~built & ~vegetation & ~cropland & ~rangeland

    cls[snow] = 7
    cls[water] = 1
    cls[built] = 2
    cls[vegetation] = 3
    cls[cropland] = 4
    cls[rangeland] = 5
    cls[barren] = 6
    return cls


def pixel_area_km2(src, out_transform, out_shape):
    """Approximate pixel area, accounting for geographic coordinate systems."""
    h, w = out_shape
    crs = src.crs
    if crs and crs.is_projected:
        # rasterio.crs.CRS does not expose axis_info on all versions;
        # pyproj.CRS provides the projected-axis unit conversion factor.
        from pyproj import CRS as PyprojCRS
        pycrs = PyprojCRS.from_user_input(crs)
        axis_info = pycrs.axis_info
        factor = axis_info[0].unit_conversion_factor if axis_info else 1.0
        factor = factor or 1.0
        area_m2 = abs(out_transform.a * out_transform.e - out_transform.b * out_transform.d) * factor**2
    else:
        from pyproj import Geod, Transformer
        geod = Geod(ellps="WGS84")
        x0, y0 = out_transform * (w / 2, h / 2)
        corners = [
            out_transform * (w/2 - 0.5, h/2 - 0.5),
            out_transform * (w/2 + 0.5, h/2 - 0.5),
            out_transform * (w/2 + 0.5, h/2 + 0.5),
            out_transform * (w/2 - 0.5, h/2 + 0.5),
        ]
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        lonlat = [transformer.transform(x, y) for x, y in corners]
        area_m2, _ = geod.polygon_area_perimeter([p[0] for p in lonlat], [p[1] for p in lonlat])
        area_m2 = abs(area_m2)
    return area_m2 / 1_000_000.0


def write_geotiff(array, profile, dtype, nodata=None):
    bio = io.BytesIO()
    p = profile.copy()
    p.update(driver="GTiff", count=1, dtype=dtype, compress="deflate", predictor=2)
    if nodata is not None:
        p["nodata"] = nodata
    with rasterio.MemoryFile() as mem:
        with mem.open(**p) as dst:
            dst.write(array.astype(dtype), 1)
        bio.write(mem.read())
    return bio.getvalue()



@st.cache_data(ttl=1800, max_entries=64, show_spinner=False)
def _search_stac_cached(intersects_json, datetime_range, max_cloud):
    """Search Sentinel-2 with retries and a second catalogue if Earth Search is down.

    Successful results are cached for 30 minutes. Temporary gateway errors are retried
    briefly; fallback uses Microsoft's Planetary Computer STAC and signs its asset URLs.
    """
    geometry = json.loads(intersects_json)
    common = {
        "intersects": geometry,
        "datetime": datetime_range,
        "limit": 300,
        "query": {"eo:cloud_cover": {"lte": float(max_cloud)}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    }
    catalogues = [
        (
            "Earth Search",
            "https://earth-search.aws.element84.com/v1/search",
            {**common, "collections": ["sentinel-2-c1-l2a"]},
        ),
        (
            "Microsoft Planetary Computer",
            "https://planetarycomputer.microsoft.com/api/stac/v1/search",
            {**common, "collections": ["sentinel-2-l2a"]},
        ),
    ]
    errors = []
    for catalogue_name, endpoint, payload in catalogues:
        for attempt in range(2):
            try:
                response = requests.post(
                    endpoint, json=payload, timeout=(8, 25),
                    headers={"User-Agent": "LULC-WebGIS/1.0"}
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    errors.append(f"{catalogue_name}: HTTP {response.status_code}")
                    if attempt == 0:
                        time.sleep(0.8)
                        continue
                    break
                response.raise_for_status()
                features = response.json().get("features", [])
                if not features:
                    # A healthy response with no scenes is not a server failure.
                    continue
                if catalogue_name == "Microsoft Planetary Computer":
                    try:
                        import planetary_computer
                        for feature in features:
                            for asset in feature.get("assets", {}).values():
                                href = asset.get("href")
                                if href:
                                    asset["href"] = planetary_computer.sign_url(href)
                    except Exception as sign_error:
                        raise RuntimeError(
                            "Planetary Computer returned scenes but asset signing failed. "
                            "Install planetary-computer from requirements.txt."
                        ) from sign_error
                return features
            except (requests.Timeout, requests.ConnectionError) as exc:
                errors.append(f"{catalogue_name}: {type(exc).__name__}")
                if attempt == 0:
                    time.sleep(0.8)
                    continue
                break
            except requests.HTTPError as exc:
                errors.append(f"{catalogue_name}: {exc}")
                # A 4xx response generally won't improve on retry; try next catalogue.
                break
            except ValueError as exc:
                errors.append(f"{catalogue_name}: invalid JSON ({exc})")
                break

    details = "; ".join(errors[-4:]) or "both catalogues returned no matching scenes"
    raise RuntimeError(
        "Sentinel-2 catalogue is temporarily unavailable or returned no matching scenes. "
        "Please retry in a minute, widen the date range, or increase the cloud-cover limit. "
        f"Details: {details}"
    )


def search_sentinel2_items(gdf, start_date, end_date, max_cloud, progress=None):
    """Select a same-day set of Sentinel-2 L2A granules whose union covers the AOI.

    Earth Search returns individual granules. A polygon crossing granule boundaries
    must be processed as a mosaic; selecting one intersecting item can create a thin
    strip and invalid area statistics. The function groups items by acquisition date,
    ranks groups by AOI coverage and cloud cover, and returns all usable granules for
    the best date.
    """
    if start_date > end_date:
        raise ValueError("Start date must be on or before end date.")
    geom = gdf.to_crs("EPSG:4326").geometry.union_all()
    if geom.is_empty:
        raise ValueError("The study boundary geometry is empty.")
    intersects_json = json.dumps(mapping(geom), sort_keys=True, separators=(",", ":"))
    datetime_range = f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z"
    features = _search_stac_cached(intersects_json, datetime_range, float(max_cloud))
    if not features:
        raise ValueError(
            "No Sentinel-2 L2A scenes matched the boundary, date range and cloud-cover limit. "
            "Try widening the date range or increasing the cloud-cover limit."
        )

    # Keep only items with all required spectral assets.
    required_aliases = {
        "blue": ("blue", "B02", "b02"), "green": ("green", "B03", "b03"),
        "red": ("red", "B04", "b04"), "nir": ("nir", "B08", "b08"),
        "swir1": ("swir16", "B11", "b11"), "swir2": ("swir22", "B12", "b12"),
    }
    usable = []
    for feature in features:
        assets = feature.get("assets", {})
        hrefs = {}
        for band, aliases in required_aliases.items():
            for alias in aliases:
                if alias in assets and assets[alias].get("href"):
                    hrefs[band] = assets[alias]["href"]
                    break
        if len(hrefs) != 6 or not feature.get("geometry"):
            continue
        feature["_lulc_band_hrefs"] = hrefs
        feature["_lulc_date"] = str(feature.get("properties", {}).get("datetime", ""))[:10]
        usable.append(feature)
    if not usable:
        raise ValueError("Scenes were found, but none exposed all six required bands (B02, B03, B04, B08, B11, B12).")

    aoi_equal = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs("EPSG:6933")
    aoi_geom = aoi_equal.geometry.iloc[0]
    aoi_area = float(aoi_geom.area)
    if not np.isfinite(aoi_area) or aoi_area <= 0:
        raise ValueError("Study-area polygon has zero or invalid area after projection.")

    # Group by acquisition date so the mosaic avoids combining different seasons.
    groups = {}
    for feature in usable:
        groups.setdefault(feature["_lulc_date"], []).append(feature)
    ranked = []
    for date, items in groups.items():
        footprints = []
        for item in items:
            try:
                footprint = gpd.GeoDataFrame(geometry=[shape(item["geometry"])], crs="EPSG:4326").to_crs("EPSG:6933").geometry.iloc[0]
                if footprint.intersects(aoi_geom):
                    footprints.append(footprint.intersection(aoi_geom))
            except Exception:
                continue
        coverage_geom = gpd.GeoSeries(footprints, crs="EPSG:6933").union_all() if footprints else None
        coverage = (100.0 * coverage_geom.area / aoi_area) if coverage_geom is not None else 0.0
        clouds = [float(x.get("properties", {}).get("eo:cloud_cover") if x.get("properties", {}).get("eo:cloud_cover") is not None else 100) for x in items]
        # Sort lower-cloud granules first so overlaps use the cleaner source first.
        items.sort(key=lambda x: float(x.get("properties", {}).get("eo:cloud_cover") if x.get("properties", {}).get("eo:cloud_cover") is not None else 100))
        ranked.append((coverage, float(np.mean(clouds)), date, items))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    coverage, mean_cloud, date, items = ranked[0]
    if coverage < 98.0:
        raise ValueError(
            f"No same-day Sentinel-2 mosaic covers at least 98% of this study area. "
            f"The best available acquisition date ({date or 'unknown'}) covers {coverage:.1f}%. "
            "Try a wider date range or use a smaller boundary. A partial scene is not accepted as a complete classification."
        )
    for item in items:
        item["_lulc_aoi_coverage_pct"] = float(coverage)
    return items


@st.cache_data(ttl=86400, max_entries=48, show_spinner=False)
def _warp_remote_band_cached(href, height, width, transform_values, target_crs_wkt):
    """Cached COG window read/reprojection; identical runs can reuse remote reads."""
    target_transform = Affine(*transform_values)
    target_crs = rasterio.crs.CRS.from_wkt(target_crs_wkt)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        GDAL_HTTP_MAX_RETRY="2",
        GDAL_HTTP_RETRY_DELAY="1",
        GDAL_HTTP_MULTIRANGE="YES",
        GDAL_NUM_THREADS="1",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_CACHEMAX=128,
    ):
        with rasterio.open(href) as src:
            if src.crs is None:
                raise ValueError(f"Remote band has no CRS: {href}")
            out = np.zeros((height, width), dtype="uint16")
            reproject(
                source=rasterio.band(src, 1), destination=out,
                src_transform=src.transform, src_crs=src.crs,
                src_nodata=src.nodata if src.nodata is not None else 0,
                dst_transform=target_transform, dst_crs=target_crs,
                dst_nodata=0, resampling=Resampling.bilinear,
                num_threads=1, init_dest_nodata=True,
            )
            return out


def _warp_remote_band(href, height, width, target_transform, target_crs):
    """Read/reproject one COG band, reusing cached results for identical requests."""
    return _warp_remote_band_cached(
        href, int(height), int(width), tuple(target_transform), target_crs.to_wkt()
    )


def download_sentinel_stack(items, gdf, max_pixels=2500000, progress=None):
    """Build a same-day Sentinel-2 mosaic, automatically lowering grid resolution when a 10 m grid would exceed a conservative
    memory/pixel limit suitable for Streamlit Community Cloud. 10 m is retained for small AOIs;
    larger AOIs use 20/30/40/60 m to reduce peak memory and avoid worker restarts.
    Only each granule's overlapping AOI window is fetched, and band reads run concurrently.
    """
    if isinstance(items, dict):
        items = [items]
    if not items:
        raise ValueError("No Sentinel-2 granules were selected for mosaicking.")
    items = sorted(
        items,
        key=lambda x: float(x.get("properties", {}).get("eo:cloud_cover")
                             if x.get("properties", {}).get("eo:cloud_cover") is not None else 100)
    )
    band_order = ["blue", "green", "red", "nir", "swir1", "swir2"]

    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="1",
        GDAL_HTTP_RETRY_DELAY="1", GDAL_HTTP_MULTIRANGE="YES",
        GDAL_NUM_THREADS="1", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
    ):
        first_hrefs = items[0]["_lulc_band_hrefs"]
        with rasterio.open(first_hrefs["blue"]) as first_src:
            if first_src.crs is None:
                raise ValueError("The Sentinel-2 blue asset has no CRS.")
            target_crs = first_src.crs
            aoi_projected = gdf.to_crs(target_crs)
            minx, miny, maxx, maxy = aoi_projected.total_bounds
            from rasterio.transform import from_origin
            # Keep native 10 m output for normal AOIs. For very large areas, use a
            # coarser grid automatically rather than failing after scene search.
            resolution_m = 10.0
            grid = None
            for candidate_resolution in (10.0, 20.0, 30.0, 40.0, 60.0):
                candidate_left = math.floor(minx / candidate_resolution) * candidate_resolution
                candidate_top = math.ceil(maxy / candidate_resolution) * candidate_resolution
                candidate_width = max(1, int(math.ceil((maxx - candidate_left) / candidate_resolution)))
                candidate_height = max(1, int(math.ceil((candidate_top - miny) / candidate_resolution)))
                if candidate_height * candidate_width <= max_pixels:
                    resolution_m = candidate_resolution
                    grid = (candidate_left, candidate_top, candidate_width, candidate_height)
                    break
            if grid is None:
                raise ValueError(
                    f"Study area is too large even at 60 m resolution (safe limit {max_pixels:,} pixels). "
                    "Please split the boundary into smaller areas."
                )
            left, top, width, height = grid
            target_transform = from_origin(left, top, resolution_m, resolution_m)
            target_profile = first_src.profile.copy()
            target_profile.update(
                driver="GTiff", height=height, width=width, count=6,
                transform=target_transform, crs=target_crs, dtype="uint16",
                nodata=0, compress="deflate", tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER"
            )
            target_profile.update(tags={"lulc_processing_resolution_m": str(int(resolution_m))})

    # Compute a compact target window for each granule instead of reprojecting each
    # source band to the full AOI-sized array. This is the main performance improvement.
    windows = []
    for item in items:
        try:
            footprint = gpd.GeoSeries([shape(item["geometry"])], crs="EPSG:4326").to_crs(target_crs).iloc[0]
            bx0, by0, bx1, by1 = footprint.bounds
            c0 = max(0, int(math.floor((bx0 - left) / resolution_m)))
            c1 = min(width, int(math.ceil((bx1 - left) / resolution_m)))
            r0 = max(0, int(math.floor((top - by1) / resolution_m)))
            r1 = min(height, int(math.ceil((top - by0) / resolution_m)))
            if c1 <= c0 or r1 <= r0:
                windows.append(None)
                continue
            local_transform = target_transform * Affine.translation(c0, r0)
            windows.append((r0, r1, c0, c1, local_transform))
        except Exception:
            windows.append(None)

    if not any(w is not None for w in windows):
        raise ValueError("None of the selected Sentinel-2 granules overlap the target raster grid.")

    stack = np.zeros((6, height, width), dtype="uint16")
    filled = np.zeros((height, width), dtype=bool)
    task_map = {}
    for item_i, item in enumerate(items):
        win = windows[item_i]
        if win is None:
            continue
        r0, r1, c0, c1, local_transform = win
        for band_i, band in enumerate(band_order):
            task_map[(item_i, band_i)] = (item["_lulc_band_hrefs"][band], r1-r0, c1-c0, local_transform)

    total_tasks = len(task_map)
    completed = 0
    # Keep concurrency high enough to overlap network latency without flooding public COG servers.
    worker_count = min(12, total_tasks)
    per_item = {i: [None] * 6 for i in range(len(items))}
    with ThreadPoolExecutor(max_workers=max(1, worker_count)) as pool:
        future_map = {
            pool.submit(_warp_remote_band, href, out_h, out_w, local_transform, target_crs): (item_i, band_i)
            for (item_i, band_i), (href, out_h, out_w, local_transform) in task_map.items()
        }
        for future in as_completed(future_map):
            item_i, band_i = future_map[future]
            per_item[item_i][band_i] = future.result()
            completed += 1
            if progress:
                progress.progress(
                    min(0.94, completed / max(1, total_tasks)),
                    text=f"Reading/caching overlapping Sentinel-2 windows… {completed}/{total_tasks} band reads complete"
                )

    # Preserve spectral consistency: pixels are filled from a granule only when all six
    # bands are valid there. Granules were sorted by cloud cover, so the clearest wins.
    # per_item is a dictionary keyed by granule index; iterating it directly yields
    # integer keys, which caused the reported "int object is not iterable" error.
    for item_i, band_arrays in per_item.items():
        win = windows[item_i]
        if win is None or any(a is None for a in band_arrays):
            continue
        r0, r1, c0, c1, _ = win
        local_stack = np.stack(band_arrays, axis=0)
        local_valid = np.all(local_stack > 0, axis=0)
        target_filled = filled[r0:r1, c0:c1]
        use = local_valid & ~target_filled
        if np.any(use):
            for band_i in range(6):
                target_view = stack[band_i, r0:r1, c0:c1]
                target_view[use] = local_stack[band_i][use]
            target_filled[use] = True
        del local_stack

    shapes = [mapping(g) for g in aoi_projected.geometry if g is not None and not g.is_empty]
    inside = geometry_mask(shapes, out_shape=(height, width), transform=target_transform, invert=True, all_touched=False)
    stack[:, ~inside] = 0
    valid_coverage = int(np.count_nonzero(filled & inside))
    total_inside = int(np.count_nonzero(inside))
    if total_inside == 0 or valid_coverage / total_inside < 0.98:
        pct = 100.0 * valid_coverage / max(1, total_inside)
        raise ValueError(
            f"The same-day Sentinel-2 mosaic has only {pct:.1f}% valid spectral coverage inside the boundary. "
            "This can happen with data gaps or missing band assets. Try another date range or smaller boundary."
        )
    out = io.BytesIO()
    with rasterio.MemoryFile() as mem:
        with mem.open(**target_profile) as dst:
            dst.write(stack)
            dst.update_tags(
                source="Sentinel-2 L2A via Element 84 Earth Search STAC; same-day multi-granule mosaic",
                item_ids=",".join(x.get("id", "unknown") for x in items),
                acquisition_date=str(items[0].get("_lulc_date", "unknown")),
                mean_scene_cloud_cover=str(np.mean([
                    float(x.get("properties", {}).get("eo:cloud_cover")
                          if x.get("properties", {}).get("eo:cloud_cover") is not None else 100)
                    for x in items
                ])),
                bands="blue,green,red,nir,swir1,swir2",
            )
        out.write(mem.read())
    if progress:
        progress.progress(1.0, text=f"Mosaic complete: {len(items)} granule(s), windowed reads.")
    return out.getvalue(), target_profile, inside


ESRI_LULC_SERVICE = "https://ic.imagery1.arcgis.com/arcgis/rest/services/Sentinel2_10m_LandCover/ImageServer"

def fetch_esri_lulc_for_grid(target_profile, year, aoi_gdf):
    """Fetch Esri land-cover classes in <=3500 px tiles and align to target grid.

    Esri ImageServer limits exportImage dimensions. Tiling keeps large AOIs supported
    instead of failing when either full-grid dimension exceeds 4000 pixels.
    """
    from rasterio.windows import Window, transform as window_transform

    height = int(target_profile["height"])
    width = int(target_profile["width"])
    transform = target_profile["transform"]
    crs = target_profile["crs"]
    out = np.zeros((height, width), dtype=np.uint8)
    tile_size = 3500
    session = requests.Session()
    mosaic_rule = {
        "mosaicMethod": "esriMosaicAttribute",
        "where": f"Year = {int(year)}",
        "sortField": "Year",
        "sortValue": str(int(year)),
        "ascending": True,
        "mosaicOperation": "MT_FIRST",
    }
    rendering_rule = {"rasterFunction": "None"}

    for row_off in range(0, height, tile_size):
        tile_h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            tile_w = min(tile_size, width - col_off)
            window = Window(col_off, row_off, tile_w, tile_h)
            tile_transform = window_transform(window, transform)
            west, south, east, north = rasterio.transform.array_bounds(tile_h, tile_w, tile_transform)
            bbox_3857 = transform_bounds(crs, "EPSG:3857", west, south, east, north, densify_pts=11)
            params = {
                "f": "image",
                "format": "tiff",
                "pixelType": "U8",
                "interpolation": "RSP_NearestNeighbor",
                "bbox": ",".join(f"{v:.8f}" for v in bbox_3857),
                "bboxSR": 3857,
                "imageSR": 3857,
                "size": f"{tile_w},{tile_h}",
                "mosaicRule": json.dumps(mosaic_rule, separators=(",", ":")),
                "renderingRule": json.dumps(rendering_rule, separators=(",", ":")),
            }
            response = session.get(f"{ESRI_LULC_SERVICE}/exportImage", params=params, timeout=(20, 120))
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if "json" in content_type or response.content[:1] == b"{":
                try:
                    message = response.json()
                except Exception:
                    message = {"error": response.text[:500]}
                raise RuntimeError(f"Esri Land Cover service returned an error for tile at row {row_off}, column {col_off}: {message}")

            try:
                with rasterio.MemoryFile(response.content) as mem:
                    with mem.open() as src:
                        if src.count < 1:
                            raise ValueError("Esri comparison response contains no raster band.")
                        src_array = src.read(1)
                        src_transform = src.transform
                        src_crs = src.crs or rasterio.crs.CRS.from_epsg(3857)
                        dst = np.zeros((tile_h, tile_w), dtype=np.uint8)
                        reproject(
                            source=src_array,
                            destination=dst,
                            src_transform=src_transform,
                            src_crs=src_crs,
                            src_nodata=src.nodata,
                            dst_transform=tile_transform,
                            dst_crs=crs,
                            dst_nodata=0,
                            resampling=Resampling.nearest,
                            num_threads=2,
                        )
                        out[row_off:row_off + tile_h, col_off:col_off + tile_w] = dst
            except Exception as exc:
                raise RuntimeError(f"Could not read/align Esri comparison tile ({row_off}, {col_off}): {exc}") from exc
    return out


def normalize_esri_classes(esri_raw):
    """
    Convert Esri/Impact Observatory source classes to the app's 7 broad classes.
    Source class IDs documented by Esri: water=1, trees=2, flooded vegetation=4,
    crops=5, built=7, bare ground=8, snow/ice=9, clouds=10, rangeland=11.
    Values 3 and 6 are unused in this schema and remain NoData.
    """
    out = np.zeros(esri_raw.shape, dtype=np.uint8)
    out[esri_raw == 1] = 1             # Water
    out[np.isin(esri_raw, [7])] = 2    # Built-up
    out[np.isin(esri_raw, [2, 4])] = 3  # Trees / flooded vegetation grouped as vegetation
    out[esri_raw == 5] = 4             # Cropland
    out[esri_raw == 11] = 5            # Rangeland
    out[esri_raw == 8] = 6             # Bare ground
    out[esri_raw == 9] = 7             # Snow / ice
    # Esri class 10 is clouds; it is intentionally left as NoData for fair comparison.
    return out


def compare_with_esri(local_lulc, valid_mask, esri_raw, area_per_pixel):
    esri_classes = normalize_esri_classes(esri_raw)
    compare_mask = valid_mask & (local_lulc > 0) & (esri_classes > 0)
    n = int(compare_mask.sum())
    if n == 0:
        raise ValueError(
            "No comparable pixels were returned by Esri for this year. "
            "Try another year between 2017 and 2025 or check the boundary."
        )
    local = local_lulc[compare_mask].astype(np.uint8)
    ref = esri_classes[compare_mask].astype(np.uint8)
    overall = float(np.mean(local == ref) * 100.0)

    codes = list(range(1, 8))
    matrix = np.zeros((7, 7), dtype=np.int64)
    for a, b in zip(ref, local):
        matrix[int(a) - 1, int(b) - 1] += 1
    row_sums = matrix.sum(axis=1)
    col_sums = matrix.sum(axis=0)
    expected = float(np.sum(row_sums * col_sums) / (n * n))
    observed = overall / 100.0
    kappa = ((observed - expected) / (1.0 - expected)) if expected < 1 else float("nan")

    cm_rows = []
    for i, ref_code in enumerate(codes):
        row = {"Esri reference class": CLASS_NAMES[ref_code]}
        for j, local_code in enumerate(codes):
            row[f"App: {CLASS_NAMES[local_code]}"] = int(matrix[i, j])
        cm_rows.append(row)
    cm_df = pd.DataFrame(cm_rows)

    stats_rows = []
    for code in codes:
        app_count = int(np.count_nonzero(local_lulc[compare_mask] == code))
        esri_count = int(np.count_nonzero(esri_classes[compare_mask] == code))
        stats_rows.append({
            "Class": CLASS_NAMES[code],
            "App area (km²)": app_count * area_per_pixel,
            "Esri area (km²)": esri_count * area_per_pixel,
            "Area difference (km²)": (app_count - esri_count) * area_per_pixel,
            "App pixels": app_count,
            "Esri pixels": esri_count,
        })
    esri_rgba = np.zeros((*esri_classes.shape, 4), dtype=np.uint8)
    for class_code, color in CLASS_COLORS.items():
        esri_rgba[esri_classes == class_code] = color
    esri_png_buf = io.BytesIO()
    plt.imsave(esri_png_buf, esri_rgba)
    return {
        "esri_raw": esri_raw,
        "esri_classes": esri_classes,
        "esri_png": esri_png_buf.getvalue(),
        "compare_pixels": n,
        "agreement_pct": overall,
        "kappa": kappa,
        "confusion_matrix": cm_df,
        "class_comparison": pd.DataFrame(stats_rows),
        "esri_valid_mask": compare_mask,
    }


def process_raster(raster_bytes, filename, gdf, band_map, chosen_indices):
    """Clip an uploaded multiband raster, calculate indices, classify and build exports."""
    with tempfile.TemporaryDirectory(prefix="lulc_raster_") as td:
        raster_path = Path(td) / Path(filename).name
        raster_path.write_bytes(raster_bytes)
        with rasterio.open(raster_path) as src:
            if src.crs is None:
                raise ValueError("The raster has no CRS. Define its correct CRS before processing.")
            if src.count < 1:
                raise ValueError("The raster contains no bands.")
            if any(v < 1 or v > src.count for v in band_map.values()):
                raise ValueError(f"Band selection is invalid. Raster contains {src.count} band(s).")
            geom = gdf.to_crs(src.crs)
            geoms = [mapping(x) for x in geom.geometry if x is not None and not x.is_empty]
            clipped, out_transform = mask(src, geoms, crop=True, filled=False, all_touched=False)
            profile = src.profile.copy()
            profile.update(
                height=clipped.shape[1], width=clipped.shape[2],
                transform=out_transform, count=1
            )
            raw = {}
            valid_mask = ~np.ma.getmaskarray(clipped).any(axis=0)
            for key, band_num in band_map.items():
                data = clipped[band_num - 1].astype("float32").filled(np.nan)
                if src.nodata is not None:
                    data[data == src.nodata] = np.nan
                raw[key] = normalize_reflectance(data)
                valid_mask &= np.isfinite(raw[key])

            # Require all six inputs for consistent index calculations.
            missing = [k for k in ("blue", "green", "red", "nir", "swir1", "swir2") if k not in raw]
            if missing:
                raise ValueError("Missing band assignments: " + ", ".join(missing))
            idx = calculate_indices(raw)
            for key in idx:
                idx[key][~valid_mask] = np.nan
            lulc = classify_lulc(idx, valid_mask)
            valid_pixels = int(np.count_nonzero(lulc))
            if valid_pixels == 0:
                raise ValueError("No valid pixels fall inside the boundary. Check the boundary location, CRS and raster coverage.")

            area_per_pixel = pixel_area_km2(src, out_transform, lulc.shape)
            total_area = valid_pixels * area_per_pixel
            rows = []
            for code, name in CLASS_NAMES.items():
                if code == 0:
                    continue
                count = int(np.count_nonzero(lulc == code))
                rows.append({
                    "Class code": code,
                    "Land-cover class": name,
                    "Pixel count": count,
                    "Area (km²)": count * area_per_pixel,
                    "Area (%)": (count / valid_pixels * 100.0) if valid_pixels else 0.0,
                })
            stats = pd.DataFrame(rows)
            # GeoTIFF exports
            clipped_bytes = write_geotiff(
                clipped[0].astype("float32").filled(-9999), profile, "float32", -9999
            )
            lulc_profile = profile.copy()
            lulc_profile.update(dtype="uint8", nodata=0)
            lulc_bytes = write_geotiff(lulc, lulc_profile, "uint8", 0)
            index_bytes = {}
            for name in chosen_indices:
                index_bytes[name] = write_geotiff(idx[name], profile, "float32", np.nan)

            # Render classification preview; axes are hidden for a clean preview.
            rgba = np.zeros((*lulc.shape, 4), dtype=np.uint8)
            for code, color in CLASS_COLORS.items():
                rgba[lulc == code] = color
            png_buf = io.BytesIO()
            plt.imsave(png_buf, rgba)
            png = png_buf.getvalue()
            bounds = transform_bounds(src.crs, "EPSG:4326", *rasterio.transform.array_bounds(
                lulc.shape[0], lulc.shape[1], out_transform
            ), densify_pts=11)
            return {
                "stats": stats, "indices": idx, "lulc": lulc,
                "png": png, "bounds": bounds, "profile": profile,
                "clipped_tif": clipped_bytes, "lulc_tif": lulc_bytes,
                "index_tifs": index_bytes, "area_per_pixel": area_per_pixel,
                "valid_pixels": valid_pixels, "total_area": total_area,
            }


def render_print_map(result, title, show_north, show_scale, show_legend, dpi, format_name, boundary_gdf=None, class_labels=None, class_colors=None):
    """Create a print-ready map image/PDF from the classified raster and editable layout options."""
    from matplotlib.patches import Patch
    west, south, east, north = result["bounds"]
    lulc = result["lulc"]
    rgba = np.zeros((*lulc.shape, 4), dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        rgba[lulc == code] = color
    # Keep export canvas bounded: 800 dpi is supported, but a compact page avoids enormous files.
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.78 if show_legend else 0.94, top=0.86, bottom=0.12)
    ax.imshow(rgba, extent=(west, east, south, north), origin="upper", interpolation="nearest")
    if boundary_gdf is not None:
        try:
            boundary = boundary_gdf.to_crs("EPSG:4326")
            boundary.boundary.plot(ax=ax, color="#111111", linewidth=0.7)
        except Exception:
            pass
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_title("")
    ax.grid(True, linewidth=0.25, alpha=0.3, linestyle="--")
    fig.suptitle(title.strip() or "Land Use / Land Cover Map", fontsize=15, fontweight="bold", y=0.96)
    if show_north:
        ax.annotate("N", xy=(0.94, 0.91), xytext=(0.94, 0.77), xycoords="axes fraction",
                    textcoords="axes fraction", ha="center", va="center", fontsize=12, fontweight="bold",
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color="black"))
    if show_scale:
        # Approximate ground distance for a longitude segment at map-center latitude.
        center_lat = (south + north) / 2
        width_km = max(0.1, (east-west) * 111.32 * math.cos(math.radians(center_lat)))
        scale_km = 1 if width_km < 15 else (2 if width_km < 40 else (5 if width_km < 100 else (10 if width_km < 250 else 25)))
        frac = scale_km / width_km if width_km else 0.1
        x0, y0 = 0.06, 0.06
        x1 = min(0.06 + frac, 0.35)
        ax.plot([x0, x1], [y0, y0], transform=ax.transAxes, color="black", linewidth=3, solid_capstyle="butt")
        ax.text((x0+x1)/2, y0+0.025, f"{scale_km} km (approx.)", transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8, color="black")
    if show_legend:
        class_labels = class_labels or CLASS_NAMES
        class_colors = class_colors or {i: "#%02x%02x%02x" % CLASS_COLORS[i][:3] for i in range(1, 8)}
        handles = [Patch(facecolor=class_colors[i], edgecolor="#555", label=class_labels[i]) for i in range(1, 8)]
        ax.legend(handles=handles, title="Land-cover classes", loc="center left", bbox_to_anchor=(1.01, 0.5),
                  frameon=True, fontsize=8, title_fontsize=9)
    fig.text(0.07, 0.035, "Prepared by Nouman Saleem | GIS Specialist", fontsize=8, color="#444444")
    output = io.BytesIO()
    fmt = format_name.lower()
    if fmt == "jpg":
        fig.savefig(output, format="jpeg", dpi=int(dpi), bbox_inches="tight", pil_kwargs={"quality": 95, "subsampling": 0})
    else:
        fig.savefig(output, format=fmt, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    return output.getvalue()


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_elevation_samples(latitudes, longitudes):
    """Fetch elevation samples in small batches from Open-Meteo's public elevation API."""
    values = []
    session = requests.Session()
    for i in range(0, len(latitudes), 100):
        lat_batch = latitudes[i:i+100]
        lon_batch = longitudes[i:i+100]
        response = session.get(
            "https://api.open-meteo.com/v1/elevation",
            params={"latitude": ",".join(f"{x:.6f}" for x in lat_batch),
                    "longitude": ",".join(f"{x:.6f}" for x in lon_batch)},
            timeout=(15, 45),
        )
        response.raise_for_status()
        payload = response.json()
        # Open-Meteo may return a list of location objects for multi-coordinate
        # requests, or one object with an elevation array depending on API response mode.
        if isinstance(payload, list):
            elev = [item.get("elevation") for item in payload]
        else:
            elev = payload.get("elevation")
        if not isinstance(elev, list):
            elev = [elev] if len(lat_batch) == 1 and elev is not None else []
        if len(elev) != len(lat_batch):
            raise ValueError("Elevation service returned an unexpected number of samples.")
        values.extend([float(x) if x is not None else np.nan for x in elev])
    return values


def build_3d_lulc_figure(result, grid_size=30):
    """Render LULC classes draped over real elevation samples on a Plotly 3D surface."""
    from pyproj import Transformer
    lulc = result["lulc"]
    profile = result["profile"]
    h, w = lulc.shape
    rows = np.unique(np.linspace(0, h-1, min(grid_size, h)).round().astype(int))
    cols = np.unique(np.linspace(0, w-1, min(grid_size, w)).round().astype(int))
    sampled_classes = lulc[np.ix_(rows, cols)].astype(float)
    trans = profile["transform"]
    xs, ys = [], []
    for r in rows:
        row_x, row_y = [], []
        for c in cols:
            x, y = trans * (int(c) + 0.5, int(r) + 0.5)
            row_x.append(x); row_y.append(y)
        xs.append(row_x); ys.append(row_y)
    transformer = Transformer.from_crs(profile["crs"], "EPSG:4326", always_xy=True)
    lon = np.zeros((len(rows), len(cols)), dtype=float)
    lat = np.zeros_like(lon)
    for i in range(len(rows)):
        for j in range(len(cols)):
            lon[i, j], lat[i, j] = transformer.transform(xs[i][j], ys[i][j])
    flat_lat = lat.ravel().tolist()
    flat_lon = lon.ravel().tolist()
    elevation = np.asarray(fetch_elevation_samples(flat_lat, flat_lon), dtype=float).reshape(lat.shape)
    elevation[sampled_classes <= 0] = np.nan
    # Build a discrete colour scale for integer class codes.
    class_hex = {
        1: "#0066ff", 2: "#e62d2d", 3: "#009641", 4: "#ffdc00",
        5: "#bedc78", 6: "#919191", 7: "#ffffff"
    }
    colorscale = []
    for code in range(1, 8):
        lo = (code - 1.0) / 7.0
        hi = code / 7.0
        colorscale.append([max(0.0, lo), class_hex[code]])
        colorscale.append([min(1.0, hi), class_hex[code]])
    fig = go.Figure(go.Surface(
        x=lon, y=lat, z=elevation,
        surfacecolor=sampled_classes,
        cmin=0.5, cmax=7.5, colorscale=colorscale,
        colorbar={"title": "LULC class", "tickmode": "array", "tickvals": list(range(1,8)),
                  "ticktext": [CLASS_NAMES[i] for i in range(1,8)]},
        connectgaps=False, showscale=True, hovertemplate="Lon %{x:.4f}<br>Lat %{y:.4f}<br>Elevation %{z:.0f} m<extra></extra>"
    ))
    fig.update_layout(
        title="3D Terrain with LULC Classification Overlay",
        scene={"xaxis_title": "Longitude", "yaxis_title": "Latitude", "zaxis_title": "Elevation (m)",
               "aspectmode": "manual", "aspectratio": {"x": 1, "y": 1, "z": 0.18}},
        height=650, margin={"l": 0, "r": 0, "t": 55, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


st.markdown("""
<div class="hero-lulc">
  <h1>🌍 LULC WebGIS</h1>
  <p>Satellite-powered land-cover mapping · Spectral indices · Water detection · Esri benchmarking · 3D terrain view</p>
  <p style="font-size:.9rem;margin-top:.65rem"><b>Prepared by Nouman Saleem | GIS Specialist</b></p>
</div>
""", unsafe_allow_html=True)


ov1, ov2, ov3, ov4 = st.columns(4)
with ov1:
    st.markdown('<div class="section-card"><b>🛰️ Data source</b><br><span class="small-note">Sentinel-2 L2A</span></div>', unsafe_allow_html=True)
with ov2:
    st.markdown('<div class="section-card"><b>💧 Water analysis</b><br><span class="small-note">NDWI · MNDWI · AWEI</span></div>', unsafe_allow_html=True)
with ov3:
    st.markdown('<div class="section-card"><b>🗺️ Mapping</b><br><span class="small-note">OSM · Satellite · Topographic</span></div>', unsafe_allow_html=True)
with ov4:
    st.markdown('<div class="section-card"><b>🏔️ 3D visualization</b><br><span class="small-note">Elevation-draped LULC</span></div>', unsafe_allow_html=True)
st.write("")

if "boundary_gdf" not in st.session_state:
    st.session_state.boundary_gdf = None
if "results" not in st.session_state:
    st.session_state.results = None
if "drawn_boundary_feature" not in st.session_state:
    st.session_state.drawn_boundary_feature = None

with st.sidebar:
    st.header("GIS Controls")
    st.subheader("Study Area")
    st.caption("Upload a ZIP, GeoJSON/KML, matching Shapefile parts, or draw a polygon on the map.")
    boundary_files = st.file_uploader(
        "Boundary file(s)",
        type=["shp", "shx", "dbf", "prj", "cpg", "zip", "geojson", "json", "kml"],
        accept_multiple_files=True,
        key="boundary_upload",
    )
    if st.button("Load / Update Boundary", use_container_width=True):
        gdf, err = load_boundary(boundary_files)
        if err:
            st.error(err)
        else:
            st.session_state.boundary_gdf = gdf
            st.session_state.results = None
            st.success(f"Loaded {len(gdf)} feature(s).")
    if st.session_state.boundary_gdf is not None:
        st.success("Boundary loaded.")
        if st.button("Remove boundary", use_container_width=True):
            st.session_state.boundary_gdf = None
            st.session_state.results = None
            st.rerun()

    st.divider()
    st.subheader("Satellite / Raster Data")
    data_mode = st.radio(
        "Data input",
        ["Automatically download Sentinel-2", "Upload local multiband GeoTIFF"],
        index=0,
        help="Automatic mode searches public Sentinel-2 Level-2A catalog data intersecting your boundary and retrieves the six bands needed for the selected indices."
    )
    today = pd.Timestamp.now(tz="UTC").date()
    default_start = today - pd.Timedelta(days=60)
    if data_mode == "Automatically download Sentinel-2":
        start_date = st.date_input("Start date", value=default_start, key="auto_start_date")
        end_date = st.date_input("End date", value=today, key="auto_end_date")
        cloud_cover = st.slider("Maximum scene cloud cover (%)", 0, 100, 20, key="auto_cloud")
        st.caption("Cloud cover is scene-level metadata; the selected scene can still contain clouds over your boundary.")
        raster_file = None
    else:
        raster_file = st.file_uploader(
            "Upload multiband GeoTIFF (.tif/.tiff)",
            type=["tif", "tiff"],
            accept_multiple_files=False,
            key="raster_upload",
        )
        start_date, end_date, cloud_cover = None, None, 20

    st.subheader("Band Assignment")
    st.caption("Local raster only: verify each band against raster metadata. Automatic Sentinel-2 mode assigns bands by their catalog asset names.")
    if data_mode == "Upload local multiband GeoTIFF":
        # Defaults: standard Sentinel-2 band positions for >=12-band stacks;
        # sequential 1..6 for exactly six-band stacks; otherwise first six bands.
        default_map = {"blue": 2, "green": 3, "red": 4, "nir": 8, "swir1": 11, "swir2": 12}
        band_map = {}
        max_bands = 12
        if raster_file:
            try:
                with rasterio.MemoryFile(raster_file.getvalue()) as mem:
                    with mem.open() as ds:
                        max_bands = ds.count
                        st.caption(f"Raster: {ds.width} × {ds.height} pixels · {ds.count} band(s) · CRS: {ds.crs}")
            except Exception as exc:
                st.error(f"Could not read raster metadata: {exc}")
        if max_bands == 6:
            default_map = {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6}
        elif max_bands < 12:
            default_map = {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6}
        if raster_file and max_bands < 6:
            st.warning("This raster has fewer than six bands. Use a multiband raster containing blue, green, red, NIR, SWIR1 and SWIR2, or the index calculations will not be valid.")
        for key, label in [
            ("blue", "Blue (B2)"), ("green", "Green (B3)"), ("red", "Red (B4)"),
            ("nir", "Near infrared (B8)"), ("swir1", "SWIR 1 (B11)"), ("swir2", "SWIR 2 (B12)")
        ]:
            default = min(default_map[key], max_bands)
            band_map[key] = st.number_input(
                label, min_value=1, max_value=max(1, max_bands),
                value=max(1, default), step=1, key=f"band_{key}"
            )

    st.subheader("Esri reference comparison")
    compare_esri = st.checkbox("Compare with Esri Sentinel-2 10 m Land Cover (slower; optional)", value=False)
    esri_year = st.selectbox("Esri reference year", list(range(2025, 2016, -1)), index=0,
                             help="Esri's global Sentinel-2 10 m land-cover time series currently covers annual maps from 2017 onward.")
    st.caption("The comparison excludes Esri cloud pixels. It is a benchmark comparison, not independent ground-truth validation.")

    st.subheader("Analysis Options")
    chosen_indices = st.multiselect(
        "Indices to export",
        INDEX_NAMES,
        default=["NDVI", "NDWI", "NDBI", "BSI", "EVI", "MNDWI", "SAVI", "NDMI", "NDSI", "AWEIsh", "AWEInsh"],
    )
    st.caption("Classification rules: snow NDSI ≥ 0.40 and NDVI < 0.20; water uses a multi-index vote from MNDWI, NDWI, AWEIsh and AWEInsh, with NDVI < 0.35; built-up NDBI ≥ 0.05, NDVI < 0.30 and BSI > -0.10; vegetation NDVI ≥ 0.50; cropland NDVI 0.30–0.50; rangeland NDVI ≥ 0.15 after prior classes; otherwise barren. Thresholds remain scene/season dependent and require independent validation.")
    st.caption("Performance: satellite band windows are cached for 24 hours; repeat runs of the same area/date/resolution can reuse them.")
    with st.expander("Performance / cache"):
        st.caption("Satellite band windows are cached for 24 hours. Re-running the same area, date and resolution can skip repeated satellite downloads.")
        if st.button("Clear cached data", use_container_width=True, help="Clears cached satellite reads and elevation samples; the next run will download data again."):
            st.cache_data.clear()
            st.success("Cached data cleared.")

    run = st.button("Run Analysis", type="primary", use_container_width=True)

map_col, summary_col = st.columns([3, 1])
with map_col:
    st.subheader("Interactive Study Map")
    view_mode = st.radio("Map view", ["2D Map", "3D Terrain"], horizontal=True, key="lulc_map_view")
    m = folium.Map(location=[30.5, 69.3], zoom_start=5, tiles=None, control_scale=True)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap", overlay=False, control=True).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satellite Imagery", overlay=False, control=True, max_zoom=19,
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="Map data: © OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)",
        name="Topographic Map", overlay=False, control=True, max_zoom=17,
    ).add_to(m)
    if st.session_state.boundary_gdf is not None:
        gdf = st.session_state.boundary_gdf
        folium.GeoJson(
            json.loads(gdf.to_json()),
            name="Study boundary",
            style_function=lambda _: {"color": "#e53935", "weight": 3, "fillColor": "#ffcc80", "fillOpacity": 0.12},
            tooltip=folium.GeoJsonTooltip(fields=[c for c in gdf.columns if c != "geometry"][:3]) if len([c for c in gdf.columns if c != "geometry"]) else None,
        ).add_to(m)
        minx, miny, maxx, maxy = gdf.total_bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]])
    result = st.session_state.results
    if result is not None:
        scene_meta = result.get("scene", {})
        if scene_meta and float(scene_meta.get("aoi_coverage_pct", 100.0)) < 98.0:
            st.error(
                "This result covers only part of the study area and must not be used as a final LULC map. "
                "Re-run with a fully covering scene or a multi-scene mosaic."
            )
        west, south, east, north = result["bounds"]
        # Folium serializes ImageOverlay inputs to JSON; raw PNG bytes are not JSON serializable.
        # A base64 data URI is a valid string URL and works without writing temporary image files.
        png_data_uri = "data:image/png;base64," + base64.b64encode(result["png"]).decode("ascii")
        folium.raster_layers.ImageOverlay(
            image=png_data_uri,
            bounds=[[south, west], [north, east]],
            opacity=0.75,
            name="LULC classification",
            interactive=True,
            cross_origin=False,
            zindex=2,
        ).add_to(m)
    if result is not None and result.get("esri_comparison"):
        esri_png_uri = "data:image/png;base64," + base64.b64encode(
            result["esri_comparison"]["esri_png"]
        ).decode("ascii")
        folium.raster_layers.ImageOverlay(
            image=esri_png_uri,
            bounds=[[result["bounds"][1], result["bounds"][0]],
                    [result["bounds"][3], result["bounds"][2]]],
            opacity=0.65,
            name=f"Esri LULC reference ({result['esri_comparison']['year']})",
            interactive=True,
            cross_origin=False,
            zindex=3,
        ).add_to(m)
    Draw(
        draw_options={
            "polyline": False, "polygon": True, "rectangle": False,
            "circle": False, "circlemarker": False, "marker": False
        },
        edit_options={"edit": True, "remove": True},
        export=False,
        position="topleft",
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    if view_mode == "2D Map" or st.session_state.results is None:
        map_state = st_folium(
            m, height=540, use_container_width=True, key="main_map",
            returned_objects=["last_active_drawing"]
        )
        if map_state and map_state.get("last_active_drawing"):
            st.session_state["drawn_boundary_feature"] = map_state["last_active_drawing"]
        draw_col1, draw_col2 = st.columns(2)
        with draw_col1:
            if st.button("Use drawn polygon as study area", use_container_width=True,
                         disabled=not bool(st.session_state.get("drawn_boundary_feature"))):
                feature = st.session_state.get("drawn_boundary_feature")
                try:
                    geom_obj = feature.get("geometry") if feature.get("type") == "Feature" else feature
                    drawn_geom = shape(geom_obj)
                    if drawn_geom.geom_type not in ("Polygon", "MultiPolygon") or drawn_geom.is_empty:
                        st.error("Draw a polygon on the map first.")
                    else:
                        st.session_state.boundary_gdf = gpd.GeoDataFrame(
                            {"source": ["User-drawn polygon"]}, geometry=[drawn_geom], crs="EPSG:4326"
                        )
                        st.session_state.results = None
                        st.success("Drawn polygon set as the study area.")
                        st.rerun()
                except Exception as exc:
                    st.error(f"Could not use drawn polygon: {exc}")
        with draw_col2:
            if st.button("Clear drawn polygon", use_container_width=True):
                st.session_state["drawn_boundary_feature"] = None
                st.rerun()
    else:
        st.caption("3D terrain view drapes the classified LULC classes over a sampled elevation surface. Use the 2D map to change basemaps or draw/edit the study polygon.")
        try:
            terrain_fig = build_3d_lulc_figure(st.session_state.results)
            st.plotly_chart(terrain_fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
        except Exception as exc:
            st.error(f"Could not build 3D terrain view: {exc}")
            st.info("Switch back to 2D Map or retry when the public elevation service is reachable.")

with summary_col:
    st.subheader("Analysis Summary")
    st.metric("Boundary", "Loaded" if st.session_state.boundary_gdf is not None else "Not loaded")
    st.metric("Indices selected", len(chosen_indices))
    if result is not None:
        st.metric("Valid pixels", f'{result["valid_pixels"]:,}')
        st.metric("Mapped area", f'{result["total_area"]:,.2f} km²')
    else:
        if st.session_state.boundary_gdf is None:
            st.info("Start by uploading a boundary or drawing a polygon on the map.")
        elif data_mode == "Automatically download Sentinel-2":
            st.info("Boundary is ready. Choose a date range and click Run Analysis to retrieve a same-day Sentinel-2 mosaic automatically.")
        else:
            st.info("Boundary is ready. Upload a georeferenced multiband GeoTIFF, assign the spectral bands, and click Run Analysis.")
    st.subheader("Workflow")
    for item in [
        "1. Upload or draw study boundary",
        "2. Retrieve Sentinel-2 or upload raster",
        "3. Clip to boundary",
        "4. Calculate indices",
        "5. Classify LULC",
        "6. Review and export",
    ]:
        st.write(item)

if run:
    if st.session_state.boundary_gdf is None:
        st.error("Please load a valid study-area boundary first.")
    elif not chosen_indices:
        st.error("Select at least one index to export.")
    elif data_mode == "Upload local multiband GeoTIFF" and raster_file is None:
        st.error("Upload a georeferenced multiband GeoTIFF or select automatic Sentinel-2 download.")
    elif data_mode == "Upload local multiband GeoTIFF" and len(set(int(v) for v in band_map.values())) < 6:
        st.error("Each of the six spectral inputs must point to a different band.")
    elif data_mode == "Upload local multiband GeoTIFF" and max_bands < 6:
        st.error("This raster has fewer than six bands and cannot provide all required inputs.")
    elif data_mode == "Automatically download Sentinel-2" and start_date > end_date:
        st.error("Start date must be on or before end date.")
    else:
        stage = "starting analysis"
        try:
            if data_mode == "Automatically download Sentinel-2":
                stage = "searching Sentinel-2 scenes"
                st.info("Searching Sentinel-2 scenes. Band reads are parallelized and cached for repeat runs; Esri comparison is optional.")
                search_bar = st.progress(0.05, text="Searching Sentinel-2 scenes…")
                items = search_sentinel2_items(
                    st.session_state.boundary_gdf, start_date, end_date, cloud_cover
                )
                if not items:
                    raise ValueError(
                        "No usable Sentinel-2 scenes were found for this boundary and date range. "
                        "Expand the date range, increase the cloud-cover limit, or upload a local multiband GeoTIFF."
                    )
                item = items[0]
                props = item.get("properties", {})
                mosaic_coverage = float(item.get("_lulc_aoi_coverage_pct", 0.0))
                mean_cloud = float(np.mean([float(x.get("properties", {}).get("eo:cloud_cover") if x.get("properties", {}).get("eo:cloud_cover") is not None else 100) for x in items]))
                st.success(
                    f"Selected same-day mosaic: {len(items)} granule(s) · "
                    f"Date: {item.get('_lulc_date', str(props.get('datetime', 'unknown'))[:10])} · "
                    f"Mean scene cloud cover: {mean_cloud:.1f}% · AOI footprint coverage: {mosaic_coverage:.1f}%"
                )
                stack_bar = st.progress(0.0, text="Retrieving and mosaicking Sentinel-2 bands…")
                stage = "downloading and assembling Sentinel-2 mosaic"
                stack_bytes, stack_profile, _ = download_sentinel_stack(
                    items, st.session_state.boundary_gdf, progress=stack_bar
                )
                processing_resolution_m = float(abs(stack_profile["transform"].a))
                if processing_resolution_m > 10.0:
                    st.warning(
                        f"Large study area: using an automatic {processing_resolution_m:.0f} m analysis grid "
                        "to stay within memory limits. Smaller boundaries retain 10 m processing. "
                        "The coarser grid is faster and safer, but reduces small-feature detail."
                    )
                stage = "clipping raster and calculating spectral indices / LULC"
                with st.spinner("Clipping bands and calculating indices / LULC…"):
                    st.session_state.pop("layout_png", None)
                    st.session_state.pop("layout_jpg", None)
                    st.session_state.pop("layout_pdf", None)
                    st.session_state.results = process_raster(
                        stack_bytes, "sentinel2_aoi_stack.tif",
                        st.session_state.boundary_gdf,
                        {"blue": 1, "green": 2, "red": 3, "nir": 4, "swir1": 5, "swir2": 6},
                        chosen_indices,
                    )
                st.session_state.results["scene"] = {
                    "id": ", ".join(x.get("id", "unknown") for x in items),
                    "date": item.get("_lulc_date", str(props.get("datetime", "unknown"))[:10]),
                    "cloud_cover": mean_cloud,
                    "aoi_coverage_pct": mosaic_coverage,
                    "granule_count": len(items),
                    "processing_resolution_m": processing_resolution_m,
                    "platform": props.get("platform", "unknown"),
                    "source": "Element 84 Earth Search STAC / Sentinel-2 L2A same-day mosaic",
                }
                if compare_esri:
                    stage = "fetching and comparing Esri reference land cover"
                    with st.spinner(f"Retrieving Esri Sentinel-2 10 m Land Cover for {esri_year} and comparing classes…"):
                        res = st.session_state.results
                        raw_esri = fetch_esri_lulc_for_grid(res["profile"], esri_year, st.session_state.boundary_gdf)
                        res["esri_comparison"] = compare_with_esri(
                            res["lulc"], (res["lulc"] > 0), raw_esri, res["area_per_pixel"]
                        )
                        res["esri_comparison"]["year"] = int(esri_year)
                stage = "finalizing results"
                st.success("Automatic Sentinel-2 retrieval and analysis completed.")
                st.rerun()
            else:
                with st.spinner("Clipping raster and calculating indices…"):
                    st.session_state.pop("layout_png", None)
                st.session_state.pop("layout_jpg", None)
                st.session_state.pop("layout_pdf", None)
                st.session_state.results = process_raster(
                        raster_file.getvalue(), raster_file.name,
                        st.session_state.boundary_gdf,
                        {k: int(v) for k, v in band_map.items()},
                        chosen_indices,
                    )
                if compare_esri:
                    with st.spinner(f"Retrieving Esri Sentinel-2 10 m Land Cover for {esri_year} and comparing classes…"):
                        res = st.session_state.results
                        raw_esri = fetch_esri_lulc_for_grid(res["profile"], esri_year, st.session_state.boundary_gdf)
                        res["esri_comparison"] = compare_with_esri(
                            res["lulc"], (res["lulc"] > 0), raw_esri, res["area_per_pixel"]
                        )
                        res["esri_comparison"]["year"] = int(esri_year)
                st.success("Processing completed.")
                st.rerun()
        except Exception as exc:
            error_text = str(exc)
            st.error(f"Analysis failed during {stage}: {error_text}")
            if "DATABASE.LAYOUT.VERSION" in error_text or "proj.db" in error_text.lower() or "EPSG code is unknown" in error_text:
                st.error(
                    "PROJ/EPSG database conflict detected. This app configures Rasterio's bundled PROJ data path "
                    "at startup. Stop the old Streamlit process (Ctrl+C), activate the project's venv, and restart "
                    "this updated app.py. If it persists, reinstall rasterio, pyproj and geopandas inside that venv."
                )
                st.code(error_text)
            else:
                st.caption(f"Technical detail: {type(exc).__name__}: {error_text}")
            st.caption(
                "Check internet access, date range, cloud-cover limit, boundary overlap, "
                "CRS information and raster band assignments. Large study areas may need to be split."
            )

st.divider()
st.subheader("LULC Results")
if st.session_state.results is None:
    st.info("Results will appear here after you run an analysis.")
else:
    res = st.session_state.results
    if res.get("scene"):
        scene = res["scene"]
        st.caption(
            f"Source: {scene['source']} · Scene: {scene['id']} · "
            f"Acquisition date: {scene['date']} · Cloud cover: {scene['cloud_cover']}% · "
            f"Granules: {scene.get('granule_count', 1)} · AOI coverage: {scene.get('aoi_coverage_pct', 100.0):.1f}%"
        )
    tab_map, tab_stats, tab_indices, tab_compare, tab_downloads = st.tabs(
        ["Classification Map", "Area Statistics", "Spectral Indices", "Esri Comparison", "Downloads"]
    )
    with tab_map:
        st.image(res["png"], caption="Threshold-based preliminary LULC classification preview", use_container_width=True)
        st.subheader("Editable Map Layout & Export")
        with st.expander("Configure print layout", expanded=True):
            layout_title = st.text_input("Map title", "Land Use / Land Cover (LULC) Map", key="layout_title")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                layout_north = st.checkbox("Show north arrow", value=True, key="layout_north")
            with lc2:
                layout_scale = st.checkbox("Show scale bar", value=True, key="layout_scale")
            with lc3:
                layout_legend = st.checkbox("Show editable class legend", value=True, key="layout_legend")
            export_dpi = st.select_slider("Export resolution (DPI)", options=[150, 300, 600, 800], value=800, key="layout_dpi")
            st.markdown("**Editable legend labels and colors**")
            class_labels = {}
            class_colors = {}
            for code in range(1, 8):
                ec1, ec2 = st.columns([2, 1])
                with ec1:
                    class_labels[code] = st.text_input(f"Class {code} label", CLASS_NAMES[code], key=f"layout_class_label_{code}")
                with ec2:
                    default_hex = "#%02x%02x%02x" % CLASS_COLORS[code][:3]
                    class_colors[code] = st.color_picker(f"Class {code} color", default_hex, key=f"layout_class_color_{code}")
            st.caption("800 DPI is available for print-quality output. Large study areas can produce very large files. The scale bar is approximate for geographic maps.")
            ex1, ex2, ex3 = st.columns(3)
            with ex1:
                if st.button("Prepare PNG", key="prepare_png", use_container_width=True):
                    try:
                        st.session_state["layout_png"] = render_print_map(res, layout_title, layout_north, layout_scale, layout_legend, export_dpi, "png", st.session_state.boundary_gdf, class_labels, class_colors)
                        st.session_state["layout_export_title"] = layout_title
                        st.success("PNG layout prepared.")
                    except Exception as exc:
                        st.error(f"PNG export failed: {type(exc).__name__}: {exc}")
            with ex2:
                if st.button("Prepare JPG", key="prepare_jpg", use_container_width=True):
                    try:
                        st.session_state["layout_jpg"] = render_print_map(res, layout_title, layout_north, layout_scale, layout_legend, export_dpi, "jpg", st.session_state.boundary_gdf, class_labels, class_colors)
                        st.session_state["layout_export_title"] = layout_title
                        st.success("JPG layout prepared.")
                    except Exception as exc:
                        st.error(f"JPG export failed: {type(exc).__name__}: {exc}")
            with ex3:
                if st.button("Prepare PDF", key="prepare_pdf", use_container_width=True):
                    try:
                        st.session_state["layout_pdf"] = render_print_map(res, layout_title, layout_north, layout_scale, layout_legend, export_dpi, "pdf", st.session_state.boundary_gdf, class_labels, class_colors)
                        st.session_state["layout_export_title"] = layout_title
                        st.success("PDF layout prepared.")
                    except Exception as exc:
                        st.error(f"PDF export failed: {type(exc).__name__}: {exc}")
            safe_title = "".join(c.lower() if c.isalnum() else "_" for c in layout_title).strip("_")[:60] or "lulc_map"
            dc1, dc2, dc3 = st.columns(3)
            if st.session_state.get("layout_png"):
                dc1.download_button("Download PNG (800 DPI)", st.session_state["layout_png"], f"{safe_title}.png", "image/png", key="download_layout_png", use_container_width=True)
            if st.session_state.get("layout_jpg"):
                dc2.download_button("Download JPG (800 DPI)", st.session_state["layout_jpg"], f"{safe_title}.jpg", "image/jpeg", key="download_layout_jpg", use_container_width=True)
            if st.session_state.get("layout_pdf"):
                dc3.download_button("Download PDF (print quality)", st.session_state["layout_pdf"], f"{safe_title}.pdf", "application/pdf", key="download_layout_pdf", use_container_width=True)
        legend_cols = st.columns(4)
        for i, (code, name) in enumerate(CLASS_NAMES.items()):
            if code == 0:
                continue
            with legend_cols[(i - 1) % 4]:
                color = CLASS_COLORS[code]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:7px;margin:4px 0">'
                    f'<span style="width:14px;height:14px;background:rgba({color[0]},{color[1]},{color[2]},1);display:inline-block;border:1px solid #777"></span>'
                    f'<span>{name}</span></div>',
                    unsafe_allow_html=True,
                )
    with tab_stats:
        stats = res["stats"]
        st.dataframe(stats.style.format({"Area (km²)": "{:,.4f}", "Area (%)": "{:.2f}%"}), use_container_width=True, hide_index=True)
        chart_stats = stats[stats["Pixel count"] > 0]
        if not chart_stats.empty:
            fig = px.bar(chart_stats, x="Land-cover class", y="Area (km²)", color="Land-cover class",
                         title="Area by land-cover class")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.download_button("Download area statistics (CSV)", stats.to_csv(index=False).encode("utf-8-sig"),
                           "lulc_area_statistics.csv", "text/csv")
    with tab_indices:
        available = [x for x in chosen_indices if x in res["indices"]]
        if available:
            selected_view = st.selectbox("Preview index", available)
            arr = res["indices"][selected_view]
            fig, ax = plt.subplots(figsize=(9, 5))
            index_styles = {
                "NDVI": ("RdYlGn", -1, 1, "Low / no vegetation  ←  NDVI  →  dense vegetation"),
                "EVI": ("RdYlGn", -1, 1, "Low vegetation response  ←  EVI  →  high vegetation response"),
                "SAVI": ("RdYlGn", -1, 1, "Low vegetation response  ←  SAVI  →  high vegetation response"),
                "NDWI": ("BrBG", -1, 1, "Low water signal  ←  NDWI  →  high water signal"),
                "MNDWI": ("BrBG", -1, 1, "Low open-water signal  ←  MNDWI  →  high open-water signal"),
                "NDBI": ("BrBG", -1, 1, "Low built-up signal  ←  NDBI  →  high built-up signal"),
                "BSI": ("RdYlBu_r", -1, 1, "Low bare-soil signal  ←  BSI  →  high bare-soil signal"),
                "NDMI": ("BrBG", -1, 1, "Low moisture  ←  NDMI  →  high vegetation moisture"),
                "NDSI": ("PuBuGn", -1, 1, "Low snow signal  ←  NDSI  →  high snow signal"),
                "AWEIsh": ("BrBG", -1, 1, "Low water likelihood  ←  AWEIsh  →  high water likelihood"),
                "AWEInsh": ("BrBG", -1, 1, "Low water likelihood  ←  AWEInsh  →  high water likelihood"),
            }
            cmap, vmin, vmax, scale_label = index_styles[selected_view]
            im = ax.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_title(f"{selected_view} — index-specific colour ramp")
            ax.axis("off")
            cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
            cbar.set_label(scale_label, fontsize=8)
            st.pyplot(fig, clear_figure=True)
            plt.close(fig)
    with tab_compare:
        comp = res.get("esri_comparison")
        if not comp:
            st.info("Enable Esri comparison in the sidebar and run the analysis to compare classes.")
        else:
            st.caption(
                f"Reference: Esri / Impact Observatory Sentinel-2 10 m Land Cover, {comp['year']}. "
                "Cloud pixels are excluded. Class groupings are harmonized to the seven classes shown in this app."
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Comparable pixels", f"{comp['compare_pixels']:,}")
            k2.metric("Pixel agreement", f"{comp['agreement_pct']:.2f}%")
            k3.metric("Cohen's kappa", "N/A" if not np.isfinite(comp["kappa"]) else f"{comp['kappa']:.3f}")
            st.subheader("Class-area comparison")
            st.dataframe(
                comp["class_comparison"].style.format({
                    "App area (km²)": "{:,.4f}",
                    "Esri area (km²)": "{:,.4f}",
                    "Area difference (km²)": "{:+,.4f}",
                }),
                use_container_width=True, hide_index=True
            )
            st.subheader("Confusion matrix (rows = Esri, columns = this app)")
            st.dataframe(comp["confusion_matrix"], use_container_width=True, hide_index=True)
            st.download_button(
                "Download Esri comparison statistics (CSV)",
                comp["class_comparison"].to_csv(index=False).encode("utf-8-sig"),
                "esri_lulc_comparison.csv", "text/csv"
            )
            st.download_button(
                "Download confusion matrix (CSV)",
                comp["confusion_matrix"].to_csv(index=False).encode("utf-8-sig"),
                "esri_lulc_confusion_matrix.csv", "text/csv"
            )
            esri_class_tif = write_geotiff(comp["esri_classes"], res["profile"], "uint8", 0)
            st.download_button(
                "Download harmonized Esri classes GeoTIFF",
                esri_class_tif, "esri_lulc_harmonized_classes.tif", "image/tiff"
            )
            st.warning(
                "Agreement with Esri is a comparison between two mapped products, not a formal accuracy assessment. "
                "Independent reference samples are required to estimate true classification accuracy."
            )
    with tab_downloads:
        st.download_button("Download LULC classification GeoTIFF", res["lulc_tif"],
                           "lulc_classification.tif", "image/tiff")
        st.download_button("Download clipped raster GeoTIFF (first band)", res["clipped_tif"],
                           "clipped_raster_first_band.tif", "image/tiff")
        st.download_button("Download area statistics CSV", res["stats"].to_csv(index=False).encode("utf-8-sig"),
                           "lulc_area_statistics.csv", "text/csv")
        for name, payload in res["index_tifs"].items():
            st.download_button(f"Download {name} GeoTIFF", payload, f"{name.lower()}.tif", "image/tiff", key=f"download_{name}")
        st.warning("The clipped raster download currently contains the first source band only. Index and classification GeoTIFFs are separate outputs.")

st.divider()
st.caption("Note: Class colors: Water blue; Built-up red; Vegetation green; Cropland yellow; Rangeland light yellow-green; Barren land gray; Snow/Ice white. The threshold classifier is preliminary; Esri comparison is a benchmark comparison, not independent ground-truth validation.")
