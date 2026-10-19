# LULC WebGIS — Remote Sensing Indices

A Streamlit WebGIS for drawing/uploading a study boundary, retrieving Sentinel-2 L2A data or uploading a multiband GeoTIFF, calculating spectral indices, producing a preliminary rules-based LULC classification, reviewing area statistics, and exporting GIS outputs.

## Features
- Draw/edit polygon boundaries on the interactive map or upload ZIP, GeoJSON/JSON, KML, or matching Shapefile components.
- OpenStreetMap, satellite and topographic basemaps.
- Automatic Sentinel-2 L2A search and same-day mosaicking, with adaptive processing resolution to reduce memory use on Streamlit Community Cloud.
- Local multiband GeoTIFF processing with editable band assignments.
- NDVI, NDWI, NDBI, BSI, EVI, MNDWI, SAVI, NDMI, NDSI, AWEIsh and AWEInsh.
- LULC classification, class area statistics, index previews, optional Esri reference comparison and 3D terrain visualization.
- Download classification/index GeoTIFFs and statistics CSV.
- Editable print layout: title, north arrow, scale bar, class legend and export resolution; prepare/download PNG, JPG and PDF, including 800 DPI output.

## Deploy on Streamlit Community Cloud
1. Keep `app.py` and `requirements.txt` in the repository root.
2. In Streamlit Community Cloud, set the app entrypoint to `app.py`.
3. Push these files to GitHub; Community Cloud will install dependencies and restart the app.
4. If the deployed app shows “Oh no. Error running app”, open **Manage app → Logs** and inspect the traceback. If logs are empty, resource exhaustion may be the cause.

## Run locally (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run app.py
```

## Important notes
- The application uses public web services for Sentinel-2 and elevation, so access depends on service availability and network connectivity.
- Large polygons automatically use coarser analysis grids when the native 10 m grid exceeds the conservative pixel budget. This reduces crash risk but may reduce small-feature detail. Split very large areas for higher spatial detail.
- The rules-based classifier is preliminary, not a trained or independently validated model. Use independent reference samples for formal accuracy assessment.
- The map layout is rasterized for export. Text, title, legend and map decorations are editable before export, but the exported JPG/PNG/PDF are not GIS-layer-editable documents. The 800 DPI setting can create large files; the scale bar is approximate in geographic coordinates.
