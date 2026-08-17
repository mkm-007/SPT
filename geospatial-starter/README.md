# Geospatial starter (SPT)

**Why:** Mercy Corps Geospatial Analytics & Mapping Intern asks for map examples + geo-Python practice. CS bridge without inventing ArcGIS YOE.

## Learning loop
1. Load public boundary data  
2. Reproject / check CRS  
3. Make **static** map (PNG) + **interactive** map (HTML)  
4. Write short notes a non-technical teammate could follow  

## Run
```bash
cd geospatial-starter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/make_maps.py
```

Outputs land in `outputs/` (gitignored binaries OK to attach to Handshake; keep scripts + README in git).

## Next (after first maps)
- Install QGIS; recreate one map by hand; screenshot  
- One Google Earth Engine hello-world notebook  

## Honesty bar
This is **self-practical training**, not Mercy Corps production GIS. On the resume: building toward QGIS/ArcGIS/GEE; attach these maps as examples of current practice.
