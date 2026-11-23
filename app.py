"""
CanopyAI — interactive tree-planting optimizer (Streamlit + Folium).

Run:  streamlit run app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import altair as alt
import folium
import numpy as np
import pandas as pd
import streamlit as st
from branca.colormap import LinearColormap
from branca.element import MacroElement, Template
from dotenv import load_dotenv
from folium.plugins import Draw
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from app_core.cities import CITIES, CITY_BY_NAME, MAX_CUSTOM_AREA_KM2  # noqa: E402
from app_core.geo import bbox_area_km2, colorize, grid_latlon_bounds, to_mercator_raster  # noqa: E402
from app_core.plan import (  # noqa: E402
    STRATEGIES,
    PlanParams,
    plan_cache_path,
    plan_geojson,
    plan_table,
    prepare_area,
    run_plan,
)
from gee.extract import ExtractParams, extract_grid, find_cached  # noqa: E402
from gee.landsat import has_ee_credentials  # noqa: E402
from models.lst_model import TARGET  # noqa: E402

st.set_page_config(page_title="CanopyAI · Tree Planting Optimizer", page_icon="🌳", layout="wide")
load_dotenv(ROOT / ".env")
os.environ["CANOPYAI_HEADLESS"] = "1"  # a web server must never wait for a browser login


def _load_secrets() -> None:
    """Streamlit Community Cloud: copy st.secrets into the env vars the Earth Engine code reads."""
    try:
        secrets = st.secrets
        if "gcp_service_account" in secrets:
            os.environ["GEE_SERVICE_ACCOUNT_JSON"] = json.dumps(dict(secrets["gcp_service_account"]))
        for name in ("GEE_PROJECT_ID", "CANOPYAI_ALLOW_CUSTOM", "CANOPYAI_MAX_DOWNLOADS"):
            if name in secrets:
                os.environ[name] = str(secrets[name])
    except Exception:  # no secrets.toml locally: .env is used instead
        pass


_load_secrets()
ALLOW_CUSTOM = os.getenv("CANOPYAI_ALLOW_CUSTOM", "1") != "0"
MAX_DOWNLOADS = int(os.getenv("CANOPYAI_MAX_DOWNLOADS", "3"))  # new Earth Engine areas per browser session

LST_STOPS = ["#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7", "#fddbc7", "#ef8a62", "#b2182b"]
COOL_STOPS = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
TREE_STOPS = ["#f7fcf5", "#c7e9c0", "#74c476", "#238b45", "#00441b"]
STRATEGY_COLORS = {"Lazy Greedy": "#1b7837", "Random": "#9e9e9e", "Hottest first": "#d6604d"}
MAP_HEIGHT = 520
SATELLITE = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
# Free, key-less light basemap (Esri World Light Gray Canvas); native tiles stop at zoom 16
LIGHT_GRAY = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"
LIGHT_GRAY_ATTR = "Esri, HERE, Garmin, © OpenStreetMap contributors"

st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500;600&display=swap');

      /* Inter on text elements only: a blanket selector would also replace the Material Symbols
         font that Streamlit uses for its icons (sidebar chevrons, expander arrows) */
      html, body, .stApp, p, li, label, input, textarea, select, button p,
      .stMarkdown, [data-testid="stCaptionContainer"], [data-baseweb="select"] div,
      [data-testid="stWidgetLabel"], [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
      }
      [data-testid="stIconMaterial"], span[class*="material"], .material-symbols-rounded {
        font-family: 'Material Symbols Rounded' !important;
      }
      h1, h2, h3, h4, h5 { font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.02em; color: #0F172A; }
      h1 { font-weight: 700; }
      .block-container { padding-top: 2.2rem; padding-bottom: 2.5rem; max-width: 1500px; }

      /* Streamlit chrome off: this is a product page, not a dev server */
      #MainMenu, footer, header [data-testid="stToolbar"], [data-testid="stDeployButton"],
      [data-testid="stStatusWidget"] { visibility: hidden; height: 0; }
      header[data-testid="stHeader"] { background: transparent; }

      /* KPI cards */
      div[data-testid="stMetric"] {
        background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px;
        padding: 14px 16px; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      div[data-testid="stMetricLabel"] p {
        font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748B;
      }
      div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 1.45rem; font-weight: 600; color: #0F172A;
      }
      div[data-testid="stMetricDelta"] { font-family: 'JetBrains Mono', ui-monospace, monospace; font-size: 0.8rem; }

      /* hero */
      .eyebrow { font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase;
                 color: #0F766E; margin-bottom: -0.4rem; }
      .lede { color: #475569; font-size: 1.02rem; line-height: 1.6; max-width: 900px; }
      .section-label { font-size: 0.78rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
                       color: #64748B; margin: 0.6rem 0 0.3rem 0; }

      /* sidebar */
      [data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
      [data-testid="stSidebar"] h1 { font-size: 1.35rem; }
      [data-testid="stSidebar"] h2 { font-size: 0.78rem; font-weight: 600; text-transform: uppercase;
                                     letter-spacing: 0.08em; color: #64748B; margin-top: 0.8rem; }

      /* tabs */
      button[data-baseweb="tab"] p { font-weight: 500; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_grid(bbox, years, months):
    return extract_grid(bbox, ExtractParams(years=years, months=months), verbose=False)


@st.cache_resource(show_spinner=False)
def load_area_model(cache_key, _grid):
    return prepare_area(_grid)


@st.cache_data(show_spinner=False, max_entries=32)
def load_plan(cache_key, params_key: str, _params: PlanParams, _grid, _area):
    return run_plan(_grid, _area, _params)


@st.cache_data(show_spinner=False, max_entries=64)
def overlay(cache_key, layer, _values, _meta, vmin, vmax, stops, transparent_below=None):
    raster, bounds = to_mercator_raster(_values, _meta)
    return colorize(raster, vmin, vmax, list(stops), alpha=0.78, transparent_below=transparent_below), bounds


# ---------------------------------------------------------------------------
# Map helpers
# ---------------------------------------------------------------------------
def base_map(bounds_latlon) -> folium.Map:
    lon_min, lat_min, lon_max, lat_max = bounds_latlon
    m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], tiles=None,
                   control_scale=True, prefer_canvas=True)
    folium.TileLayer(LIGHT_GRAY, attr=LIGHT_GRAY_ATTR, name="Light basemap", max_native_zoom=16, max_zoom=19,
                     show=True).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Streets", show=False).add_to(m)
    folium.TileLayer(SATELLITE, attr="Esri World Imagery", name="Satellite", show=False).add_to(m)
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    return m


def add_raster(m, rgba, bounds, name, show=True):
    folium.raster_layers.ImageOverlay(rgba, bounds=bounds, name=name, opacity=1.0, show=show,
                                      interactive=False, zindex=2).add_to(m)


LEGEND_CSS = """
<style>
  /* branca draws the colour bar inside <div class="legend leaflet-control">; the white card keeps
     the black ticks and caption readable over the dark satellite basemap */
  .legend.leaflet-control {
    background-color: rgba(255, 255, 255, 0.88);
    padding: 6px 10px;
    border-radius: 6px;
  }
  .legend.leaflet-control text.caption { font-weight: 600; }
</style>
"""

class LegendPolish(MacroElement):
    """
    Runs right after branca's legend script (it is added to the map after the colormap, so its script
    comes later in the same block). branca's caption is an SVG <text> with no x, so it sits at the left
    of the colour bar; the bar spans [0, width − 50] inside a group translated by 25 px, so the caption
    is centred on it. The white card is also set inline, in case the page <head> CSS is not carried
    into the streamlit-folium component.
    """

    _template = Template("""
        {% macro script(this, kwargs) %}
        document.querySelectorAll(".legend.leaflet-control").forEach(function (box) {
            box.style.backgroundColor = "rgba(255, 255, 255, 0.88)";
            box.style.padding = "6px 10px";
            box.style.borderRadius = "6px";
            box.querySelectorAll("svg").forEach(function (svg) {
                var w = parseFloat(svg.getAttribute("width")) || 450;
                svg.querySelectorAll("text.caption").forEach(function (t) {
                    t.setAttribute("x", (w - 50) / 2);
                    t.setAttribute("text-anchor", "middle");
                    t.style.fontWeight = "600";
                });
            });
        });
        {% endmacro %}
    """)


def legend(m, stops, vmin, vmax, caption):
    m.get_root().header.add_child(folium.Element(LEGEND_CSS))
    LinearColormap(stops, vmin=vmin, vmax=vmax, caption=caption).add_to(m)
    LegendPolish().add_to(m)


def bbox_from_drawing(drawing) -> tuple[float, float, float, float] | None:
    if not drawing or drawing.get("geometry", {}).get("type") != "Polygon":
        return None
    ring = drawing["geometry"]["coordinates"][0]
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return (round(min(lons), 5), round(min(lats), 5), round(max(lons), 5), round(max(lats), 5))


def fmt_ratio(a, b):
    return f"{a / b:.2f}×" if b > 0 else "—"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌳 CanopyAI")
st.sidebar.caption("Prescriptive tree planting for urban heat")

st.sidebar.header("Study area")
SOURCE_LABELS = {"Preloaded city": "Demo city", "Custom area": "Draw area"}
source = st.sidebar.radio("Source", list(SOURCE_LABELS), horizontal=True, format_func=SOURCE_LABELS.get,
                          help="Demo cities are cached and open instantly. "
                               "Drawn areas are downloaded from Google Earth Engine.")

bbox = years = months = None
area_label = ""
if source == "Preloaded city":
    names = [c.name for c in CITIES]
    labels = {c.name: (c.name if c.is_cached else f"{c.name} (not cached)") for c in CITIES}
    city = CITY_BY_NAME[st.sidebar.selectbox("City", names, format_func=labels.get)]
    st.sidebar.caption(city.note)
    if city.is_cached:
        bbox, years, months, area_label = city.bbox, city.years, city.months, city.name
    else:
        st.sidebar.warning("This city is not cached yet. Run `python scripts/precache_cities.py`.")
else:
    st.sidebar.caption(f"Draw a rectangle on the map · max {MAX_CUSTOM_AREA_KM2:.0f} km²")
    season = st.sidebar.select_slider(
        "Season", options=list(range(1, 13)), value=(6, 9),
        format_func=lambda m: ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1],
        help="Months used for the median Landsat composite. Pick a dry, low-cloud season.",
    )
    custom = st.session_state.get("custom_bbox")
    if custom and custom["months"] == tuple(season):
        bbox, years, months = custom["bbox"], (2022, 2025), custom["months"]
        area_label = f"Custom area {bbox}"

st.sidebar.header("Scenario")
budget = st.sidebar.slider("Trees", 500, 20000, 5000, step=500,
                           help="Number of new trees to allocate (N).")
canopy = st.sidebar.slider("Crown area · m²", 20, 120, 50, step=5,
                           help="Mature canopy area per tree. 50 m² ≈ an 8 m crown diameter.")
OBJECTIVE_LABELS = {"Total cooling (Σ ΔLST)": "Total cooling", "Population exposure (WorldPop)": "People-weighted"}
objective_label = st.sidebar.radio(
    "Objective", list(OBJECTIVE_LABELS), horizontal=True, format_func=OBJECTIVE_LABELS.get,
    help="Total cooling sums the predicted temperature drop (Σ ΔLST) over every 30 m cell. "
         "People-weighted multiplies each cell by its WorldPop residents (person·°C).",
)
objective = "population" if objective_label.startswith("Population") else "cooling"
with st.sidebar.expander("Advanced"):
    street_share = st.slider("Plantable built-up share", 0.0, 0.4, 0.15, step=0.05,
                             help="Sidewalks, medians and parking lots inside 'built-up' cells where a tree can fit.")
    step = st.slider("Trees per move", 1, 10, 5, help="Granularity of the lazy-greedy optimizer.")
    pool = st.select_slider("Candidate pool", options=[2000, 4000, 6000, 8000, 12000], value=6000,
                            help="Half chosen by a linear screen, half sampled at random; larger is slower but better.")
params = PlanParams(budget=budget, canopy_area_m2=float(canopy), objective=objective, street_share=street_share,
                    step=step, pool=pool // 2, explore=pool // 2)
run_clicked = st.sidebar.button("Run optimizer", type="primary", width="stretch", disabled=bbox is None)

st.sidebar.divider()
st.sidebar.caption(
    "Landsat 8/9 · ESA WorldCover · SRTM · WorldPop 2025 via Google Earth Engine  \n"
    "Monotonic gradient boosting · lazy-greedy allocation"
)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="eyebrow">Urban heat · Prescriptive planning</p>', unsafe_allow_html=True)
st.title("Where should a city plant its next trees?")
st.markdown(
    '<p class="lede">CanopyAI predicts <b>land-surface temperature</b> at 30 m from satellite data, then '
    "places a tree budget where it cools the most with a <b>lazy-greedy optimizer</b>, and benchmarks the "
    "plan against random planting and a naive <i>hottest-first</i> rule.</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Custom area picker
# ---------------------------------------------------------------------------
if source == "Custom area":
    st.subheader("Select a study area")
    prev = st.session_state.get("custom_bbox", {}).get("bbox")
    center = [((prev[1] + prev[3]) / 2), ((prev[0] + prev[2]) / 2)] if prev else [-23.57, -46.66]
    pick = folium.Map(location=center, zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(LIGHT_GRAY, attr=LIGHT_GRAY_ATTR, name="Light basemap", max_native_zoom=16,
                     max_zoom=19).add_to(pick)
    Draw(draw_options={"rectangle": {"shapeOptions": {"color": "#1b7837"}}, "polyline": False, "polygon": False,
                       "circle": False, "marker": False, "circlemarker": False},
         edit_options={"edit": False}).add_to(pick)
    if prev:
        folium.Rectangle([[prev[1], prev[0]], [prev[3], prev[2]]], color="#1b7837", fill=False,
                         tooltip="Loaded area").add_to(pick)
    out = st_folium(pick, height=420, use_container_width=True, key="picker",
                    returned_objects=["last_active_drawing"])
    drawn = bbox_from_drawing(out.get("last_active_drawing") if out else None)
    c1, c2 = st.columns([3, 1])
    if drawn:
        area_km2 = bbox_area_km2(drawn)
        too_big = area_km2 > MAX_CUSTOM_AREA_KM2
        cached = find_cached(drawn, ExtractParams(months=tuple(season))) is not None
        can_download = bool(os.getenv("GEE_PROJECT_ID", "").strip()) and has_ee_credentials() and ALLOW_CUSTOM
        used = st.session_state.get("downloads", 0)
        msg = f"Selected **{area_km2:.1f} km²** · bbox `{drawn}`"
        blocked = too_big
        if too_big:
            c1.error(f"{msg} — larger than the {MAX_CUSTOM_AREA_KM2:.0f} km² limit that protects the "
                     "Earth Engine quota. Draw a smaller rectangle.")
        elif cached:
            c1.success(f"{msg} — already cached, no Earth Engine call needed.")
        elif not can_download:
            blocked = True
            c1.warning(f"{msg} — downloading new areas is not enabled on this deployment "
                       "(Earth Engine credentials not configured). The preloaded cities still work.")
        elif used >= MAX_DOWNLOADS:
            blocked = True
            c1.warning(f"{msg} — this session already used its {MAX_DOWNLOADS} new-area downloads.")
        else:
            c1.info(f"{msg} — about {area_km2 / 100 * 20:.0f}–{area_km2 / 100 * 40:.0f} s to download "
                    f"({MAX_DOWNLOADS - used} download(s) left in this session).")
        if c2.button("Load this area", type="primary", width="stretch", disabled=blocked):
            with st.spinner("Downloading the 30 m grid from Google Earth Engine…" if not cached else "Loading…"):
                try:
                    load_grid(drawn, (2022, 2025), tuple(season))
                except Exception as exc:  # GEE errors must not crash the app
                    st.error(f"Earth Engine request failed: {exc}")
                else:
                    if not cached:
                        st.session_state["downloads"] = used + 1
                    st.session_state["custom_bbox"] = {"bbox": drawn, "months": tuple(season)}
                    st.rerun()
    else:
        c1.info("Use the ▭ tool on the left of the map to draw a rectangle.")

if bbox is None:
    if source == "Preloaded city":
        st.info("Pick a cached city in the sidebar.")
    st.stop()

# ---------------------------------------------------------------------------
# Area data
# ---------------------------------------------------------------------------
with st.spinner("Loading area and fitting the LST model…"):
    grid = load_grid(bbox, years, months)
    key = grid.meta["cache_key"]
    try:
        area = load_area_model(key, grid)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

df = area.feat
land = ~df["is_water"].to_numpy()
st.subheader(area_label)
a1, a2, a3, a4, a5 = st.columns(5)
a1.metric("Area", f"{grid.shape[0] * grid.shape[1] * grid.scale_m ** 2 / 1e6:.0f} km²")
a2.metric("30 m cells", f"{grid.shape[0] * grid.shape[1]:,}")
a3.metric("Median LST", f"{area.lst_median:.1f} °C",
          help=f"Median {months[0]:02d}–{months[1]:02d} land-surface temperature, {years[0]}–{years[1]}, "
               f"{grid.meta['n_scenes']} Landsat scenes.")
a4.metric("Tree cover", f"{df.loc[land, 'tree_frac'].mean():.0%}")
a5.metric("Residents (WorldPop)", f"{df['population'].sum():,.0f}" if "population" in df else "—")

# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------
if run_clicked:
    st.session_state["plan_request"] = (key, params)
requested = st.session_state.get("plan_request")
want_plan = (requested == (key, params)) or plan_cache_path(grid, params).exists()

result = None
if want_plan:
    fresh = not plan_cache_path(grid, params).exists()
    label = ("Optimizing… a new scenario takes about 1–2 minutes (lazy greedy + 2 baselines)."
             if fresh else "Loading cached plan…")
    with st.spinner(label):
        result = load_plan(key, repr(params), params, grid, area)

metric_key = "soma_ponderada" if objective == "population" else "soma_dT"
unit = "person·°C" if objective == "population" else "°C·cell"

if result:
    m = result["metrics"]
    g, r, h = (m[s][metric_key] for s in STRATEGIES)
    st.markdown("#### Optimized plan")
    b1, b2, b3, b4, b5 = st.columns(5)
    b1.metric("Objective (Lazy Greedy)", f"{g:,.0f} {unit}",
              help="Sum of predicted LST reduction over all 30 m cells"
                   + (", weighted by residents." if objective == "population" else "."))
    b2.metric("vs random planting", fmt_ratio(g, r), f"{(g / r - 1):+.0%}" if r > 0 else None)
    b3.metric("vs hottest-first", fmt_ratio(g, h), f"{(g / h - 1):+.0%}" if h > 0 else None)
    b4.metric("Cells planted", f"{m['Lazy Greedy']['celulas_plantadas']:,}",
              help=f"{result['budget_used']:,} trees placed.")
    b5.metric("Peak local cooling", f"{m['Lazy Greedy']['dT_max']:.2f} °C")
    if result["budget_used"] < budget:
        st.warning(f"Only {result['budget_used']:,} trees fit in the plantable area "
                   f"(capacity {result['capacity_total']:,}).")
else:
    st.info("Adjust the scenario in the sidebar and click **Run optimizer**. "
            "New scenarios take about 1–2 minutes; the default scenario of each preloaded city is cached.")

# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------
bounds_ll = grid_latlon_bounds(grid.meta)
left, right = st.columns(2)

with left:
    st.markdown("##### Current LST anomaly")
    anom = df[TARGET].to_numpy().reshape(grid.shape)
    lim = float(np.nanpercentile(np.abs(anom[land.reshape(grid.shape)]), 98))
    rgba, bnds = overlay(key, "lst", np.where(land.reshape(grid.shape), anom, np.nan), grid.meta,
                         -lim, lim, tuple(LST_STOPS))
    m1 = base_map(bounds_ll)
    add_raster(m1, rgba, bnds, "LST anomaly (°C)")
    tree = df["tree_frac"].to_numpy().reshape(grid.shape)
    rgba_t, _ = overlay(key, "tree", tree, grid.meta, 0.0, 1.0, tuple(TREE_STOPS))
    add_raster(m1, rgba_t, bnds, "Tree cover (WorldCover)", show=False)
    legend(m1, LST_STOPS, -lim, lim, "LST anomaly vs. area median (°C)")
    folium.LayerControl(collapsed=True).add_to(m1)
    st_folium(m1, height=MAP_HEIGHT, use_container_width=True, returned_objects=[], key=f"lst_{key}")

with right:
    st.markdown("##### Prescriptive planting plan")
    m2 = base_map(bounds_ll)
    if result:
        cool = result["cooling_greedy"].reshape(grid.shape)
        positive = cool[cool > 0.005]
        cmax = float(np.percentile(positive, 99)) if positive.size else 0.1
        rgba_c, bnds = overlay(f"{key}|{params!r}", "cool", cool, grid.meta, 0.0, cmax, tuple(COOL_STOPS),
                               transparent_below=0.005)
        add_raster(m2, rgba_c, bnds, "Predicted ΔLST (°C cooler)")
        legend(m2, COOL_STOPS, 0.0, cmax, "Predicted cooling (°C)")
        table = plan_table(grid, area, result, float(canopy))
        pts = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [rec.lon, rec.lat]},
                 "properties": {"trees": int(rec.trees), "cooling": f"{rec.predicted_cooling_c:.2f} °C",
                                "cover": f"{rec.tree_cover_before:.0%} → {min(rec.tree_cover_after, 1):.0%}"}}
                for rec in table.itertuples()
            ],
        }
        tmax = max(int(table["trees"].max()), 1)
        folium.GeoJson(
            pts, name="Planting sites",
            marker=folium.CircleMarker(radius=3.5, weight=0.6, color="#ffffff", fill=True, fill_opacity=0.95),
            style_function=lambda f: {"fillColor": TREE_STOPS[1 + min(3, int(3 * f["properties"]["trees"] / tmax))]},
            tooltip=folium.GeoJsonTooltip(["trees", "cooling", "cover"],
                                          aliases=["Trees", "Cooling at cell", "Tree cover"]),
        ).add_to(m2)
    folium.LayerControl(collapsed=True).add_to(m2)
    st_folium(m2, height=MAP_HEIGHT, use_container_width=True, returned_objects=[],
              key=f"plan_{key}_{abs(hash(repr(params))) if result else 'empty'}")

# ---------------------------------------------------------------------------
# Details
# ---------------------------------------------------------------------------
tab_cmp, tab_plan, tab_method, tab_audit = st.tabs(
    ["📊 Strategy comparison", "🌱 Planting plan & export", "🧠 Method & assumptions", "🔍 Robustness audit"])

with tab_cmp:
    if not result:
        st.caption("Run the optimizer to compare strategies.")
    else:
        m = result["metrics"]
        comp = pd.DataFrame([
            {"Strategy": s, "Objective": m[s][metric_key],
             "Σ ΔLST (°C·cell)": m[s]["soma_dT"],
             "Mean ΔLST over area (m°C)": m[s]["dT_medio_mC"],
             "Peak cooling (°C)": m[s]["dT_max"],
             "Cells ≥ 0.1 °C cooler": m[s]["celulas_dT_ge_0.1"],
             "Cells planted": m[s]["celulas_plantadas"]}
            for s in STRATEGIES
        ])
        c1, c2 = st.columns([1, 1])
        bars = alt.Chart(comp).mark_bar(cornerRadiusEnd=4).encode(
            x=alt.X("Objective:Q", title=f"Objective ({unit})"),
            y=alt.Y("Strategy:N", sort=list(STRATEGIES), title=None),
            color=alt.Color("Strategy:N", scale=alt.Scale(domain=list(STRATEGY_COLORS),
                                                          range=list(STRATEGY_COLORS.values())), legend=None),
            tooltip=["Strategy", alt.Tooltip("Objective:Q", format=",.0f")],
        ).properties(height=200, title="Total objective by strategy")
        c1.altair_chart(bars.configure(font="Inter").configure_view(stroke=None), width="stretch")

        hist = pd.DataFrame(result["history"], columns=["Trees", "Objective"])
        lines = [hist.assign(Strategy="Lazy Greedy")]
        n = result["budget_used"]
        for s in ("Random", "Hottest first"):
            lines.append(pd.DataFrame({"Trees": [0, n], "Objective": [0, m[s][metric_key]], "Strategy": s}))
        curve = alt.Chart(pd.concat(lines)).mark_line().encode(
            x=alt.X("Trees:Q", title="Trees planted"),
            y=alt.Y("Objective:Q", title=f"Cumulative objective ({unit})"),
            color=alt.Color("Strategy:N", scale=alt.Scale(domain=list(STRATEGY_COLORS),
                                                          range=list(STRATEGY_COLORS.values()))),
            strokeDash=alt.condition(alt.datum.Strategy == "Lazy Greedy", alt.value([1, 0]), alt.value([5, 4])),
        ).properties(height=200, title="Diminishing returns (baselines shown as their average rate)")
        c2.altair_chart(curve.configure(font="Inter").configure_view(stroke=None), width="stretch")
        st.dataframe(comp.style.format({"Objective": "{:,.0f}", "Σ ΔLST (°C·cell)": "{:,.0f}",
                                        "Mean ΔLST over area (m°C)": "{:.1f}", "Peak cooling (°C)": "{:.2f}",
                                        "Cells ≥ 0.1 °C cooler": "{:,.0f}", "Cells planted": "{:,.0f}"}),
                     hide_index=True, width="stretch")
        rsd = m["Random"].get("soma_dT_sd", 0.0)
        st.caption(f"Random = mean of {params.n_random} draws (Σ ΔLST sd {rsd:,.0f}). "
                   f"Greedy evaluated {result['n_evaluations']:,} candidate moves; "
                   f"computed in {result['runtime_s']:.0f} s"
                   + (" (loaded from cache)." if result.get("from_cache") else "."))

with tab_plan:
    if not result:
        st.caption("Run the optimizer to get a planting plan.")
    else:
        table = plan_table(grid, area, result, float(canopy))
        st.markdown(f"**{len(table):,} cells · {int(table['trees'].sum()):,} trees** — sorted by trees per cell.")
        d1, d2, _ = st.columns([1, 1, 3])
        slug = "".join(ch if ch.isalnum() else "_" for ch in area_label.lower()).strip("_")[:40]
        d1.download_button("⬇ Download CSV", table.to_csv(index=False).encode(), f"canopyai_plan_{slug}.csv",
                           "text/csv", width="stretch")
        d2.download_button("⬇ Download GeoJSON", plan_geojson(table, grid).encode(),
                           f"canopyai_plan_{slug}.geojson", "application/geo+json", width="stretch")
        st.dataframe(table.drop(columns=["cell_id", "utm_x", "utm_y"]).head(500), hide_index=True,
                     width="stretch", height=360)
        st.caption("GeoJSON exports one 30 m square per cell (WGS84). Coordinates are cell centres in the CSV.")

with tab_method:
    st.markdown(
        f"""
**Data (30 m grid, {grid.meta['grid']['crs']})** — median {months[0]:02d}–{months[1]:02d} composite of
cloud-masked Landsat 8/9 Collection 2 Level 2 ({years[0]}–{years[1]}): LST from the thermal band, NDVI and
NDBI. ESA WorldCover v200 gives tree / built-up / water fractions per cell, SRTM the elevation and WorldPop
2025 the residents.

**Features** — besides the cell itself, mean NDVI and tree cover within 90, 150, 300 and 600 m
(normalized convolution that skips water and handles the area edges).

**Model** — a gradient-boosting regressor of the LST anomaly with *monotonic constraints*: more vegetation
(in the cell or around it) can never predict a warmer surface, and more built-up area never a cooler one.
Here it was fitted on {area.n_train:,} land cells (in-sample R² {area.r2_in_sample:.2f}). In the São Paulo
benchmark it reaches R² 0.44 under *spatial* block cross-validation (1.5 km blocks), against 0.49 under
random K-fold — the gap is the optimism caused by spatial autocorrelation.

**Intervention** — a tree adds its mature crown area ({canopy} m²) to a cell's tree cover, using open land
first and then up to {street_share:.0%} of the built-up land (sidewalks, medians). NDVI moves toward the
95th percentile of pure-canopy cells; neighbourhood features are recomputed.

**Optimizer** — lazy greedy (CELF) with a priority queue keyed on predicted cooling *per tree*; each move's
effect is evaluated exactly on all cells within 600 m. Baselines: random cells, and the hottest cells first.

**Limitations** — predictions are of *surface* temperature, not air temperature, for mature trees; the
model is correlational and learned from one season and area; NDBI is kept fixed after planting
(conservative); the plantable share of built-up land is an assumption you can change under *Advanced*.
"""
    )

with tab_audit:
    audit_path = ROOT / "reports" / "cross_model_audit.json"
    if not audit_path.exists():
        st.caption("Run `python scripts/audit_cross_model.py` to generate the cross-model audit.")
    else:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        st.markdown(
            "Does the optimizer find real cooling, or just exploit the step functions of its own tree-based "
            "model? The São Paulo plan (5,000 trees) optimized with the boosted model was re-scored by models "
            "it never saw."
        )
        g = audit["greedy_m2_vs_random"]
        tbl = audit["table"]
        names = {
            "M2": ("Boosted model used by the optimizer", "reference, not independent"),
            "M2-alt": ("Independent boosted model", "other hyper-parameters, other half of the data"),
            "M1": ("Ridge, unconstrained", "smooth, but collinearity flips signs: predicts some planting warms"),
            "M1-sinal": ("Linear with physical signs", "smooth and physically consistent (added post hoc)"),
        }
        st.dataframe(pd.DataFrame([
            {"Evaluator": names[k][0], "Lazy Greedy vs random": f"{v:.2f}×",
             "Random plan, Σ ΔLST (°C·cell)": f"{tbl[f'{k}|Aleatória']['soma_dT']:,.0f}", "Note": names[k][1]}
            for k, v in g.items() if k in names
        ]), hide_index=True, width="stretch")
        verdict = "passes" if audit["robust"] else "does **not** pass"
        st.markdown(
            f"Under the pre-registered criterion (≥ {audit['robust_threshold']}× random in both the Ridge and the "
            f"independent boosted model) the advantage {verdict}. The ranking of strategies is far more stable "
            "across models than the *size* of the cooling, which varies several-fold — treat absolute ΔLST as "
            "uncertain. Full report: `reports/cross_model_audit.md`."
        )
