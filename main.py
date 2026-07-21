import io
import time
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------------------------------------------------------
# پیکربندی محدوده جغرافیایی ایران و شبکه نمونه‌برداری
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 24.5, 40.5
LON_MIN, LON_MAX = 43.5, 63.5
GRID_STEP = 1.0  # فاصله بین نقاط نمونه‌برداری (درجه) - هرچه کمتر، نقشه دقیق‌تر ولی کندتر

IRAN_GEOJSON_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

_boundary_cache = {"data": None, "ts": 0}
_map_cache = {}  # key: (type, days) -> {"png": bytes, "ts": time}
CACHE_TTL = 3 * 3600  # ۳ ساعت - چون داده‌های مدل هر چند ساعت یک‌بار آپدیت می‌شوند


def get_iran_boundary():
    now = time.time()
    if _boundary_cache["data"] is None or now - _boundary_cache["ts"] > 24 * 3600:
        try:
            r = requests.get(IRAN_GEOJSON_URL, timeout=15)
            r.raise_for_status()
            world = r.json()
            iran_features = [
                f for f in world.get("features", [])
                if "iran" in str(f.get("properties", {}).get("name", "")).lower()
            ]
            if iran_features:
                _boundary_cache["data"] = {"type": "FeatureCollection", "features": iran_features}
                _boundary_cache["ts"] = now
        except Exception:
            pass  # اگر نشد، نقشه بدون مرز رسم می‌شود (بهتر از خطا دادن کامل است)
    return _boundary_cache["data"]


def build_grid():
    lats = np.arange(LAT_MIN, LAT_MAX + 0.001, GRID_STEP)
    lons = np.arange(LON_MIN, LON_MAX + 0.001, GRID_STEP)
    return [(round(float(la), 3), round(float(lo), 3)) for la in lats for lo in lons]


def fetch_weather_batch(points, days, var_names, resolution="daily"):
    """داده‌ها را به‌صورت دسته‌ای می‌گیرد. resolution: 'daily' یا 'hourly' (برای متغیرهایی مثل CAPE که فقط ساعتی موجودند)."""
    results = {}
    batch_size = 100
    var_param = ",".join(var_names) if isinstance(var_names, list) else var_names

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        lat_str = ",".join(str(p[0]) for p in batch)
        lon_str = ",".join(str(p[1]) for p in batch)
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            resolution: var_param,
            "forecast_days": days,
            "timezone": "auto",
        }
        resp = requests.get("https://api.open-meteo.com/v1/ecmwf", params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        # وقتی چند مکان درخواست می‌شود، خروجی یک آرایه است؛ برای یک مکان، یک آبجکت است
        if isinstance(data, list):
            for loc_data, pt in zip(data, batch):
                results[pt] = loc_data
        else:
            results[batch[0]] = data

    return results


def precip_colormap():
    # شبیه‌سازی جدول رنگی معمول نقشه‌های بارش هواشناسی (سبز/آبی کم تا بنفش/قرمز زیاد)
    colors = [
        (1.00, 1.00, 1.00),  # بدون بارش - سفید
        (0.65, 0.85, 1.00),
        (0.30, 0.60, 1.00),
        (0.10, 0.80, 0.30),
        (1.00, 1.00, 0.30),
        (1.00, 0.65, 0.00),
        (0.90, 0.10, 0.10),
        (0.60, 0.00, 0.60),
    ]
    return LinearSegmentedColormap.from_list("precip", colors, N=256)


def cape_colormap():
    # جدول رنگی معمول شاخص CAPE: سبز کم تا بنفش/سیاه برای ناپایداری شدید
    colors = [
        (1.00, 1.00, 1.00),  # بدون ناپایداری - سفید
        (0.70, 0.90, 0.70),
        (0.30, 0.75, 0.30),
        (1.00, 1.00, 0.30),
        (1.00, 0.65, 0.00),
        (0.90, 0.10, 0.10),
        (0.60, 0.00, 0.60),
        (0.20, 0.00, 0.20),
    ]
    return LinearSegmentedColormap.from_list("cape", colors, N=256)


MAJOR_CITIES = [
    ("Tehran", 35.6892, 51.3890), ("Mashhad", 36.2970, 59.6060), ("Isfahan", 32.6546, 51.6680),
    ("Karaj", 35.8400, 50.9391), ("Shiraz", 29.5918, 52.5837), ("Tabriz", 38.0800, 46.2919),
    ("Qom", 34.6401, 50.8764), ("Ahvaz", 31.3183, 48.6706), ("Kermanshah", 34.3142, 47.0650),
    ("Urmia", 37.5527, 45.0761), ("Rasht", 37.2808, 49.5832), ("Zahedan", 29.4963, 60.8629),
    ("Kerman", 30.2839, 57.0834), ("Arak", 34.0917, 49.6892), ("Yazd", 31.8974, 54.3569),
    ("Ardabil", 38.2498, 48.2933), ("Bandar Abbas", 27.1865, 56.2808), ("Sanandaj", 35.3219, 46.9862),
    ("Zanjan", 36.6736, 48.4787), ("Qazvin", 36.2688, 50.0041), ("Khorramabad", 33.4878, 48.3558),
    ("Gorgan", 36.8427, 54.4400), ("Sari", 36.5633, 53.0601), ("Bushehr", 28.9234, 50.8203),
    ("Ilam", 33.6374, 46.4227), ("Birjand", 32.8663, 59.2211), ("Shahrekord", 32.3256, 50.8644),
    ("Semnan", 35.5729, 53.3971), ("Yasuj", 30.6683, 51.5878), ("Bojnurd", 37.4747, 57.3291),
]


def render_city_temp_map(days):
    """نقشه دما به سبک شبکه‌های هواشناسی - دایره زرد با عدد دما رو هر شهر اصلی."""
    lat_str = ",".join(str(c[1]) for c in MAJOR_CITIES)
    lon_str = ",".join(str(c[2]) for c in MAJOR_CITIES)
    params = {
        "latitude": lat_str,
        "longitude": lon_str,
        "daily": "temperature_2m_max",
        "forecast_days": days,
        "timezone": "auto",
    }
    resp = requests.get("https://api.open-meteo.com/v1/ecmwf", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    city_data = data if isinstance(data, list) else [data]

    fig, ax = plt.subplots(figsize=(10, 12), facecolor="#aee0f5")
    ax.set_facecolor("#aee0f5")  # آبی ملایم برای دریاها/پس‌زمینه

    boundary = get_iran_boundary()
    if boundary:
        for feature in boundary["features"]:
            geom = feature["geometry"]
            polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                # اولین حلقه، مرز بیرونی است؛ با رنگ سبز روشن پر می‌شود
                outer_ring = poly[0]
                xs = [c[0] for c in outer_ring]
                ys = [c[1] for c in outer_ring]
                ax.fill(xs, ys, color="#a8d9a0", zorder=1)
                ax.plot(xs, ys, color="#4a4a4a", linewidth=1.2, zorder=2)

    # برچسب دریاها (انگلیسی، چون فونت فارسی روی سرور نصب نیست)
    ax.text(50.5, 38.8, "Caspian Sea", fontsize=10, color="#1a4d6d", style="italic", ha="center")
    ax.text(52.0, 26.8, "Persian Gulf", fontsize=10, color="#1a4d6d", style="italic", ha="center")
    ax.text(59.0, 25.3, "Gulf of Oman", fontsize=9, color="#1a4d6d", style="italic", ha="center")

    for idx, (name, lat, lon) in enumerate(MAJOR_CITIES):
        d = city_data[idx] if idx < len(city_data) else None
        temp_val = None
        if d and "daily" in d and d["daily"].get("temperature_2m_max"):
            vals = [v for v in d["daily"]["temperature_2m_max"] if v is not None]
            if vals:
                temp_val = round(max(vals))
        if temp_val is None:
            continue
        ax.scatter([lon], [lat], s=550, color="#ffcc00", edgecolor="#e08900", linewidth=1.5, zorder=3)
        ax.text(lon, lat, str(temp_val), fontsize=10, fontweight="bold", color="#1a1a4d",
                 ha="center", va="center", zorder=4)

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(f"Max Temperature Forecast - Iran (Next {days} Days, °C)", fontsize=14, pad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="#aee0f5")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/debug-boundary")
def debug_boundary():
    b = get_iran_boundary()
    if not b:
        return JSONResponse({"found": False})
    return JSONResponse({"found": True, "feature_count": len(b["features"]), "names": [f["properties"].get("name") for f in b["features"]]})


@app.get("/map")
def generate_map(
    type: str = Query(..., pattern="^(precip|temp|cape)$"),
    days: int = Query(3, ge=1, le=10),
):
    cache_key = (type, days)
    cached = _map_cache.get(cache_key)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        return Response(content=cached["png"], media_type="image/png")

    if type == "temp":
        try:
            png_bytes = render_city_temp_map(days)
        except requests.RequestException as e:
            raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {e}")
        _map_cache[cache_key] = {"png": png_bytes, "ts": time.time()}
        return Response(content=png_bytes, media_type="image/png")

    points = build_grid()

    try:
        if type == "precip":
            raw = fetch_weather_batch(points, days, "precipitation_sum", resolution="daily")
            values, valid_points = [], []
            for pt in points:
                d = raw.get(pt)
                if d and "daily" in d and d["daily"].get("precipitation_sum"):
                    total = sum(v for v in d["daily"]["precipitation_sum"] if v is not None)
                    values.append(total)
                    valid_points.append(pt)
            cmap = precip_colormap()
            vmax = max(max(values) if values else 1, 10)
            levels = np.linspace(0, vmax, 15)
            title = f"Total Precipitation - {days} Day Forecast (mm)"
            unit = "mm"
        elif type == "cape":
            # فقط ۳ روز اول را برای CAPE در نظر می‌گیریم چون این شاخص برای پیش‌بینی کوتاه‌مدت معنادار است
            cape_days = min(days, 3)
            raw = fetch_weather_batch(points, cape_days, "cape", resolution="hourly")
            values, valid_points = [], []
            for pt in points:
                d = raw.get(pt)
                if d and "hourly" in d and d["hourly"].get("cape"):
                    vals = [v for v in d["hourly"]["cape"] if v is not None]
                    if vals:
                        values.append(max(vals))
                        valid_points.append(pt)
            cmap = cape_colormap()
            levels = np.array([0, 250, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000])
            title = f"Max CAPE (Convective Instability) - {cape_days} Day Forecast (J/kg)"
            unit = "J/kg"
        else:
            raw = fetch_weather_batch(points, days, ["temperature_2m_max"], resolution="daily")
            values, valid_points = [], []
            for pt in points:
                d = raw.get(pt)
                if d and "daily" in d and d["daily"].get("temperature_2m_max"):
                    vals = [v for v in d["daily"]["temperature_2m_max"] if v is not None]
                    if vals:
                        values.append(max(vals))
                        valid_points.append(pt)
            cmap = "turbo"
            levels = np.linspace(-10, 50, 25)
            title = f"Max 2m Temperature - {days} Day Forecast (°C)"
            unit = "°C"
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Weather data fetch failed: {e}")

    if len(valid_points) < 10:
        raise HTTPException(status_code=502, detail="Not enough valid data points returned from weather API")

    lats = np.array([p[0] for p in valid_points])
    lons = np.array([p[1] for p in valid_points])
    vals = np.array(values)

    grid_lon, grid_lat = np.mgrid[LON_MIN:LON_MAX:300j, LAT_MIN:LAT_MAX:300j]
    grid_vals = griddata((lons, lats), vals, (grid_lon, grid_lat), method="cubic")
    # نرم‌سازی سبک برای حذف حالت پلکانی ناشی از فاصله نقاط نمونه‌برداری
    nan_mask = np.isnan(grid_vals)
    filled = np.nan_to_num(grid_vals, nan=0.0)
    smoothed = gaussian_filter(filled, sigma=3)
    grid_vals = np.where(nan_mask, np.nan, smoothed)

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="black")
    ax.set_facecolor("black")
    cf = ax.contourf(grid_lon, grid_lat, grid_vals, levels=levels, cmap=cmap, extend="both")

    boundary = get_iran_boundary()
    if boundary:
        try:
            for feature in boundary["features"]:
                geom = feature["geometry"]
                polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
                for poly in polys:
                    for ring in poly:
                        xs = [c[0] for c in ring]
                        ys = [c[1] for c in ring]
                        ax.plot(xs, ys, color="black", linewidth=1.4)
        except Exception:
            pass

    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_title(title, fontsize=15, color="white", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")

    cbar = plt.colorbar(cf, ax=ax, shrink=0.75)
    cbar.set_label(unit, color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="white")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    buf.seek(0)
    png_bytes = buf.read()

    _map_cache[cache_key] = {"png": png_bytes, "ts": time.time()}
    return Response(content=png_bytes, media_type="image/png")
