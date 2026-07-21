import io
import time
import requests
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from fastapi import FastAPI, Query, Response, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI()

# ---------------------------------------------------------------------------
# پیکربندی محدوده جغرافیایی ایران و شبکه نمونه‌برداری
# ---------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 24.5, 40.5
LON_MIN, LON_MAX = 43.5, 63.5
GRID_STEP = 1.0  # فاصله بین نقاط نمونه‌برداری (درجه) - هرچه کمتر، نقشه دقیق‌تر ولی کندتر

IRAN_GEOJSON_URL = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/iran.geojson"

_boundary_cache = {"data": None, "ts": 0}
_map_cache = {}  # key: (type, days) -> {"png": bytes, "ts": time}
CACHE_TTL = 3 * 3600  # ۳ ساعت - چون داده‌های مدل هر چند ساعت یک‌بار آپدیت می‌شوند


def get_iran_boundary():
    now = time.time()
    if _boundary_cache["data"] is None or now - _boundary_cache["ts"] > 24 * 3600:
        try:
            r = requests.get(IRAN_GEOJSON_URL, timeout=15)
            r.raise_for_status()
            _boundary_cache["data"] = r.json()
            _boundary_cache["ts"] = now
        except Exception:
            pass  # اگر نشد، نقشه بدون مرز رسم می‌شود (بهتر از خطا دادن کامل است)
    return _boundary_cache["data"]


def build_grid():
    lats = np.arange(LAT_MIN, LAT_MAX + 0.001, GRID_STEP)
    lons = np.arange(LON_MIN, LON_MAX + 0.001, GRID_STEP)
    return [(round(float(la), 3), round(float(lo), 3)) for la in lats for lo in lons]


def fetch_weather_batch(points, days, daily_vars):
    """داده‌ها را به‌صورت دسته‌ای (چون Open-Meteo تعداد مکان محدود در هر درخواست دارد) می‌گیرد."""
    results = {}
    batch_size = 100
    var_param = ",".join(daily_vars) if isinstance(daily_vars, list) else daily_vars

    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        lat_str = ",".join(str(p[0]) for p in batch)
        lon_str = ",".join(str(p[1]) for p in batch)
        params = {
            "latitude": lat_str,
            "longitude": lon_str,
            "daily": var_param,
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


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/map")
def generate_map(
    type: str = Query(..., pattern="^(precip|temp)$"),
    days: int = Query(3, ge=1, le=10),
):
    cache_key = (type, days)
    cached = _map_cache.get(cache_key)
    if cached and time.time() - cached["ts"] < CACHE_TTL:
        return Response(content=cached["png"], media_type="image/png")

    points = build_grid()

    try:
        if type == "precip":
            raw = fetch_weather_batch(points, days, "precipitation_sum")
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
        else:
            raw = fetch_weather_batch(points, days, ["temperature_2m_max"])
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
                        ax.plot(xs, ys, color="white", linewidth=1.2)
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
