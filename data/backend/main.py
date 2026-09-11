from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import xarray as xr
from pathlib import Path
import math

app = FastAPI(title="OceanLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# DATA PATHS
# ============================================================

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR.parent

MODEL_PATH = DATA_DIR / "arabian_sea_test.nc"
ARGO_PATH = DATA_DIR / "argo" / "arabian_sea_argo.nc"

# ============================================================
# LOAD DATA
# ============================================================

ds = xr.open_dataset(MODEL_PATH)
argo_ds = xr.open_dataset(ARGO_PATH)
# //api endpoints 

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return {
        "message": "OceanLens API running",
        "model": "Copernicus GLORYS",
        "argo": "IFREMER Argo"
    }


# ============================================================
# MODEL METADATA
# ============================================================

@app.get("/metadata")
def metadata():
    return {
        "variables": list(ds.data_vars),
        "depths": [
            float(x)
            for x in ds.depth.values
        ],
        "times": [
            str(x.astype("datetime64[s]"))
            for x in ds.time.values
        ],
        "latitude_range": [
            float(ds.latitude.min()),
            float(ds.latitude.max())
        ],
        "longitude_range": [
            float(ds.longitude.min()),
            float(ds.longitude.max())
        ]
    }


# ============================================================
# MODEL 2D SLICE
# ============================================================

@app.get("/slice")
def get_slice(
    variable: str = "thetao",
    depth: float = 0.494,
    time_index: int = 0
):
    if variable not in ds.data_vars:
        return {
            "error": "Invalid variable"
        }

    selected = ds[variable].isel(
        time=time_index
    ).sel(
        depth=depth,
        method="nearest"
    )

    return {
        "variable": variable,
        "depth": float(selected.depth.values),
        "time": str(
            selected.time.values.astype("datetime64[s]")
        ),
        "latitude": [
            float(x)
            for x in selected.latitude.values
        ],
        "longitude": [
            float(x)
            for x in selected.longitude.values
        ],
        "values": selected.values.tolist()
    }


# ============================================================
# MODEL 3D VOLUME
# ============================================================

@app.get("/volume")
def get_volume(
    variable: str = "thetao",
    time_index: int = 0
):
    if variable not in ds.data_vars:
        return {
            "error": "Invalid variable"
        }

    selected = ds[variable].isel(
        time=time_index
    )

    return {
        "variable": variable,
        "time": str(
            selected.time.values.astype("datetime64[s]")
        ),
        "depth": [
            float(x)
            for x in selected.depth.values
        ],
        "latitude": [
            float(x)
            for x in selected.latitude.values
        ],
        "longitude": [
            float(x)
            for x in selected.longitude.values
        ],
        "values": selected.values.tolist()
    }


# ============================================================
# ARGO OBSERVATIONS
# ============================================================

@app.get("/argo")
def get_argo():
    observations = []

    # Extract the Argo observations
    for i in range(argo_ds.sizes["N_POINTS"]):
        latitude = argo_ds["LATITUDE"].values[i]
        longitude = argo_ds["LONGITUDE"].values[i]
        time = argo_ds["TIME"].values[i]
        pressure = argo_ds["PRES"].values[i]
        temperature = argo_ds["TEMP"].values[i]
        salinity = argo_ds["PSAL"].values[i]

        try:
            if not (
                float(latitude) == float(latitude)
                and float(longitude) == float(longitude)
                and float(pressure) == float(pressure)
                and float(temperature) == float(temperature)
                and float(salinity) == float(salinity)
            ):
                continue
        except (TypeError, ValueError):
            continue

        observations.append({
            "latitude": float(latitude),
            "longitude": float(longitude),
            "time": str(
                time.astype("datetime64[s]")
            ),
            "pressure": float(pressure),
            "temperature": float(temperature),
            "salinity": float(salinity),
            "platform": int(
                argo_ds["PLATFORM_NUMBER"].values[i]
            ),
            "cycle": int(
                argo_ds["CYCLE_NUMBER"].values[i]
            )
        })

    return {
        "count": len(observations),
        "observations": observations
    }


# ============================================================
# FAST POINT PROFILE
# ============================================================
#
# This endpoint is specifically for the normal globe.
#
# Instead of downloading four complete 3D volumes to the browser
# when the user clicks one point, it extracts ONLY one vertical
# model column here on the server.
#
# Returned:
#   - 26 real Copernicus depth values
#   - temperature profile
#   - salinity profile
#   - current-speed profile
#   - nearest real Argo profile, if available
#   - model-vs-Argo MAE/RMSE
#
# This dramatically reduces the amount of data transferred on
# every click while keeping the plotted values unchanged.
# ============================================================

def _finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _distance_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two geographic coordinates."""
    r = 6371.0

    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)
    dlat = lat2 - lat1
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    return 2 * r * math.asin(math.sqrt(a))


def _interpolate_profile(profile, pressure, field):
    """
    Linearly interpolate a real Argo variable to a model depth.

    Returns None outside the observed pressure range or when the
    required observations are invalid.
    """
    if not profile:
        return None

    if pressure < profile[0]["pressure"]:
        return None

    if pressure > profile[-1]["pressure"]:
        return None

    for i in range(1, len(profile)):
        p0 = profile[i - 1]["pressure"]
        p1 = profile[i]["pressure"]

        if pressure <= p1:
            v0 = profile[i - 1][field]
            v1 = profile[i][field]

            if not _finite(v0) or not _finite(v1):
                return None

            if p1 == p0:
                return float(v1)

            fraction = (pressure - p0) / (p1 - p0)

            return float(
                v0 + fraction * (v1 - v0)
            )

    return None


def _error_stats(model_values, observation_values):
    pairs = []

    for model, observation in zip(
        model_values,
        observation_values
    ):
        if (
            model is not None
            and observation is not None
            and _finite(model)
            and _finite(observation)
        ):
            pairs.append(
                (
                    float(model),
                    float(observation)
                )
            )

    if not pairs:
        return {
            "mae": None,
            "rmse": None,
            "count": 0
        }

    errors = [
        model - observation
        for model, observation in pairs
    ]

    mae = sum(abs(e) for e in errors) / len(errors)
    rmse = math.sqrt(
        sum(e * e for e in errors) / len(errors)
    )

    return {
        "mae": mae,
        "rmse": rmse,
        "count": len(pairs)
    }


@app.get("/point-profile")
def get_point_profile(
    latitude: float,
    longitude: float,
    time_index: int = 0
):
    # --------------------------------------------------------
    # Validate requested model timestep
    # --------------------------------------------------------

    if time_index < 0 or time_index >= ds.sizes["time"]:
        return {
            "error": "Invalid time_index"
        }

    # --------------------------------------------------------
    # Find nearest REAL Copernicus model grid cell
    # --------------------------------------------------------

    lat_index = int(
        abs(ds.latitude - latitude).argmin().values
    )

    lon_index = int(
        abs(ds.longitude - longitude).argmin().values
    )

    model_lat = float(ds.latitude.values[lat_index])
    model_lon = float(ds.longitude.values[lon_index])

    # --------------------------------------------------------
    # Extract ONLY the selected vertical columns
    # --------------------------------------------------------

    theta_column = (
        ds["thetao"]
        .isel(
            time=time_index,
            latitude=lat_index,
            longitude=lon_index
        )
        .values
    )

    salinity_column = (
        ds["so"]
        .isel(
            time=time_index,
            latitude=lat_index,
            longitude=lon_index
        )
        .values
    )

    u_column = (
        ds["uo"]
        .isel(
            time=time_index,
            latitude=lat_index,
            longitude=lon_index
        )
        .values
    )

    v_column = (
        ds["vo"]
        .isel(
            time=time_index,
            latitude=lat_index,
            longitude=lon_index
        )
        .values
    )

    depths = [
        float(x)
        for x in ds.depth.values
    ]

    temperature = [
        float(v) if _finite(v) else None
        for v in theta_column
    ]

    salinity = [
        float(v) if _finite(v) else None
        for v in salinity_column
    ]

    current_speed = []

    for u, v in zip(u_column, v_column):
        if _finite(u) and _finite(v):
            current_speed.append(
                math.hypot(float(u), float(v))
            )
        else:
            current_speed.append(None)

    model_time = str(
        ds.time.values[time_index].astype(
            "datetime64[s]"
        )
    )

    # --------------------------------------------------------
    # Find the nearest REAL Argo profile.
    #
    # We use the existing IFREMER Argo NetCDF already loaded by
    # OceanLens. Profiles are grouped by platform + cycle.
    # --------------------------------------------------------

    argo_groups = {}

    for i in range(argo_ds.sizes["N_POINTS"]):
        try:
            lat = float(argo_ds["LATITUDE"].values[i])
            lon = float(argo_ds["LONGITUDE"].values[i])
            pressure = float(argo_ds["PRES"].values[i])
            temp = float(argo_ds["TEMP"].values[i])
            psal = float(argo_ds["PSAL"].values[i])
            platform = int(
                argo_ds["PLATFORM_NUMBER"].values[i]
            )
            cycle = int(
                argo_ds["CYCLE_NUMBER"].values[i]
            )
            obs_time = str(
                argo_ds["TIME"].values[i].astype(
                    "datetime64[s]"
                )
            )
        except (TypeError, ValueError, OverflowError):
            continue

        if not all([
            _finite(lat),
            _finite(lon),
            _finite(pressure),
            _finite(temp),
            _finite(psal),
        ]):
            continue

        key = (platform, cycle)

        argo_groups.setdefault(key, []).append({
            "latitude": lat,
            "longitude": lon,
            "pressure": pressure,
            "temperature": temp,
            "salinity": psal,
            "time": obs_time,
            "platform": platform,
            "cycle": cycle,
        })

    nearest_profile = None
    nearest_score = float("inf")

    model_time_value = ds.time.values[time_index]

    for profile in argo_groups.values():
        if not profile:
            continue

        representative = profile[0]

        spatial_km = _distance_km(
            model_lat,
            model_lon,
            representative["latitude"],
            representative["longitude"]
        )

        try:
            obs_time_value = xr.coding.times.decode_cf_datetime(
                representative["time"],
                units="seconds since 1970-01-01"
            )
        except Exception:
            obs_time_value = None

      
        time_penalty = 0.0

        if obs_time_value is not None:
            try:
                delta_days = abs(
                    (
                        obs_time_value
                        - model_time_value
                    ).astype("timedelta64[s]").astype(float)
                ) / 86400.0

                time_penalty = delta_days * 1.5
            except Exception:
                time_penalty = 0.0

        score = spatial_km + time_penalty

        if score < nearest_score:
            nearest_score = score
            nearest_profile = sorted(
                profile,
                key=lambda x: x["pressure"]
            )

    # --------------------------------------------------------
    # Build model-vs-Argo comparison
    # --------------------------------------------------------

    argo_result = None

    if nearest_profile:
        argo_temperature = [
            _interpolate_profile(
                nearest_profile,
                depth,
                "temperature"
            )
            for depth in depths
        ]

        argo_salinity = [
            _interpolate_profile(
                nearest_profile,
                depth,
                "salinity"
            )
            for depth in depths
        ]

        temperature_stats = _error_stats(
            temperature,
            argo_temperature
        )

        salinity_stats = _error_stats(
            salinity,
            argo_salinity
        )

        representative = nearest_profile[0]

        argo_result = {
            "platform": representative["platform"],
            "cycle": representative["cycle"],
            "latitude": representative["latitude"],
            "longitude": representative["longitude"],
            "time": representative["time"],
            "distance_km": _distance_km(
                model_lat,
                model_lon,
                representative["latitude"],
                representative["longitude"]
            ),
            "temperature": argo_temperature,
            "salinity": argo_salinity,
            "temperature_stats": temperature_stats,
            "salinity_stats": salinity_stats,
        }

    # --------------------------------------------------------
    # Return ONLY the small profile payload
    # --------------------------------------------------------

    return {
        "requested_location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "model": {
            "latitude": model_lat,
            "longitude": model_lon,
            "time": model_time,
            "depth": depths,
            "temperature": temperature,
            "salinity": salinity,
            "current_speed": current_speed,
        },
        "argo": argo_result
    }
