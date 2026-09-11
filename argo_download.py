import argopy

# Use IFREMER's Argo ERDDAP backend
argopy.set_options(src="erddap")

# Arabian Sea region
# [lon_min, lon_max, lat_min, lat_max, depth_min, depth_max, date_min, date_max]
region = [
    65, 70,
    10, 15,
    0, 2000,
    "2025-01-01",
    "2025-01-07"
]

print("Downloading Argo data...")
print("Region: 65–70°E, 10–15°N")
print("Period: 2025-01-01 → 2025-01-07")

fetcher = argopy.DataFetcher(
    mode="standard"
).region(region)

ds = fetcher.to_xarray()

print("\nDownload complete!")
print(ds)

ds.to_netcdf(
    "data/argo/arabian_sea_argo.nc"
)

print("\nSaved:")
print("data/argo/arabian_sea_argo.nc")