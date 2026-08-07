```
The customer provides input Zarr files in a GCS bucket path via the
`--cns_data_dir` flag. 

#### How to Use

##### Flag: `--input_data_gcs_dir`

`--input_data_gcs_dir` flag for specifying custom input data:

```
--input_data_gcs_dir=gs://customer-bucket/path-to-input-data
```

##### GCS Bucket Setup

We recommend using the **same GCS bucket** for both input data and output
predictions.

The customer should place their input Zarr files under a path within their
existing output bucket, e.g.:

```
gs://customer-bucket/custom-inputs/   ← input Zarr files go here
gs://customer-bucket/outputs/         ← model predictions are written here
```

#### Input File Format

##### Zarr V3 Format Requirement

**All custom input Zarr files MUST be in Zarr V3 format.**

When creating custom input files, ensure they are saved in Zarr V3 format:

```python
import xarray as xa

# When creating input files, ensure they are saved in Zarr V3 format
dataset.to_zarr("path/to/output.zarr", zarr_format=3)
```

If V2 format files are provided as custom input, the job raises
an error when attempting to read them.

##### Example Dataset

Opening an example input Zarr file with xarray (init time 2026-03-18 00:00 UTC):

```python
>>> import xarray as xa
>>> ds = xa.open_zarr("2026_A1D03180000031800011.zarr")
>>> ds
<xarray.Dataset> Size: ...
Dimensions:  (isobaricInhPa: 13, latitude: 721, longitude: 1440)
Coordinates:
  * isobaricInhPa      (isobaricInhPa) float64 13 ...
  * latitude           (latitude) float64 721 ...
  * longitude          (longitude) float64 1440 ...
    number             int64 ...
    step               int64 ...
    surface            float64 ...
    time               int64 ...
    valid_time         int64 ...
Data variables: (13 total)
    msl                (latitude, longitude) float32 ...
    q                  (isobaricInhPa, latitude, longitude) float32 ...
    sst                (latitude, longitude) float32 ...
    t                  (isobaricInhPa, latitude, longitude) float32 ...
    t2m                (latitude, longitude) float32 ...
    u                  (isobaricInhPa, latitude, longitude) float32 ...
    u10                (latitude, longitude) float32 ...
    u100               (latitude, longitude) float32 ...
    v                  (isobaricInhPa, latitude, longitude) float32 ...
    v10                (latitude, longitude) float32 ...
    v100               (latitude, longitude) float32 ...
    w                  (isobaricInhPa, latitude, longitude) float32 ...
    z                  (isobaricInhPa, latitude, longitude) float32 ...
```

##### Variables

All data variables are `float32`.

Only the variables and levels needed for inference are listed here. Your zarr
may contain additional variables and levels.

###### Pressure Level Variables (3D: `isobaricInhPa` × `latitude` × `longitude`)

Shape: `[13, 721, 1440]`

These variables are used for input at the following pressure levels (hPa):
`50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000`

| Short Name | Description |
| :--- | :--- |
| `q` | specific humidity |
| `t` | temperature |
| `u` | u component of wind |
| `v` | v component of wind |
| `w` | vertical velocity |
| `z` | geopotential |

###### Surface Variables (2D: `latitude` × `longitude`)

Shape: `[721, 1440]`

| Short Name | Description |
| :--- | :--- |
| `msl` | mean sea level pressure |
| `sst` | sea surface temperature |
| `t2m` | 2m temperature |
| `u10` | 10m u component of wind |
| `u100` | 100m u component of wind |
| `v10` | 10m v component of wind |
| `v100` | 100m v component of wind |

###### Scalar Coordinates

| Name         | dtype     | Description                                |
|--------------|-----------|----------------------------------------------|
| `number`     | `int64`   | Ensemble member number                     |
| `step`       | `int64`   | Forecast step                              |
| `surface`    | `float64` | Surface level indicator                    |
| `time`       | `int64`   | Timestamp (units: days since init time, calendar: proleptic_gregorian) |
| `valid_time` | `int64`   | Validity time                              |

##### Dimension Coordinates

| Coordinate      | dtype     | Shape    | Description                    |
|-----------------|-----------|----------|--------------------------------|
| `isobaricInhPa` | `float64` | `[13]`   | 13 pressure levels in hPa      |
| `latitude`      | `float64` | `[721]`  | 0.25° resolution, 721 points   |
| `longitude`     | `float64` | `[1440]` | 0.25° resolution, 1440 points  |

##### Spatial Resolution

The data is at **0.25° resolution** globally:

- Latitude: 721 points (90°N to 90°S)
- Longitude: 1440 points (0° to 359.75°E)

#### File Naming and Structure

##### File Naming Convention

The inference binary expects input Zarr files to follow a specific file naming
convention. Each file corresponds to a specific forecast initialization time and
uses the following format:

```
<year>_<config><stream><MMDDHHMMMMDDHHMMEE>.zarr
```

Where:

- `<year>`: 4-digit year (e.g., `2025`)
- `<config>`: Data config name (default: `A1`)
- `<stream>`: `D` for 00/12 UTC init times, `S` for 06/18 UTC
- First `MMDDHHMM`: Month, day, hour, minute of the forecast init time
- Second `MMDDHHMM`: Month, day, hour, minute of the validity time
- `EE`: Experiment version (default: `1`)

The validity minute is hardcoded to `01`.

###### Examples

For a forecast initialized at **2026-03-18 00:00 UTC** using fc0:
```
2026_A1D03180000031800011.zarr
```

For a forecast initialized at **2026-03-17 06:00 UTC** using fc0:
```
2026_A1S03170600031706011.zarr
```

##### Success Sentinels

Each Zarr file directory must contain a `success` sentinel file to signal that
the data is complete and ready for reading:

```
gs://<bucket>/custom-inputs/
├── 2026_A1D03180000031800011.zarr/
│   ├── .zmetadata
│   ├── <array data>
│   └── success    ← required sentinel file
├── 2026_A1D03171200031712011.zarr/
│   ├── .zmetadata
│   ├── <array data>
│   └── success
```

The sentinel is a zero-byte file named `success` placed inside each `.zarr`
directory. The job will not proceed with inference until all required input
sentinels exist.

##### Number of Input Files

The model typically requires **2 input timestamps** (the forecast init time and
6 hours prior). For example, for a forecast initialized at 2026-03-18 12:00 UTC,
the binary expects:

1. `2026_A1D03181200031812011.zarr` (init time: 12:00 UTC)
2. `2026_A1D03180600031806011.zarr` (6 hours prior: 06:00 UTC)

#### Caveats and Limitations

1. **No fine-tuning guarantee**: The model is trained on ECMWF HRES
   data. Using custom inputs from a different source may degrade
   forecast quality, especially for features sensitive to the initial
   condition source.

2. **Variable completeness**: All variables listed above must be present in the
   custom input files. Missing variables will cause the inference to fail.

3. **Temporal alignment**: Custom input timestamps must align with valid HRES
   forecast hours (00, 06, 12, or 18 UTC).

4. **File format**: Only Zarr V3 format is accepted.