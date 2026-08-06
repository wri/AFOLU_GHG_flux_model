# Land GHG Monitoring System

NOTE: This readme is preliminary. It will be refined prior to publication of the associated journal article. 

A global, pixel-based greenhouse gas (GHG) emissions and removal framework for the **A**griculture, **F**orestry, and **O**ther **L**and **U**se (AFOLU) sector. 
The current focus of the repository is Land Use, Land-Use Change, and Forestry (LULUCF): 
annual, spatially-explicit CO2/CH4/N2O fluxes from vegetation (aboveground/belowground carbon, deadwood, litter), 
mineral soil organic carbon (0–30cm depth), and organic soils, 
synthesized into combined LULUCF outputs. The LULUCF framework currently covers 2016-2024.

## Inputs

Inputs include a variety of global geospatial datasets, including annual vegetation height (=>5 m), annual land cover,
starting carbon densities by pool, soil carbon density in mineral soil, 
annual burned area, forest age, age-dependent growth curves for trees, and more. 
All of these are resampled to a common resolution of 0.00025x0.00025° (roughly 30x30 m at the equator). 

A few inputs are tabular: ratios for deadwood and litter carbon to aboveground carbon, 
emission factors for partially disturbed vegetation, etc. These are spatialized in the code. 

## Outputs

The framework outputs gross emissions, gross removals, and net flux by carbon pool and GHG at 0.00025x0.00025° resolution.
Fluxes are reported by framework component (vegetation, mineral soil, organic soil) and their sub-components. 
Outputs are created as geotifs (always 1x1° geotifs and sometimes 10x10° geotifs, both at 0.00025x0.00025° resolution), 
as well as zarrs. 
Geotifs are mostly for sharing/data distribution and QC, while zarrs are used for subsequent processing steps. 
Also, global resampled/aggregated geotifs at 0.04x0.04° (~4x4 km) resolution are created for static displays, 
like maps in publications and presentations. 

The repository also includes code to perform zonal statistics on the output datasets with various contextual layers. 

## Model versioning

Each framework component is versioned independently. 
Versions are defined in [`src/utilities/constants_and_names.py`](src/utilities/constants_and_names.py). 
That way, each component can be updated and refined on its own timeline. 


## Repository layout

```
src/
  LULUCF/                 All code related to LULUCF specifically
    scripts/
      preprocessing/       Build model inputs (Hansenized/standardized rasters, starting carbon pools,
                            starting forest age, burned area, mangrove extent)
      vegetation_model/    Vegetation (AGC/BGC/deadwood/litter) flux model
      mineral_soil_organic_carbon/
                           Mineral soil (SOC, 0-30cm) flux model
      postprocessing/      Combine vegetation + organic soil + mineral soil into LULUCF totals;
                            IPCC land-use/land-use-change classification (LUC/)
      zonal_statistics/    Zonal stats by shapefile/bounding box/IPCC land-use class;
                            figure & table notebooks; Postgres ingestion
      test_code/           Ad hoc QC scripts and Coiled cluster experiments (not a pytest suite)
    LULUCF_state_node_lookup_table.xlsx   Lookup table for land-state decision-tree node codes

  Agriculture/            Early-stage module (currently: COG conversion of cropland-emission
                            rasters only; no model pipeline yet)

  synthesis/              Combines LULUCF (and eventually other AFOLU sectors) into unified
                            annual outputs, numbered pipeline 1-3

  utilities/              Shared code: constants/paths, S3/raster/zarr helpers, logging,
                            Numba calculations, Coiled cluster management

conda_envs/               Conda environment definitions (see Environments below)
chunk_stats/              QC spreadsheets of per-chunk raster statistics from model runs
logs/                     Run logs (mostly gitignored; KEEP_definitive_runs kept for the record)
```

All raw inputs, intermediate products, and final outputs live on AWS S3; 
nothing data-related is included in this repository.

## Pipeline overview

There is no single script that runs the entire framework end-to-end, by design. Each framework component is being
developed (and will be updated) on its own timeline. Thus, there is a separate workflow for each component:

1. **Preprocessing** (`LULUCF/scripts/preprocessing/`) — "Hansenize" raw inputs to the model's native grid (0.00025°, WGS84, 10x10° tiles), 
and build starting-condition rasters: starting (2015) carbon pools, starting forest age, burned area, mangrove extent.
2. **Vegetation model** (`LULUCF/scripts/vegetation_model/`) and **mineral soil model** (`LULUCF/scripts/mineral_soil_organic_carbon/`) run independently, 
each computing gross/net stock changes and fluxes, aggregating to 10x10° and 0.04° display resolution, and producing zonal statistics.
3. **Organic soil model** — outputs are consumed from pre-existing S3 locations; the organic soil model's own processing code does not live in this repository.
4. **LULUCF postprocessing** (`LULUCF/scripts/postprocessing/`) sums vegetation + organic soil + mineral soil into combined LULUCF totals, and classifies IPCC land-use/land-use-change categories (`postprocessing/LUC/`).
5. **Synthesis** (`synthesis/scripts/`) reconciles the annual-resolution vegetation outputs with the 5-year-interval soil outputs into unified annual LULUCF products, and builds global display maps.
6. **Zonal statistics** (`LULUCF/scripts/zonal_statistics/`) summarizes results by shapefile, bounding box, or IPCC land-use class/transition into Parquet tables; some tables are loaded into PostgreSQL for downstream querying.

## Spatial and compute conventions

- **Tiles** are 10x10° (40,000 x 40,000 px); **chunks** are 1x1° (4,000 x 4,000 px) — 
the unit most processing loops iterate over. Native resolution is 0.00025° (~30m); display outputs are also produced at 0.04° (~4km).
- A global `fishnet_1x1deg` shapefile (joined to GADM 4.1 admin boundaries) is the standard unit for iterating full global runs.
- Dask (using Coiled-managed clusters) drives parallel processing; pixel-level calculations use Numba (`@jit(nopython=True)`) for speed.
- Raster I/O uses rasterio, rioxarray, and GDAL; large multi-temporal cubes are stored as zarr. 
Two zarr versions are used side-by-side (v3.1.3 for zonal-stats/flox clusters, v3.1.6 for everything else) because of a version incompatibility — 
see Environments below.
- Zonal statistics use flox against zarr cubes and export Parquet; some tables are ingested into PostgreSQL via `psycopg2`/`duckdb`/`pyarrow`.

## Environments

Environment YAMLs live in `conda_envs/`. Two environments are required and must stay separate due to a zarr version conflict:

- `AFOLU_not_zonal_stats_*.yml` — the general-purpose environment (zarr v3.1.6) for preprocessing, model runs, and postprocessing.
- `AFOLU_zonal_stats_*.yml` — a lightweight environment (zarr v3.1.3) for Coiled clusters running zonal statistics with flox.

`src/utilities/create_coiled_software_environment.py` documents the conda-lock workflow used to build and upload these environments to Coiled.

## Running scripts

Scripts are run as modules from the repository root, e.g.:

```bash
python -m src.LULUCF.scripts.vegetation_model.1_calculate_veg_fluxes <args>
```

Most scripts take `argparse` CLI arguments and include example invocations 
(local test, small Coiled test, shapefile-based run, full global run) in their module docstring — 
check the top of the script you want to run for exact usage. 
Cluster lifecycle (create/resize/terminate) is managed via `src/utilities/create_cluster.py`, `resize_cluster.py`, and `terminate_cluster.py`.

## Notes

- `src/Agriculture` is a placeholder module — it currently only contains a script to convert cropland-emission rasters to Cloud-Optimized GeoTIFFs. `src/synthesis` is designed to eventually combine LULUCF with Agriculture and other AFOLU sectors once those mature.
- `src/LULUCF/scripts/test_code/` is a collection of ad hoc QC and cluster-experiment scripts, not a formal test suite.
- `src/LULUCF/scripts/zonal_statistics/` contains many dated, exploratory notebooks under `old_notebooks/`, `old_zonal_stats_notebooks/`, and `old_figure_table_creation_notebooks/` documenting iterative development history rather than serving as reference docs.
