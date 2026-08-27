import logging
import posixpath
from datetime import datetime, timezone
import numpy as np

from src.scripts.utilities.tile_index import load_tile_ids_from_s3
from src.scripts.utilities import local_output_paths as lop

# ---------------------------------------------------
# 1. General Configuration
# ---------------------------------------------------

# ── version helpers ──────────────────────────────────────────────
model_version = "1.0.1"              # dotted string
model_version_underscore = model_version.replace(".", "_")   # "1_0_1"

s3_bucket_name = 'gfw2-data'

# Coiled workspace that clusters are created in
Coiled_workspace = "wri-land-research"

logger = logging.getLogger(__name__)
full_bucket_prefix = f"s3://{s3_bucket_name}"

climate_domain_dir = (
    f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/"
    "processed/fao_ecozones_bor_tem_tro/20190418/"
)
climate_domain_pattern = "fao_ecozones_bor_tem_tro_processed"

short_bucket_prefix = "gfw2-data"

project_dir = 'climate/AFOLU_flux_model/organic_soils'
raw_dir = posixpath.join(project_dir, 'inputs/raw')
processed_dir = posixpath.join(project_dir, 'inputs/processed')

date_date_range_pattern = r'_\d{4}(_\d{4})?'

# organic‑soils constants file
outputs_path = posixpath.join(
    full_bucket_prefix,                      # "s3://gfw2-data"
    project_dir,                             # "climate/AFOLU_flux_model/organic_soils"
    "outputs",
    f"version_{model_version_underscore}"    # "version_1_0_1"
)   # organic-soils model output prefix

drainage_outputs_path_mega_zarr = posixpath.join(outputs_path, "mega_zarr")

drainage_outputs_to_zarr = [
    "organic_soil",
    "combined_state",
    "burned_years_count",
    "drained_co2_Mg_CO2_ha_yr",
    "drained_n2o_Mg_CO2e_ha_yr",
    "drained_ch4_land_Mg_CO2e_ha_yr",
    "drained_ch4_ditch_Mg_CO2e_ha_yr",
    "drained_co2_offsite_Mg_CO2_ha_yr",
    "drained_total_Mg_CO2e_ha_yr",
    "burned_co2_Mg_CO2_ha_yr",
    "burned_co_Mg_CO_ha_yr",
    "burned_ch4_Mg_CO2e_ha_yr",
    "burned_total_Mg_CO2e_ha_yr",
]

# Optional component categorical state outputs.
# These are computed for combined-state packing but excluded from default outputs.
drainage_optional_state_outputs = [
    "drained_state",
    "burned_state",
]

drainage_output_dtypes = {
    "organic_soil": "uint8",
    "drained_soil": "uint32",
    "drained_state": "uint32",
    "burned_state": "uint32",
    "combined_state": "uint32",
    "burned_years_count": "uint32",
    "drained_co2_Mg_CO2_ha_yr": "float32",
    "drained_n2o_Mg_CO2e_ha_yr": "float32",
    "drained_ch4_land_Mg_CO2e_ha_yr": "float32",
    "drained_ch4_ditch_Mg_CO2e_ha_yr": "float32",
    "drained_co2_offsite_Mg_CO2_ha_yr": "float32",
    "drained_total_Mg_CO2e_ha_yr": "float32",
    "burned_co2_Mg_CO2_ha_yr": "float32",
    "burned_co_Mg_CO_ha_yr": "float32",
    "burned_ch4_Mg_CO2e_ha_yr": "float32",
    "burned_total_Mg_CO2e_ha_yr": "float32",
}


# Shapefile of global 1x1 degree chunks with GADM ISO codes
fishnet_1x1deg_uri = posixpath.join(
    full_bucket_prefix,
    'climate/AFOLU_flux_model/fishnet_1x1deg/20250429',
    'fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp'
)


# Global 10x10 degree tile index shapefile used to drive preprocessing and
# extent selection in organic-soils workflows.
tile_index_shapefile_prefix = posixpath.join(
    'climate/AFOLU_flux_model/fishnet_10x10deg/20250925',
    'fishnet_10x10deg_20251107_intersect'
)
tile_index_shapefile_uri = posixpath.join(
    full_bucket_prefix,
    f"{tile_index_shapefile_prefix}.shp"
)


local_log_path = f"{lop.local_output_path('logs')}/"

local_temp_dir = '/tmp'

today_date = datetime.now(timezone.utc).strftime('%Y%m%d')


# ---------------------------------------------------
# 2. Tile and Chunk Patterns
# ---------------------------------------------------

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
sample_tile_id = '{tile_id}'

full_raster_dims = 40000
local_chunk_stats_path = lop.chunk_stats_root()
pixel_area_dir = f"{full_bucket_prefix}/analyses/area_28m/"
pixel_area_pattern = "hanson_2013_area"

# conversion factor from square meters to hectares
m2_to_ha = 1e-4


# ---------------------------------------------------
# 3. Dataset File Patterns
# ---------------------------------------------------

# TODO add all peat patterns (and look at thresholds)
patterns = {
    'land_cover': "{tile_id}__lc_ipcc.tif",
    'peat': "{tile_id}.tif",
    'dadap': "{tile_id}__dadap_canals_distance.tif",
    'engert': "engert_{tile_id}.tif",
    'grip': "{tile_id}__grip_roads_distance.tif",
    'osm_roads': "{tile_id}__osm_roads_distance.tif",
    'osm_canals': "{tile_id}__osm_canals_distance.tif",
    'planted_forest_type': "{tile_id}__sdpt.tif",
    'extraction': "{tile_id}_extraction.tif",
    'climate_domain': f"{{tile_id}}_{climate_domain_pattern}.tif",
    'descals_type': "plantation_type_{tile_id}.tif",
    'mangrove_extent': "{tile_id}__gmw_mangrove_any_year.tif",
    'tidal_marsh': "{tile_id}.tif",
    'ogh': "{tile_id}_ogh_mask.tif",
    'burned_area_final': "{tile_id}_burned_area_final_{year}.tif"
}

# ---------------------------------------------------
# 4. Dataset Directories
# ---------------------------------------------------

dirs = {
    'land_cover': posixpath.join(
        full_bucket_prefix,
        'climate/AFOLU_flux_model/LULUCF/landcover/composite/{interval_type}',
        '{land_cover_version}',
        'raw'
    ),
    'peat': posixpath.join(full_bucket_prefix, raw_dir, 'soils/GFW_Global_Peatlands'),
    # Dadap canals are now consumed as a distance-to-canal surface (metres),
    # built by dadap_canal_distance.py, so the model treats Dadap like osm_canals
    # and the drainage-distance sensitivity sweep applies in SE Asia.
    'dadap': posixpath.join(full_bucket_prefix, processed_dir, 'dadap_density/distance/40000_pixels/20260602'),
    'engert': posixpath.join(full_bucket_prefix, processed_dir, 'engert_density/30m/20240925'),
    'grip': posixpath.join(full_bucket_prefix, processed_dir, f'grip_density/distance/{full_raster_dims}_pixels/20260513'),
    'osm_roads': posixpath.join(full_bucket_prefix, processed_dir, f'osm_roads_density/distance/{full_raster_dims}_pixels/20260513'),
    'osm_canals': posixpath.join(full_bucket_prefix, processed_dir, f'osm_canals_density/distance/{full_raster_dims}_pixels/20260513'),
    'planted_forest_type': posixpath.join(full_bucket_prefix, processed_dir, f'sdpt/{full_raster_dims}_pixels/20250531'),
    'extraction': posixpath.join(full_bucket_prefix, processed_dir, 'extraction/20260602'),
    'climate_domain': climate_domain_dir,
    'descals_type': posixpath.join(full_bucket_prefix, processed_dir, 'descals_plantation/extent/20241105'),
    'mangrove_extent': posixpath.join(full_bucket_prefix, processed_dir, 'mangrove_extent/hansen/20251112'),
    'tidal_marsh': posixpath.join(full_bucket_prefix, processed_dir, 'tidal_marshes/hansen/20251112'),
    'burned_area_final': posixpath.join(full_bucket_prefix, processed_dir, 'fires/{year}')
}

# directories for 30 m peat mask datasets
peat_mask_dirs = {
    # TODO add full gfw path here, change all references
    'gfw': dirs['peat'],
    'gpd': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/GPD/tiles/20251110',
    ),
    'peatmap': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/PEATMAP/tiles'),
    'peatml': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/PEATML/tiles'),
    'ogh': posixpath.join(full_bucket_prefix, processed_dir, 'peat_mask/OGH/tiles'),
    'ogh_unthresholded': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/OGH/tiles_unthresholded/20260513',
    ),
    'union_mask': posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'peat_mask/union/30m/tiles/20260513',
    ),
}

peat_dataset_choices = tuple(peat_mask_dirs.keys())

# ---------------------------------------------------
# 5. Classification and Conversion Constants
# ---------------------------------------------------

ipcc_codes = {
    'forest': 1,
    'cropland': 2,
    'settlement': 3,
    'wetland': 4,
    'grassland': 5,
    'otherland': 6
}

ecozone_codes = {
    'unknown': 0,
    'tropical': 1,
    'boreal': 2,
    'temperate': 3
}

# Mapping from FAO ecozone classes to simplified climate domain codes
climate_domain_remap = {
    0: ecozone_codes['unknown'],
    1: ecozone_codes['tropical'],
    2: ecozone_codes['boreal'],
    3: ecozone_codes['temperate'],
}

nutrient_status_codes = {
    'unknown': 0,
    'poor': 1,
    'rich': 2
}

plantation_type_codes = {
    'oil_palm': 1,
    'short_rotation': 2,
    'long_rotation': 3
}


combined_log = "combined_log"

# Do not apply this to organic-soil fire factors. Those factor tables store
# IPCC Table 2.6 fuel consumption values that already represent MB * Cf.
combustion_factor = np.float32(0.75)

# Global warming potentials (GWP) and emission conversions
gwp_ch4 = np.float32(27.0)
gwp_n2o = np.float32(273.0)
c_to_co2 = np.float32(3.67)
n2o_n_to_n2o = np.float32(1.571)

# ---------------------------------------------------
# 6. Time Interval Handling Constants
# ---------------------------------------------------

intervals_annual = "annual"
intervals_five_year = "five_year"

# Annual IPCC land-cover products currently have complete global coverage only
# for 2024. Keep the explicit roster so callers cannot infer that the sparse
# 2015-2023 prefixes are production-ready.
annual_land_cover_years = [2024]
annual_land_cover_start_year = annual_land_cover_years[0]

# Version mapping for land cover composites
land_cover_version_map = {
    intervals_annual: "v2",
    intervals_five_year: "v1",
}

# Date sub-folders for IPCC land cover composites by interval type
land_cover_ipcc_dates = {
    intervals_annual: "20250715",
    intervals_five_year: "20250710",
}

# End years backed by complete five-year land-cover composites. The final
# 2021-2024 inventory period uses the complete annual 2024 composite instead.
five_year_land_cover_years = [2005, 2010, 2015, 2020]

# The complete annual 2024 composite defines the expected 10x10-degree tile
# footprint for every land-cover input. This lets the model distinguish valid
# ocean/edge gaps from an accidentally sparse processing prefix.
land_cover_coverage_reference_period = (2024, 2024)

# Fingerprint for the complete processed land-cover model domain.
# Update only after an intentional input-footprint revision and verification.
land_cover_reference_tile_count = 266
land_cover_reference_tile_ids_sha256 = (
    "9b817f485f856ed9811a0f4293f762830311dd92a84674e398a9e81574d7922b"
)

# Convenience list of five year inventory periods. The first period now uses the 2005 land cover composite (2000–2004). The final period uses the 2024 land cover composite for the 2024 inventory year.
five_year_inventory_periods = [
    (2001, 2005),
    (2006, 2010),
    (2011, 2015),
    (2016, 2020),
    (2021, 2024),
]

# -------------------------------------------------------------------
# Convenience list of all 10x10 degree tile IDs used by the model
# -------------------------------------------------------------------
# Duplicated from ``preprocessing_constants.tile_id_list`` so that core
# model code can access the list without importing preprocessing modules.
_FALLBACK_TILE_ID_LIST = [
    '00N_000E', '00N_010E', '00N_020E', '00N_020W', '00N_030E', '00N_040E',
    '00N_040W', '00N_050E', '00N_050W', '00N_060W', '00N_070E', '00N_070W',
    '00N_080W', '00N_090E', '00N_090W', '00N_100E', '00N_100W', '00N_110E',
    '00N_120E', '00N_130E', '00N_140E', '00N_140W', '00N_150E', '00N_150W',
    '00N_160E', '00N_160W', '00N_170E', '00N_170W', '00N_180W', '10N_000E',
    '10N_010E', '10N_010W', '10N_020E', '10N_020W', '10N_030E', '10N_040E',
    '10N_050E', '10N_050W', '10N_060W', '10N_070E', '10N_070W', '10N_080E',
    '10N_080W', '10N_090E', '10N_090W', '10N_100E', '10N_100W', '10N_110E',
    '10N_110W', '10N_120E', '10N_120W', '10N_130E', '10N_140E', '10N_150E',
    '10N_160E', '10N_160W', '10N_170E', '10N_170W', '10N_180W', '10S_010E',
    '10S_010W', '10S_020E', '10S_030E', '10S_030W', '10S_040E', '10S_040W',
    '10S_050E', '10S_050W', '10S_060E', '10S_060W', '10S_070W', '10S_080W',
    '10S_090E', '10S_100E', '10S_110E', '10S_120E', '10S_130E', '10S_140E',
    '10S_140W', '10S_150E', '10S_150W', '10S_160E', '10S_160W', '10S_170E',
    '10S_170W', '10S_180W', '20N_000E', '20N_010E', '20N_010W', '20N_020E',
    '20N_020W', '20N_030E', '20N_030W', '20N_040E', '20N_050E', '20N_060E',
    '20N_060W', '20N_070E', '20N_070W', '20N_080E', '20N_080W', '20N_090E',
    '20N_090W', '20N_100E', '20N_100W', '20N_110E', '20N_110W', '20N_120E',
    '20N_120W', '20N_130E', '20N_140E', '20N_150E', '20N_160E', '20N_160W',
    '20N_170E', '20N_170W', '20N_180W', '20S_010E', '20S_020E', '20S_030E',
    '20S_030W', '20S_040E', '20S_040W', '20S_050E', '20S_050W', '20S_060E',
    '20S_060W', '20S_070W', '20S_080W', '20S_090W', '20S_110E', '20S_110W',
    '20S_120E', '20S_120W', '20S_130E', '20S_130W', '20S_140E', '20S_140W',
    '20S_150E', '20S_150W', '20S_160E', '20S_160W', '20S_170E', '20S_170W',
    '20S_180W', '30N_000E', '30N_010E', '30N_010W', '30N_020E', '30N_020W',
    '30N_030E', '30N_040E', '30N_050E', '30N_060E', '30N_070E', '30N_070W',
    '30N_080E', '30N_080W', '30N_090E', '30N_090W', '30N_100E', '30N_100W',
    '30N_110E', '30N_110W', '30N_120E', '30N_120W', '30N_130E', '30N_140E',
    '30N_150E', '30N_160E', '30N_160W', '30N_170W', '30N_180W', '30S_010E',
    '30S_010W', '30S_020E', '30S_020W', '30S_030E', '30S_050W', '30S_060W',
    '30S_070E', '30S_070W', '30S_080W', '30S_090W', '30S_110E', '30S_120E',
    '30S_130E', '30S_140E', '30S_150E', '30S_160E', '30S_170E', '30S_180W',
    '40N_000E', '40N_010E', '40N_010W', '40N_020E', '40N_020W', '40N_030E',
    '40N_030W', '40N_040E', '40N_040W', '40N_050E', '40N_060E', '40N_070E',
    '40N_070W', '40N_080E', '40N_080W', '40N_090E', '40N_090W', '40N_100E',
    '40N_100W', '40N_110E', '40N_110W', '40N_120E', '40N_120W', '40N_130E',
    '40N_130W', '40N_140E', '40S_010W', '40S_020W', '40S_040E', '40S_050E',
    '40S_060E', '40S_070E', '40S_070W', '40S_080W', '40S_140E', '40S_160E',
    '40S_170E', '40S_180W', '50N_000E', '50N_010E', '50N_010W', '50N_020E',
    '50N_020W', '50N_030E', '50N_030W', '50N_040E', '50N_040W', '50N_050E',
    '50N_060E', '50N_060W', '50N_070E', '50N_070W', '50N_080E', '50N_080W',
    '50N_090E', '50N_090W', '50N_100E', '50N_100W', '50N_110E', '50N_110W',
    '50N_120E', '50N_120W', '50N_130E', '50N_130W', '50N_140E', '50N_140W',
    '50N_150E', '50S_000E', '50S_030W', '50S_040W', '50S_050W', '50S_060E',
    '50S_060W', '50S_070E', '50S_070W', '50S_080W', '50S_150E', '50S_160E',
    '50S_170E', '60N_000E', '60N_010E', '60N_010W', '60N_020E', '60N_020W',
    '60N_030E', '60N_040E', '60N_050E', '60N_050W', '60N_060E', '60N_060W',
    '60N_070E', '60N_070W', '60N_080E', '60N_080W', '60N_090E', '60N_090W',
    '60N_100E', '60N_100W', '60N_110E', '60N_110W', '60N_120E', '60N_120W',
    '60N_130E', '60N_130W', '60N_140E', '60N_140W', '60N_150E', '60N_150W',
    '60N_160E', '60N_160W', '60N_170E', '60N_170W', '60N_180W', '70N_000E',
    '70N_010E', '70N_010W', '70N_020E', '70N_020W', '70N_030E', '70N_030W',
    '70N_040E', '70N_040W', '70N_050E', '70N_050W', '70N_060E', '70N_060W',
    '70N_070E', '70N_070W', '70N_080E', '70N_080W', '70N_090E', '70N_090W',
    '70N_100E', '70N_100W', '70N_110E', '70N_110W', '70N_120E', '70N_120W',
    '70N_130E', '70N_130W', '70N_140E', '70N_140W', '70N_150E', '70N_150W',
    '70N_160E', '70N_160W', '70N_170E', '70N_170W', '70N_180W', '80N_000E',
    '80N_010E', '80N_010W', '80N_020E', '80N_020W', '80N_030E', '80N_030W',
    '80N_040E', '80N_040W', '80N_050E', '80N_050W', '80N_060E', '80N_060W',
    '80N_070E', '80N_070W', '80N_080E', '80N_080W', '80N_090E', '80N_090W',
    '80N_100E', '80N_100W', '80N_110E', '80N_110W', '80N_120E', '80N_120W',
    '80N_130E', '80N_130W', '80N_140E', '80N_140W', '80N_150E', '80N_150W',
    '80N_160E', '80N_160W', '80N_170E', '80N_170W', '80N_180W'
]

tile_id_list = list(_FALLBACK_TILE_ID_LIST)
tile_id_list_source = "fallback_tile_roster"
_tile_id_list_resolved = False


def get_tile_id_list(*, refresh: bool = False) -> list[str]:
    """Return the model tile roster, loading the shared index on first use."""

    global tile_id_list, tile_id_list_source, _tile_id_list_resolved
    if _tile_id_list_resolved and not refresh:
        return list(tile_id_list)

    loaded_tile_ids = load_tile_ids_from_s3(
        s3_bucket_name=s3_bucket_name,
        s3_prefix=tile_index_shapefile_prefix,
        local_dir=local_temp_dir,
    )
    _tile_id_list_resolved = True
    if loaded_tile_ids:
        tile_id_list = loaded_tile_ids
        tile_id_list_source = tile_index_shapefile_uri
    else:
        tile_id_list = list(_FALLBACK_TILE_ID_LIST)
        tile_id_list_source = "fallback_tile_roster"
        logger.warning(
            "Falling back to the built-in tile roster because the shared "
            "tile index could not be loaded."
        )
    return list(tile_id_list)

burned_area_final_pattern = "burned_area_final"
land_cover_pattern = "land_cover"

# ---------------------------------------------------
# 7. Dynamic Download Dictionary Function
# ---------------------------------------------------

def resolve_land_cover_interval_type(
    interval_start_year: int,
    interval_end_year: int,
) -> str:
    """Return the source interval used for an inventory period's land cover.

    Multi-year periods ending in 2005, 2010, 2015, or 2020 use the complete
    five-year composites. Annual model runs use annual composites, and the
    shortened 2021-2024 inventory period uses the complete annual 2024 source.
    """

    period = (int(interval_start_year), int(interval_end_year))
    if period in five_year_inventory_periods:
        return (
            intervals_annual
            if period[1] in annual_land_cover_years
            else intervals_five_year
        )
    if period[0] == period[1] and period[1] in annual_land_cover_years:
        return intervals_annual
    raise ValueError(
        "Unsupported land-cover period "
        f"{period[0]}-{period[1]}. Supported inventory periods are "
        f"{five_year_inventory_periods}; supported annual years are "
        f"{annual_land_cover_years}."
    )


def get_dynamic_download_dict(tile_id, interval_start_year, interval_end_year=None, peat_dataset='ogh'):
    if interval_end_year is None:
        interval_end_year = interval_start_year

    lc_year = interval_end_year
    interval_type = resolve_land_cover_interval_type(
        interval_start_year,
        interval_end_year,
    )

    # Updated IPCC land cover directory
    pixel_resolution = "40000_pixels"
    land_cover_ipcc_dir = posixpath.join(
        full_bucket_prefix,
        processed_dir,
        'land_cover_ipcc',
        land_cover_ipcc_dates[interval_type],
        interval_type,
        str(lc_year),
        pixel_resolution,
    )


    peat_dataset = peat_dataset.lower()
    if peat_dataset not in peat_mask_dirs:
        raise ValueError(f"Unknown peat dataset: {peat_dataset}")

    peat_path_dataset = peat_dataset
    if peat_dataset == 'ogh':
        # Use the unthresholded probability surface for OGH and apply
        # thresholds downstream in the core model.
        peat_path_dataset = 'ogh_unthresholded'

    if peat_path_dataset == 'gfw':
        peat_path = posixpath.join(
            peat_mask_dirs['gfw'], patterns['peat'].format(tile_id=tile_id)
        )
    elif peat_path_dataset == 'union_mask':
        peat_path = posixpath.join(
            peat_mask_dirs['union_mask'], f"{tile_id}_union_mask.tif"
        )
    else:
        peat_path = posixpath.join(
            peat_mask_dirs[peat_path_dataset],
            f"{tile_id}_{peat_path_dataset}_mask.tif"
        )

    dynamic_dict = {
        'land_cover': posixpath.join(
            land_cover_ipcc_dir,
            patterns['land_cover'].format(tile_id=tile_id)
        ),
        'peat': peat_path,
        'dadap': posixpath.join(dirs['dadap'], patterns['dadap'].format(tile_id=tile_id)),
        'engert': posixpath.join(dirs['engert'], patterns['engert'].format(tile_id=tile_id)),
        'grip': posixpath.join(dirs['grip'], patterns['grip'].format(tile_id=tile_id)),
        'osm_roads': posixpath.join(dirs['osm_roads'], patterns['osm_roads'].format(tile_id=tile_id)),
        'osm_canals': posixpath.join(dirs['osm_canals'], patterns['osm_canals'].format(tile_id=tile_id)),
        'planted_forest_type': posixpath.join(dirs['planted_forest_type'], patterns['planted_forest_type'].format(tile_id=tile_id)),
        'extraction': posixpath.join(dirs['extraction'], patterns['extraction'].format(tile_id=tile_id)),
        'climate_domain': posixpath.join(
            dirs['climate_domain'],
            patterns['climate_domain'].format(tile_id=tile_id),
        ),
        'descals_type': posixpath.join(dirs['descals_type'], patterns['descals_type'].format(tile_id=tile_id)),
        'mangrove_extent': posixpath.join(dirs['mangrove_extent'], patterns['mangrove_extent'].format(tile_id=tile_id)),
        'tidal_marsh': posixpath.join(dirs['tidal_marsh'], patterns['tidal_marsh'].format(tile_id=tile_id)),
    }

    # Add burned area layers for each year in the interval
    for yr in range(interval_start_year, interval_end_year + 1):
        burned_key = f"burned_area_final_{yr}"
        dynamic_dict[burned_key] = posixpath.join(
            dirs['burned_area_final'].format(year=yr),
            patterns['burned_area_final'].format(tile_id=tile_id, year=yr)
        )

    return dynamic_dict
