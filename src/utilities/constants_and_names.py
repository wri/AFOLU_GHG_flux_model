import math
import boto3
import numpy as np

# Project imports
from src.utilities import universal_utilities as uu

########
### Constants
########

### Model versions. Written order should be as below for consistency.

AFOLU_model_version = "1.0.0"
AFOLU_model_version_underscore = AFOLU_model_version.replace(".", "_")

LULUCF_model_version = "1.0.0"
LULUCF_model_version_underscore = LULUCF_model_version.replace(".", "_")

veg_model_version = "1.0.6"
veg_model_version_underscore = veg_model_version.replace(".", "_")

organic_soil_model_version = "1.0.1"
organic_soil_model_version_underscore = organic_soil_model_version.replace(".", "_")

SOC_model_version = "1.0.1"
SOC_model_version_underscore = SOC_model_version.replace(".", "_")

LULUCF_full_version_underscore = (f"LULUCF_version_{LULUCF_model_version_underscore}_MODEL_TYPE__MODEL_PATH_DESCRIPTION__veg_v{veg_model_version_underscore}__org_soil_v{organic_soil_model_version_underscore}__min_soil_v{SOC_model_version_underscore}")


### s3 buckets
s3 = boto3.resource('s3')
short_bucket_prefix = "gfw2-data"
full_bucket_prefix = "s3://" + short_bucket_prefix
full_bucket_prefix_length = len(full_bucket_prefix)+1
s3_client = boto3.client("s3")

### Pattern for tile_ids in regex form
tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
small_chunk_pattern = r'__-?\d+_-?\d+_-?\d+_-?\d+__'

Coiled_workspace = "wri-land-research"

### m^2 to hectares
m2_to_ha = 1/10000

### Vegetation model years
LC_first_year = 2015  # First year of annual LC/veg height data
LC_last_year = 2024   # Last year of annual LC/veg height data
veg_modeL_increment = 1  # Timestep for the vegetation model (years)
LC_years = list(range(LC_first_year, LC_last_year, veg_modeL_increment))  # All years of the LC/veg height data

veg_outputs_years = LC_years[1:]  # Output years for the vegetation model
veg_end_year_count = len(veg_outputs_years)  # Number of years output from the veg moel
veg_year_range_str = f"{veg_outputs_years[0]}_{LC_last_year}"


# Seconds until a file download timeouts (and potentially retries)
download_timeout = 300


### Carbon constants

# Carbon to CO2 (data type needs to be specified because of use in numba)
C_to_CO2 = 44/12
C_to_CO2_numba = np.float32(C_to_CO2)

# Biomass to carbon ratios
biomass_to_carbon_non_mangrove = 0.47   # Conversion of biomass to carbon for non-mangrove forests
biomass_to_carbon_mangrove = 0.45   # Conversion of biomass to carbon for mangroves (IPCC 2013 Wetlands Supplement table 4.2)

# Default root:shoot when no Huang et al. 2021 is available. The average slope of the AGB:BGB relationship in Figure 3 of Mokany et al. 2006.
# and is only used where Huang et al. 2021 can't reach (remote Pacific islands).
default_r_s_non_mang = 0.26

# Non-mangrove deadwood C:AGC and litter C:AGC constants (unitless)
# Deadwood and litter carbon as fractions of AGC are from
# https://cdm.unfccc.int/methodologies/ARmethodologies/tools/ar-am-tool-12-v3.0.pdf
# "Clean Development Mechanism A/R Methodological Tool:
# Estimation of carbon stocks and change in carbon stocks in dead wood and litter in A/R CDM project activities version 03.0"
# Tables on pages 18 (deadwood) and 19 (litter).
# They depend on the climate domain, elevation, and precipitation.
tropical_low_elev_low_precip_deadwood_c_ratio = 0.02
tropical_low_elev_low_precip_litter_c_ratio = 0.04
tropical_low_elev_med_precip_deadwood_c_ratio = 0.01
tropical_low_elev_med_precip_litter_c_ratio = 0.01
tropical_low_elev_high_precip_deadwood_c_ratio = 0.06
tropical_low_elev_high_precip_litter_c_ratio = 0.01
tropical_high_elev_deadwood_c_ratio = 0.07
tropical_high_elev_litter_c_ratio = 0.01
non_tropical_deadwood_c_ratio = 0.08
non_tropical_litter_c_ratio = 0.04

# s3 folder with Excel workbooks that contain lookup tables for emission factors, removal factors, and carbon pool constants
EF_RF_C_ratio_spreadsheet_URL = "http://gfw2-data.s3.amazonaws.com/climate/AFOLU_flux_model/LULUCF/rate_ratio_lookup_tables/"

# Removal factor and carbon pool constant workbook
RF_C_ratio_spreadsheet_name = "RF_rate_and_C_ratio_lookup_tables_20250430.xlsx"
RF_C_ratio_spreadsheet_full_path = f"{EF_RF_C_ratio_spreadsheet_URL}{RF_C_ratio_spreadsheet_name}"

# IPCC Tier 1 removal factor spreadsheet by continent-ecozone-age category combination
# (IPCC 2019, Table 4.9, with corrigenda 4 temperate forest revision) (Mg AGB/ha/yr)
# Currently only used for primary forests.
IPCC_removal_factor_table_tab = "natrl fores gain, for std model"

# IPCC Tier 1 mangrove removal factors and AGC:BGC, AGC:deadwood C, and AGC:litter C, from IPCC 2013 Wetlands Supplement
mangrove_rate_ratio_tab = 'mang gain and Cratios,for model'

# Emission factors for partial disturbances (by 1km driver)
partial_disturbance_emission_factor_table_name = "partial_disturbance_emission_factors_LULUCF_model_w_sensit_anlys__20260627.xlsx"
partial_disturbance_emission_factor_table_full_path = f"{EF_RF_C_ratio_spreadsheet_URL}{partial_disturbance_emission_factor_table_name}"
partial_disturbance_emission_factor_table_tab_standard = "EF_combined_std"
partial_disturbance_emission_factor_table_tab_low_EF = "EF_combined_low"
partial_disturbance_emission_factor_table_tab_high_EF = "EF_combined_high"

# Aboveground carbon removal factor for oil palm (Mg C/ha/yr) (IPCC 2019 Cropland Table 5.3)
oil_palm_agc_rf = 2.4
oil_palm_bgc_rf = oil_palm_agc_rf * default_r_s_non_mang

# One-time annual cropland removal factor (Mg C/ha) (IPCC 2019 Cropland Section 5.3.1.2)
cropland_rf = 4.7

# Cropland aboveground carbon density (global constant, static value) (Mg C/ha)
cropland_agc_dens = cropland_rf

# Aboveground carbon removal factor for trees outside forests (Mg C/ha/yr), assuming that the entire hectare is ToF
# (IPCC 2019 Settlements Section 8.2.1.2 (p. 8.5))
trees_outside_forests_agc_rf_max = 2.8

# Global warming potentials (GWP)
gwp_ch4 = 27 # AR6 WG1 Table 7.15
gwp_n2o = 273 # AR6 WG1 Table 7.15

# Combustion factor for trees that had fire but no height reduction or other sign of disturbance
# (i.e. undisturbed trees remaining trees).
# From IPCC 2019, Table 2.6, "Boreal forest- surface fire" (applied globally, though boreal)
Cf_forest_undisturbed_standard = 0.15  # Standard model and sensitivity analyses in which EFs are not changed
Cf_forest_undisturbed_low = (0.15-0.08)  # For sensitivity analysis: value-st dev
Cf_forest_undisturbed_high = (0.15+0.08)  # For sensitivity analysis: value+st dev

other_landcover_node = 7


### Crop residue and grassland burning constants

# Value for cropland nodes in land state node decision tree (for gain, loss, or remaining)
cropland_node = 5

# Ratio of aboveground residue dry matter to harvested yield (Rag(T) (IPCC 2019, V4, Ch. 11, Table 11.1A- generic value)
cropland_residue_harvest_ratio = 1.0

# Emission factors for crop residue burning (IPCC 2019, V4, Ch. 2, Table 2.5-- agricultural residues)
Gef_CH4_crop_residue_standard = 2.7  # Standard model and sensitivity analyses in which EFs are not changed
Gef_N2O_crop_residue_standard = 0.07  # Standard model and sensitivity analyses in which EFs are not changed
Gef_CH4_crop_residue_low_EF = (2.7-1.5)  # For sensitivity analysis: value-best professional judgement st dev because IPCC has no st dev
Gef_N2O_crop_residue_low_EF = (0.07-0.03)  # For sensitivity analysis: value-best professional judgement st dev because IPCC has no st dev
Gef_CH4_crop_residue_high_EF = (2.7+1.5)  # For sensitivity analysis: value+best professional judgement st dev because IPCC has no st dev
Gef_N2O_crop_residue_high_EF = (0.07+0.03)  # For sensitivity analysis: value+best professional judgement st dev because IPCC has no st dev

# Combustion factor for crop residue burning (IPCC 2019, V4, Ch. 2, Table 2.6-- agricultural residues, other crops)
Cf_crop_residue_standard = 0.85  # Standard model and sensitivity analyses in which EFs are not changed
Cf_crop_residue_low = (0.85-0.15)  # For sensitivity analysis: value-best professional judgement st dev because IPCC has no st dev
Cf_crop_residue_high = (0.85+0.15)  # For sensitivity analysis: value-best professional judgement st dev because IPCC has no st dev

# Value for short/medium vegetation nodes in land state node decision tree (for gain, loss, or remaining)
grassland_node = 6

# Emission factors for savanna and grassland burning (IPCC 2019, V4, Ch. 2, Table 2.5-- savanna and grassland)
Gef_CH4_grassland_standard = 2.3  # Standard model and sensitivity analyses in which EFs are not changed
Gef_N2O_grassland_standard = 0.21  # Standard model and sensitivity analyses in which EFs are not changed
Gef_CH4_grassland_low_EF = (2.3-0.9)  # For sensitivity analysis: value-st dev
Gef_N2O_grassland_low_EF = (0.21-0.10)  # For sensitivity analysis: value-st dev
Gef_CH4_grassland_high_EF = (2.3+0.9)  # For sensitivity analysis: value+st dev
Gef_N2O_grassland_high_EF = (0.21+0.10)  # For sensitivity analysis: value+st dev

# Combustion factor for savanna and grassland burning (IPCC 2019, V4, Ch. 2, Table 2.6-- all savanna grasslands (mid/late dry season burns)
Cf_grassland_standard = 0.77  # Standard model and sensitivity analyses in which EFs are not changed
Cf_grassland_low = (0.77-0.26)  # For sensitivity analysis: value-st dev
Cf_grassland_high = 1  # For sensitivity analysis: value+st dev (0.77+0.26>1, so capping at 1).


### GLCLU cover codes
### Classifications proposed by Elise Mazur 2025-06-05 by email. Agreed on 2025-06-09
# Bare ground
bare_ground_dry_min_code = 0
bare_ground_dry_max_code = 4
bare_ground_wet_min_code = 100
bare_ground_wet_max_code = 104

# Short vegetation
short_veg_dry_min_code = 5
short_veg_dry_max_code = 26
short_veg_wet_min_code = 105
short_veg_wet_max_code = 126

# Tall vegetation
tall_veg_dry_min_code = 27
tall_veg_dry_max_code = 48
tall_veg_wet_min_code = 127
tall_veg_wet_max_code = 148

water_min_code = 200
water_max_code = 208

cropland = 244
builtup = 250


### Miscellaneous
chunk_dims = 4000           # Size of a 1x1 deg raster in pixels
full_raster_dims = 40000    # Size of a 10x10 deg raster in pixels

resolution = 0.00025  # Decimal degrees
global_geotif_resolution = 0.04

# Pixel aggregation parameters
global_aggregation_factor = int(global_geotif_resolution / resolution)  # 160 native pixels per 0.04 deg

#Dimensions for global zarrs (80N to 80S)
global_width = int(round(360.0 / resolution))
global_height = int(round(160.0 / resolution))
origin_x = -180.0
origin_y = 80.0

# Threshold for height loss to be counted as disturbed (m)
sig_height_loss_threshold_abs = 5

# Threshold for height gain to be counted as regrowth in the same interval as a disturbance
sig_height_gain_threshold_abs = -5

# Height minimum for GLAD trees (meters)
tree_threshold = 5

# Height minimum for short vegetation using Global Pasture Watch (meters) (Hunter et al. 2025)
GPW_short_veg_threshold = 2

# Converts grams to kilograms for burning of dry matter
g_to_kg = 10 ** -3

# Which carbon pools are emitted under different circumstances for full tree loss: AGC, BGC, deadwood C, litter C.
# Need to specify numpy datatype because they're used in the Numba functions, which need explicit datatypes.
# Based on LULUCF model framework slides: https://onewri-my.sharepoint.com/:p:/g/personal/david_gibbs_wri_org/EWwyxRfgdeVJi4ezwX7LrfcBT4k1CY-vHRtVDjJIAsgsJg?e=6nDCkA
# 1 means full emissions, 0 means no emissions.
agc_emissions_only = np.array([1, 0, 0, 0]).astype('uint8')  # AGC only
biomass_emissions_only = np.array([1, 1, 0, 0]).astype('uint8')  # AGC and BGC
all_but_bgc_emissions = np.array([1, 0, 1, 1]).astype('uint8')  # AGC, deadwood C, and litter C
deadwood_litter_emissions = np.array([0, 0, 1, 1]).astype('uint8')  # deadwood C and litter C
all_non_soil_pools = np.array([1, 1, 1, 1]).astype('uint8')  # AGC, BGC, deadwood C, and litter C
no_carbon_pools = np.array([0, 0, 0, 0]).astype('uint8')  # None

# SDPT v2.0 planted forest type codes
SDPT_oil_palm_code = 1
SDPT_wood_fiber_code = 2
SDPT_other_code = 3

# Threshold for including GAMI v2.1 in primary forest composite (inclusive, >=)
primary_age_threshold = 100



########
### File name paths and patterns
########

##### Miscellaneous

# Pattern for date (XXXX) or date range (XXXX_YYYY_XXXX_YYYY, SOC only) with 1 or 2 leading _ in output file names
date_date_range_pattern = r'_{1,2}(?:\d{4}(?:_\d{4})*)'

AFOLU_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/"
LULUCF_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/"
cropland_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/cropland_emissions/"

local_log_path = "logs/"
s3_log_path = "climate/AFOLU_flux_model/LULUCF/model_logs/"
combined_log = "AFOLU"

# Local path for chunk stats
local_chunk_stats_path = "chunk_stats/"
s3_chunk_stats_path = "climate/AFOLU_flux_model/LULUCF/chunk_stats/"

# Chunk stats table name patterns (Excel tabs or Parquet tables)
annual_1x1_inputs = "annual_1x1_inputs"
other_1x1_inputs = "other_1x1_inputs"
gross_outputs_1x1 = "gross_outputs_1x1"
net_outputs_1x1 = "net_outputs_1x1"
other_outputs_1x1 = "other_outputs_1x1"
min_max_for_layers_1x1 = "min_max_for_layers_1x1"
counts_1x1_in_10x10 = "1x1_counts_in_10x10"

# 1x1 deg fishnet between 80N and 60N, 180W and 180E that intersects GADM4.1 and has GADM iso joined to it
fishnet_1x1deg_s3_dir = f"{AFOLU_dir}fishnet_1x1deg/20250429/"

fishnet_1x1deg_all_land_name = "fishnet_GADM41_1x1deg__spatial_join_intersect__20250428__center_in.shp"

fishnet_1x1deg_uri = f"{fishnet_1x1deg_s3_dir}{fishnet_1x1deg_all_land_name}"

progress_tracking_path = "climate/AFOLU_flux_model/task_progress_txts/"

# Pixel meanings: values per-hectare, per-pixel, or per 0.04x0.04 deg aggregation.
# Different for carbon pools vs. fluxes because of the temporal component.
C_density_pixel_meaning = "_ha"
C_per_pixel_pixel_meaning = "_pixel"
C_density_aggreg_pixel_meaning = "_0_04deg"

flux_density_pixel_meaning = "_ha_yr"
flux_per_pixel_pixel_meaning = "_pixel_yr"
flux_aggreg_pixel_meaning = "_0_04deg_yr"

# The four possible pixel meanings.
# Order is important; the ones with _yr come first because otherwise a function that looks for these meanings in order
# will always find the C meanings, even for flux outputs because flux outputs always have _ha or _pixel.
# That is, it avoids issues of partial matches when this list is being iterated through.
pixel_meanings = [flux_density_pixel_meaning, flux_per_pixel_pixel_meaning,
                  C_density_pixel_meaning, C_per_pixel_pixel_meaning, flux_aggreg_pixel_meaning]

##### Inputs

land_cover_5_year_path = f"{LULUCF_dir}landcover/composite/five_year/v1/raw/"
land_cover_annual_path = f"{LULUCF_dir}landcover/composite/annual/v2/raw/"
land_cover_pattern = "land_cover_composite"  # Raw tifs don't have a pattern; this is just for use in the numba data dictionary

vegetation_height_annual_GLAD_path = "https://glad.geog.umd.edu/Potapov/Global_TCH_2015-23"
vegetation_height_5_year_path = f"{LULUCF_dir}landcover/vegetation_height/five_year/v1/raw/"
vegetation_height_5_year_pattern = "vegetation_height"
vegetation_height_annual_path = f"{LULUCF_dir}landcover/vegetation_height/annual/v2_20250716/raw/"
vegetation_height_annual_pattern = ""
vegetation_height_pattern = "vegetation_height"  # Raw tifs don't have a pattern; this is just for use in the numba data dictionary

forest_disturbance_annual_dir = f"{LULUCF_dir}landcover/annual_forest_disturbance/raw/"
forest_disturbance_layer_name = "forest_disturbance"

### Biomass and carbon densities

agb_2000_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/Processed/"
agb_2000_pattern = "t_aboveground_biomass_ha_2000"

# From https://catalogue.ceda.ac.uk/uuid/bf535053562141c6bb7ad831f5998d77/
# https://data.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v5.01/geotiff
# Bulk downloaded to computer (/mnt/c/GIS/AFOLU_flux_model/ESA_CCI_2015/) using WSL Ubuntu:
# wget -e robots=off --mirror --no-parent -r https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v6.0/geotiff/2015/
esa_AGB_v = 'v6_0'
agb_2015_dir_raw = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/AGB/raw/"
agb_2015_pattern_raw = "ESACCI-BIOMASS-L4-AGB-MERGED-100m-2015-fv6.0"
agb_2015_dir_processed = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/AGB/processed/20260120/"
agb_2015_pattern = "AGB_2015_ESA_CCI_Mg_AGB_ha"

agb_stdev_2015_dir_raw = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/AGB_stdev/raw/"
agb_stdev_2015_pattern_raw = "ESACCI-BIOMASS-L4-AGB_SD-MERGED-100m-2015-fv6.0"
agb_stdev_2015_dir_processed = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/AGB_stdev/processed/20260120/"
agb_stdev_2015_pattern = "AGB_stdev_2015_ESA_CCI_Mg_AGB_ha"

mangrove_agb_2000_dir = f"{full_bucket_prefix}/climate/carbon_model/mangrove_biomass/processed/standard/20190220/"
mangrove_agb_2000_pattern = "mangrove_agb_t_ha_2000"

# Data available at https://ctrees-agb-100m-global.s3.us-west-2.amazonaws.com/index.html#cogs/
# Description available at https://registry.opendata.aws/ctrees-agb-100m-global/
# CTrees 2015 AGB and uncertainty global COGs (used for starting carboon pool sensitivity analysis)
# aws s3 cp s3://ctrees-agb-100m-global/cogs/global_agb_100m_landsat0024_all_2015_densenet_l1_agb_mosaic_100m_base_cd_ts.tif s3://gfw2-data/climate/Ctrees_biomass/2015/AGB/raw/ --source-region us-west-2
# aws s3 cp s3://ctrees-agb-100m-global/cogs/global_agb_100m_landsat0024_all_2015_densenet_l1_agb_mosaic_100m_base_cd_ts_uncertainty_sem.tif s3://gfw2-data/climate/Ctrees_biomass/2015/AGB_uncertainty/raw/ --source-region us-west-2
# Note: Raw pixel values are multiplied by 10 to save space. To retrieve the actual biomass density in Mg/ha, divide the raw pixel value by 10
ctrees_run_date = '20260629'
ctrees_agb_raw_nodata = -9999
ctrees_agb_processed_nodata = 0
ctrees_agb_scale_factor = 0.1
ctrees_agb_output_dtype = "uint16"

ctrees_agb_2015_dir_raw = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/AGB/raw/"
ctrees_agb_2015_pattern_raw = "global_agb_100m_landsat0024_all_2015_densenet_l1_agb_mosaic_100m_base_cd_ts"
ctrees_agb_2015_dir_processed = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/AGB/processed/{ctrees_run_date}/"
ctrees_agb_2015_pattern = "AGB_2015_Ctrees_Mg_AGB_ha"

ctrees_agb_uncertainty_2015_dir_raw = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/AGB_uncertainty/raw/"
ctrees_agb_uncertainty_2015_pattern_raw = "global_agb_100m_landsat0024_all_2015_densenet_l1_agb_mosaic_100m_base_cd_ts_uncertainty_sem"



# Carbon density patterns (also used in path names)
agb_dens_pattern = "AGB_density_MgAGB_ha"

# Raw carbon density patterns, not masked by landcover composite
agc_raw_dens_pattern = "carbon_density__AGC__raw__MgC"
bgc_raw_dens_pattern = "carbon_density__BGC__raw__MgC"
deadwood_c_raw_dens_pattern = "carbon_density__deadwood_C__raw__MgC"
litter_c_raw_dens_pattern = "carbon_density__litter_C__raw__MgC"
non_soil_c_raw_dens_pattern = "carbon_density__non_soil__raw__MgC"
soil_c_dens_raw_pattern = "carbon_density__soil_C__raw__MgC"

# Carbon density patterns when masked by landcover composite
agc_LC_masked_dens_pattern = "carbon_density__AGC__landcover_masked__MgC"
bgc_LC_masked_dens_pattern = "carbon_density__BGC__landcover_masked__MgC"
deadwood_c_LC_masked_dens_pattern = "carbon_density__deadwood_C__landcover_masked__MgC"
litter_c_LC_masked_dens_pattern = "carbon_density__litter_C__landcover_masked__MgC"
non_soil_c_LC_masked_dens_pattern = "carbon_density__non_soil__landcover_masked__MgC"
soil_c_dens_LC_masked_pattern = "carbon_density__soil_C__landcover_masked__MgC"

# Carbon density pattern for vegetation model outputs
agc_modeled_dens_pattern = "carbon_density__AGC__MgC"
bgc_modeled_dens_pattern = "carbon_density__BGC__MgC"
deadwood_c_modeled_dens_pattern = "carbon_density__deadwood_C__MgC"
litter_c_modeled_dens_pattern = "carbon_density__litter_C__MgC"
non_soil_c_modeled_dens_pattern = "carbon_density__non_soil__MgC"

### Carbon pools in starting year (2000/2015)

## 2000
carbon_2000_creation_date = '20250930'

# Raw carbon density, not masked by landcover composite
agc_2000_raw_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{agc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
agc_2000_raw_pattern = f"{agc_raw_dens_pattern}_2000"

bgc_2000_raw_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{bgc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
bgc_2000_raw_pattern = f"{bgc_raw_dens_pattern}_2000"

deadwood_c_2000_raw_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{deadwood_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
deadwood_c_2000_raw_pattern = f"{deadwood_c_raw_dens_pattern}_2000"

litter_c_2000_raw_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{litter_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
litter_c_2000_raw_pattern = f"{litter_c_raw_dens_pattern}_2000"

non_soil_c_2000_raw_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{non_soil_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
non_soil_c_2000_raw_pattern = f"{non_soil_c_raw_dens_pattern}_2000"

# Carbon density, masked by landcover composite
agc_2000_LC_masked_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{agc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
agc_2000_LC_masked_pattern = f"{agc_LC_masked_dens_pattern}_2000"

bgc_2000_LC_masked_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{bgc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
bgc_2000_LC_masked_pattern = f"{bgc_LC_masked_dens_pattern}_2000"

deadwood_c_2000_LC_masked_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{deadwood_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
deadwood_c_2000_LC_masked_pattern = f"{deadwood_c_LC_masked_dens_pattern}_2000"

litter_c_2000_LC_masked_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{litter_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
litter_c_2000_LC_masked_pattern = f"{litter_c_LC_masked_dens_pattern}_2000"

non_soil_c_2000_LC_masked_dir = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/{non_soil_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2000_creation_date}/"
non_soil_c_2000_LC_masked_pattern = f"{non_soil_c_LC_masked_dens_pattern}_2000"

starting_C_densities_2000_path_zarr = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/year_2000_derived_carbon_pools/mega_zarr/CHUNK_SIZE_pixels/RUN_DATE/starting_C_densities_zarr.zarr"


## 2015
carbon_2015_creation_date = '20260807'

agc_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{agc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
agc_2015_raw_pattern = f"{agc_raw_dens_pattern}_2015"

bgc_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{bgc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
bgc_2015_raw_pattern = f"{bgc_raw_dens_pattern}_2015"

deadwood_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{deadwood_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
deadwood_c_2015_raw_pattern = f"{deadwood_c_raw_dens_pattern}_2015"

litter_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{litter_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
litter_c_2015_raw_pattern = f"{litter_c_raw_dens_pattern}_2015"

non_soil_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{non_soil_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
non_soil_c_2015_raw_pattern = f"{non_soil_c_raw_dens_pattern}_2015"

# Carbon density, masked by landcover composite
agc_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{agc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
agc_2015_LC_masked_pattern = f"{agc_LC_masked_dens_pattern}_2015"

bgc_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{bgc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
bgc_2015_LC_masked_pattern = f"{bgc_LC_masked_dens_pattern}_2015"

deadwood_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{deadwood_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
deadwood_c_2015_LC_masked_pattern = f"{deadwood_c_LC_masked_dens_pattern}_2015"

litter_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{litter_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
litter_c_2015_LC_masked_pattern = f"{litter_c_LC_masked_dens_pattern}_2015"

non_soil_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{non_soil_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
non_soil_c_2015_LC_masked_pattern = f"{non_soil_c_LC_masked_dens_pattern}_2015"

starting_C_densities_2015_path_zarr = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/mega_zarr/CHUNK_SIZE_pixels/RUN_DATE/starting_C_densities_zarr.zarr"

# Code that describes the source for the starting carbon densities in the landcover-masked outputs
starting_C_pools_LC_masked_source_flag_pattern = "carbon_density_source_flag_landcover_masked"
starting_C_pools_LC_masked_state_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/{esa_AGB_v}/2015/year_2015_derived_carbon_pools/{starting_C_pools_LC_masked_source_flag_pattern}/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"


### 2015 sensitivity analysis
agc_2015_ctrees_raw_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{agc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
bgc_2015_ctrees_raw_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{bgc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
deadwood_c_2015_ctrees_raw_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{deadwood_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
litter_c_2015_ctrees_raw_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{litter_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
non_soil_c_2015_ctrees_raw_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{non_soil_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"

agc_2015_ctrees_LC_masked_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{agc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
bgc_2015_ctrees_LC_masked_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{bgc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
deadwood_c_2015_ctrees_LC_masked_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{deadwood_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
litter_c_2015_ctrees_LC_masked_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{litter_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"
non_soil_c_2015_ctrees_LC_masked_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{non_soil_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{ctrees_run_date}/"

starting_C_densities_2015_ctrees_path_zarr = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/mega_zarr/CHUNK_SIZE_pixels/RUN_DATE/starting_C_densities_zarr.zarr"

starting_C_pools_ctrees_LC_masked_state_dir = f"{full_bucket_prefix}/climate/Ctrees_biomass/2015/year_2015_derived_carbon_pools/{starting_C_pools_LC_masked_source_flag_pattern}/CHUNK_SIZE_pixels/{ctrees_run_date}/"


### Other inputs

elevation_dir = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/elevation/20190418/"
elevation_pattern = "elevation"

climate_domain_dir = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418/"
climate_domain_pattern = "fao_ecozones_bor_tem_tro_processed"

climate_zone_raw_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/climate_zone/raw/20250422/"
climate_zone_raw_pattern = "ipcc_climate_1985-2015.tif"

climate_zone_processed_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/climate_zone/processed/20250422/"
climate_zone_pattern = "climate_zone_2019_corrigenda_processed"
"""
1: tropical montane
2: tropical wet
3: tropical moist
4: tropical dry
5: warm temperate moist
6: warm temperate dry
7: cool temperate moist
8: cool temperate dry
9: boreal moist
10: boreal dry
11: polar moist
12: polar dry
"""


precipitation_dir = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/precip/20190418/"
precipitation_pattern = "precip_mm_annual"

r_s_ratio_non_mang_dir = f"{full_bucket_prefix}/climate/carbon_model/BGB_AGB_ratio/processed/20230216/"
r_s_ratio_non_mang_pattern = "BGB_AGB_ratio_non_mang"

continent_ecozone_dir = f"{full_bucket_prefix}/climate/carbon_model/fao_ecozones/ecozone_continent/20190116/processed/"
continent_ecozone_pattern = "fao_ecozones_continents_processed"

TCL_dir = f"{full_bucket_prefix}/forest_change/hansen_2024/"
TCL_pattern = "GFW2024"

## Forest age
# Forest age in 2010 and 2015 (created together)
forest_age_2010_2015_run_date = '20260126'
forest_age_2010_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v3_1/2010/standard/not_gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2010_pattern = "forest_age_not_gap_filled_2010"

forest_age_2015_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v3_1/2015/standard/not_gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2015_pattern = "forest_age_not_gap_filled_2015"

forest_age_2010_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v3_1/2010/standard/gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2010_gap_filled_pattern = "forest_age_gap_filled_2010"

forest_age_2015_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v3_1/2015/standard/gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2015_gap_filled_pattern = "forest_age_gap_filled_2015"

# Forest age in 2000 (age in 2000 is derived from gap-filled age in 2010, so there is no non-gap-filled age in 2000 data)
forest_age_2000_run_date = '20250707'
forest_age_2000_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2000/standard/age__years/gap_filled/CHUNK_SIZE_pixels/{forest_age_2000_run_date}/"
forest_age_2000_gap_filled_pattern = "forest_age_gap_filled_2000"

forest_age_2000_gap_filled_source_flag_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2000/standard/age__source_flag/gap_filled/CHUNK_SIZE_pixels/{forest_age_2000_run_date}/"
forest_age_2000_gap_filled_source_flag_pattern = "forest_age_gap_filled_2000__source_flag"

# Age at disturbance (1x1 deg resolution) from forthcoming Besnard et al. paper
global_age_at_disturbance_file = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/forest_age/age_pre_disturbance_Besnard_et_al/global_geotif/20250702/age_pre_disturbance_median_1deg_global__20250703.tif"

# Forest age pattern for use in the LULUCF model
forest_age_start_year_pattern = "forest_age_gap_filled_start_year"
forest_age_output_pattern = "forest_age_at_end_of_interval"

# Starting composite primary forest (2015)
starting_composite_primary_forest_run_date = '20260806'
starting_composite_primary_forest_pattern = "starting_composite_primary_forest"
starting_composite_primary_forest_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/{starting_composite_primary_forest_pattern}/{LC_first_year}/presence/{chunk_dims}_pixels/{starting_composite_primary_forest_run_date}/"
starting_composite_primary_forest_zarr_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/{starting_composite_primary_forest_pattern}/{LC_first_year}/presence/zarr/{chunk_dims}_pixels/{starting_composite_primary_forest_run_date}/{starting_composite_primary_forest_pattern}.zarr"

# GEE script that the global rasters are from is https://code.earthengine.google.com/805896f7a511c13eb873c4804a683abc (each file takes about 15 minutes to export to Google Drive).
# NOTE: GEE export function splits the exported global raster into two pieces. I merged the two pieces into a single file in ArcPro,
# then applied ArcPro nibble command to fill in gaps so that all pixels would have rates, for example:
# out_raster = arcpy.sa.Nibble(
#     in_raster=r"Robinson et al. rates\natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_40_years.tif",
#     in_mask_raster=r"Robinson et al. rates\natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_40_years.tif",
#     nibble_values="DATA_ONLY",
#     nibble_nodata="PROCESS_NODATA",
#     in_zone_raster=None
# )
# out_raster.save(r"C:\GIS\Carbon_seqr_mapping\secondary_forests\average_rates_for_LULUCF_model\natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_40_years__nibble_20250516.tif")
# Then, uploaded to s3.

#Robinson regrowth rates (5-year rates for ages 0-20 and 20-year rates for ages 20+)
secondary_forest_curve_run_date = '20250616'
natural_forest_growth_curve_base_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/"
natural_forest_growth_curve_raw_dir = f"{natural_forest_growth_curve_base_dir}raw/{secondary_forest_curve_run_date}/"
natural_forest_growth_curve_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{secondary_forest_curve_run_date}/"
natural_forest_growth_curve_pattern = "natural_forest_mean_growth_rate__Mg_AGC_ha_yr"
natural_forest_growth_curve_intervals = ['0_5', '6_10', '11_15', '16_20', '21_40', '41_60', '61_80', '81_100']

# Xu et al. 2026 Chapman-Richards regrowth curves (other degraded forest, 1-degree resolution)
# From https://drive.google.com/drive/folders/1ANjxrL4LItXtteHIGFRU70_MfZiFVF9n?usp=sharing
# Using on the "other degraded" curves for the sensitivity analysis
Xu_global_date = '20260629'
Xu_regrowth_base_dir = f"{full_bucket_prefix}/climate/regrowth_curves__Xu_et_al/from_Yidi_Xu_20260628/"
Xu_regrowth_raw_dir = f"{Xu_regrowth_base_dir}raw/"
Xu_regrowth_raw_pattern = "growthcurve_otherdeg.tif"
Xu_regrowth_AGB_rate_dir = f"{Xu_regrowth_base_dir}AGB_density_by_age/all_ages_global/"
Xu_regrowth_AGB_rate_pattern = "Xu_other_degrad_regrowth_density__Mg_AGB_ha"   # append __{age}_years_{date}.tif

Xu_regrowth_AGC_rate_base_dir = f"{Xu_regrowth_base_dir}AGC_rate/"
Xu_regrowth_AGC_rate_global_all_ages_dir = f"{Xu_regrowth_AGC_rate_base_dir}all_ages_global/"
Xu_regrowth_AGC_rate_tiles_dir = f"{Xu_regrowth_AGC_rate_base_dir}all_ages_tiled/"
Xu_regrowth_AGC_rate_pattern = "Xu_other_degrad_regrowth_rate__Mg_AGC_ha_yr"   # append __{age_min}_{age_max}_years_{date}.tif
Xu_regrowth_intervals = ["0_5_years", "5_10_years", "10_15_years", "15_20_years", "20_40_years", "40_60_years", "60_80_years", "80_100_years"]

drivers_run_date = '20250414'
drivers_raw_dir = f"{full_bucket_prefix}/drivers_of_loss/1_km/raw/update2023_20241218/"
drivers_raw_pattern = "drivers_forest_loss_1km_2023_band1.tif"
drivers_processed_dir = f"{full_bucket_prefix}/drivers_of_loss/1_km/processed/{drivers_run_date}/"
drivers_processed_pattern = f"drivers_of_TCL_1_km_{drivers_run_date}.tif"

drivers_path = f"{full_bucket_prefix}/drivers_of_loss/1_km/processed/{drivers_run_date}/"
drivers_pattern = f"drivers_of_TCL_1_km_{drivers_run_date}"

pixel_area_dir = f"{full_bucket_prefix}/analyses/area_28m/"
pixel_area_pattern = "hanson_2013_area"

'''
From Radost Stanimirova via Slack 2024-10-18:
//1: Permanent agriculture
//2: Hard commodities
//3: Shifting cultivation
//4: Forest management
//5: Wildfire
//6: Settlements & Infrastructure
//7: Other natural disturbances
'''
permanent_agriculture = 1
hard_commodities = 2
shifting_cultivation = 3
forest_management = 4
wildfire = 5
settlements_and_infrastruct = 6
other_natural_disturbances = 7

# Drivers categorized by what carbon pools are emitted from stand-replacing non-fire disturbances
# Need to be tuples rather than lists because the numba function can't check list membership but can check tuple membership
drivers_biomass_C_only = (forest_management, wildfire, other_natural_disturbances)
drivers_non_soil_C = (permanent_agriculture, hard_commodities, shifting_cultivation, settlements_and_infrastruct)

# Humid tropical primary forest in 2001
# Copied from s3://gfw-data-lake/umd_regional_primary_forest_2001/v201901/raster/epsg-4326/10/40000/is/geotiff
primary_2001_dir = f"{full_bucket_prefix}/forest_cover/primary_forest/umd_regional_primary_forest_2001_from_gfw-data-lake/v201901/raster/epsg-4326/10/40000/is/geotiff/"
primary_2001_pattern = "primary_2001"

# IFL 2016 (from Peter Potapov 9/18/25). Not using IFL2016 geotifs from data-lake because they don't match Peter's.
# I think there are NoData issues in the gfw-data-lake version.
ifl_2016_dir = f"{full_bucket_prefix}/forest_cover/IFL_2016/20250918/raw/"
ifl_2016_pattern = "ifl_2016"

# Annual Hansen tree cover loss tiles (2001-2024) (used to mask out loss from primary forest before 2015)
tree_cover_loss_dir = f"{full_bucket_prefix}/forest_change/hansen_2024/"
tree_cover_loss_pattern = 'GFW2024'

# Composite of humid tropical primary forest for 20001 and IFL for 2000, created for the forest carbon flux model.
# Used for LULUCF vegetation model starting in 2000.
ifl_primary_2000_dir = f"{full_bucket_prefix}/climate/carbon_model/ifl_primary_merged/processed/20200724/"
ifl_primary_2000_pattern = "ifl_2000_primary_2001_merged"

planted_forest_type_dir = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/"
planted_forest_type_pattern = "plantation_type_oilpalm_woodfiber_other"

planted_forest_AGC_removal_factor_dir = f"{full_bucket_prefix}/climate/carbon_model/annual_removal_factor_planted_forest/SDPTv2_AGC/20230911/"
planted_forest_AGC_removal_factor_pattern = "annual_gain_rate_AGC_Mg_ha_planted_forest"

planted_forest_AGC_BGC_removal_factor_dir = f"{full_bucket_prefix}/climate/carbon_model/annual_removal_factor_planted_forest/SDPTv2_AGC_BGC/20230911/"
planted_forest_AGC_BGC_removal_factor_pattern = "annual_gain_rate_AGC_BGC_Mg_ha_planted_forest"

oil_palm_2000_extent_dir = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/IDN_MYS_plantation_pre_2000/processed/20200724/"
oil_palm_2000_extent_pattern = "plantation_2000_or_earlier_processed"

# Descals et al. 2024: https://essd.copernicus.org/articles/16/5111/2024/essd-16-5111-2024-discussion.html
oil_palm_first_year_dir = f"{AFOLU_dir}organic_soils/inputs/processed/descals_plantation/year/20241105/"
oil_palm_first_year_pattern = "plantation_year"

# Originally from gfw-data-lake, so it's in 400x400 windows
planted_forest_tree_crop_dir = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/"
planted_forest_tree_crop_pattern = "planted_forest_tree_crop"

# hdf and raw raster paths don't include bucket prefix because of special processing code
burned_area_hdf_dir = "fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/"
burned_area_hdf_converted_to_raw_raster_dir = "fires/MODIS_burned_area/MCD64A1.061/1_intermediate_outputs__hdf_converted_to_raster/"
burned_area_final_dir = "fires/MODIS_burned_area/MCD64A1.061/2_final_outputs__Hansenized/"  # With each year in its own folder
burned_area_final_pattern = "burned_area_final"

# GMW mangrove extent
GMW_version = "v3"
mangrove_extent_years = [1996, 2007, 2008, 2009, 2010, 2015, 2016, 2017, 2018, 2019, 2020]

mangrove_extent_raw_dir = f"{full_bucket_prefix}/global-mangrove-extent/version3/raw/raster/"
mangrove_extent_raw_pattern = r"GMW_[NS][0-9]{2}[EW][0-9]{3}_[0-9]{4}_v3.tif"       # e.g "GMW_N00E008_1996_v3.tif" in regex form

mangrove_extent_hansenized_dir = f"{full_bucket_prefix}/global-mangrove-extent/version3/hansenized/raster/"
mangrove_extent_hansenized_pattern = f"GMW{GMW_version}_mangrove_extent"

mangrove_1x1deg_smoothed_dir = f"{full_bucket_prefix}/global-mangrove-extent/version3/smoothed_1x1deg/raster/"

mangrove_extent_processed_dir = f"{full_bucket_prefix}/global-mangrove-extent/version3/smoothed/raster/"
mangrove_extent_processed_pattern = f"GMW{GMW_version}_smoothed_mangrove_extent"

# Global pasture watch grasslands extent (1 = cultivated grassland, 2 = natural/semi-natural grassland)
GPW_version = "v1.1"
GPW_years = range(2000, 2025)

GPW_extent_raw_dir = f"{full_bucket_prefix}/lcl/gpw/grasslands/{GPW_version}/raw/"
GPW_extent_raw_pattern = "grasslands"

GPW_extent_processed_dir = f"{full_bucket_prefix}/lcl/gpw/grasslands/{GPW_version}/processed/"
GPW_extent_processed_pattern = f"GPW_grasslands_extent"

# Global Pasture Watch median vegetation height (https://stac.openlandmap.org/gpw_gsvh-30m/collection.json?.language=en,
# from Hunter et al. 2025 (https://www.nature.com/articles/s41597-025-05739-6)
GPW_MVH_uri = f"https://s3.opengeohub.org/gpw/arco/gpw_short.veg.height_egbt_m_30m_s_YYYY0101_YYYY1231_go_epsg.4326_v1.tif"
GPW_MVH_pattern = f"GPW_height"


# Organic Soils
# Organic soil mask, created by Erin Glen based on Hengl et al. 2026 and
# https://opengeohub.medium.com/global-organic-soils-extent-and-peat-depth-at-30-m-spatial-resolution-based-on-multisource-eo-data-c6e00f390069
# Erin confirmed that it didn't matter which interval I used for the organic soil mask; all are equivalent.
organic_soil_extent_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{organic_soil_model_version_underscore}/organic_soil/ogh_mixed_f1_f15_f2_20260513/five_year_intervals/2021_2024/40000_pixels/20260525/"
organic_soil_extent_pattern = "organic_soil__2021_2024"
organic_soil_burned_pattern   = "burned_total_Mg_CO2e"
organic_soil_drained_pattern  = "drained_total_Mg_CO2e"
organic_soil_year_intervals   = ['2016_2020', '2021_2024']  # update when a new interval is added

organic_soil_zarr_path = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{organic_soil_model_version_underscore}/mega_zarr/ogh_mixed_f1_f15_f2_20260513/five_year/4000_pixels/20260525/mega.zarr"


# Cropland emissions
cropland_emis_run_date =  '20241204'
global_cropland_emissions_raw_dir = f"{AFOLU_dir}cropland_emissions/raw__from_Cornell/20241126/year_2020/all_sources/"
global_cropland_emissions_processed_dir = f"{AFOLU_dir}cropland_emissions/processed/{cropland_emis_run_date}/year_2020/all_sources"

global_cropland_mean_rate_harvest_area_all_crops_peat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_2006_kg_ha_CO2.tif"
global_cropland_mean_rate_harvest_area_all_crops_peat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/including_peatland/2006/harvest_area/"
global_cropland_mean_rate_harvest_area_all_crops_peat_2006_processed_pattern = f"all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_2006_kg_ha_CO2.tif"

global_cropland_mean_rate_harvest_area_all_crops_peat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_2019_kg_ha_CO2.tif"
global_cropland_mean_rate_harvest_area_all_crops_peat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/including_peatland/2019/harvest_area/"
global_cropland_mean_rate_harvest_area_all_crops_peat_2019_processed_pattern = f"all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_2019_kg_ha_CO2.tif"

global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_NonPeatland_2006_kg_ha_CO2.tif"
global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/non_peatland/2006/harvest_area/"
global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2006_processed_pattern = f"all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_nonpeatland_2006_kg_ha_CO2.tif"

global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_NonPeatland_2019_kg_ha_CO2.tif"
global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/non_peatland/2019/harvest_area/"
global_cropland_mean_rate_harvest_area_all_crops_nonpeat_2019_processed_pattern = f"all_GHGs_cropland_mean_rate_harvest_area_CO2eq_all_crops_nonpeatland_2019_kg_ha_CO2.tif"

global_cropland_mean_rate_physical_area_all_crops_peat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2006_kg_ha_CO2.tif"
global_cropland_mean_rate_physical_area_all_crops_peat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/including_peatland/2006/physical_area/"
global_cropland_mean_rate_physical_area_all_crops_peat_2006_processed_pattern = f"all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2006_kg_ha_CO2.tif"

global_cropland_mean_rate_physical_area_all_crops_peat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2019_kg_ha_CO2.tif"
global_cropland_mean_rate_physical_area_all_crops_peat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/including_peatland/2019/physical_area/"
global_cropland_mean_rate_physical_area_all_crops_peat_2019_processed_pattern = f"all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_2019_kg_ha_CO2.tif"

global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_NonPeatland_2006_kg_ha_CO2.tif"
global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/non_peatland/2006/physical_area/"
global_cropland_mean_rate_physical_area_all_crops_nonpeat_2006_processed_pattern = f"all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_nonpeatland_2006_kg_ha_CO2.tif"

global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_NonPeatland_2019_kg_ha_CO2.tif"
global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/mean_rate/non_peatland/2019/physical_area/"
global_cropland_mean_rate_physical_area_all_crops_nonpeat_2019_processed_pattern = f"all_GHGs_cropland_mean_rate_physical_area_CO2eq_all_crops_nonpeatland_2019_kg_ha_CO2.tif"

global_cropland_total_amount_all_crops_peat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_2006_kg_CO2.tif"
global_cropland_total_amount_all_crops_peat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/total_amount/including_peatland/2006/"
global_cropland_total_amount_all_crops_peat_2006_processed_pattern = f"all_GHGs_cropland_total_amount_CO2eq_all_crops_2006_kg_CO2.tif"

global_cropland_total_amount_all_crops_peat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_2019_kg_CO2.tif"
global_cropland_total_amount_all_crops_peat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/total_amount/including_peatland/2019/"
global_cropland_total_amount_all_crops_peat_2019_processed_pattern = f"all_GHGs_cropland_total_amount_CO2eq_all_crops_2019_kg_CO2.tif"

global_cropland_total_amount_all_crops_nonpeat_2006_raw_pattern = "Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2006_kg_CO2.tif"
global_cropland_total_amount_all_crops_nonpeat_2006_processed_dir = f"{global_cropland_emissions_processed_dir}/total_amount/non_peatland/2006/"
global_cropland_total_amount_all_crops_nonpeat_2006_processed_pattern = f"all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2006_kg_CO2.tif"

global_cropland_total_amount_all_crops_nonpeat_2019_raw_pattern = "Global_grid_all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif"
global_cropland_total_amount_all_crops_nonpeat_2019_processed_dir = f"{global_cropland_emissions_processed_dir}/total_amount/non_peatland/2019/"
global_cropland_total_amount_all_crops_nonpeat_2019_processed_pattern = f"all_GHGs_cropland_total_amount_CO2eq_all_crops_NonPeatland_2019_kg_CO2.tif"



##### Outputs

#IPCC LUC translation version number
IPCC_LU_version = "1.0.0"
IPCC_LU_version_underscore = IPCC_LU_version.replace(".", "_")

### IPCC classes and change
IPCC_class_path = "IPCC_class"
IPCC_class_pattern = "IPCC_class"
IPCC_node_path = "IPCC_node_code"
IPCC_node_pattern = "IPCC_node_code"
IPCC_change_path = "IPCC_change"
IPCC_change_pattern = "IPCC_change"
IPCC_summary_path = "IPCC_summary"
IPCC_summary_pattern = "IPCC_summary"

### IPCC codes
# Based in IPCC LU heirarchy: Settlements > Cropland > Forest Land > Grassland > Wetland > Other Land
settlement_IPCC = 1
cropland_IPCC = 2
forest_IPCC = 3
grassland_IPCC = 4
wetland_IPCC = 5
otherland_IPCC = 6

#IPCC_class_max_val = 6  # Maximum value of IPCC class codes

IPCC_outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/land_use/{IPCC_LU_version}"
IPCC_class_dir = f"{IPCC_outputs_path}/{IPCC_class_path}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
IPCC_node_dir = f"{IPCC_outputs_path}/{IPCC_node_path}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
IPCC_change_dir = f"{IPCC_outputs_path}/{IPCC_change_path}/START_END/CHUNK_SIZE_pixels/RUN_DATE/"
IPCC_summary_dir = f"{IPCC_outputs_path}/{IPCC_summary_path}/2015_2024/CHUNK_SIZE_pixels/RUN_DATE/"

IPCC_outputs_path_mega_zarr = f"{IPCC_outputs_path}/mega_zarr/CHUNK_SIZE_pixels/RUN_DATE/land_use_zarr.zarr"

land_state_pattern = "veg_land_state_node"
land_state_node_fire_value = 9  # State nodes that end in this value had fire

agc_rf_pre_dist_pattern = "veg_removal_factor__AGC__MgC"

# Gross and net fluxes (fluxes are in Mg CO2/ha/yr or Mg CO2e/ha/yr)
agc_gross_emis_pattern = "veg_gross_emissions__AGC__MgCO2"
bgc_gross_emis_pattern = "veg_gross_emissions__BGC__MgCO2"
deadwood_c_gross_emis_pattern = "veg_gross_emissions__deadwood_C__MgCO2"
litter_c_gross_emis_pattern = "veg_gross_emissions__litter_C__MgCO2"

ch4_gross_emis_pattern = "veg_gross_emissions__CH4__MgCO2e"
n2o_gross_emis_pattern = "veg_gross_emissions__N2O__MgCO2e"

agc_gross_removals_pattern = "veg_gross_removals__AGC__MgCO2"
bgc_gross_removals_pattern = "veg_gross_removals__BGC__MgCO2"
deadwood_c_gross_removals_pattern = "veg_gross_removals__deadwood_C__MgCO2"
litter_c_gross_removals_pattern = "veg_gross_removals__litter_C__MgCO2"

net_flux_agc_pattern = "veg_net_flux__AGC__MgCO2"
net_flux_bgc_pattern = "veg_net_flux__BGC__MgCO2"
net_flux_deadwood_c_pattern = "veg_net_flux__deadwood_C__MgCO2"
net_flux_litter_c_pattern = "veg_net_flux__litter_C__MgCO2"

gross_emis_all_C_pools_CO2_only_pattern = "veg_gross_emissions__all_C_pools__CO2_only__MgCO2"
gross_emis_all_C_pools_non_CO2_only_pattern = "veg_gross_emissions__all_C_pools__non_CO2_only__MgCO2e"
gross_emis_all_C_pools_all_gases_pattern = "veg_gross_emissions__all_C_pools__all_gases__MgCO2e"

gross_removals_all_C_pools_pattern = "veg_gross_removals__all_C_pools__MgCO2"

net_flux_all_C_pools_CO2_only_pattern = "veg_net_flux__all_C_pools__CO2_only__MgCO2"
net_flux_all_C_pools_all_gases_pattern = "veg_net_flux__all_C_pools__all_gases__MgCO2e"

# Intermediate outputs
gain_year_count_pattern = "veg_gain_year_count_during_interval"
most_recent_year_not_tall_veg = "veg_most_recent_year_not_tall_veg"
year_of_forest_loss = "veg_year_of_forest_loss"
max_height_since_last_time_not_tall_veg = "veg_max_height_since_last_time_not_tall_veg"
first_time_sig_loss_from_max_height = "veg_first_time_sig_loss_from_max_height"
part_or_full_dist_in_earlier_intervals = "veg_partial_or_full_dist_in_earlier_intervals"
part_or_full_dist_in_curr_interval = "veg_partial_or_full_dist_in_current_interval"
times_burned_in_interval = "veg_times_burned_in_current_interval"
agc_emission_factor = "veg_AGC_emission_factor_CO2_only__fraction"
composite_primary_forest = "veg_composite_primary_forest"

# Tolerance for difference between model and zarr chunk stat metrics.
# There's often some rounding/float error between them, so a small difference (~10^-8) is expected.
zarr_difference_tolerance = 0.05

model_version_type_description_placeholder = 'version_MODEL_VERSION__TYPE__DESCRIPTION'

veg_outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs_vegetation/{model_version_type_description_placeholder}/"
veg_outputs_path_zarr = f"{veg_outputs_path}zarr/CHUNK_SIZE_pixels/RUN_DATE/vegetation_zarr.zarr"

# List of output directories from vegetation model with placeholders for parts of the directory
veg_core_output_dirs = [
    f"{veg_outputs_path}{agc_modeled_dens_pattern}/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{bgc_modeled_dens_pattern}/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{deadwood_c_modeled_dens_pattern}/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{litter_c_modeled_dens_pattern}/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{agc_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{bgc_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{deadwood_c_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{litter_c_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{agc_gross_removals_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{bgc_gross_removals_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{deadwood_c_gross_removals_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{litter_c_gross_removals_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{ch4_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{n2o_gross_emis_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{land_state_pattern}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{agc_rf_pre_dist_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
]


# Intermediate outputs from vegetation model
veg_intermediate_output_dirs = [
    f"{veg_outputs_path}{forest_age_output_pattern}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gain_year_count_pattern}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{most_recent_year_not_tall_veg}/RUNSTART_END/CHUNK_SIZE_pixels/RUN_DATE/", # Years represent from model start to current interval end
    f"{veg_outputs_path}{max_height_since_last_time_not_tall_veg}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{first_time_sig_loss_from_max_height}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{part_or_full_dist_in_earlier_intervals}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{part_or_full_dist_in_curr_interval}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{times_burned_in_interval}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{agc_emission_factor}/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{composite_primary_forest}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
]

# Summative outputs from core vegetation model
veg_summative_output_patterns = [
    gross_emis_all_C_pools_CO2_only_pattern, gross_emis_all_C_pools_non_CO2_only_pattern, gross_emis_all_C_pools_all_gases_pattern,
    gross_removals_all_C_pools_pattern,
    net_flux_agc_pattern, net_flux_bgc_pattern, net_flux_deadwood_c_pattern, net_flux_litter_c_pattern,
    net_flux_all_C_pools_CO2_only_pattern, net_flux_all_C_pools_all_gases_pattern,
    non_soil_c_modeled_dens_pattern
]

# Only specific datasets output to zarrs.
# Starts with non-summative outputs from the vegetation model
core_veg_outputs_to_zarr = [
    agc_modeled_dens_pattern, bgc_modeled_dens_pattern, deadwood_c_modeled_dens_pattern, litter_c_modeled_dens_pattern,
    agc_gross_emis_pattern, bgc_gross_emis_pattern, deadwood_c_gross_emis_pattern, litter_c_gross_emis_pattern,
    ch4_gross_emis_pattern, n2o_gross_emis_pattern,
    agc_gross_removals_pattern, bgc_gross_removals_pattern, deadwood_c_gross_removals_pattern, litter_c_gross_removals_pattern,
    land_state_pattern, composite_primary_forest, forest_age_output_pattern, agc_rf_pre_dist_pattern, agc_emission_factor
]

# Also want to add the metadata for the summative outputs to the global zarr upfront for simplicity,
# rather than having to add more empty layers to the zarr at the summative stage
full_veg_outputs_to_zarr = core_veg_outputs_to_zarr + veg_summative_output_patterns

# Summative outputs from core vegetation model
veg_summative_output_dirs = [
    f"{veg_outputs_path}{gross_emis_all_C_pools_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_emis_all_C_pools_non_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_emis_all_C_pools_all_gases_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_removals_all_C_pools_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_agc_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_bgc_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_deadwood_c_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_litter_c_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_all_C_pools_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_all_C_pools_all_gases_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{non_soil_c_modeled_dens_pattern}/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"  # Only per interval
]


### LULUCF summation

burned_organic_soils_total_pattern = "burned_total_Mg_CO2e_ha_yr"
burned_organic_soils_total_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{organic_soil_model_version_underscore}/{burned_organic_soils_total_pattern}/ogh_sensitivity_500m_10/five_year_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/"

drained_organic_soils_total_pattern = "drained_total_Mg_CO2e_ha_yr"
drained_organic_soils_total_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/outputs/version_{organic_soil_model_version_underscore}/{drained_organic_soils_total_pattern}/ogh_sensitivity_500m_10/five_year_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/"


# Soil total patterns
gross_emis_all_C_pools_CO2_only_soil_pattern = "soil_gross_emissions__all_C_pools__CO2_only__MgCO2"
gross_emis_all_C_pools_non_CO2_only_soil_pattern = "soil_gross_emissions__all_C_pools__non_CO2_only__MgCO2e"
gross_emis_all_C_pools_all_gases_soil_pattern = "soil_gross_emissions__all_C_pools__all_gases__MgCO2e"

net_flux_all_C_pools_CO2_only_soil_pattern = "soil_net_flux__all_C_pools__CO2_only__MgCO2"
net_flux_all_C_pools_all_gases_soil_pattern = "soil_net_flux__all_C_pools__all_gases__MgCO2e"

# LULUCF patterns
gross_emis_all_C_pools_CO2_only_LULUCF_pattern = "LULUCF_gross_emissions__all_C_pools__CO2_only__MgCO2"
gross_emis_all_C_pools_non_CO2_only_LULUCF_pattern = "LULUCF_gross_emissions__all_C_pools__non_CO2_only__MgCO2e"
gross_emis_all_C_pools_all_gases_LULUCF_pattern = "LULUCF_gross_emissions__all_C_pools__all_gases__MgCO2e"

gross_removals_all_C_pools_LULUCF_pattern = "LULUCF_gross_removals__all_C_pools__MgCO2"

net_flux_all_C_pools_CO2_only_LULUCF_pattern = "LULUCF_net_flux__all_C_pools__CO2_only__MgCO2"
net_flux_all_C_pools_all_gases_LULUCF_pattern = "LULUCF_net_flux__all_C_pools__all_gases__MgCO2e"

# Summative outputs used for LULUCF totals
veg_summative_for_LULUCF_output_dirs = [
    f"{veg_outputs_path}{gross_emis_all_C_pools_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_emis_all_C_pools_non_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_emis_all_C_pools_all_gases_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{gross_removals_all_C_pools_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_all_C_pools_CO2_only_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{veg_outputs_path}{net_flux_all_C_pools_all_gases_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
]

### Soil organic carbon (SOC) timeseries from OpenGeoHub (OGH) (URIs from https://github.com/openlandmap/soildb/blob/main/tables/OpenLandMap_soildb_COGS.csv)

# Value for organic soil, from Erin Glen's organic soil mask
organic_soil_mask_val = 1

# From Hengl et al. 2026 (https://essd.copernicus.org/articles/18/989/2026/)
# Confirmed to be up-to-date by Tom Hengl on 2025-12-19 via email.
# Units of raw global COGs are kg C/m^3 for 0-30 cm, multiplied by 10 (rescale factor) to make COGs ints.
# Values in dict need to be lists (even with just 1 element) because of how uu.prepare_to_download_chunk works.
SOC_COGS = {
    "2005": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20000101_20051231_g_epsg.4326_v20250204.tif"],
    "2010": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20050101_20101231_g_epsg.4326_v20250204.tif"],
    "2015": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20100101_20151231_g_epsg.4326_v20250204.tif"],
    "2020": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20150101_20201231_g_epsg.4326_v20250204.tif"],
    "2022": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20200101_20221231_g_epsg.4326_v20250204.tif"]
}

SOC_outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs_soil_organic_carbon/{model_version_type_description_placeholder}/"

# Value refers to the end year of the OGH reporting block
SOC_density_intervals = [2005, 2010, 2015, 2020, 2022]
# Value refers to the end year of the second OGH reporting block, e.g., 2010 is the comparison of 2000-2005 block vs. 2005-2010 block
SOC_change_intervals = [2010, 2015, 2020, 2022]
SOC_change_interval_length = 5 # years

SOC_density_intervals_annual = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]
SOC_change_intervals_annual = [2016, 2017, 2018, 2019, 2020, 2021, 2022]

SOC_path_zarr = f"{SOC_outputs_path}zarr/CHUNK_SIZE_pixels/RUN_DATE/SOC_zarr.zarr"

# Converts the raw COG's kg C/m^3 (top 30 cm) that is rescaled by 10 -> Mg C/ha without the rescaling.
# OGH rescaled the global COGs by 10 to make them ints instead of floats to save storage.
# OGH COG encoding: raw integer × 0.1 = kg C/m³ volumetric density.
# Depth: 0–30 cm = 0.3 m.
# Conversion to Mg C per pixel: density × depth × pixel_area_m² / 1000 (=0.3)
OGH_SCALE  = np.float64(0.1)    # raw → kg C/m³
DEPTH_M    = np.float64(0.3)    # 0–30 cm layer
M2_PER_HA  = np.float64(10000) # m²/ha
KG_TO_MG   = np.float64(1000)  # kg/Mg
SOC_conversion_factor = np.float32(OGH_SCALE * DEPTH_M * M2_PER_HA / KG_TO_MG)


# Extent of raw COGs
SOC_density_full_extent_pattern = "SOC_density__full_extent__0-30cm_MgC"
SOC_density_full_extent_dir = f"{SOC_outputs_path}{SOC_density_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_loss_full_extent_pattern = "SOC_loss__full_extent__0-30cm_MgCO2"
SOC_loss_full_extent_dir = f"{SOC_outputs_path}{SOC_loss_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_gain_full_extent_pattern = "SOC_gain__full_extent__0-30cm_MgCO2"
SOC_gain_full_extent_dir = f"{SOC_outputs_path}{SOC_gain_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_net_full_extent_pattern = "SOC_net__full_extent__0-30cm_MgCO2"
SOC_net_full_extent_dir = f"{SOC_outputs_path}{SOC_net_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"

# Extent of mineral soil (excludes thresholded organic soil extent created by Erin Glen)
SOC_density_min_soil_extent_pattern = "SOC_density__mineral_soil_extent__0-30cm_MgC"
SOC_density_min_soil_extent_dir = f"{SOC_outputs_path}{SOC_density_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_loss_min_soil_extent_pattern = "SOC_loss__mineral_soil_extent__0-30cm_MgCO2"
SOC_loss_min_soil_extent_dir = f"{SOC_outputs_path}{SOC_loss_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_gain_min_soil_extent_pattern = "SOC_gain__mineral_soil_extent__0-30cm_MgCO2"
SOC_gain_min_soil_extent_dir = f"{SOC_outputs_path}{SOC_gain_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_net_min_soil_extent_pattern = "SOC_net__mineral_soil_extent__0-30cm_MgCO2"
SOC_net_min_soil_extent_dir = f"{SOC_outputs_path}{SOC_net_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"

SOC_outputs_to_zarr = [
    SOC_density_full_extent_pattern, SOC_density_min_soil_extent_pattern,
    SOC_net_full_extent_pattern,  SOC_net_min_soil_extent_pattern,
    SOC_loss_full_extent_pattern, SOC_loss_min_soil_extent_pattern,
    SOC_gain_full_extent_pattern, SOC_gain_min_soil_extent_pattern
]


# S3 base path (without s3://gfw2-data/)
SOC_uncertainty_output_base = (f'{SOC_outputs_path}mineral_soil_uncertainty/')
SOC_uncertainty_output_intermediates = (f'{SOC_uncertainty_output_base}intermediates/')


### Soil summative outputs

soil_output_patterns = [
    gross_emis_all_C_pools_CO2_only_soil_pattern,
    gross_emis_all_C_pools_non_CO2_only_soil_pattern,
    gross_emis_all_C_pools_all_gases_soil_pattern,
    net_flux_all_C_pools_CO2_only_soil_pattern,
    net_flux_all_C_pools_CO2_only_soil_pattern
]
soil_output_dirs = [
    f"{SOC_outputs_path}{gross_emis_all_C_pools_CO2_only_soil_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{SOC_outputs_path}{gross_emis_all_C_pools_non_CO2_only_soil_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{SOC_outputs_path}{gross_emis_all_C_pools_all_gases_soil_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{SOC_outputs_path}{net_flux_all_C_pools_CO2_only_soil_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{SOC_outputs_path}{net_flux_all_C_pools_CO2_only_soil_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
]


### LULUCF summative outputs

LULUCF_outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs_LULUCF_totals/{model_version_type_description_placeholder}/"
LULUCF_outputs_path_zarr = f"{LULUCF_outputs_path}zarr/CHUNK_SIZE_pixels/RUN_DATE/"

LULUCF_output_patterns = [
    gross_emis_all_C_pools_CO2_only_LULUCF_pattern,
    gross_emis_all_C_pools_non_CO2_only_LULUCF_pattern,
    gross_emis_all_C_pools_all_gases_LULUCF_pattern,
    gross_removals_all_C_pools_LULUCF_pattern,
    net_flux_all_C_pools_CO2_only_LULUCF_pattern,
    net_flux_all_C_pools_all_gases_LULUCF_pattern
]
LULUCF_output_dirs = [
    f"{LULUCF_outputs_path}{gross_emis_all_C_pools_CO2_only_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{LULUCF_outputs_path}{gross_emis_all_C_pools_non_CO2_only_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{LULUCF_outputs_path}{gross_emis_all_C_pools_all_gases_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{LULUCF_outputs_path}{gross_removals_all_C_pools_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{LULUCF_outputs_path}{net_flux_all_C_pools_CO2_only_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{LULUCF_outputs_path}{net_flux_all_C_pools_all_gases_LULUCF_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
]

LULUCF_annual_zarr_name = "LULUCF_annual.zarr"
LULUCF_avg_zarr_name = f"LULUCF_avg_{veg_outputs_years[0]}_{veg_outputs_years[-1]}.zarr"


#######
### Zonal stats resources
#######

### Contextual layer zarrs

contextual_zarr_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/contextual_layer_global_zarr/"

adm0_zarr_date = '20251209'
adm0_zarr_dtype = 'uint16'
adm0_pattern = 'adm0'
adm0_geotif_path = "s3://gfw2-data/gadm_administrative_boundaries/v4.1/v4.1.64__from_gfw-data-lake/raster/epsg-4326/10/40000/adm0/gdal-geotiff/"
adm0_zarr_path = f"{contextual_zarr_path}GADM4_1_adm0_global/{adm0_zarr_date}_fillValue_removed/global_GADM41_adm0_{adm0_zarr_date}.zarr"
adm0_test_chunk = [13, 48, 14, 49]  # Three countries meet in Europe, with different values in three corners (50N_010E)

pixel_area_zarr_date = '20260531'
pixel_area_zarr_dtype = 'float32'
pixel_area_zstats_pattern = 'pixel_area'
pixel_area_geotif_path = pixel_area_dir
pixel_area_zarr_path = f"{contextual_zarr_path}pixel_area/{pixel_area_zarr_date}_fillValue_removed/global_pixel_area_{pixel_area_zarr_date}.zarr"
pixel_area_test_chunk = [13, 48, 14, 49]  # 50N_010E

WDPA_zarr_date = '20251229'
WDPA_zarr_dtype = 'uint8'
WDPA_pattern = 'WDPA'
WDPA_geotif_path = "s3://gfw2-data/conservation/wdpa_licensed_proteced_areas__from_data_lake/v202511/raster/epsg-4326/10/40000/detailed_iucn_cat/gdal-geotiff/"
WDPA_zarr_path = f"{contextual_zarr_path}WDPAv202511/{WDPA_zarr_date}_fillValue_removed/wdpa_{WDPA_zarr_date}.zarr"
WDPA_test_chunk = [21, -3, 22, -2]  # Has WDPA 0, 3 (bottom left, top right), and 9 (top left) (00N_020E)

BRA_biomes_zarr_date = '20251229'
BRA_biomes_zarr_dtype = 'uint8'
BRA_biomes_pattern = 'BRA_biomes'
BRA_biomes_geotif_path = "s3://gfw2-data/country/bra/bra_biomes_geotif/"
BRA_biomes_zarr_path = f"{contextual_zarr_path}BRA_biomes/{BRA_biomes_zarr_date}_fillValue_removed/BRA_biomes_{BRA_biomes_zarr_date}.zarr"
BRA_biomes_test_chunk = [-58, -16, -57, -15]  # Three biomes meet, with different values in three corners (10S_060W)

cont_eco_zarr_date = '20260206'
cont_eco_zarr_dtype = 'uint16'
cont_eco_zstats_pattern = 'cont_eco'
cont_eco_geotif_path = "s3://gfw2-data/climate/carbon_model/fao_ecozones/ecozone_continent/20190116/processed/"
cont_eco_zarr_path = f"{contextual_zarr_path}FAO_ecozone_continents/{cont_eco_zarr_date}_fillValue_removed/FAO_ecozone_continents_{cont_eco_zarr_date}.zarr"
cont_eco_test_chunk = [119, -6, 120, -5]  # Mix of 0, 4018 and 4020, with 4020 in upper right (00N_110E)=

landmark_zarr_date = '20260213'
landmark_zarr_dtype = 'uint8'
landmark_pattern = 'Landmark'
landmark_geotif_path = "s3://gfw2-data/landmark/gfw-data-lake/landmark_ip_lc_and_indicative_poly/v20250909/raster/epsg-4326/10/40000/is/geotiff/"
landmark_zarr_path = f"{contextual_zarr_path}landmark/v20250909/{landmark_zarr_date}_fillValue_removed/landmark_{landmark_zarr_date}.zarr"
landmark_test_chunk = [29, -1, 30, 0]  # 1 in upper left and lower left (00N_020E)

KBA_zarr_date = '20260213'
KBA_zarr_dtype = 'uint16'
KBA_pattern = 'KBA'
KBA_geotif_path = "s3://gfw2-data/conservation/Key_Biodiversity_Areas/KBA_2024_09/KBA_v20240903__from_gfw-data-lake/raster/epsg-4326/10/40000/is/geotiff/"
KBA_zarr_path = f"{contextual_zarr_path}KBA/v20240903/{KBA_zarr_date}_fillValue_removed/KBA_{KBA_zarr_date}.zarr"
KBA_test_chunk = [29, -1, 30, 0]  # 1 in upper right; roughly 1/3-1/2 of chunk is KBA (00N_020E)

# watersheds_zarr_date = '20260213'
watersheds_zarr_date = '20260508'
watersheds_zarr_dtype = 'uint16'
watersheds_pattern = 'watershed'
watersheds_geotif_path = "s3://gfw2-data/water/mapbox_river_basins__from_gfw-data-lake/v2018/raster/epsg-4326/10/40000/id/gdal-geotiff/"
watersheds_zarr_path = f"{contextual_zarr_path}river_basins/v2018/{watersheds_zarr_date}_fillValue_removed/river_basins_{watersheds_zarr_date}.zarr"
watersheds_test_chunk = [29, -1, 30, 0]  # 7005 in upper and lower left corners, 7003 in upper and lower right corners; should have nearly full coverage (00N_020E)

managed_land_CAN_zarr_date = '20260322'
managed_land_CAN_zarr_dtype = 'uint8'  # 1=managed, 2=unmanaged
managed_land_CAN_pattern = 'managed_land_Canada'
managed_land_CAN_geotif_path = "s3://gfw2-data/climate/jrc_managed_land_can__from_gfw-data-lake/v20260218/raster/epsg-4326/10/40000/managed_land_extent/geotiff/"
managed_land_CAN_zarr_path = f"{contextual_zarr_path}jrc_managed_land_can/v20260218/{managed_land_CAN_zarr_date}_fillValue_removed/jrc_managed_land_can_{managed_land_CAN_zarr_date}.zarr"
managed_land_CAN_test_chunk = [-141, 61, -140, 62]  # 1 (managed) in bottom corners, 2 (unmanaged) in top corners. Should have full coverage. (70N_150W)

managed_land_USA_zarr_date = '20260219'
managed_land_USA_zarr_dtype = 'uint8'  # 1=managed, 2=unmanaged
managed_land_USA_pattern = 'managed_land_USA'
managed_land_USA_geotif_path = "s3://gfw2-data/climate/jrc_managed_land_usa__from_gfw-data-lake/v20260218/raster/epsg-4326/10/40000/managed_land_extent/geotiff/"
managed_land_USA_zarr_path = f"{contextual_zarr_path}jrc_managed_land_USA/v20260218/{managed_land_USA_zarr_date}_fillValue_removed/jrc_managed_land_USA_{managed_land_USA_zarr_date}.zarr"
managed_land_USA_test_chunk = [-143, 61, -142, 62]  # 1 (managed) in top right, 2 (unmanaged) in other corners. Should have full coverage. (70N_150W)

drivers_of_loss_zarr_date = '20260507'
drivers_of_loss_zarr_dtype = 'uint8'
drivers_of_loss_pattern = 'drivers_of_TCL_1_km'
drivers_of_loss_geotif_path = drivers_processed_dir
drivers_of_loss_zarr_path = f"{contextual_zarr_path}drivers_of_TCL_1_km/v{drivers_run_date}/update2023_20241218__run_{drivers_of_loss_zarr_date}_fillValue_removed/drivers_of_TCL_1_km_{drivers_of_loss_zarr_date}.zarr"
drivers_of_loss_test_chunk = [27, -9, 28, -8]  # 7 in top-left, 3 in top-right, 1 in bottom-right, NoData in bottom-left (00N_020E)

first_year_LC_composite_zarr_date = '20260611'
first_year_LC_composite_zarr_dtype = 'uint8'
first_year_LC_composite_pattern = 'first_year_LC_composite'
first_year_LC_composite_geotif_path = f'{land_cover_annual_path}{LC_first_year}/'
first_year_LC_composite_zarr_path = f"{contextual_zarr_path}{first_year_LC_composite_pattern}/v2/{first_year_LC_composite_zarr_date}_fillValue_removed/{first_year_LC_composite_pattern}_{first_year_LC_composite_zarr_date}.zarr"
first_year_LC_composite_test_chunk = [19, 44, 20, 45] # 145 in top-left, 244 in top right, 24 in bottom right, 24 in bottom left


### Value options for contextual layer values.
### Every contextual layer needs to have all possible values listed here.

state_node_lookup_table_local = "/mnt/c/GIS/git/AFOLU_GHG_flux_model/src/LULUCF/LULUCF_state_node_lookup_table.xlsx"
state_node_lookup_table_s3 = "http://gfw2-data.s3.amazonaws.com/climate/AFOLU_flux_model/LULUCF/state_node_lookup_tables/LULUCF_state_node_lookup_table.xlsx"
sheet = "v105_20260601"

primary_forest_IFL_codes = np.array([0, 1], dtype=np.uint8)

BRA_biomes_codes = np.array([0, 1, 2, 3, 4, 5, 6], dtype=np.uint8)

WDPA_codes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=np.uint8)

# GADM v4.1 adm0 IDs (from Solomon Negusse's notebook)
gadm_adm0_ids = np.array([  0.,   4.,   8.,  10.,  12.,  16.,  20.,  24.,  28.,  31.,  32.,
        36.,  40.,  44.,  48.,  50.,  51.,  52.,  56.,  60.,  64.,  68.,
        70.,  72.,  74.,  76.,  84.,  86.,  90.,  92.,  96., 100., 104.,
       108., 112., 116., 120., 124., 132., 136., 140., 144., 148., 152.,
       156., 158., 162., 166., 170., 174., 175., 178., 180., 184., 188.,
       191., 192., 196., 203., 204., 208., 212., 214., 218., 222., 226.,
       231., 232., 233., 234., 238., 239., 242., 246., 248., 250., 254.,
       258., 260., 262., 266., 268., 270., 275., 276., 288., 292., 296.,
       300., 304., 308., 312., 316., 320., 324., 328., 332., 334., 336.,
       340., 348., 352., 356., 360., 364., 368., 372., 376., 380., 384.,
       388., 392., 398., 400., 404., 408., 410., 414., 417., 418., 422.,
       426., 428., 430., 434., 438., 440., 442., 450., 454., 458., 462.,
       466., 470., 474., 478., 480., 484., 492., 496., 498., 499., 500.,
       504., 508., 512., 516., 520., 524., 528., 531., 533., 534., 535.,
       540., 548., 554., 558., 562., 566.,70., 574., 578., 580., 581.,
       583., 584., 585., 586., 591., 598., 600., 604., 608., 612., 616.,
       620., 624., 626., 630., 634., 638., 642., 643., 646., 652., 654.,
       659., 660., 662., 663., 666., 670., 674., 678., 682., 686., 688.,
       690., 694., 702., 703., 704., 705., 706., 710., 716., 724., 728.,
       729., 732., 740., 744., 748., 752., 756., 760., 762., 764., 768.,
       772., 776., 780., 784., 788., 792., 795., 796., 798., 800., 804.,
       807., 818., 826., 831., 832., 833., 834., 840., 850., 854., 858.,
       860., 862., 876., 882., 887., 894.], dtype=np.uint16)

cont_eco_codes = np.array([0,
    1004, 1007, 1008, 1009, 1010, 1014, 1016, 1017, 1018, 1018,
    1019, 1020, 1021, 1022,
    2001, 2002, 2003, 2004, 2004, 2005, 2006, 2007, 2007, 2008,
    2008, 2009, 2009, 2010, 2010, 2011, 2012, 2013, 2013,
    2014, 2014, 2014, 2014, 2015, 2015, 2016, 2017, 2017,
    2018, 2018, 2018, 2019, 2019, 2020, 2020, 2021, 2021, 2022,
    3004, 3005,
    4001, 4002, 4003, 4004, 4005, 4006, 4008, 4009, 4011, 4012,
    4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020, 4022,
    5007, 5010, 5021,
    6007, 6010, 6021,
    7001, 7002, 7003, 7004, 7005, 7007, 7009, 7011, 7012,
    7013, 7014, 7015, 7022,
    8004, 8008, 8013, 8014
], dtype=np.uint16)

landmark_codes = np.array([0, 1], dtype=np.uint8)

composite_primary_codes = np.array([0, 1], dtype=np.uint8)

KBA_codes = np.array([0, 1], dtype=np.uint8)

watershed_codes = np.array([0,
    1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010,
    1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
    2011, 2012, 2013, 2014, 2015,
    3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008, 3009, 3010,
    3011, 3012, 3013, 3014, 3015, 3016, 3017, 3018, 3019, 3020,
    3021, 3022, 3023, 3024, 3025,
    4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010,
    4011, 4012, 4013, 4014, 4015, 4016, 4017, 4018, 4019, 4020,
    4021, 4022, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4030,
    4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040,
    4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051,
    4052,
    5001, 5002, 5003, 5004, 5005, 5006, 5007, 5008, 5009, 5010,
    5011, 5012, 5013, 5014, 5015, 5016, 5017, 5018, 5019, 5020,
    5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029, 5030,
    5031, 5032, 5033, 5034, 5035, 5036, 5037, 5038, 5039, 5040,
    5041, 5042, 5043, 5044, 5045, 5046, 5047, 5048, 5049, 5050,
    5051, 5052, 5053, 5054, 5055, 5056, 5057, 5058, 5059, 5060,
    5061, 5062, 5063, 5064, 5065, 5066, 5067, 5068,
    6001, 6002, 6003, 6004, 6005, 6006, 6007, 6008, 6009, 6010,
    6011, 6012, 6013, 6014, 6015, 6016, 6017, 6018, 6019,
    7001, 7002, 7003, 7004, 7005, 7006, 7007, 7008, 7009, 7010,
    7011, 7014, 7015, 7016, 7017, 7018, 7019, 7020, 7021, 7022,
    7023, 7024, 7025, 7026, 7027,
    8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009
], dtype=np.uint16)

drivers_codes = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8)

managed_land_codes = np.array([0, 1, 2], dtype=np.uint8)

forest_age_category_pattern = 'forest_age_category_end_of_interval'
forest_age_category_codes = np.array([0, 1, 6, 21, 41, 61, 81, 101], dtype=np.uint8)

ipcc_class_codes = np.arange(0, 9, dtype=np.uint8)

ipcc_node_codes = np.array([
    0,
    10, 11, 12,
    20, 21, 22, 23, 24, 29,
    30, 301, 31, 32, 333, 334, 335, 337, 34, 351, 352, 353, 357, 358, 39,
    40, 41, 430, 432, 436, 44, 451, 452, 453, 457, 458, 49,
    50, 51, 52, 53, 59,
    60, 61, 62, 69,
    70, 71, 79,
    80, 81, 89,
], dtype=np.uint16)
# Description of each node code in numeric_to_ipcc_node_code (below)

ipcc_change_codes = np.array(
    [0] + [10 * start + end for start in range(1, 9) for end in range(1, 9)],
    dtype=np.uint8,
)

ipcc_summary_codes = np.array(
    [0] + [10 * start + end for start in range(1, 9) for end in range(1, 9)],
    dtype=np.uint16,
)


# Converts numeric classes into IPCC land use categories
numeric_to_ipcc_class = {
    1: "Settlements",
    2: "Cropland",
    3: "Forest Land",
    4: "Grassland",
    5: "Wetlands",
    6: "Other Land (Bare)",
    7: "Other Land (Water)",
    8: "Other Land (Snow/Ice)"
}

# Converts numeric classes into IPCC land use change categories
numeric_to_ipcc_change = {
    int(f"{from_code}{to_code}"): (
        f"{from_class} remaining {to_class}"
        if from_code == to_code
        else f"{from_class} to {to_class}"
    )
    for from_code, from_class in numeric_to_ipcc_class.items()
    for to_code, to_class in numeric_to_ipcc_class.items()
}

# Converts numeric node codes into text description of rule applied for land use classification
numeric_to_ipcc_node_code = {
    # 1) Settlements and Infrastructure:
        10: "Built from GLAD data",
        11: "Built following tall vegetation loss before built LC",
        12: "Built after first built LC",

    # 2) Cropland:
        20: "Crop from GLAD data",
        21: "Crop from oil palm extent or planting year",
        22: "Crop from SDPT tree crop extent (not oil palm)",
        23: "Crop following tall vegetation loss before crop LC",
        24: "Crop from TCL + permanent agriculture driver",

    # 3) Forest:
        30 : "Forest from GLAD tall vegetation",
        31 : "Forest from SDPT planted forest extent",
        32 : "Forest from GMW mangrove extent",
        333: "Forest from shifting cultivation driver",
        334: "Forest from logging driver",
        335: "Forest from wildfire driver",
        337: "Forest from natural disturbance driver",
        34 : "Unstocked forest after TCL and before oil palm planting",
        353: "Forest from mixed tall/short vegetation rule",
        357: "Forest from vegetation/water transition rule",
        358: "Forest from mixed snow/ice rule",
        39 : "Forest from majority years rule",

    # 4) Grassland:
        40 : "Grass from GLAD short vegetation",
        41 : "Grass from TCL + permanent agriculture driver + GPW cultivated grassland extent",
        430: "Grass from TCL + unknown driver",
        432: "Grass from TCL + hard commodities driver",
        436: "Grass from TCL + settlements/infrastructure driver",
        44 : "Grass prior to oil palm establishment",
        453: "Grass from mixed tall/short vegetation rule",
        457: "Grass from vegetation/water transition rule",
        458: "Grass from mixed snow/ice rule",
        49 : "Grass from majority years rule",

    # 5) Wetland:
        50: "Wetland from GLAD data",
        51: "Wetland from water/wetland/built transition rule",
        52: "Wetland from vegetation/water transition rule",
        53: "Wetland from bare/ice to water/wetland transition rule",
        59: "Wetland from majority years in mixed water rule",

    # 6) Other Land:
        60: "Bare from GLAD data",
        61: "Bare from mixed bare + tall/short vegetation rule",
        62: "Bare from mixed snow/ice rule",
        69: "Bare from majority years rule",

        70: "Water from GLAD data",
        79: "Water from majority years in mixed water rule",

        80: "Snow/ice from GLAD data",
        89: "Snow/ice from majority years rule",

}



# Converts numeric ISO values to ISO codes
# From https://github.com/wri/project-zeno-data-infra/blob/main/notebooks/grasslands_areas_gadm_2000-2022.ipynb
numeric_to_alpha3 = {
    4: 'AFG', 248: 'ALA', 8: 'ALB', 12: 'DZA', 16: 'ASM', 20: 'AND', 24: 'AGO', 660: 'AIA',
    10: 'ATA', 28: 'ATG', 32: 'ARG', 51: 'ARM', 533: 'ABW', 36: 'AUS', 40: 'AUT', 31: 'AZE',
    44: 'BHS', 48: 'BHR', 50: 'BGD', 52: 'BRB', 112: 'BLR', 56: 'BEL', 84: 'BLZ', 204: 'BEN',
    60: 'BMU', 64: 'BTN', 68: 'BOL', 535: 'BES', 70: 'BIH', 72: 'BWA', 74: 'BVT', 76: 'BRA',
    86: 'IOT', 96: 'BRN', 100: 'BGR', 854: 'BFA', 108: 'BDI', 132: 'CPV', 116: 'KHM', 120: 'CMR',
    124: 'CAN', 136: 'CYM', 140: 'CAF', 148: 'TCD', 152: 'CHL', 156: 'CHN', 162: 'CXR', 166: 'CCK',
    170: 'COL', 174: 'COM', 178: 'COG', 180: 'COD', 184: 'COK', 188: 'CRI', 384: 'CIV', 191: 'HRV',
    192: 'CUB', 531: 'CUW', 196: 'CYP', 203: 'CZE', 208: 'DNK', 262: 'DJI', 212: 'DMA', 214: 'DOM',
    218: 'ECU', 818: 'EGY', 222: 'SLV', 226: 'GNQ', 232: 'ERI', 233: 'EST', 748: 'SWZ', 231: 'ETH',
    238: 'FLK', 234: 'FRO', 242: 'FJI', 246: 'FIN', 250: 'FRA', 254: 'GUF', 258: 'PYF', 260: 'ATF',
    266: 'GAB', 270: 'GMB', 268: 'GEO', 276: 'DEU', 288: 'GHA', 292: 'GIB', 300: 'GRC', 304: 'GRL',
    308: 'GRD', 312: 'GLP', 316: 'GUM', 320: 'GTM', 831: 'GGY', 324: 'GIN', 624: 'GNB', 328: 'GUY',
    332: 'HTI', 334: 'HMD', 336: 'VAT', 340: 'HND', 344: 'HKG', 348: 'HUN', 352: 'ISL', 356: 'IND',
    360: 'IDN', 364: 'IRN', 368: 'IRQ', 372: 'IRL', 833: 'IMN', 376: 'ISR', 380: 'ITA', 388: 'JAM',
    392: 'JPN', 832: 'JEY', 400: 'JOR', 398: 'KAZ', 404: 'KEN', 296: 'KIR', 408: 'PRK', 410: 'KOR',
    414: 'KWT', 417: 'KGZ', 418: 'LAO', 428: 'LVA', 422: 'LBN', 426: 'LSO', 430: 'LBR', 434: 'LBY',
    438: 'LIE', 440: 'LTU', 442: 'LUX', 446: 'MAC', 450: 'MDG', 454: 'MWI', 458: 'MYS', 462: 'MDV',
    466: 'MLI', 470: 'MLT', 584: 'MHL', 474: 'MTQ', 478: 'MRT', 480: 'MUS', 175: 'MYT', 484: 'MEX',
    583: 'FSM', 498: 'MDA', 492: 'MCO', 496: 'MNG', 499: 'MNE', 500: 'MSR', 504: 'MAR', 508: 'MOZ',
    104: 'MMR', 516: 'NAM', 520: 'NRU', 524: 'NPL', 528: 'NLD', 540: 'NCL', 554: 'NZL', 558: 'NIC',
    562: 'NER', 566: 'NGA', 570: 'NIU', 574: 'NFK', 807: 'MKD', 580: 'MNP', 578: 'NOR', 512: 'OMN',
    586: 'PAK', 585: 'PLW', 275: 'PSE', 591: 'PAN', 598: 'PNG', 600: 'PRY', 604: 'PER', 608: 'PHL',
    612: 'PCN', 616: 'POL', 620: 'PRT', 630: 'PRI', 634: 'QAT', 638: 'REU', 642: 'ROU', 643: 'RUS',
    646: 'RWA', 652: 'BLM', 654: 'SHN', 659: 'KNA', 662: 'LCA', 663: 'MAF', 666: 'SPM', 670: 'VCT',
    882: 'WSM', 674: 'SMR', 678: 'STP', 682: 'SAU', 686: 'SEN', 688: 'SRB', 690: 'SYC', 694: 'SLE',
    702: 'SGP', 534: 'SXM', 703: 'SVK', 705: 'SVN', 90: 'SLB', 706: 'SOM', 710: 'ZAF', 239: 'SGS',
    728: 'SSD', 724: 'ESP', 144: 'LKA', 729: 'SDN', 740: 'SUR', 744: 'SJM', 752: 'SWE', 756: 'CHE',
    760: 'SYR', 158: 'TWN', 762: 'TJK', 834: 'TZA', 764: 'THA', 626: 'TLS', 768: 'TGO', 772: 'TKL',
    776: 'TON', 780: 'TTO', 788: 'TUN', 792: 'TUR', 795: 'TKM', 796: 'TCA', 798: 'TUV', 800: 'UGA',
    804: 'UKR', 784: 'ARE', 826: 'GBR', 840: 'USA', 581: 'UMI', 858: 'URY', 860: 'UZB', 548: 'VUT',
    862: 'VEN', 704: 'VNM', 92: 'VGB', 850: 'VIR', 876: 'WLF', 732: 'ESH', 887: 'YEM', 894: 'ZMB',
    716: 'ZWE', 0: 'NA'
}

iso_to_country = {
    'ABW': 'Aruba', 'AFG': 'Afghanistan', 'AGO': 'Angola', 'AIA': 'Anguilla', 'ALA': 'Åland Islands', 'ALB': 'Albania', 'AND': 'Andorra', 'ARE': 'United Arab Emirates', 'ARG': 'Argentina',
    'ARM': 'Armenia', 'ATF': 'French Southern Territories', 'ATG': 'Antigua and Barbuda', 'AUS': 'Australia', 'AUT': 'Austria', 'AZE': 'Azerbaijan', 'BDI': 'Burundi', 'BEL': 'Belgium', 'BEN': 'Benin',
    'BES': 'Bonaire', 'BFA': 'Burkina Faso', 'BGD': 'Bangladesh', 'BGR': 'Bulgaria', 'BHR': 'Bahrain', 'BHS': 'Bahamas', 'BIH': 'Bosnia and Herzegovina', 'BLM': 'Saint Barthélemy', 'BLR': 'Belarus',
    'BLZ': 'Belize', 'BMU': 'Bermuda', 'BOL': 'Bolivia', 'BRA': 'Brazil', 'BRB': 'Barbados', 'BRN': 'Brunei', 'BTN': 'Bhutan', 'BWA': 'Botswana', 'CAF': 'Central African Republic', 'CAN': 'Canada',
    'CHE': 'Switzerland', 'CHL': 'Chile', 'CHN': 'China', 'CIV': 'Côte d Ivoire', 'CMR': 'Cameroon', 'COD': 'DR Congo', 'COG': 'Republic of Congo', 'COL': 'Colombia',
    'COM': 'Comoros', 'CPV': 'Cape Verde', 'CRI': 'Costa Rica', 'CUB': 'Cuba', 'CUW': 'Curaçao', 'CYM': 'Cayman Islands', 'CYP': 'Cyprus', 'CZE': 'Czechia', 'DEU': 'Germany', 'DJI': 'Djibouti',
    'DMA': 'Dominica', 'DNK': 'Denmark', 'DOM': 'Dominican Republic', 'DZA': 'Algeria', 'ECU': 'Ecuador', 'EGY': 'Egypt', 'ERI': 'Eritrea', 'ESH': 'Western Sahara', 'ESP': 'Spain', 'EST': 'Estonia',
    'ETH': 'Ethiopia', 'FIN': 'Finland', 'FJI': 'Fiji', 'FLK': 'Falkland Islands', 'FRA': 'France', 'FRO': 'Faroe Islands', 'FSM': 'Micronesia', 'GAB': 'Gabon', 'GBR': 'United Kingdom',
    'GEO': 'Georgia', 'GGY': 'Guernsey', 'GHA': 'Ghana', 'GIB': 'Gibraltar', 'GIN': 'Guinea', 'GLP': 'Guadeloupe', 'GMB': 'Gambia', 'GNB': 'Guinea-Bissau', 'GNQ': 'Equatorial Guinea', 'GRC': 'Greece',
    'GRD': 'Grenada', 'GRL': 'Greenland', 'GTM': 'Guatemala', 'GUF': 'French Guiana', 'GUY': 'Guyana', 'HKG': 'Hong Kong', 'HND': 'Honduras', 'HRV': 'Croatia', 'HTI': 'Haiti', 'HUN': 'Hungary',
    'IDN': 'Indonesia', 'IMN': 'Isle of Man', 'IND': 'India', 'IRL': 'Ireland', 'IRN': 'Iran', 'IRQ': 'Iraq', 'ISL': 'Iceland', 'ISR': 'Israel', 'ITA': 'Italy', 'JAM': 'Jamaica', 'JEY': 'Jersey',
    'JOR': 'Jordan', 'JPN': 'Japan', 'KAZ': 'Kazakhstan', 'KEN': 'Kenya', 'KGZ': 'Kyrgyzstan', 'KHM': 'Cambodia', 'KIR': 'Kiribati', 'KNA': 'Saint Kitts and Nevis', 'KOR': 'Korea', 'KWT': 'Kuwait',
    'LAO': 'Laos', 'LBN': 'Lebanon', 'LBR': 'Liberia', 'LBY': 'Libya', 'LCA': 'Saint Lucia', 'LIE': 'Liechtenstein', 'LKA': 'Sri Lanka', 'LSO': 'Lesotho', 'LTU': 'Lithuania', 'LUX': 'Luxembourg',
    'LVA': 'Latvia', 'MAC': 'Macao', 'MAF': 'Saint Martin (French part)', 'MAR': 'Morocco', 'MCO': 'Monaco', 'MDA': 'Moldova', 'MDG': 'Madagascar', 'MDV': 'Maldives',
    'MEX': 'Mexico', 'MKD': 'North Macedonia', 'MLI': 'Mali', 'MLT': 'Malta', 'MMR': 'Myanmar', 'MNE': 'Montenegro', 'MNG': 'Mongolia', 'MOZ': 'Mozambique', 'MRT': 'Mauritania', 'MSR': 'Montserrat', 'MTQ': 'Martinique', 'MUS': 'Mauritius',
    'MWI': 'Malawi', 'MYS': 'Malaysia', 'MYT': 'Mayotte', 'NAM': 'Namibia', 'NCL': 'New Caledonia', 'NER': 'Niger', 'NFK': 'Norfolk Island', 'NGA': 'Nigeria', 'NIC': 'Nicaragua', 'NLD': 'Netherlands',
    'NOR': 'Norway', 'NPL': 'Nepal', 'NRU': 'Nauru', 'NZL': 'New Zealand', 'OMN': 'Oman', 'PAK': 'Pakistan', 'PAN': 'Panama', 'PER': 'Peru', 'PHL': 'Philippines', 'PLW': 'Palau', 'PNG': 'Papua New Guinea',
    'POL': 'Poland', 'PRI': 'Puerto Rico', 'PRK': 'Korea (the Democratic Peoples Republic of)', 'PRT': 'Portugal', 'PRY': 'Paraguay', 'PSE': 'Palestine, State of', 'QAT': 'Qatar', 'REU': 'Réunion',
    'ROU': 'Romania', 'RUS': 'Russia', 'RWA': 'Rwanda', 'SAU': 'Saudi Arabia', 'SDN': 'Sudan', 'SEN': 'Senegal',
    'SGP': 'Singapore','SJM': 'Svalbard and Jan Mayen', 'SLB': 'Solomon Islands', 'SLE': 'Sierra Leone', 'SLV': 'El Salvador', 'SMR': 'San Marino', 'SOM': 'Somalia', 'SPM': 'Saint Pierre and Miquelon',
    'SRB': 'Serbia', 'SSD': 'South Sudan', 'STP': 'Sao Tome and Principe', 'SUR': 'Suriname', 'SVK': 'Slovakia', 'SVN': 'Slovenia', 'SWE': 'Sweden',
    'SWZ': 'Swaziland', 'SXM': 'Sint Maarten', 'SYC': 'Seychelles', 'SYR': 'Syria', 'TCA': 'Turks and Caicos Islands', 'TCD': 'Chad', 'TGO': 'Togo', 'THA': 'Thailand', 'TJK': 'Tajikistan',
    'TKM': 'Turkmenistan', 'TLS': 'East Timor', 'TTO': 'Trinidad and Tobago', 'TUN': 'Tunisia', 'TUR': 'Turkey', 'TUV': 'Tuvalu', 'TWN': 'Taiwan', 'TZA': 'Tanzania',
    'UGA': 'Uganda', 'UKR': 'Ukraine', 'UMI': 'United States Minor Outlying Islands', 'URY': 'Uruguay', 'USA': 'USA', 'UZB': 'Uzbekistan', 'VAT': 'Holy See', 'VCT': 'Saint Vincent and the Grenadines',
    'VEN': 'Venezuela', 'VGB': 'British Virgin Islands', 'VIR': 'Virgin Islands, U.S.', 'VNM': 'Vietnam', 'VUT': 'Vanuatu', 'XAD': 'nan', 'XCA': 'nan', 'XCL': 'nan',
    'XKO': 'nan', 'XNC': 'nan', 'XPI': 'nan', 'XSP': 'nan', 'YEM': 'Yemen', 'ZAF': 'South Africa', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe',
    'NA': 'no_country'
}

iso_to_region = {
    'ABW': 'Tropical LAC', 'AFG': 'Non-tropical Asia', 'AGO': 'Tropical Africa', 'AIA': 'Tropical LAC', 'ALA': 'Europe', 'ALB': 'Europe', 'AND': 'Europe', 'ARE': 'Non-tropical Asia',
    'ARG': 'Non-tropical LAC', 'ARM': 'Non-tropical Asia', 'ATF': 'Non-tropical Africa', 'ATG': 'Tropical LAC', 'AUS': 'Non-tropical Asia', 'AUT': 'Europe', 'AZE': 'Non-tropical Asia',
    'BDI': 'Tropical Africa', 'BEL': 'Europe', 'BEN': 'Tropical Africa', 'BES': 'Tropical LAC', 'BFA': 'Tropical Africa', 'BGD': 'Tropical Asia', 'BGR': 'Europe', 'BHR': 'Non-tropical Asia',
    'BHS': 'Tropical LAC', 'BIH': 'Europe', 'BLM': 'Tropical LAC', 'BLR': 'Europe', 'BLZ': 'Tropical LAC', 'BMU': 'Tropical LAC', 'BOL': 'Tropical LAC', 'BRA': 'Tropical LAC', 'BRB': 'Tropical LAC',
    'BRN': 'Tropical Asia', 'BTN': 'Tropical Asia', 'BWA': 'Tropical Africa', 'CAF': 'Tropical Africa', 'CAN': 'North America', 'CHE': 'Europe', 'CHL': 'Non-tropical LAC', 'CHN': 'Non-tropical Asia',
    'CIV': 'Tropical Africa', 'CMR': 'Tropical Africa', 'COD': 'Tropical Africa', 'COG': 'Tropical Africa', 'COL': 'Tropical LAC', 'COM': 'Tropical Africa', 'CPV': 'Tropical Africa',
    'CRI': 'Tropical LAC', 'CUB': 'Tropical LAC', 'CUW': 'Tropical LAC', 'CYM': 'Tropical LAC', 'CYP': 'Europe', 'CZE': 'Europe', 'DEU': 'Europe', 'DJI': 'Tropical Africa', 'DMA': 'Tropical LAC',
    'DNK': 'Europe', 'DOM': 'Tropical LAC', 'DZA': 'Non-tropical Africa', 'ECU': 'Tropical LAC', 'EGY': 'Non-tropical Africa', 'ERI': 'Tropical Africa', 'ESH': 'Non-tropical Africa', 'ESP': 'Europe', 'EST': 'Europe',
    'ETH': 'Tropical Africa', 'FIN': 'Europe', 'FJI': 'Tropical Asia', 'FLK': 'Non-tropical LAC', 'FRA': 'Europe', 'FRO': 'Europe', 'FSM': 'Tropical Asia', 'GAB': 'Tropical Africa', 'GBR': 'Europe',
    'GEO': 'Non-tropical Asia', 'GGY': 'Europe', 'GHA': 'Tropical Africa', 'GIB': 'Europe', 'GIN': 'Tropical Africa', 'GLP': 'Tropical LAC', 'GMB': 'Tropical Africa', 'GNB': 'Tropical Africa',
    'GNQ': 'Tropical Africa', 'GRC': 'Europe', 'GRD': 'Tropical LAC', 'GRL': 'Europe', 'GTM': 'Tropical LAC', 'GUF': 'Tropical LAC', 'GUY': 'Tropical LAC', 'HKG': 'Non-tropical Asia', 'HND': 'Tropical LAC',
    'HRV': 'Europe', 'HTI': 'Tropical LAC', 'HUN': 'Europe', 'IDN': 'Tropical Asia', 'IMN': 'Europe', 'IND': 'Tropical Asia', 'IRL': 'Europe', 'IRN': 'Non-tropical Asia', 'IRQ': 'Non-tropical Asia',
    'ISL': 'Europe', 'ISR': 'Non-tropical Asia', 'ITA': 'Europe', 'JAM': 'Tropical LAC', 'JEY': 'Europe', 'JOR': 'Non-tropical Asia', 'JPN': 'Non-tropical Asia', 'KAZ': 'Non-tropical Asia', 'KEN': 'Tropical Africa',
    'KGZ': 'Non-tropical Asia', 'KHM': 'Tropical Asia', 'KIR': 'Tropical Asia', 'KNA': 'Tropical LAC', 'KOR': 'Non-tropical Asia', 'KWT': 'Non-tropical Asia', 'LAO': 'Tropical Asia',
    'LBN': 'Non-tropical Asia', 'LBR': 'Tropical Africa', 'LBY': 'Non-tropical Africa', 'LCA': 'Tropical LAC', 'LIE': 'Europe', 'LKA': 'Tropical Asia', 'LSO': 'Non-tropical Africa', 'LTU': 'Europe',
    'LUX': 'Europe', 'LVA': 'Europe', 'MAC': 'Tropical Asia', 'MAF': 'Tropical LAC', 'MAR': 'Non-tropical Africa', 'MCO': 'Europe', 'MDA': 'Europe', 'MDG': 'Tropical Africa', 'MDV': 'Tropical Africa',
    'MEX': 'Tropical LAC', 'MKD': 'Europe', 'MLI': 'Tropical Africa', 'MLT': 'Europe', 'MMR': 'Tropical Asia', 'MNE': 'Europe', 'MNG': 'Non-tropical Asia', 'MOZ': 'Tropical Africa', 'MRT': 'Tropical Africa',
    'MSR': 'Tropical LAC', 'MTQ': 'Tropical LAC', 'MUS': 'Tropical Africa', 'MWI': 'Tropical Africa', 'MYS': 'Tropical Asia', 'MYT': 'Tropical Africa', 'NAM': 'Tropical Africa', 'NCL': 'Tropical Asia', 'NER': 'Tropical Africa',
    'NFK': 'Non-tropical Asia', 'NGA': 'Tropical Africa', 'NIC': 'Tropical LAC', 'NLD': 'Europe', 'NOR': 'Europe', 'NPL': 'Tropical Asia', 'NRU': 'Non-tropical Asia', 'NZL': 'Non-tropical Asia',
    'OMN': 'Non-tropical Asia', 'PAK': 'Non-tropical Asia', 'PAN': 'Tropical LAC', 'PER': 'Tropical LAC', 'PHL': 'Tropical Asia', 'PLW': 'Tropical Asia', 'PNG': 'Tropical Asia', 'POL': 'Europe',
    'PRI': 'Tropical LAC', 'PRK': 'Non-tropical Asia', 'PRT': 'Europe', 'PRY': 'Tropical LAC', 'PSE': 'Non-tropical Asia', 'QAT': 'Non-tropical Asia', 'REU': 'Tropical Africa', 'ROU': 'Europe',
    'RUS': 'Non-tropical Asia', 'RWA': 'Tropical Africa', 'SAU': 'Non-tropical Asia', 'SDN': 'Tropical Africa', 'SEN': 'Tropical Africa', 'SGP': 'Tropical Asia', 'SJM': 'Europe', 'SLB': 'Tropical Asia',
    'SLE': 'Tropical Africa', 'SLV': 'Tropical LAC', 'SMR': 'Europe', 'SOM': 'Tropical Africa', 'SPM': 'North America', 'SRB': 'Europe', 'SSD': 'Tropical Africa', 'STP': 'Non-tropical Africa', 'SUR': 'Tropical LAC',
    'SVK': 'Europe', 'SVN': 'Europe', 'SWE': 'Europe', 'SWZ': 'Tropical Africa', 'SXM': 'Tropical LAC', 'SYC': 'Tropical Africa', 'SYR': 'Non-tropical Asia', 'TCA': 'Tropical LAC', 'TCD': 'Tropical Africa',
    'TGO': 'Tropical Africa', 'THA': 'Tropical Asia', 'TJK': 'Non-tropical Asia', 'TKM': 'Non-tropical Asia', 'TLS': 'Tropical Asia', 'TTO': 'Tropical LAC', 'TUN': 'Non-tropical Africa',
    'TUR': 'Non-tropical Asia', 'TUV': 'Tropical Asia', 'TWN': 'Non-tropical Asia', 'TZA': 'Tropical Africa', 'UGA': 'Tropical Africa', 'UKR': 'Europe', 'UMI': 'Tropical Asia', 'URY': 'Non-tropical LAC',
    'USA': 'North America', 'UZB': 'Non-tropical Asia', 'VAT': 'Europe', 'VCT': 'Tropical LAC', 'VEN': 'Tropical LAC', 'VGB': 'Tropical LAC', 'VIR': 'Tropical LAC', 'VNM': 'Tropical Asia',
    'VUT': 'Tropical Asia', 'XAD': 'Not tropical misc', 'XCA': 'Not tropical misc', 'XCL': 'Not tropical misc', 'XKO': 'Not tropical misc', 'XNC': 'Not tropical misc', 'XPI': 'Not tropical misc',
    'XSP': 'Not tropical misc', 'YEM': 'Non-tropical Asia', 'ZAF': 'Non-tropical Africa', 'ZMB': 'Tropical Africa', 'ZWE': 'Tropical Africa', 'NA': 'no_country'
}

# UN geoscheme regions from https://unstats.un.org/unsd/methodology/m49/ ("geographic regions" tab)
# Converted to dictionary by https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6a0bc6ac-dc2c-8327-9154-3e49e70c5022
iso_to_region_UN_geoscheme_L1 = {

    # Africa
    "DZA": "Africa", "EGY": "Africa", "LBY": "Africa", "MAR": "Africa", "SDN": "Africa", "TUN": "Africa", "ESH": "Africa", "IOT": "Africa",
    "BDI": "Africa", "COM": "Africa", "DJI": "Africa", "ERI": "Africa", "ETH": "Africa", "ATF": "Africa", "KEN": "Africa", "MDG": "Africa",
    "MWI": "Africa", "MUS": "Africa", "MYT": "Africa", "MOZ": "Africa", "REU": "Africa", "RWA": "Africa", "SYC": "Africa", "SOM": "Africa",
    "SSD": "Africa", "UGA": "Africa", "TZA": "Africa", "ZMB": "Africa", "ZWE": "Africa", "AGO": "Africa", "CMR": "Africa", "CAF": "Africa",
    "TCD": "Africa", "COG": "Africa", "COD": "Africa", "GNQ": "Africa", "GAB": "Africa", "STP": "Africa", "BWA": "Africa", "SWZ": "Africa",
    "LSO": "Africa", "NAM": "Africa", "ZAF": "Africa", "BEN": "Africa", "BFA": "Africa", "CPV": "Africa", "CIV": "Africa", "GMB": "Africa",
    "GHA": "Africa", "GIN": "Africa", "GNB": "Africa", "LBR": "Africa", "MLI": "Africa", "MRT": "Africa", "NER": "Africa", "NGA": "Africa",
    "SHN": "Africa", "SEN": "Africa", "SLE": "Africa", "TGO": "Africa",

    # North America
    "AIA": "North America", "ATG": "North America", "ABW": "North America", "BHS": "North America", "BRB": "North America", "BES": "North America",
    "VGB": "North America", "CYM": "North America", "CUB": "North America", "CUW": "North America", "DMA": "North America", "DOM": "North America",
    "GRD": "North America", "GLP": "North America", "HTI": "North America", "JAM": "North America", "MTQ": "North America", "MSR": "North America",
    "PRI": "North America", "BLM": "North America", "KNA": "North America", "LCA": "North America", "MAF": "North America", "VCT": "North America",
    "SXM": "North America", "TTO": "North America", "TCA": "North America", "VIR": "North America", "BLZ": "North America", "CRI": "North America",
    "SLV": "North America", "GTM": "North America", "HND": "North America", "MEX": "North America", "NIC": "North America", "PAN": "North America",
    "BMU": "North America", "CAN": "North America", "GRL": "North America", "SPM": "North America", "USA": "North America",

    # South America
    "ARG": "South America", "BOL": "South America", "BVT": "South America", "BRA": "South America", "CHL": "South America", "COL": "South America",
    "ECU": "South America", "FLK": "South America", "GUF": "South America", "GUY": "South America", "PRY": "South America", "PER": "South America",
    "SGS": "South America", "SUR": "South America", "URY": "South America", "VEN": "South America",

    # Antarctica
    "ATA": "Antarctica",

    # Asia
    "KAZ": "Asia", "KGZ": "Asia", "TJK": "Asia", "TKM": "Asia", "UZB": "Asia", "CHN": "Asia", "HKG": "Asia", "MAC": "Asia",
    "PRK": "Asia", "JPN": "Asia", "MNG": "Asia", "KOR": "Asia", "BRN": "Asia", "KHM": "Asia", "IDN": "Asia", "LAO": "Asia",
    "MYS": "Asia", "MMR": "Asia", "PHL": "Asia", "SGP": "Asia", "THA": "Asia", "TLS": "Asia", "VNM": "Asia", "AFG": "Asia",
    "BGD": "Asia", "BTN": "Asia", "IND": "Asia", "IRN": "Asia", "MDV": "Asia", "NPL": "Asia", "PAK": "Asia", "LKA": "Asia",
    "ARM": "Asia", "AZE": "Asia", "BHR": "Asia", "CYP": "Asia", "GEO": "Asia", "IRQ": "Asia", "ISR": "Asia", "JOR": "Asia",
    "KWT": "Asia", "LBN": "Asia", "OMN": "Asia", "QAT": "Asia", "SAU": "Asia", "PSE": "Asia", "SYR": "Asia", "TUR": "Asia",
    "ARE": "Asia", "YEM": "Asia",

    # Europe
    "BLR": "Europe", "BGR": "Europe", "CZE": "Europe", "HUN": "Europe", "POL": "Europe", "MDA": "Europe", "ROU": "Europe", "RUS": "Europe",
    "SVK": "Europe", "UKR": "Europe", "ALA": "Europe", "DNK": "Europe", "EST": "Europe", "FRO": "Europe", "FIN": "Europe", "GGY": "Europe",
    "ISL": "Europe", "IRL": "Europe", "IMN": "Europe", "JEY": "Europe", "LVA": "Europe", "LTU": "Europe", "NOR": "Europe", "SJM": "Europe",
    "SWE": "Europe", "GBR": "Europe", "ALB": "Europe", "AND": "Europe", "BIH": "Europe", "HRV": "Europe", "GIB": "Europe", "GRC": "Europe",
    "VAT": "Europe", "ITA": "Europe", "MLT": "Europe", "MNE": "Europe", "MKD": "Europe", "PRT": "Europe", "SMR": "Europe", "SRB": "Europe",
    "SVN": "Europe", "ESP": "Europe", "AUT": "Europe", "BEL": "Europe","FRA": "Europe", "DEU": "Europe", "LIE": "Europe", "LUX": "Europe",
    "MCO": "Europe", "NLD": "Europe", "CHE": "Europe",

    # Oceania
    "AUS": "Oceania", "CXR": "Oceania", "CCK": "Oceania", "HMD": "Oceania", "NZL": "Oceania", "NFK": "Oceania", "FJI": "Oceania", "NCL": "Oceania",
    "PNG": "Oceania", "SLB": "Oceania", "VUT": "Oceania", "GUM": "Oceania", "KIR": "Oceania", "MHL": "Oceania", "FSM": "Oceania", "NRU": "Oceania",
    "MNP": "Oceania", "PLW": "Oceania", "UMI": "Oceania", "ASM": "Oceania", "COK": "Oceania", "PYF": "Oceania", "NIU": "Oceania", "PCN": "Oceania",
    "WSM": "Oceania", "TKL": "Oceania", "TON": "Oceania", "TUV": "Oceania","WLF": "Oceania",
}

iso_to_region_UN_geoscheme_L2_L3 = {

    # Northern Africa
    "DZA": "Northern Africa", "EGY": "Northern Africa", "LBY": "Northern Africa", "MAR": "Northern Africa", "SDN": "Northern Africa", "TUN": "Northern Africa", "ESH": "Northern Africa",

    # Eastern Africa
    "IOT": "Eastern Africa", "BDI": "Eastern Africa", "COM": "Eastern Africa", "DJI": "Eastern Africa", "ERI": "Eastern Africa", "ETH": "Eastern Africa", "ATF": "Eastern Africa", "KEN": "Eastern Africa",
    "MDG": "Eastern Africa", "MWI": "Eastern Africa", "MUS": "Eastern Africa", "MYT": "Eastern Africa", "MOZ": "Eastern Africa", "REU": "Eastern Africa", "RWA": "Eastern Africa", "SYC": "Eastern Africa",
    "SOM": "Eastern Africa", "SSD": "Eastern Africa", "UGA": "Eastern Africa", "TZA": "Eastern Africa", "ZMB": "Eastern Africa", "ZWE": "Eastern Africa",

    # Middle Africa
    "AGO": "Middle Africa", "CMR": "Middle Africa", "CAF": "Middle Africa", "TCD": "Middle Africa", "COG": "Middle Africa", "COD": "Middle Africa",
    "GNQ": "Middle Africa", "GAB": "Middle Africa", "STP": "Middle Africa",

    # Southern Africa
    "BWA": "Southern Africa", "SWZ": "Southern Africa", "LSO": "Southern Africa", "NAM": "Southern Africa", "ZAF": "Southern Africa",

    # Western Africa
    "BEN": "Western Africa", "BFA": "Western Africa", "CPV": "Western Africa", "CIV": "Western Africa", "GMB": "Western Africa", "GHA": "Western Africa",
    "GIN": "Western Africa", "GNB": "Western Africa", "LBR": "Western Africa", "MLI": "Western Africa", "MRT": "Western Africa", "NER": "Western Africa",
    "NGA": "Western Africa", "SHN": "Western Africa", "SEN": "Western Africa", "SLE": "Western Africa", "TGO": "Western Africa",

    # Caribbean
    "AIA": "Caribbean", "ATG": "Caribbean", "ABW": "Caribbean", "BHS": "Caribbean", "BRB": "Caribbean", "BES": "Caribbean", "VGB": "Caribbean", "CYM": "Caribbean",
    "CUB": "Caribbean", "CUW": "Caribbean", "DMA": "Caribbean", "DOM": "Caribbean", "GRD": "Caribbean", "GLP": "Caribbean", "HTI": "Caribbean", "JAM": "Caribbean",
    "MTQ": "Caribbean", "MSR": "Caribbean", "PRI": "Caribbean", "BLM": "Caribbean", "KNA": "Caribbean", "LCA": "Caribbean", "MAF": "Caribbean", "VCT": "Caribbean",
    "SXM": "Caribbean", "TTO": "Caribbean", "TCA": "Caribbean", "VIR": "Caribbean",

    # Central America
    "BLZ": "Central America", "CRI": "Central America", "SLV": "Central America", "GTM": "Central America", "HND": "Central America", "MEX": "Central America",
    "NIC": "Central America", "PAN": "Central America",

    # South America
    "ARG": "South America", "BOL": "South America", "BVT": "South America", "BRA": "South America", "CHL": "South America", "COL": "South America",
    "ECU": "South America", "FLK": "South America", "GUF": "South America", "GUY": "South America", "PRY": "South America", "PER": "South America",
    "SGS": "South America", "SUR": "South America", "URY": "South America", "VEN": "South America",

    # Northern America
    "BMU": "Northern America", "CAN": "Northern America", "GRL": "Northern America", "SPM": "Northern America", "USA": "Northern America",

    # Antarctica
    "ATA": "Antarctica",

    # Central Asia
    "KAZ": "Central Asia", "KGZ": "Central Asia", "TJK": "Central Asia", "TKM": "Central Asia", "UZB": "Central Asia",

    # Eastern Asia
    "CHN": "Eastern Asia", "HKG": "Eastern Asia", "MAC": "Eastern Asia", "PRK": "Eastern Asia", "JPN": "Eastern Asia", "MNG": "Eastern Asia", "KOR": "Eastern Asia",

    # South-eastern Asia
    "BRN": "South-eastern Asia", "KHM": "South-eastern Asia", "IDN": "South-eastern Asia", "LAO": "South-eastern Asia", "MYS": "South-eastern Asia", "MMR": "South-eastern Asia",
    "PHL": "South-eastern Asia", "SGP": "South-eastern Asia", "THA": "South-eastern Asia", "TLS": "South-eastern Asia", "VNM": "South-eastern Asia",

    # Southern Asia
    "AFG": "Southern Asia", "BGD": "Southern Asia", "BTN": "Southern Asia", "IND": "Southern Asia", "IRN": "Southern Asia", "MDV": "Southern Asia",
    "NPL": "Southern Asia", "PAK": "Southern Asia", "LKA": "Southern Asia",

    # Western Asia
    "ARM": "Western Asia", "AZE": "Western Asia", "BHR": "Western Asia", "CYP": "Western Asia", "GEO": "Western Asia", "IRQ": "Western Asia", "ISR": "Western Asia", "JOR": "Western Asia",
    "KWT": "Western Asia", "LBN": "Western Asia", "OMN": "Western Asia", "QAT": "Western Asia", "SAU": "Western Asia", "PSE": "Western Asia",
    "SYR": "Western Asia", "TUR": "Western Asia", "ARE": "Western Asia", "YEM": "Western Asia",

    # Eastern Europe
    "BLR": "Eastern Europe", "BGR": "Eastern Europe", "CZE": "Eastern Europe", "HUN": "Eastern Europe", "POL": "Eastern Europe", "MDA": "Eastern Europe",
    "ROU": "Eastern Europe", "RUS": "Eastern Europe", "SVK": "Eastern Europe", "UKR": "Eastern Europe",

    # Northern Europe
    "ALA": "Northern Europe", "DNK": "Northern Europe", "EST": "Northern Europe", "FRO": "Northern Europe", "FIN": "Northern Europe", "GGY": "Northern Europe",
    "ISL": "Northern Europe", "IRL": "Northern Europe", "IMN": "Northern Europe", "JEY": "Northern Europe", "LVA": "Northern Europe", "LTU": "Northern Europe",
    "NOR": "Northern Europe", "SJM": "Northern Europe", "SWE": "Northern Europe", "GBR": "Northern Europe",

    # Southern Europe
    "ALB": "Southern Europe", "AND": "Southern Europe", "BIH": "Southern Europe", "HRV": "Southern Europe", "GIB": "Southern Europe", "GRC": "Southern Europe",
    "VAT": "Southern Europe", "ITA": "Southern Europe", "MLT": "Southern Europe", "MNE": "Southern Europe", "MKD": "Southern Europe", "PRT": "Southern Europe",
    "SMR": "Southern Europe", "SRB": "Southern Europe", "SVN": "Southern Europe", "ESP": "Southern Europe",

    # Western Europe
    "AUT": "Western Europe", "BEL": "Western Europe", "FRA": "Western Europe", "DEU": "Western Europe", "LIE": "Western Europe", "LUX": "Western Europe",
    "MCO": "Western Europe", "NLD": "Western Europe", "CHE": "Western Europe",

    # Australia and New Zealand
    "AUS": "Australia and New Zealand", "CXR": "Australia and New Zealand", "CCK": "Australia and New Zealand", "HMD": "Australia and New Zealand",
    "NZL": "Australia and New Zealand", "NFK": "Australia and New Zealand",

    # Melanesia
    "FJI": "Melanesia", "NCL": "Melanesia", "PNG": "Melanesia", "SLB": "Melanesia", "VUT": "Melanesia",

    # Micronesia
    "GUM": "Micronesia", "KIR": "Micronesia", "MHL": "Micronesia", "FSM": "Micronesia", "NRU": "Micronesia", "MNP": "Micronesia", "PLW": "Micronesia", "UMI": "Micronesia",

    # Polynesia
    "ASM": "Polynesia", "COK": "Polynesia", "PYF": "Polynesia", "NIU": "Polynesia", "PCN": "Polynesia", "WSM": "Polynesia", "TKL": "Polynesia", "TON": "Polynesia",
    "TUV": "Polynesia", "WLF": "Polynesia",
}

# Converts continent-ecozone codes to continent and ecozone labels
cont_eco_to_text = {
    1004: {"ecozone": "No data", "continent": "Africa"},
    1007: {"ecozone": "Subtropical dry forest", "continent": "Africa"},
    1008: {"ecozone": "Subtropical humid forest", "continent": "Africa"},
    1009: {"ecozone": "Subtropical mountain system", "continent": "Africa"},
    1010: {"ecozone": "Subtropical steppe", "continent": "Africa"},
    1014: {"ecozone": "No data- Assigned to temperate oceanic forest", "continent": "Africa"},
    1016: {"ecozone": "Tropical desert", "continent": "Africa"},
    1017: {"ecozone": "Tropical dry forest", "continent": "Africa"},
    1018: {"ecozone": "Tropical moist deciduous forest", "continent": "Africa"},
    1019: {"ecozone": "Tropical mountain system", "continent": "Africa"},
    1020: {"ecozone": "Tropical rainforest", "continent": "Africa"},
    1021: {"ecozone": "Tropical shrubland", "continent": "Africa"},
    1022: {"ecozone": "Water", "continent": "Africa"},

    2001: {"ecozone": "Boreal coniferous forest", "continent": "America North"},
    2002: {"ecozone": "Boreal mountain system", "continent": "America North"},
    2003: {"ecozone": "Boreal tundra woodland", "continent": "America North"},
    2004: {"ecozone": "No data", "continent": "America South"},
    2005: {"ecozone": "Polar", "continent": "America North"},
    2006: {"ecozone": "Subtropical desert", "continent": "America North"},
    2007: {"ecozone": "Subtropical dry forest", "continent": "America South"},
    2008: {"ecozone": "Subtropical humid forest", "continent": "America South"},
    2009: {"ecozone": "Subtropical mountain system", "continent": "America South"},
    2010: {"ecozone": "Subtropical steppe", "continent": "America South"},
    2011: {"ecozone": "Temperate continental forest", "continent": "America North"},
    2012: {"ecozone": "Temperate desert", "continent": "America North"},
    2013: {"ecozone": "Temperate mountain system", "continent": "America South"},
    2014: {"ecozone": "Temperate oceanic forest", "continent": "America South"},
    2015: {"ecozone": "Temperate steppe", "continent": "America South"},
    2016: {"ecozone": "Tropical desert", "continent": "America South"},
    2017: {"ecozone": "Tropical dry forest", "continent": "America South"},
    2018: {"ecozone": "Tropical moist deciduous forest", "continent": "America South"},
    2019: {"ecozone": "Tropical mountain system", "continent": "America South"},
    2020: {"ecozone": "Tropical rainforest", "continent": "America South"},
    2021: {"ecozone": "Tropical shrubland", "continent": "America South"},
    2022: {"ecozone": "Water", "continent": "America North"},

    3004: {"ecozone": "No data", "continent": "Antarctica"},
    3005: {"ecozone": "Polar", "continent": "Antarctica"},

    4001: {"ecozone": "Boreal coniferous forest", "continent": "Asia"},
    4002: {"ecozone": "Boreal mountain system", "continent": "Asia"},
    4003: {"ecozone": "Boreal tundra woodland", "continent": "Asia"},
    4004: {"ecozone": "No data", "continent": "Asia"},
    4005: {"ecozone": "Polar", "continent": "Asia"},
    4006: {"ecozone": "Subtropical desert", "continent": "Asia"},
    4008: {"ecozone": "Subtropical humid forest", "continent": "Asia"},
    4009: {"ecozone": "Subtropical mountain system", "continent": "Asia"},
    4011: {"ecozone": "Temperate continental forest", "continent": "Asia"},
    4012: {"ecozone": "Temperate desert", "continent": "Asia"},
    4013: {"ecozone": "Temperate mountain system", "continent": "Asia"},
    4014: {"ecozone": "Temperate oceanic forest", "continent": "Asia"},
    4015: {"ecozone": "Temperate steppe", "continent": "Asia"},
    4016: {"ecozone": "Tropical desert", "continent": "Asia"},
    4017: {"ecozone": "Tropical dry forest", "continent": "Asia"},
    4018: {"ecozone": "Tropical moist deciduous forest", "continent": "Asia"},
    4019: {"ecozone": "Tropical mountain system", "continent": "Asia"},
    4020: {"ecozone": "Tropical rainforest", "continent": "Asia"},
    4022: {"ecozone": "Water", "continent": "Asia"},

    5007: {"ecozone": "Subtropical dry forest", "continent": "Asia continental"},
    5010: {"ecozone": "Subtropical steppe", "continent": "Asia continental"},
    5021: {"ecozone": "Tropical shrubland", "continent": "Asia continental"},

    6007: {"ecozone": "Subtropical dry forest", "continent": "Asia insular"},
    6010: {"ecozone": "Subtropical steppe", "continent": "Asia insular"},
    6021: {"ecozone": "Tropical shrubland", "continent": "Asia insular"},

    7001: {"ecozone": "Boreal coniferous forest", "continent": "Europe"},
    7002: {"ecozone": "Boreal mountain system", "continent": "Europe"},
    7003: {"ecozone": "Boreal tundra woodland", "continent": "Europe"},
    7004: {"ecozone": "No data", "continent": "Europe"},
    7005: {"ecozone": "Polar", "continent": "Europe"},
    7007: {"ecozone": "Subtropical dry forest", "continent": "Europe"},
    7009: {"ecozone": "Subtropical mountain system", "continent": "Europe"},
    7011: {"ecozone": "Temperate continental forest", "continent": "Europe"},
    7012: {"ecozone": "Temperate desert", "continent": "Europe"},
    7013: {"ecozone": "Temperate mountain system", "continent": "Europe"},
    7014: {"ecozone": "Temperate oceanic forest", "continent": "Europe"},
    7015: {"ecozone": "Temperate steppe", "continent": "Europe"},
    7022: {"ecozone": "Water", "continent": "Europe"},

    8004: {"ecozone": "No data", "continent": "New Zealand"},
    8008: {"ecozone": "Subtropical humid forest", "continent": "New Zealand"},
    8013: {"ecozone": "Temperate mountain system", "continent": "New Zealand"},
    8014: {"ecozone": "Temperate oceanic forest", "continent": "New Zealand"},
}

# Converts the watershed code to name
watershed_to_text = {
    1001: "Gulf of Mexico, North Atlantic Coast",
    1002: "United States, North Atlantic Coast",
    1003: "Mississippi - Missouri",
    1004: "Gulf Coast",
    1005: "California",
    1006: "Great Basin",
    1007: "North America, Colorado",
    1008: "Columbia and Northwestern United States",
    1009: "Fraser",
    1010: "Pacific and Arctic Coast",
    1011: "Saskatchewan - Nelson",
    1012: "Northwest Territories",
    1013: "Hudson Bay Coast",
    1014: "Atlantic Ocean Seaboard",
    1015: "Churchill",
    1016: "St Lawrence",
    1017: "St John",
    1018: "Mackenzie",
    2001: "Río Grande - Bravo",
    2002: "Mexico, Northwest Coast",
    2003: "Baja California",
    2004: "Mexico, Interior",
    2005: "North Gulf",
    2006: "Río Verde",
    2007: "Río Lerma",
    2008: "Pacific Central Coast",
    2009: "Río Balsas",
    2010: "Papaloapan",
    2011: "Isthmus of Tehuantepec",
    2012: "Grijalva - Usumacinta",
    2013: "Yucatán Peninsula",
    2014: "Southern Central America",
    2015: "Caribbean",
    3001: "Caribbean Coast",
    3002: "Magdalena",
    3003: "Orinoco",
    3004: "Northeast South America, South Atlantic Coast",
    3005: "Amazon",
    3006: "Tocantins",
    3007: "North Brazil, South Atlantic Coast",
    3008: "Parnaiba",
    3009: "East Brazil, South Atlantic Coast",
    3010: "Sao Francisco",
    3011: "Uruguay - Brazil, South Atlantic Coast",
    3012: "La Plata",
    3013: "North Argentina, South Atlantic Coast",
    3014: "South America, Colorado",
    3015: "Negro",
    3016: "South Argentina, South Atlantic Coast",
    3017: "Central Patagonia Highlands",
    3018: "Colombia - Ecuador, Pacific Coast",
    3019: "Peru, Pacific Coast",
    3020: "North Chile, Pacific Coast",
    3021: "South Chile, Pacific Coast",
    3022: "La Puna Region",
    3023: "Salinas Grandes",
    3024: "Mar Chiquita",
    3025: "Pampas Region",
    4001: "Spain - Portugal, Atlantic Coast",
    4002: "Douro",
    4003: "Tagus",
    4004: "Guadiana",
    4005: "Spain, South and East Coast",
    4006: "Guadalquivir",
    4007: "Ebro",
    4008: "Gironde",
    4009: "France, West Coast",
    4010: "Loire",
    4011: "Seine",
    4012: "Rhône",
    4013: "France, South Coast",
    4014: "England and Wales",
    4015: "Ireland",
    4016: "Scotland",
    4017: "Scheldt",
    4018: "Rhine",
    4019: "Maas",
    4020: "Ems - Weser",
    4021: "Po",
    4022: "Italy, West Coast",
    4023: "Tiber",
    4024: "Italy, East Coast",
    4025: "Danube",
    4026: "Elbe",
    4027: "Denmark - Germany Coast",
    4028: "Sweden",
    4029: "Wisla",
    4030: "Oder",
    4031: "Adriatic Sea - Greece - Black Sea Coast",
    4032: "Dnieper",
    4033: "Poland Coast",
    4034: "Neman",
    4035: "Dniester",
    4036: "Don",
    4037: "Volga",
    4038: "Ural",
    4039: "Daugava",
    4040: "Narva",
    4042: "Black Sea, North Coast",
    4043: "Caspian Sea Coast",
    4044: "Baltic Sea Coast",
    4045: "Neva",
    4046: "Mediterranean Sea Islands",
    4047: "Scandinavia, North Coast",
    4048: "Finland",
    4049: "Russia, Barents Sea Coast",
    4050: "Arctic Ocean Islands",
    4051: "Northern Dvina",
    4052: "Iceland",
    5001: "Amur",
    5002: "Bo Hai - Korean Bay, North Coast",
    5003: "Russia, South East Coast",
    5004: "Gobi Interior",
    5005: "Ziya He, Interior",
    5006: "Huang He",
    5007: "Tarim Interior",
    5008: "Plateau of Tibet Interior",
    5009: "Yangtze",
    5010: "China Coast",
    5011: "Xun Jiang",
    5012: "South China Sea Coast",
    5013: "Kiribati - Nauru",
    5014: "North and South Korea",
    5015: "Andaman - Nicobar Islands",
    5016: "Hong (Red River)",
    5017: "Viet Nam, Coast",
    5018: "Mekong",
    5019: "Gulf of Thailand Coast",
    5020: "Chao Phraya",
    5021: "Peninsula Malaysia",
    5022: "Salween",
    5023: "Sittang",
    5024: "Irrawaddy",
    5025: "Bay of Bengal, North East Coast",
    5026: "Hainan",
    5027: "Sumatra",
    5028: "Java - Timor",
    5029: "Irian Jaya Coast",
    5030: "Taiwan",
    5031: "Sulawesi",
    5032: "Kalimantan",
    5033: "North Borneo Coast",
    5034: "Philippines",
    5035: "Ganges - Bramaputra",
    5036: "Yasai",
    5037: "Brahamani",
    5038: "Mahandi",
    5039: "India North East Coast",
    5040: "Godavari",
    5041: "Krishna",
    5042: "Pennar",
    5043: "India East Coast",
    5044: "Cauvery",
    5045: "India South Coast",
    5046: "India West Coast",
    5047: "Tapti",
    5048: "Narmada",
    5049: "Mahi",
    5050: "Sabarmati",
    5051: "Indus",
    5052: "Sri Lanka",
    5053: "Fly",
    5054: "Papua New Guinea Coast",
    5055: "Wake - Marshall Islands",
    5056: "Japan",
    5057: "Palau and East Indonesia",
    5058: "North Marina Islands and Guam",
    5059: "Sepik",
    5060: "Micronesia",
    5061: "Tuvalu",
    5062: "Solomon Islands",
    5063: "Lena",
    5064: "Siberia, North Coast",
    5065: "Yenisey",
    5066: "Kara Sea Coast",
    5067: "Ob",
    5068: "Siberia, West Coast",
    6001: "Black Sea, South Coast",
    6002: "Mediterranean Sea, East Coast",
    6003: "Caspian Sea, South West Coast",
    6004: "Tigris - Euphrates",
    6005: "Eastern Jordan - Syria",
    6006: "Dead Sea",
    6007: "Sinai Peninsula",
    6008: "Red Sea, East Coast",
    6009: "Arabian Peninsula",
    6010: "Persian Gulf Coast",
    6011: "Central Iran",
    6012: "Arabian Sea Coast",
    6013: "Hamun-i-Mashkel",
    6014: "Helmand",
    6015: "Farahrud",
    6016: "Caspian Sea, East Coast",
    6017: "Amu Darya",
    6018: "Syr Darya",
    6019: "Lake Balkash",
    7001: "Senegal",
    7002: "Niger",
    7003: "Nile",
    7004: "Shebelli - Juba",
    7005: "Congo",
    7006: "Zambezi",
    7007: "Limpopo",
    7008: "Orange",
    7009: "Lake Chad",
    7010: "Rift Valley",
    7011: "Africa, South Interior",
    7014: "Africa, North Interior",
    7015: "Madasgacar",
    7016: "South Africa, South Coast",
    7017: "Africa, Indian Ocean Coast",
    7018: "Africa, East Central Coast",
    7019: "Africa, Red Sea - Gulf of Aden Coast",
    7020: "Mediterranean South Coast",
    7021: "Africa, West Coast",
    7022: "Gulf of Guinea",
    7023: "Angola, Coast",
    7024: "South Africa, West Coast",
    7025: "Namibia, Coast",
    7026: "Africa, North West Coast",
    7027: "Volta",
    8001: "Australia, West Coast",
    8002: "Australia, North Coast",
    8003: "Australia, Interior",
    8004: "Australia, South Coast",
    8005: "Australia, East Coast",
    8006: "Murray - Darling",
    8007: "South Pacific Islands",
    8008: "New Zealand",
    8009: "Tasmania"
}
drivers_to_text = {
    1: "Permanent agriculture",
    2: "Hard commodities",
    3: "Shifting cultivation",
    4: "Logging",
    5: "Wildfire",
    6: "Settlements and infrastructure",
    7: "Other natural disturbances"
}

# Converts the WDPA code to type
WDPA_to_text = {
    0: "NA",
    1: "Category Ia",
    2: "Category Ib",
    3: "Category II",
    4: "Category III",
    5: "Category IV",
    6: "Category V",
    7: "Category VI",
    8: "UNESCO-MAB Biosphere Reserve",
    9: "World Heritage Site (natural or mixed)",
    10: "Ramsar Site, Wetland of International Importance",
    11: "Not Reported",
    12: "Not Applicable",
    13: "Not Assigned"
}

BRA_biomes_to_text = {
    0: "NA",
    1: "Caatinga",
    2: "Cerrado",
    3: "Pantanal",
    4: "Pampa",
    5: "Amazônia",
    6: "Mata Atlântica"
}

managed_land_to_text = {
    0: "NA",
    1: "managed",
    2: "unmanaged",
}

forest_age_category_to_text = {
    0: 'non_forest',
    1: '1_5yr',
    6: '6_20yr',
    21: '21_40yr',
    41: '41_60yr',
    61: '61_80yr',
    81: '81_100yr',
    101: '>100yr'
}

GLAD_LC_to_text = {
    0: "Unassigned",
    1: "Forest",
    2: "Cropland",
    3: "Settlement",
    4: "Wetland",
    5: "Grassland",
    6: "Other land",
    7: "Not specified"
}

land_use_zonal_stats_table_folder = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/IPCC_land_use_v{IPCC_LU_version_underscore}_standard_global/"
veg_local_zonal_stats_table_folder = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/vegetation_v{veg_model_version_underscore}_standard_global/"
SOC_local_zonal_stats_table_folder = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/SOC_v{SOC_model_version_underscore}_standard_global/"



#######
### Output jpeg creation
#######

# Country shapefile, with small islands removed for visual simplicity
original_shapefile_path = "/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/world-administrative-boundaries_simple__20250102.shp"
reprojected_shapefile_path = "/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/world-administrative-boundaries_simple__20250102_reproj.shp"

local_jpeg_folder_vegetation = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/vegetation/v{veg_model_version_underscore}_standard_global/"
local_jpeg_folder_LULUCF = f"/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/LULUCF_totals/{LULUCF_full_version_underscore}/"
local_jpeg_folder_cropland = f"/mnt/c/GIS/AFOLU_flux_model/cropland_emissions/20250828/4x4km_aggregated_maps/"
local_jpeg_folder_livestock = f"/mnt/c/GIS/AFOLU_flux_model/livestock_emissions/20251223/4x4km_aggregated_maps/"
local_jpeg_folder_AFOLU = f"/mnt/c/GIS/AFOLU_flux_model/AFOLU_totals/4x4km_aggregated_maps/v{veg_model_version_underscore}__standard__global/"

# CRS for jpegs (Robinson equal area)
Robinson_crs = "ESRI:54030"

# Graphical elements
land_bkgrnd = (245, 245, 245) # Color for land where no raster data (light gray)
# land_bkgrnd = (2, 2, 2) # Color for land where no raster data (black: for testing)
# land_bkgrnd = (245, 245, 220) # Color for land where no raster data (light yellow: for testing)
# ocean_color = (235, 235, 235) # Color for ocean (gray)
ocean_color = (255, 255, 255) # Color for ocean (white)
# ocean_color = (50, 50, 50) # Color for land where no raster data (dark gray: for testing)
boundary_color = (150, 150, 150) # Color for country boundaries (medium gray)
boundary_width = 0.2 # Width of country boundaries
panel_dims = (12, 6) # Map panel dimensions (width, height)
dpi_jpeg = 300 # dpi for output jpegs
legend_fontsize = 9 # Font size for legend titles and labels
colorbar_dimensions = [0.12, 0.17, 0.02, 0.13] # [left, bottom, width, height]

# Colors in RGB. Gross emissions and removals are subset of net flux palette.
# From https://colorbrewer2.org/#type=diverging&scheme=BrBG&n=10
net_colors_rgb = [(0, 60, 48), (1, 102, 94), (53, 151, 143), (128, 205, 193), (199, 234, 229),  # Used for removals
                  (246, 232, 195), (223, 194, 125), (191, 129, 45), (140, 81, 10), (84, 48, 5)  # Used for emissions
                  ]
# Defines desired percentiles for colors. Specifies where colors transition in the data.
# Setting neutral ends of sink and source is empirically based on the 0 value being around the 82nd percentile.
# From some experimentation, it's better not to encode a neutral percentile (or associated color) here or below.
# It dampens the colors around the neutral value (low emissions and removals) even more.
net_percentiles = [0.17, 0.25, 0.5, 0.77, 0.95,
                   1.05, 1.1, 1.2, 1.3, 1.5]
removals_percentiles = [5, 25, 50, 75, 99]
emissions_percentiles = [5, 25, 50, 75, 99]
removals_colors_rgb = net_colors_rgb[0:5]
emissions_colors_rgb = net_colors_rgb[5:]

# Percentile at which map color saturates and the min and max values on legend (0.5 -> 0.5 and 99.5 percentiles)
saturation_percentile = 0.5

veg_pres_text = f"Vegetation fluxes: v{veg_model_version}, {veg_outputs_years[0]}-{LC_last_year}"
organic_soil_pres_text = f"Organic soil: v{organic_soil_model_version}, 2021-2024"
mineral_soil_pres_text = f"Mineral soil: v{SOC_model_version}, 2021-2022"
cropland_pres_text = f"Cropland: vYYYYMMDD, ca. 2020"
livestock_pres_text = f"Livestock: vYYYYMMDD, ca. 2020"
legend_percentile_disclaimer = f"Legend value range represents {saturation_percentile} and {1-saturation_percentile} percentiles of fluxes."

# Output global aggregated jpeg names
three_panel_jpeg_base = f"three_panels__4km_aggregation__v{veg_model_version}"

# Colors for jpegs showing the fraction of gross emissions from LULUCF components
fraction_base_cmap = 'RdPu'


#######
### Sensitivity analysis
#######

# Variant names
low_EF = 'low_partial_dist_EF'
high_EF = 'high_partial_dist_EF'
alt_AGB = 'ctrees_starting_AGC'
alt_RF = 'alternative_RF'

model_type_options = ['standard', low_EF, high_EF, alt_AGB, alt_RF]
