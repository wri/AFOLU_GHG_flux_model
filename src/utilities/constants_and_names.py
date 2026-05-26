import math
import boto3
import numpy as np

########
### Constants
########

### Model version
model_version = "1.0.3_1884_chunks"
model_version_underscore = model_version.replace(".", "_")

### s3 buckets
s3 = boto3.resource('s3')
short_bucket_prefix = "gfw2-data"
full_bucket_prefix = "s3://" + short_bucket_prefix
full_bucket_prefix_length = len(full_bucket_prefix)+1
s3_client = boto3.client("s3")

### Pattern for tile_ids in regex form
tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"
small_chunk_pattern = r'__-?\d+_-?\d+_-?\d+_-?\d+__'

### m^2 to hectares
m2_to_ha = 1/10000

### Model years in 5-year intervals
first_model_year_5_years = 2000  # First year of 5-year interval data
last_model_year_5_years = 2020   # Last year of 5-year interval data

# Number of years in five-year interval
five_year_interval_duration = 5
interval_end_years_5_years = list(range(first_model_year_5_years, last_model_year_5_years + 1, five_year_interval_duration))[1:]  # 2005, 2010, 2015, 2020

# Number of years of removals in a tree cover gain pixel (3 years in a 5-year interval)
NT_T_gain_year_count_default = math.ceil(five_year_interval_duration / 2)

### Model years in annual series
first_model_year_annual = 2015  # First year of annual data
last_model_year_annual = 2024   # Last year of annual data

years_annual = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
interval_end_years_annual = years_annual[1:]

possible_task_statuses = ["pending_", "loading_", "preprocessing_", "calculating_",
                          "zarr_population_", "uploading_", "error_"]

# Model interval types
intervals_five_years = "five_years"
intervals_annual = "annual"
intervals_hybrid = "hybrid"


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
partial_disturbance_emission_factor_table_name = "partial_disturbance_emission_factors_LULUCF_model__20250415.xlsx"
partial_disturbance_emission_factor_table_full_path = f"{EF_RF_C_ratio_spreadsheet_URL}{partial_disturbance_emission_factor_table_name}"
partial_disturbance_emission_factor_table_tab = "EF_combined"

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
# From IPCC 2019, Table 2.6, "Boreal forest- ground fire" (applied globally, though boreal)
Cf_forest_undisturbed = 0.15

other_landcover_node = 7


### Crop residue and grassland burning constants

# Value for cropland nodes in land state node decision tree (for gain, loss, or remaining)
cropland_node = 5

# Ratio of aboveground residue dry matter to harvested yield (Rag(T) (IPCC 2019, V4, Ch. 11, Table 11.1A- generic value)
cropland_residue_harvest_ratio = 1.0

# Emission factors for crop residue burning (IPCC 2019, V4, Ch. 2, Table 2.5-- agricultural residues)
Gef_CH4_crop_residue = 2.7
Gef_N2O_crop_residue = 0.07

# Combustion factor for crop residue burning (IPCC 2019, V4, Ch. 2, Table 2.6-- agricultural residues, other crops)
Cf_crop_residue = 0.85

# Value for short/medium vegetation nodes in land state node decision tree (for gain, loss, or remaining)
grassland_node = 6

# Emission factors for savanna and grassland burning (IPCC 2019, V4, Ch. 2, Table 2.5-- savanna and grassland)
Gef_CH4_grassland = 2.3
Gef_N2O_grassland = 0.21

# Combustion factor for savanna and grassland burning (IPCC 2019, V4, Ch. 2, Table 2.6-- all savanna grasslands (mid/late dry season burns)
Cf_grassland = 0.77


### GLCLU cover codes
### Classifications proposed by Elise Mazur 2025-06-05 by email. Agreed on 2025-06-09
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

#Dimensions for global zarrs (80N to 80S)
global_width = int(round(360.0 / resolution))
global_height = int(round(160.0 / resolution))
origin_x = -180.0
origin_y = 80.0

agg_chunk_length_pixels = 4000 * 0.04  # Dimensions for 1x1 deg outputs at 0.04x0.04 deg resolution

# Threshold for height loss to be counted as disturbed (m)
sig_height_loss_threshold_abs = 5

# Threshold for height gain to be counted as regrowth in the same interval as a disturbance
sig_height_gain_threshold_abs = -5

# Height minimum for trees (meters)
tree_threshold = 5

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

# Pattern for date (XXXX), date range (XXXX_YYYY), or date range (XXXX_YYYY_XXXX_YYYY, SOC only) with 1 or 2 leading _ in output file names
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
# wget -e robots=off --mirror --no-parent -r https://dap.ceda.ac.uk/neodc/esacci/biomass/data/agb/maps/v5.01/geotiff/2015/
agb_2015_dir_raw = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/AGB/raw/"
agb_2015_pattern_raw = "ESACCI-BIOMASS-L4-AGB-MERGED-100m-2015-fv5.0"
agb_2015_dir_processed = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/AGB/processed/20250217/"
agb_2015_pattern = "AGB_2015_ESA_CCI_Mg_AGB_ha"

agb_stdev_2015_dir_raw = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/AGB_stdev/raw/"
agb_stdev_2015_pattern_raw = "ESACCI-BIOMASS-L4-AGB_SD-MERGED-100m-2015-fv5.0"
agb_stdev_2015_dir_processed = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/AGB_stdev/processed/20250217/"
agb_stdev_2015_pattern = "AGB_stdev_2015_ESA_CCI_Mg_AGB_ha"

mangrove_agb_2000_dir = f"{full_bucket_prefix}/climate/carbon_model/mangrove_biomass/processed/standard/20190220/"
mangrove_agb_2000_pattern = "mangrove_agb_t_ha_2000"


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

# Carbon density pattern for LULUCF model outputs
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

## 2015
carbon_2015_creation_date = '20250930'

agc_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{agc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
agc_2015_raw_pattern = f"{agc_raw_dens_pattern}_2015"

bgc_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{bgc_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
bgc_2015_raw_pattern = f"{bgc_raw_dens_pattern}_2015"

deadwood_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{deadwood_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
deadwood_c_2015_raw_pattern = f"{deadwood_c_raw_dens_pattern}_2015"

litter_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{litter_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
litter_c_2015_raw_pattern = f"{litter_c_raw_dens_pattern}_2015"

non_soil_c_2015_raw_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{non_soil_c_raw_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
non_soil_c_2015_raw_pattern = f"{non_soil_c_raw_dens_pattern}_2015"

# Carbon density, masked by landcover composite
agc_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{agc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
agc_2015_LC_masked_pattern = f"{agc_LC_masked_dens_pattern}_2015"

bgc_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{bgc_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
bgc_2015_LC_masked_pattern = f"{bgc_LC_masked_dens_pattern}_2015"

deadwood_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{deadwood_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
deadwood_c_2015_LC_masked_pattern = f"{deadwood_c_LC_masked_dens_pattern}_2015"

litter_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{litter_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
litter_c_2015_LC_masked_pattern = f"{litter_c_LC_masked_dens_pattern}_2015"

non_soil_c_2015_LC_masked_dir = f"{full_bucket_prefix}/climate/ESA_CCI_biomass/v5_01/2015/year_2015_derived_carbon_pools/{non_soil_c_LC_masked_dens_pattern}/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/{carbon_2015_creation_date}/"
non_soil_c_2015_LC_masked_pattern = f"{non_soil_c_LC_masked_dens_pattern}_2015"


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

## Forest age
# Forest age in 2010 and 2015 (created together)
forest_age_2010_2015_run_date = '20250705'
forest_age_2010_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2010/standard/not_gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2010_pattern = "forest_age_not_gap_filled_2010"

forest_age_2015_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2015/standard/not_gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2015_pattern = "forest_age_not_gap_filled_2015"

forest_age_2010_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2010/standard/gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2010_gap_filled_pattern = "forest_age_gap_filled_2010"

forest_age_2015_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2015/standard/gap_filled/CHUNK_SIZE_pixels/{forest_age_2010_2015_run_date}/"
forest_age_2015_gap_filled_pattern = "forest_age_gap_filled_2015"

# Forest age in 2000 (age in 2000 is derived from gap-filled age in 2010, so there is no non-gap-filled age in 2000 data)
forest_age_2000_run_date = '20250707'
forest_age_2000_gap_filled_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2000/standard/age__years/gap_filled/CHUNK_SIZE_pixels/{forest_age_2000_run_date}/"
forest_age_2000_gap_filled_pattern = "forest_age_gap_filled_2000"

forest_age_2000_gap_filled_source_flag_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/forest_age/GAMI_v2_1/2000/standard/age__source_flag/gap_filled/CHUNK_SIZE_pixels/{forest_age_2000_run_date}/"
forest_age_2000_gap_filled_source_flag_pattern = "forest_age_gap_filled_2000__source_flag"

# Age at disturbance (1x1 deg resolution) from forthcoming Besnard et al. paper
global_age_at_disturbance_file = "s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/forest_age/age_pre_disturbance_Besnard_et_al/global_geotif/20250702/age_pre_disturbance_median_1deg_global__20250703.tif"

# Forest age pattern for use in the LULUCF model. Applies to any starting year (2000 or 2015).
forest_age_start_year_pattern = "forest_age_gap_filled_start_year"
forest_age_output_pattern = "forest_age_at_end_of_interval"

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

#TODO: @Mel Make sure all refrences to the old commented out names are updated
# Refactor Hansenize inputs: 
# Robinson_5_year_rates_processed_date or Robinson_20_year_rates_processed_date --> Robinson_processed_date
# Robinson_processed_date --> secondary_forest_curve_run_date

Robinson_processed_date = '20250616'

#Robinson 5-year rates
Robinson_5_year_raw_date = '20250616'
secondary_natural_forest_5_year_raw_dir =  f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/raw/{Robinson_5_year_raw_date}/"
secondary_natural_forest_0_5_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__0_5_years__nibble"
secondary_natural_forest_6_10_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__6_10_years__nibble"
secondary_natural_forest_11_15_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__11_15_years__nibble"
secondary_natural_forest_16_20_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__16_20_years__nibble"

secondary_natural_forest_0_5_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_0_5/"
secondary_natural_forest_6_10_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_6_10/"
secondary_natural_forest_11_15_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_11_15/"
secondary_natural_forest_16_20_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_16_20/"

#Robinson 20+-year rates
Robinson_20_year_raw_date = '20250516'
secondary_natural_forest_20_year_raw_dir =  f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/raw/{Robinson_20_year_raw_date}/"
secondary_natural_forest_21_40_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_40_years__nibble"
secondary_natural_forest_41_60_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__41_60_years__nibble"
secondary_natural_forest_61_80_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__61_80_years__nibble"
secondary_natural_forest_81_100_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__81_100_years__nibble"
secondary_natural_forest_21_100_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_100_years__nibble"

secondary_natural_forest_21_40_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_21_40/"
secondary_natural_forest_41_60_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_41_60/"
secondary_natural_forest_61_80_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_61_80/"
secondary_natural_forest_81_100_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_81_100/"
secondary_natural_forest_21_100_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{Robinson_processed_date}/rate_21_100/"

# secondary_natural_forest_raw_dir =  f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/raw/20250516/"
# secondary_natural_forest_0_5_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__0_5_years__nibble_20250516"   # both the raw raster name and processed pattern for hansenized tiles
# secondary_natural_forest_6_10_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__6_10_years__nibble_20250516"
# secondary_natural_forest_11_15_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__11_15_years__nibble_20250516"
# secondary_natural_forest_16_20_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__16_20_years__nibble_20250516"
# secondary_natural_forest_21_40_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_40_years__nibble_20250516"
# secondary_natural_forest_41_60_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__41_60_years__nibble_20250516"
# secondary_natural_forest_61_80_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__61_80_years__nibble_20250516"
# secondary_natural_forest_81_100_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__81_100_years__nibble_20250516"
# secondary_natural_forest_21_100_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_100_years__nibble_20250516"

secondary_forest_curve_run_date = '20250616'
natural_forest_growth_curve_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/{secondary_forest_curve_run_date}/"
natural_forest_growth_curve_pattern = "natural_forest_mean_growth_rate__Mg_AGC_ha_yr"
natural_forest_growth_curve_intervals = ['0_5', '6_10', '11_15', '16_20', '21_40', '41_60', '61_80', '81_100']

#TODO: @Mel Update to path pattern instead of processed_dir/ pattern in hansenize. Delete processed after.
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
oil_palm_first_year_pattern = "descals_year"

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

# Organic Soils
# Organic soil mask, from Hengl et al. under review (https://essd.copernicus.org/preprints/essd-2025-336/)
# and thresholded by Erin Glen for peat emissions model. Includes only pixels with organic soil probability >23%,
# a cutoff determined by OpenGeoHub and implemented by Erin.
organic_soil_extent_dir = "s3://gfw2-data/climate/AFOLU_flux_model/organic_soils/inputs/processed/peat_mask/OGH/tiles/"
organic_soil_extent_pattern = "ogh_mask"

### Soil organic carbon (SOC) timeseries (URIs from https://github.com/openlandmap/soildb/blob/main/tables/OpenLandMap_soildb_COGS.csv)
# From Hengl et al. under review (https://essd.copernicus.org/preprints/essd-2025-336/)
# Units of raw global COGs are kg C/m^3 for 0-30 cm, multiplied by 10 (rescale factor) to make COGs ints.
# Values in dict need to be lists (even with just 1 element) because of how uu.prepare_to_download_chunk works.
SOC_COGS = {
    "2000_2005": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20000101_20051231_g_epsg.4326_v20250204.tif"],
    "2005_2010": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20050101_20101231_g_epsg.4326_v20250204.tif"],
    "2010_2015": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20100101_20151231_g_epsg.4326_v20250204.tif"],
    "2015_2020": ["https://s3.opengeohub.org/global-soil/global_soil_props_v20250204_mosaics/oc_iso.10694.1995.mg.cm3_m_30m_b0cm..30cm_20150101_20201231_g_epsg.4326_v20250204.tif"]
}

SOC_timeseries_run_date = '20250819'
SOC_timeseries_base_output_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/soil_organic_carbon_timeseries/"

SOC_density_intervals = ['2000_2005', '2005_2010', '2010_2015', '2015_2020']
SOC_change_intervals = ['2000_2005_2005_2010', '2005_2010_2010_2015', '2010_2015_2015_2020', '2000_2005_2015_2020']   # Last one is the full-period delta

# Extent of raw COGs
SOC_density_full_extent_pattern = "SOC_density__full_extent__0-30cm_MgC"
SOC_density_full_extent_dir = f"{SOC_timeseries_base_output_dir}{SOC_density_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_change_full_extent_pattern = "SOC_change__full_extent__0-30cm_MgC"
SOC_change_full_extent_dir = f"{SOC_timeseries_base_output_dir}{SOC_change_full_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"

# Extent of mineral soil (excludes thresholded organic soil extent created by Erin Glen)
SOC_density_min_soil_extent_pattern = "SOC_density__mineral_soil_extent__0-30cm_MgC"
SOC_density_min_soil_extent_dir = f"{SOC_timeseries_base_output_dir}{SOC_density_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
SOC_change_min_soil_extent_pattern = "SOC_change__mineral_soil_extent__0-30cm_MgC"
SOC_change_min_soil_extent_dir = f"{SOC_timeseries_base_output_dir}{SOC_change_min_soil_extent_pattern}/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"

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

model_type_placeholder = "MODEL_TYPE"

outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs/version_{model_version_underscore}/"
outputs_path_mega_zarr = f"{outputs_path}mega_zarr/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/CHUNK_SIZE_pixels/RUN_DATE/"

### IPCC classes and change
IPCC_class_path = "IPCC_basic_class"
IPCC_class_pattern = "IPCC_class"
IPCC_node_path = "IPCC_nod_code"
IPCC_node_pattern = "IPCC_node"
IPCC_change_path = "IPCC_basic_change"
IPCC_change_pattern = "IPCC_change"

### IPCC codes
settlement_IPCC = 1
cropland_IPCC = 2
forest_IPCC = 3
grassland_IPCC = 4
wetland_IPCC = 5
otherland_IPCC = 6

IPCC_class_max_val = 6  # Maximum value of IPCC class codes

IPCC_class_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/LUC/{IPCC_class_path}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
IPCC_node_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/LUC/{IPCC_node_path}/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
IPCC_change_dir = f"s3://gfw2-data/climate/AFOLU_flux_model/LULUCF/outputs/LUC/{IPCC_change_path}/START_END/CHUNK_SIZE_pixels/RUN_DATE/"

land_state_pattern = "land_state_node"
land_state_node_fire_value = 9  # State nodes that end in this value had fire

agc_rf_pre_dist_pattern = "removal_factor__AGC__MgC"

# Gross and net fluxes (fluxes are in Mg CO2/ha/yr or Mg CO2e/ha/yr)
agc_gross_emis_pattern = "gross_emissions__AGC__MgCO2"
bgc_gross_emis_pattern = "gross_emissions__BGC__MgCO2"
deadwood_c_gross_emis_pattern = "gross_emissions__deadwood_C__MgCO2"
litter_c_gross_emis_pattern = "gross_emissions__litter_C__MgCO2"

agc_gross_removals_pattern = "gross_removals__AGC__MgCO2"
bgc_gross_removals_pattern = "gross_removals__BGC__MgCO2"
deadwood_c_gross_removals_pattern = "gross_removals__deadwood_C__MgCO2"
litter_c_gross_removals_pattern = "gross_removals__litter_C__MgCO2"

net_flux_agc_pattern = "net_flux__AGC__MgCO2"
net_flux_bgc_pattern = "net_flux__BGC__MgCO2"
net_flux_deadwood_c_pattern = "net_flux__deadwood_C__MgCO2"
net_flux_litter_c_pattern = "net_flux__litter_C__MgCO2"

ch4_flux_pattern = "gross_emissions__CH4__MgCO2e"
n2o_flux_pattern = "gross_emissions__N2O__MgCO2e"

gross_emis_all_C_pools_CO2_only_pattern = "gross_emissions__all_C_pools__CO2_only__MgCO2"
gross_emis_all_C_pools_non_CO2_only_pattern = "gross_emissions__all_C_pools__non_CO2_only__MgCO2e"
gross_emis_all_C_pools_all_gases_pattern = "gross_emissions__all_C_pools__all_gases__MgCO2e"

gross_removals_all_C_pools_pattern = "gross_removals__all_C_pools__MgCO2"

net_flux_all_C_pools_CO2_only_pattern = "net_flux__all_C_pools__CO2_only__MgCO2"
net_flux_all_C_pools_all_gases_pattern = "net_flux__all_C_pools__all_gases__MgCO2e"

# Intermediate outputs
gain_year_count_pattern = "gain_year_count_during_interval"
most_recent_year_not_tall_veg = "most_recent_year_not_tall_veg"
year_of_forest_loss = "year_of_forest_loss"
max_height_since_last_time_not_tall_veg = "max_height_since_last_time_not_tall_veg"
first_time_sig_loss_from_max_height = "first_time_sig_loss_from_max_height"
part_or_full_dist_in_earlier_intervals = "partial_or_full_dist_in_earlier_intervals"
part_or_full_dist_in_curr_interval = "partial_or_full_dist_in_current_interval"
times_burned_in_interval = "times_burned_in_current_interval"
agc_emission_factor = "AGC_emission_factor_CO2_only__fraction"
composite_primary_forest = "composite_primary_forest"

zarr_output_pattern = "global_zarr"
zarr_pixel_chunks = 10000

# Tolerance for difference between model and zarr chunk stat metrics.
# There's often some rounding/float error between them, so a small difference (~10^-8) is expected.
zarr_difference_tolerance = 0.05

# List of output directories from vegetation model with placeholders for parts of the directory
veg_core_output_dirs = [
    f"{outputs_path}{agc_modeled_dens_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{bgc_modeled_dens_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{deadwood_c_modeled_dens_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{litter_c_modeled_dens_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{agc_gross_emis_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{bgc_gross_emis_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{deadwood_c_gross_emis_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{litter_c_gross_emis_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{agc_gross_removals_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{bgc_gross_removals_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{deadwood_c_gross_removals_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{litter_c_gross_removals_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{ch4_flux_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{n2o_flux_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{land_state_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{agc_rf_pre_dist_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"
]

# Intermediate outputs from vegetation model
veg_intermediate_output_dirs = [
    f"{outputs_path}{forest_age_output_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{gain_year_count_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{most_recent_year_not_tall_veg}/{model_type_placeholder}/RUNSTART_END/CHUNK_SIZE_pixels/RUN_DATE/", # Years represent from model start to current interval end
    f"{outputs_path}{max_height_since_last_time_not_tall_veg}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{first_time_sig_loss_from_max_height}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{part_or_full_dist_in_earlier_intervals}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{part_or_full_dist_in_curr_interval}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{times_burned_in_interval}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{agc_emission_factor}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{composite_primary_forest}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/CHUNK_SIZE_pixels/RUN_DATE/"
]

# Summative outputs from core vegetation model
veg_summative_output_patterns = [
    gross_emis_all_C_pools_CO2_only_pattern, gross_emis_all_C_pools_non_CO2_only_pattern, gross_emis_all_C_pools_all_gases_pattern,
    gross_removals_all_C_pools_pattern,
    net_flux_agc_pattern, net_flux_bgc_pattern, net_flux_deadwood_c_pattern, net_flux_litter_c_pattern,
    net_flux_all_C_pools_CO2_only_pattern, net_flux_all_C_pools_all_gases_pattern,
    non_soil_c_modeled_dens_pattern
]

# Summative outputs from core vegetation model
veg_summative_output_dirs = [
    # Outputs per interval
    f"{outputs_path}{gross_emis_all_C_pools_CO2_only_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{gross_emis_all_C_pools_non_CO2_only_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{gross_emis_all_C_pools_all_gases_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{gross_removals_all_C_pools_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_agc_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_bgc_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_deadwood_c_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_litter_c_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_all_C_pools_CO2_only_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{net_flux_all_C_pools_all_gases_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/START_END/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/",
    f"{outputs_path}{non_soil_c_modeled_dens_pattern}/{model_type_placeholder}/MODEL_INTERVAL_TYPE_intervals/YEAR/PER_HA_OR_PIXEL/CHUNK_SIZE_pixels/RUN_DATE/"  # Only per interval
]

# Only specific datasets output to zarrs.
# Starts with non-summative outputs from the vegetation model
core_veg_outputs_to_zarr = [
    agc_modeled_dens_pattern, bgc_modeled_dens_pattern, deadwood_c_modeled_dens_pattern, litter_c_modeled_dens_pattern,
    agc_gross_emis_pattern, bgc_gross_emis_pattern, deadwood_c_gross_emis_pattern, litter_c_gross_emis_pattern,
    ch4_flux_pattern, n2o_flux_pattern,
    agc_gross_removals_pattern, bgc_gross_removals_pattern, deadwood_c_gross_removals_pattern, litter_c_gross_removals_pattern,
    land_state_pattern, composite_primary_forest, forest_age_output_pattern
]

# Also want to add the metadata for the summative outputs to the global zarr upfront for simplicity,
# rather than having to add more empty layers to the zarr at the summative stage
full_outputs_to_zarr = core_veg_outputs_to_zarr
full_outputs_to_zarr.extend(veg_summative_output_patterns)

pixel_area_global_zarr = "s3://gfw2-data/climate/AFOLU_flux_model/contextual_layer_global_zarr/pixel_area/20251106/global_pixel_area_20251106.zarr"


#######
### Output jpeg creation
#######

# Country shapefile, with small islands removed for visual simplicity
original_shapefile_path = "/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/world-administrative-boundaries_simple__20250102.shp"
reprojected_shapefile_path = "/mnt/c/GIS/AFOLU_flux_model/LULUCF/4x4km_aggregated_maps/world-administrative-boundaries_simple__20250102_reproj.shp"

# CRS for jpegs (Robinson equal area)
Robinson_crs = "ESRI:54030"

# Graphical elements
land_bkgrnd = (245, 245, 245) # Color for land where no raster data (light gray)
# land_bkgrnd = (2, 2, 2) # Color for land where no raster data (black: for testing)
# land_bkgrnd = (245, 245, 220) # Color for land where no raster data (light yellow: for testing)
ocean_color = (235, 235, 235) # Color for land where no raster data (very light gray)
# ocean_color = (255, 255, 255) # Color for land where no raster data (white)
# ocean_color = (50, 50, 50) # Color for land where no raster data (dark gray: for testing)
boundary_color = (150, 150, 150) # Color for country boundaries (medium gray)
boundary_width = 0.2 # Width of country boundaries
panel_dims = (12, 6) # Map panel dimensions (width, height)
dpi_jpeg = 300 # dpi for output jpegs
legend_fontsize = 9 # Font size for legend titles and labels
colorbar_dimensions = [0.14, 0.17, 0.02, 0.13] # [left, bottom, width, height]

pres_text = f"Land use vegetation fluxes (model v{model_version}, 2016-2024)"

# Output global aggregated jpeg names
three_panel_jpeg_base = f"three_panels__4km_aggregation__v{model_version}"


