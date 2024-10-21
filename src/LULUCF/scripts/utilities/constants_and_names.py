import math
import boto3

import numpy as np

########
### Constants
########

### Model version
model_version = 0.1

### s3 buckets
s3 = boto3.resource('s3')
short_bucket_prefix = "gfw2-data"
full_bucket_prefix = "s3://" + short_bucket_prefix
s3_client = boto3.client("s3")

### Pattern for tile_ids in regex form
tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"

### IPCC codes
forest_IPCC = 1
cropland_IPCC = 2
settlement_IPCC = 3
wetland_IPCC = 4
grassland_IPCC = 5
otherland_IPCC = 6

IPCC_class_max_val = 6  # Maximum value of IPCC class codes

### Model years
first_model_year = 2000  # First year of model
last_model_year = 2020   # Last year of model

# Number of years in interval.
interval_years = 5    #TODO: calculate programmatically in numba function rather than coded here-- for greater flexibility.

# Number of years of removals in a tree cover gain pixel
NF_F_gain_year = math.ceil(interval_years/2)

### Carbon constants

# Biomass to carbon ratios
biomass_to_carbon_non_mangrove = 0.47   # Conversion of biomass to carbon for non-mangrove forests
biomass_to_carbon_mangrove = 0.45   # Conversion of biomass to carbon for mangroves (IPCC wetlands supplement table 4.2)

# Default root:shoot when no Huang et al. 2021 is available. The average slope of the AGB:BGB relationship in Figure 3 of Mokany et al. 2006.
# and is only used where Huang et al. 2021 can't reach (remote Pacific islands).
default_r_s_non_mang = 0.26

rate_ratio_spreadsheet = 'http://gfw2-data.s3.amazonaws.com/climate/AFOLU_flux_model/LULUCF/rate_ratio_lookup_tables/rate_and_ratio_lookup_tables_20240718.xlsx'
mangrove_rate_ratio_tab = 'mang gain C ratio, for model'

# Non-mangrove deadwood C:AGC and litter C:AGC constants
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

# Aboveground carbon removal factor for oil palm (Mg C/ha/yr).
# From IPCC 2019 Refinement Cropland Table 5.3.
oil_palm_agc_rf = 2.4

# Aboveground carbon removal factor for trees outside forests (Mg C/ha/yr), assuming that the entire hectare is ToF
#TODO Confirm value and units and add source
trees_outside_forests_agc_rf_max = 2.8

# Global warming potentials (GWP)
gwp_ch4 = 27 # AR6 WG1 Table 7.15
gwp_n2o = 273 # AR6 WG1 Table 7.15

# Removal factors for deadwood and litter carbon
deadwood_c_NT_T_rf = 0  # NT->T removal factor
litter_c_NT_T_rf = 0  # NT->T removal factor
deadwood_c_T_T_rf = 0  # T->T removal factor
litter_c_T_T_rf = 0  # T->T removal factor


### GLCLU codes
cropland = 244
builtup = 250

tree_dry_min_height_code = 27
tree_dry_max_height_code = 48
tree_wet_min_height_code = 127
tree_wet_max_height_code = 148


### Miscellaneous

full_raster_dims = 40000    # Size of a 10x10 deg raster in pixels

# Threshold for height loss to be counted as tree loss (meters)
sig_height_loss_threshold = 5

# Height minimum for trees (meters)
tree_threshold = 5

# Converts grams to kilograms for burning of dry matter
g_to_kg = 10 ** -3

# Which carbon pools are emitted under different circumstances for full tree loss: AGC, BGC, deadwood C, litter C.
# Need to specify numpy datatype because they're used in the Numba functions, which need explicit datatypes
agc_emissions_only = np.array([1, 0, 0, 0]).astype('uint8')
all_but_bgc_emissions = np.array([1, 0, 1, 1]).astype('uint8')
biomass_emissions_only = np.array([1, 1, 0, 0]).astype('uint8')
all_non_soil_pools = np.array([1, 1, 1, 1]).astype('uint8')

SDPT_oil_palm_code = 1
SDPT_wood_fiber_code = 2
SDPT_other_code = 3

########
### File name paths and patterns
########

date_date_range_pattern = r'_\d{4}(_\d{4})?'   # Pattern for date (XXXX) or date range XXXX_YYYY in output file names

s3_out_dir = 'climate/AFOLU_flux_model/LULUCF/outputs'

local_log_path = "logs/"
s3_log_path = "climate/AFOLU_flux_model/LULUCF/model_logs/"
combined_log = "AFOLU_model_log"

# Local path for chunk stats
chunk_stats_path = "chunk_stats/"

land_cover_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/landcover/composite/"
land_cover_pattern = "land_cover"

vegetation_height_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/landcover/vegetation_height/"
vegetation_height_pattern = "vegetation_height"

annual_forest_disturbance_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/landcover/annual_forest_disturbance/raw/"
forest_disturbance_layer_name = "forest_disturbance"

agb_2000_path = f"{full_bucket_prefix}/climate/WHRC_biomass/WHRC_V4/Processed/"
agb_2000_pattern = "t_aboveground_biomass_ha_2000"

mangrove_agb_2000_path = f"{full_bucket_prefix}/climate/carbon_model/mangrove_biomass/processed/standard/20190220/"
mangrove_agb_2000_pattern = "mangrove_agb_t_ha_2000"

elevation_path = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/elevation/20190418/"
elevation_pattern = "elevation"

climate_domain_path = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418/"
climate_domain_pattern = "fao_ecozones_bor_tem_tro_processed"

precipitation_path = f"{full_bucket_prefix}/climate/carbon_model/inputs_for_carbon_pools/processed/precip/20190418/"
precipitation_pattern = "precip_mm_annual"

r_s_ratio_path = f"{full_bucket_prefix}/climate/carbon_model/BGB_AGB_ratio/processed/20230216/"
r_s_ratio_pattern = "BGB_AGB_ratio"

continent_ecozone_path = f"{full_bucket_prefix}/climate/carbon_model/fao_ecozones/ecozone_continent/20190116/processed/"
continent_ecozone_pattern = "fao_ecozones_continents_processed"

natural_forest_growth_curve_path = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/"
natural_forest_growth_curve_pattern = "natural_forest_mean_growth_rate__Mg_AGC_ha_yr"
natural_forest_growth_curve_intervals = ['0_5', '6_10', '11_15', '16_20', '21_100']

drivers_path = f"{full_bucket_prefix}/drivers_of_loss/1_km/processed/20241004/"
drivers_pattern = "drivers_of_TCL_1_km_20241004"

'''
From Radost Stanimirova via Slack 2024-10-18:
//1: Permanent agriculture
//2: Hard commodities
//3: Shifting cultivation
//4: Forest management
//5: "Wildfire
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


ifl_primary_path = f"{full_bucket_prefix}/climate/carbon_model/ifl_primary_merged/processed/20200724/"
ifl_primary_pattern = "ifl_2000_primary_2001_merged"

planted_forest_type_path = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/plantation_type/SDPTv2/20230911/"
planted_forest_type_pattern = "plantation_type_oilpalm_woodfiber_other"

planted_forest_removal_factor_path = f"{full_bucket_prefix}/climate/carbon_model/annual_removal_factor_planted_forest/SDPTv2_AGC/20230911/"
planted_forest_removal_factor_pattern = "annual_gain_rate_AGC_Mg_ha_planted_forest"

oil_palm_2000_extent_path = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/IDN_MYS_plantation_pre_2000/processed/20200724/"
oil_palm_2000_extent_pattern = "plantation_2000_or_earlier_processed"

oil_palm_first_year_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/organic_soils/inputs/processed/descals_plantation/year/20240823/"
oil_palm_first_year_pattern = "descals_year"

# Originally from gfw-data-lake, so it's in 400x400 windows
planted_forest_tree_crop_path = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/plantation_simpleType__planted_forest_tree_crop/SDPTv2/20230911/"
planted_forest_tree_crop_pattern = "planted_forest_tree_crop"

burned_area_path = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/burn_year/burn_year_10x10_clip_by_year/"
burned_area_pattern = "ba"

organic_soil_extent_path = f"{full_bucket_prefix}/climate/carbon_model/other_emissions_inputs/peatlands/processed/20230315/"
organic_soil_extent_pattern = "peat_mask_processed"


##### Outputs

outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs/"


### IPCC classes and change
IPCC_class_path = "IPCC_basic_classes"
IPCC_class_pattern = "IPCC_classes"
IPCC_change_path = "IPCC_basic_change"
IPCC_change_pattern = "IPCC_change"

land_state_node_path_part = "land_state_node"

AGC_density_path_part = "AGC_density_MgC_ha"
BGC_density_path_part = "BGC_density_MgC_ha"
deadwood_c_density_path_part = "deadwood_C_density_MgC_ha"
litter_c_density_path_part = "litter_C_density_MgC_ha"

# Carbon density patterns
agb_dens_pattern = "AGB_density_MgAGB_ha"
agc_dens_pattern = "AGC_density_MgC_ha"
bgc_dens_pattern = "BGC_density_MgC_ha"
deadwood_c_dens_pattern = "deadwood_C_density_MgC_ha"
litter_c_dens_pattern = "litter_C_density_MgC_ha"
soil_c_dens_pattern = "soil_c_MgC_ha"

carbon_pool_2000_date = "20240821"

agc_2000_path = f"{outputs_path}{AGC_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
agc_2000_pattern = f"{agc_dens_pattern}_2000"

bgc_2000_path = f"{outputs_path}{BGC_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
bgc_2000_pattern = f"{bgc_dens_pattern}_2000"

deadwood_c_2000_path = f"{outputs_path}{deadwood_c_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
deadwood_c_2000_pattern = f"{deadwood_c_dens_pattern}_2000"

litter_c_2000_path = f"{outputs_path}{litter_c_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
litter_c_2000_pattern = f"{litter_c_dens_pattern}_2000"

soil_c_2000_path = f"{full_bucket_prefix}/climate/carbon_model/carbon_pools/soil_carbon/intermediate_full_extent/standard/20231108/"
soil_c_2000_pattern = "soil_C_full_extent_2000_Mg_C_ha"

land_state_pattern = "land_state_node"

gain_year_count_pattern = "gain_year_count"

agc_rf_pattern = "AGC_removal_factor_UNITS_TBD" #TODO Specify RF units here

# Gross and net fluxes
agc_gross_emis_pattern = "AGC_gross_emis_MgC_ha"
bgc_gross_emis_pattern = "BGC_gross_emis_MgC_ha"
deadwood_c_gross_emis_pattern = "deadwood_C_gross_emis_MgC_ha"
litter_c_gross_emis_pattern = "litter_C_gross_emis_MgC_ha"

agc_gross_removals_pattern = "AGC_gross_removals_MgC_ha"
bgc_gross_removals_pattern = "BGC_gross_removals_MgC_ha"
deadwood_c_gross_removals_pattern = "deadwood_C_gross_removals_MgC_ha"
litter_c_gross_removals_pattern = "litter_C_gross_removals_MgC_ha"

agc_net_flux_pattern = "AGC_net_flux_MgC_ha"
bgc_net_flux_pattern = "BGC_net_flux_MgC_ha"
deadwood_c_net_flux_pattern = "deadwood_C_net_flux_MgC_ha"
litter_c_net_flux_pattern = "litter_C_net_flux_MgC_ha"

ch4_flux_pattern = "CH4_flux_MgCO2e_ha"
n2o_flux_pattern = "N2O_flux_MgCO2e_ha"