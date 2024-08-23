import boto3

# General paths and constants

LC_uri = 's3://gfw2-data/landcover'

s3_out_dir = 'climate/AFOLU_flux_model/LULUCF/outputs'

s3 = boto3.resource('s3')
my_bucket = s3.Bucket('gfw2-data')
s3_client = boto3.client("s3")

tile_id_pattern = r"[0-9]{2}[A-Z][_][0-9]{3}[A-Z]"  # Pattern for tile_ids in regex form

IPCC_class_max_val = 6  # Maximum value of IPCC class codes

# IPCC codes
forest = 1
cropland = 2
settlement = 3
wetland = 4
grassland = 5
otherland = 6

first_year = 2000  # First year of model
last_year = 2020   # Last year of model

full_raster_dims = 40000    # Size of a 10x10 deg raster in pixels

interval_years = 5   # Number of years in interval. #TODO: calculate programmatically in numba function rather than coded here-- for greater flexibility.

# Threshold for height loss to be counted as tree loss (meters)
sig_height_loss_threshold = 5

biomass_to_carbon_non_mangrove = 0.47   # Conversion of biomass to carbon for non-mangrove forests
biomass_to_carbon_mangrove = 0.45   # Conversion of biomass to carbon for mangroves (IPCC wetlands supplement table 4.2)

# Default root:shoot when no Huang et al. 2021 is available. The average slope of the AGB:BGB relationship in Figure 3 of Mokany et al. 2006.
# and is only used where Huang et al. 2021 can't reach (remote Pacific islands).
default_r_s = 0.26

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

mang_no_data_val = 255   # NoData value in mangrove AGB raster


model_version = 0.1


# GLCLU codes
cropland = 244
builtup = 250

tree_dry_min_height_code = 27
tree_dry_max_height_code = 48
tree_wet_min_height_code = 127
tree_wet_max_height_code = 148

tree_threshold = 5   # Height minimum for trees (meters)


# File name paths and patterns

log_path = "climate/AFOLU_flux_model/LULUCF/model_logs/"
combined_log = "AFOLU_model_log"

agb_2000_path = "s3://gfw2-data/climate/WHRC_biomass/WHRC_V4/Processed/"
agb_2000_pattern = "t_aboveground_biomass_ha_2000"

mangrove_agb_2000_path = "s3://gfw2-data/climate/carbon_model/mangrove_biomass/processed/standard/20190220/"
mangrove_agb_2000_pattern = "mangrove_agb_t_ha_2000"

elevation_path = "s3://gfw2-data/climate/carbon_model/inputs_for_carbon_pools/processed/elevation/20190418/"
elevation_pattern = "elevation"

climate_domain_path = "s3://gfw2-data/climate/carbon_model/inputs_for_carbon_pools/processed/fao_ecozones_bor_tem_tro/20190418/"
climate_domain_pattern = "fao_ecozones_bor_tem_tro_processed"

precipitation_path = "s3://gfw2-data/climate/carbon_model/inputs_for_carbon_pools/processed/precip/20190418/"
precipitation_pattern = "precip_mm_annual"

r_s_ratio_path = "s3://gfw2-data/climate/carbon_model/BGB_AGB_ratio/processed/20230216/"
r_s_ratio_pattern = "BGB_AGB_ratio"

continent_ecozone_path = "s3://gfw2-data/climate/carbon_model/fao_ecozones/ecozone_continent/20190116/processed/"
continent_ecozone_pattern = "fao_ecozones_continents_processed"


### IPCC classes and change
IPCC_class_path = "IPCC_basic_classes"
IPCC_class_pattern = "IPCC_classes"
IPCC_change_path = "IPCC_basic_change"
IPCC_change_pattern = "IPCC_change"

land_state_pattern = "land_state_node"

agb_dens_pattern = "AGB_density_MgAGB_ha"
agc_dens_pattern = "AGC_density_MgC_ha"
bgc_dens_pattern = "BGC_density_MgC_ha"
deadwood_c_dens_pattern = "deadwood_C_density_MgC_ha"
litter_c_dens_pattern = "litter_C_density_MgC_ha"
agc_flux_pattern = "AGC_flux_MgC_ha"
bgc_flux_pattern = "BGC_flux_MgC_ha"
deadwood_c_flux_pattern = "deadwood_C_flux_MgC_ha"
litter_c_flux_pattern = "litter_C_flux_MgC_ha"

land_cover = "land_cover"
vegetation_height = "vegetation_height"

agb_2000 = "agb_2000"
mangrove_agb_2000 = "mangrove_agb_2000"
agc_2000 = "agc_2000"
bgc_2000 = "bgc_2000"
deadwood_c_2000 = "deadwood_c_2000"
litter_c_2000 = "litter_c_2000"
soil_c_2000 = "soil_c_2000"

r_s_ratio = "r_s_ratio"

burned_area = "burned_area"
forest_disturbance = "forest_disturbance"

planted_forest_type_layer = "planted_forest_type"
planted_forest_tree_crop_layer = "planted_forest_tree_crop"

elevation = "elevation"
climate_domain = "climate_domain"
precipitation = "precipitation"
continent_ecozone = "continent_ecozone"