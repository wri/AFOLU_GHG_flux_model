import math
import boto3

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
first_year = 2000  # First year of model
last_year = 2020   # Last year of model

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


########
### File name paths and patterns
########

# Local path for chunk stats
chunk_stats_path = "chunk_stats/"

LC_uri = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/landcover"

s3_out_dir = 'climate/AFOLU_flux_model/LULUCF/outputs'

local_log_path = "logs/"
s3_log_path = "climate/AFOLU_flux_model/LULUCF/model_logs/"
combined_log = "AFOLU_model_log"

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

natural_forest_dir =  f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/natural_forest_mean_biomass_accumulation_rates_all_years/"
natural_forest_0_5_raw_dir =  f"{natural_forest_dir}/rate_0_5/raw/"
natural_forest_6_10_raw_dir =  f"{natural_forest_dir}/rate_6_10/raw/"
natural_forest_11_15_raw_dir =  f"{natural_forest_dir}/rate_11_15/raw/"
natural_forest_16_20_raw_dir =  f"{natural_forest_dir}/rate_16_20/raw/"
natural_forest_21_100_raw_dir =  f"{natural_forest_dir}/rate_21_100/raw/"

# used as both the raw raster name and pattern for hansenized tile output following the tile ID
natural_forest_0_5_pattern =  "natural_forest_mean_biomass_accumulation_rate_0_5.tif"
natural_forest_6_10_pattern =  "natural_forest_mean_biomass_accumulation_rate_6_10.tif"
natural_forest_11_15_pattern =  "natural_forest_mean_biomass_accumulation_rate_11_15.tif"
natural_forest_16_20_pattern =  "natural_forest_mean_biomass_accumulation_rate_16_20.tif"
natural_forest_21_100_pattern =  "natural_forest_mean_biomass_accumulation_rate_21_100.tif"

drivers_raw_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/tree_cover_loss_drivers/raw/"
drivers_raw_pattern = "preliminary_drivers_1km_10032024.tif"
drivers_processed_dir = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/tree_cover_loss_drivers/processed/drivers_2023/"
drivers_processed_pattern = "tree_cover_loss_driver_processed"


### Outputs

outputs_path = f"{full_bucket_prefix}/climate/AFOLU_flux_model/LULUCF/outputs/"



land_state_node_path_part = "land_state_node"

AGC_density_path_part = "AGC_density_MgC_ha"
BGC_density_path_part = "BGC_density_MgC_ha"
deadwood_c_density_path_part = "deadwood_C_density_MgC_ha"
litter_c_density_path_part = "litter_C_density_MgC_ha"

AGC_flux_path_part = "AGC_flux_MgC_ha"
BGC_flux_path_part = "BGC_flux_MgC_ha"
deadwood_c_flux_path_part = "deadwood_C_flux_MgC_ha"
litter_c_flux_path_part = "litter_C_flux_MgC_ha"



carbon_pool_2000_date = "20240821"

agc_2000_path = f"{outputs_path}{AGC_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
agc_2000_pattern = "AGC_density_MgC_ha_2000"

bgc_2000_path = f"{outputs_path}{BGC_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
bgc_2000_pattern = "BGC_density_MgC_ha_2000"

deadwood_c_2000_path = f"{outputs_path}{deadwood_c_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
deadwood_c_2000_pattern = "deadwood_C_density_MgC_ha_2000"

litter_c_2000_path = f"{outputs_path}{litter_c_density_path_part}/2000/40000_pixels/{carbon_pool_2000_date}/"
litter_c_2000_pattern = "litter_C_density_MgC_ha_2000"


### IPCC classes and change
IPCC_class_path = "IPCC_basic_classes"
IPCC_class_pattern = "IPCC_classes"
IPCC_change_path = "IPCC_basic_change"
IPCC_change_pattern = "IPCC_change"

land_state_pattern = "land_state_node"

gain_year_count_pattern = "gain_year_count_pre_post_disturb"

agb_dens_pattern = "AGB_density_MgAGB_ha"
agc_dens_pattern = "AGC_density_MgC_ha"
bgc_dens_pattern = "BGC_density_MgC_ha"
deadwood_c_dens_pattern = "deadwood_C_density_MgC_ha"
litter_c_dens_pattern = "litter_C_density_MgC_ha"
agc_flux_pattern = "AGC_flux_MgC_ha"
bgc_flux_pattern = "BGC_flux_MgC_ha"
deadwood_c_flux_pattern = "deadwood_C_flux_MgC_ha"
litter_c_flux_pattern = "litter_C_flux_MgC_ha"

ch4_flux_pattern = "CH4_flux_MgCO2e_ha"
n2o_flux_pattern = "N2O_flux_MgCO2e_ha"

land_cover = "land_cover"
vegetation_height = "vegetation_height"


### Carbon pools

agb_2000 = "agb_2000"
mangrove_agb_2000 = "mangrove_agb_2000"
agc_2000 = "agc_2000"
bgc_2000 = "bgc_2000"
deadwood_c_2000 = "deadwood_c_2000"
litter_c_2000 = "litter_c_2000"
soil_c_2000 = "soil_c_2000"

r_s_ratio = "r_s_ratio"


### Other inputs

burned_area = "burned_area"
burned_area_pattern = "ba"
forest_disturbance = "forest_disturbance"

planted_forest_type_layer = "planted_forest_type"
planted_forest_tree_crop_layer = "planted_forest_tree_crop"

elevation = "elevation"
climate_domain = "climate_domain"
precipitation = "precipitation"
continent_ecozone = "continent_ecozone"

ifl_primary = "ifl_primary"

drivers = "drivers"