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

secondary_natural_forest_raw_dir =  f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/raw/20241004/"
secondary_natural_forest_0_5_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__0_5_years.tif"   # both the raw raster name and pattern for hansenized tiles
secondary_natural_forest_6_10_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__6_10_years.tif"
secondary_natural_forest_11_15_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__11_15_years.tif"
secondary_natural_forest_16_20_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__16_20_years.tif"
secondary_natural_forest_21_100_pattern =  "natural_forest_mean_growth_rate__Mg_AGC_ha_yr__21_100_years.tif"
secondary_natural_forest_0_5_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_0_5/"
secondary_natural_forest_6_10_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_6_10/"
secondary_natural_forest_11_15_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_11_15/"
secondary_natural_forest_16_20_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_16_20/"
secondary_natural_forest_21_100_processed_dir = f"{full_bucket_prefix}/climate/secondary_forest_carbon_curves__Robinson_et_al/processed/20241004/rate_21_100/"

drivers_raw_dir = f"{full_bucket_prefix}/drivers_of_loss/1_km/raw/20241004/"
drivers_pattern = "drivers_of_TCL_1_km_20241004.tif"   # both the raw raster name and pattern for hansenized tiles
drivers_processed_dir = f"{full_bucket_prefix}/drivers_of_loss/1_km/processed/20241004/"


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

tile_id_list = ['00N_000E', '00N_010E', '00N_020E', '00N_030E', '00N_040E', '00N_040W', '00N_050W', '00N_060W', '00N_070E',
                '00N_070W', '00N_080W', '00N_090E', '00N_090W', '00N_100E', '00N_100W', '00N_110E', '00N_120E', '00N_130E',
                '00N_140E', '00N_150E', '00N_160E', '10N_000E', '10N_010E', '10N_010W', '10N_020E', '10N_020W', '10N_030E',
                '10N_040E', '10N_050W', '10N_060W', '10N_070E', '10N_070W', '10N_080E', '10N_080W', '10N_090E', '10N_090W',
                '10N_100E', '10N_100W', '10N_110E', '10N_120E', '10N_130E', '10N_150E', '10N_160E', '10S_010E', '10S_020E',
                '10S_030E', '10S_040E', '10S_040W', '10S_050E', '10S_050W', '10S_060W', '10S_070W', '10S_080W', '10S_110E',
                '10S_120E', '10S_130E', '10S_140E', '10S_150E', '10S_160E', '10S_170E', '10S_180W', '20N_000E', '20N_010E',
                '20N_010W', '20N_020E', '20N_020W', '20N_030E', '20N_040E', '20N_050E', '20N_060W', '20N_070E', '20N_070W',
                '20N_080E', '20N_080W', '20N_090E', '20N_090W', '20N_100E', '20N_100W', '20N_110E', '20N_110W', '20N_120E',
                '20N_120W', '20N_160W', '20S_010E', '20S_020E', '20S_030E', '20S_040E', '20S_050E', '20S_050W', '20S_060W',
                '20S_070W', '20S_080W', '20S_110E', '20S_120E', '20S_130E', '20S_140E', '20S_150E', '20S_160E', '20S_180W',
                '30N_000E', '30N_010E', '30N_010W', '30N_020E', '30N_020W', '30N_030E', '30N_040E', '30N_050E', '30N_060E',
                '30N_070E', '30N_080E', '30N_080W', '30N_090E', '30N_090W', '30N_100E', '30N_100W', '30N_110E', '30N_110W',
                '30N_120E', '30N_120W', '30N_160W', '30N_170W', '30S_010E', '30S_020E', '30S_030E', '30S_060W', '30S_070W',
                '30S_080W', '30S_110E', '30S_120E', '30S_130E', '30S_140E', '30S_150E', '30S_170E', '40N_000E', '40N_010E',
                '40N_010W', '40N_020E', '40N_020W', '40N_030E', '40N_040E', '40N_050E', '40N_060E', '40N_070E', '40N_070W',
                '40N_080E', '40N_080W', '40N_090E', '40N_090W', '40N_100E', '40N_100W', '40N_110E', '40N_110W', '40N_120E',
                '40N_120W', '40N_130E', '40N_130W', '40N_140E', '40S_070W', '40S_080W', '40S_140E', '40S_160E', '40S_170E',
                '50N_000E', '50N_010E', '50N_010W', '50N_020E', '50N_030E', '50N_040E', '50N_050E', '50N_060E', '50N_060W',
                '50N_070E', '50N_070W', '50N_080E', '50N_080W', '50N_090E', '50N_090W', '50N_100E', '50N_100W', '50N_110E',
                '50N_110W', '50N_120E', '50N_120W', '50N_130E', '50N_130W', '50N_140E', '50N_150E', '50S_060W', '50S_070W',
                '50S_080W', '60N_000E', '60N_010E', '60N_010W', '60N_020E', '60N_020W', '60N_030E', '60N_040E', '60N_050E',
                '60N_060E', '60N_060W', '60N_070E', '60N_070W', '60N_080E', '60N_080W', '60N_090E', '60N_090W', '60N_100E',
                '60N_100W', '60N_110E', '60N_110W', '60N_120E', '60N_120W', '60N_130E', '60N_130W', '60N_140E', '60N_140W',
                '60N_150E', '60N_150W', '60N_160E', '60N_160W', '60N_170E', '60N_170W', '60N_180W', '70N_000E', '70N_010E',
                '70N_020E', '70N_030E', '70N_040E', '70N_050E', '70N_060E', '70N_070E', '70N_070W', '70N_080E', '70N_080W',
                '70N_090E', '70N_090W', '70N_100E', '70N_100W', '70N_110E', '70N_110W', '70N_120E', '70N_120W', '70N_130E',
                '70N_130W', '70N_140E', '70N_140W', '70N_150E', '70N_150W', '70N_160E', '70N_160W', '70N_170E', '70N_170W',
                '70N_180W', '80N_010E', '80N_020E', '80N_030E', '80N_070E', '80N_080E', '80N_090E', '80N_100E', '80N_110E',
                '80N_120E', '80N_130E', '80N_130W', '80N_140E', '80N_140W', '80N_150E', '80N_150W', '80N_160E', '80N_160W',
                '80N_170E', '80N_170W']