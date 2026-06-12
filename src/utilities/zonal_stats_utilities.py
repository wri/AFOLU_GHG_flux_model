import pandas as pd
from dask.distributed import print
import xarray as xr
import os
from datetime import date
import numpy as np
from io import BytesIO
import requests

# Project imports
from src.utilities import constants_and_names as cn
from src.utilities import universal_utilities as uu


# Creates a Pandas dataframe with the state_nodes codes and meanings from an Excel spreadsheet
def create_state_node_df(state_node_lookup_table_local, state_node_lookup_table_s3, sheet_name):

    try:
        # Tries fetching the file from the S3 URL
        # print(f"Attempting to download file from URL: {spreadsheet}")
        response = requests.get(state_node_lookup_table_s3, timeout=10)
        response.raise_for_status()
        state_node_df = pd.read_excel(BytesIO(response.content), sheet_name=sheet_name)

    except (requests.exceptions.RequestException, Exception) as e:
        print(f"Failed to download file from S3. Falling back to local file. Error: {e}")

        print(f"Reading file from local path: {state_node_lookup_table_local}")
        state_node_df = pd.read_excel(state_node_lookup_table_local, sheet_name=sheet_name)

    return state_node_df


# Crops one input to the other input's extent.
# ref is the reference dataset that is being cropped to.
# From long chat in https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/684749fe-7b30-800a-ba8b-c502377f2c3a
def safe_crop(ds, ref):
    return ds.sel(x=ref.x, y=ref.y, method="nearest")


# Fix floating-point precision issues
def round_coords(ds, decimals=5):
    ds = ds.assign_coords({
        'x': np.round(ds.coords['x'].values, decimals),
        'y': np.round(ds.coords['y'].values, decimals)
    })
    return ds


# Reclassifies age zarr for a given year into 20-year bins
# Per Claude session 'Forest age categorization in zonal stats'
def categorize_age(da):
    """Reclassify raw forest age (integer years) into 20-year category codes."""
    return xr.where(da == 0, 0,
                    xr.where(da <= 5, 1,
                    xr.where(da <= 20, 6,
                    xr.where(da <= 40, 21,
                    xr.where(da <= 60, 41,
                    xr.where(da <= 80, 61,
                    xr.where(da <= 100, 81, 101)))))))


# Reclassifies composite landcover zarr for a given year into basic landcover classes
# Adapted Claude session 'Forest age categorization in zonal stats'
def categorize_composite_LC(da):
    return xr.where(da <= 4, 6,             # Codes 0-4                                     Other land
                    xr.where(da <= 26, 5,   # Codes 5-26                                    Grassland
                    xr.where(da <= 48, 1,   # Codes 27-48                                   Forest
                    xr.where(da <= 104, 6,  # Codes 49-104, but practically just 100-104    Other land
                    xr.where(da <= 126, 5,  # Codes 105-126                                 Grassland
                    xr.where(da <= 148, 1,  # Codes 127-148                                 Other land
                    xr.where(da <= 204, 4,  # Codes 149-204, but practically just 200-204   Wetland
                    xr.where(da <= 207, 6,  # Codes 205-207                                 Other land
                    xr.where(da <= 241, 6,  # Codes 208-241, but practically just code 241  Other land
                    xr.where(da <= 244, 2,  # Codes 242-244, but practically just code 244  Cropland
                    xr.where(da <= 250, 3,  # Codes 245-250, but practically just code 250  Settlement
                    xr.where(da <= 254, 6,  # Codes 251-254, but practically just code 254  Other land
                                        7))))))))))))   # All other codes                   None of the above

# Converts results of flox to coordinate dictionary.
# This code came from Solomon Negusse and I haven't changed it in any substantial way.
def convert_to_coord_dict(flux_results, tile_id, main_logger):

    main_logger.info(f"  Creating {tile_id} coord_dict: {uu.timestr()}")
    sparse_data = flux_results.data

    dim_names = flux_results.dims
    indices = sparse_data.coords  # tuple of arrays with indices into each dim
    values = sparse_data.data  # non-zero values

    coord_dict = {
        dim: flux_results.coords[dim].values[indices[i]]
        for i, dim in enumerate(dim_names)
    }
    coord_dict["value"] = values

    return coord_dict


# Creates all summative output rows in one go, rather than one summative output type at a time (e.g., AGC net flux, removals all pools, etc.).
# Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69973d0e-2dec-832a-bc6f-8cb1f914f0f6
def add_all_summative_rows(df_other: pd.DataFrame, composites: dict[str, list[str]]):
    """
    Create all summative rows in one pass.
    df_other must already have analysis_layer cleaned (no '_ha_yr') and must contain 'value'.
    """

    # Build mapping: each (base_layer -> composite_layer) pair becomes one row
    mapping = pd.DataFrame(
        [(base, comp) for comp, bases in composites.items() for base in bases],
        columns=["analysis_layer", "composite_layer"],
    )

    # Keep only rows that participate in any composite, then attach composite labels
    df_mapped = df_other.merge(mapping, on="analysis_layer", how="inner")

    if df_mapped.empty:
        return df_other

    # Group by everything except the layer + value columns
    group_cols = [c for c in df_other.columns if c not in ["analysis_layer", "value"]]

    # Sum values for each composite and contextual combination
    summed = (
        df_mapped
        .groupby(group_cols + ["composite_layer"], dropna=False, as_index=False)["value"]
        .sum()
        .rename(columns={"composite_layer": "analysis_layer"})
    )

    # Append to df_other (original + composites)
    return pd.concat([df_other, summed], ignore_index=True)


# Assigns climate domain column
def assign_climate_domain(df):

    cont_eco = df["continent_ecozone"].astype("string")

    df["climate_domain"] = np.select(
        [
            cont_eco.str.contains("boreal", case=False, na=False) | cont_eco.str.contains("polar", case=False, na=False),
            cont_eco.str.contains("temperate", case=False, na=False),
            cont_eco.str.contains("tropical", case=False, na=False) | cont_eco.str.contains("subtropical", case=False, na=False),
        ],
        [
            "Boreal",
            "Temperate",
            "Subtropical/tropical",
        ],
        default="Unassigned"
    )

    return df

# Converts flox output to dataframe and does some processing of it:
# replaces the numeric flux type with the name
# classifies specific flux types to larger groupings
# adds the interval end year to the dataframe
# adds the state node meaning to the dataframe
# converts area from m^2 to ha
def create_df(coord_dict, state_node_df, merge_keys, tile_id, flux_type, main_logger):

    main_logger.info(f"  Creating {tile_id} data frame: {uu.timestr()}")

    df = pd.DataFrame(coord_dict)
    # print("df:", df)
    # print("df.columns:", df.columns)

    # Adds column with tile_id
    df['tile_id'] = str(tile_id)

    # Summative outputs
    layers_emissions_all_pools_CO2_only = [
        cn.agc_gross_emis_pattern.replace('_ha_yr', ''),
        cn.bgc_gross_emis_pattern.replace('_ha_yr', ''),
        cn.deadwood_c_gross_emis_pattern.replace('_ha_yr', ''),
        cn.litter_c_gross_emis_pattern.replace('_ha_yr', '')
    ]

    layers_emissions_all_pools_non_CO2 = [
        cn.ch4_gross_emis_pattern.replace('_ha_yr', ''),
        cn.n2o_gross_emis_pattern.replace('_ha_yr', '')
    ]

    layers_removals_all_pools = [
        cn.agc_gross_removals_pattern.replace('_ha_yr', ''),
        cn.bgc_gross_removals_pattern.replace('_ha_yr', ''),
        cn.deadwood_c_gross_removals_pattern.replace('_ha_yr', ''),
        cn.litter_c_gross_removals_pattern.replace('_ha_yr', '')
    ]

    layers_emissions_all_pools_all_gases = layers_emissions_all_pools_CO2_only + layers_emissions_all_pools_non_CO2

    layers_net_AGC = [cn.agc_gross_emis_pattern.replace('_ha_yr', ''), cn.agc_gross_removals_pattern.replace('_ha_yr', '')]
    layers_net_BGC = [cn.bgc_gross_emis_pattern.replace('_ha_yr', ''), cn.bgc_gross_removals_pattern.replace('_ha_yr', '')]
    layers_net_deadwood_C = [cn.deadwood_c_gross_emis_pattern.replace('_ha_yr', ''), cn.deadwood_c_gross_removals_pattern.replace('_ha_yr', '')]
    layers_net_litter_C = [cn.litter_c_gross_emis_pattern.replace('_ha_yr', ''), cn.litter_c_gross_removals_pattern.replace('_ha_yr', '')]

    # layers_net_CO2_only = layers_emissions_all_pools_CO2_only + layers_removals_all_pools
    # layers_net_all_gases = layers_emissions_all_pools_all_gases + layers_removals_all_pools

    # Dictionary of summative outputs.
    summative_outputs = {
        cn.gross_emis_all_C_pools_CO2_only_pattern: layers_emissions_all_pools_CO2_only,
        cn.gross_emis_all_C_pools_non_CO2_only_pattern: layers_emissions_all_pools_non_CO2,
        cn.gross_emis_all_C_pools_all_gases_pattern: layers_emissions_all_pools_all_gases,
        cn.gross_removals_all_C_pools_pattern: layers_removals_all_pools,
        cn.net_flux_agc_pattern: layers_net_AGC,
        cn.net_flux_bgc_pattern: layers_net_BGC,
        cn.net_flux_deadwood_c_pattern: layers_net_deadwood_C,
        cn.net_flux_litter_c_pattern: layers_net_litter_C,
        # cn.net_flux_all_C_pools_CO2_only_pattern: layers_net_CO2_only,
        # cn.net_flux_all_C_pools_all_gases_pattern: layers_net_all_gases,
    }

    # Splits df into pixel area and other analysis layers
    df_area = (
        df[df['analysis_layer'] == 'pixel_area_ha']
          .rename(columns={'value': 'pixel_area_ha'})
          [merge_keys + ['pixel_area_ha']]
    )

    # Non-pixel area analysis layers
    df_other = df[df['analysis_layer'] != 'pixel_area_ha']

    # Removes _ha_yr from analysis layer names because these are no longer per-ha values
    df_other.loc[:, 'analysis_layer'] = df_other['analysis_layer'].str.replace('_ha_yr', '', regex=False)

    # Creates all summative rows in one pass
    df_other = add_all_summative_rows(df_other, summative_outputs)

    # Merges area values into flux rows
    df_with_areas = df_other.merge(df_area, on=merge_keys, how='left')

    # Adds the state_node meaning and classifications to the dataframe
    if cn.land_state_pattern in df_with_areas.columns:
        df_with_areas = df_with_areas.merge(state_node_df[['land_state', 'land_state_meaning', 'land_state_broad_class', 'land_state_detailed_class', 'tall_veg_type']],
              left_on='land_state_node', right_on='land_state',
              how='left')
    # print("merged:", df_with_areas)

    # Column with the GHGs represented in that row
    df_with_areas["gas"] = "Unassigned"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("CH4", na=False), "gas"] = "CH4"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("N2O", na=False), "gas"] = "N2O"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("CO2_only__MgCO2", na=False), "gas"] = "CO2"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("non_CO2", na=False), "gas"] = "non-CO2"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("all_gases", na=False), "gas"] = "all gases"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("SOC", na=False), "gas"] = "CO2"
    df_with_areas.loc[df_with_areas["analysis_layer"].str.contains("C__MgCO2", na=False), "gas"] = "CO2"  # Captures individual carbon pools

    # Replaces the year index with the actual reporting year (differs for vegetation and SOC)
    if flux_type == "vegetation":
        df_with_areas['year'] = df_with_areas['year'] + cn.interval_end_years_annual[0]
    elif flux_type == "SOC":
        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69bcb169-b658-832e-b697-46d22c126cb6
        # Could also try using .map()-- may be faster (or slower) for large dfs
        df_with_areas["year"] = [
            cn.SOC_density_intervals[i] for i in df_with_areas["year"]
        ]
    else:
        main_logger.warning(f"{flux_type} not found for {tile_id}")

    # Deletes redundant state node column
    if 'land_state' in df_with_areas.columns:
        df_with_areas = df_with_areas.drop(columns=['land_state'])

    # Converts numeric codes to ISO codes
    # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/698a53aa-8674-832c-b734-4bd8afc6a6df
    # Based on https://github.com/wri/project-zeno-data-infra/blob/main/notebooks/grasslands_areas_gadm_2000-2022.ipynb originally
    if cn.adm0_pattern in df_with_areas.columns:
        df_with_areas[cn.adm0_pattern] = df_with_areas[cn.adm0_pattern].map(cn.numeric_to_alpha3)
        df_with_areas['country_name'] = df_with_areas[cn.adm0_pattern].map(cn.iso_to_country)
        df_with_areas['region_L1'] = df_with_areas[cn.adm0_pattern].map(cn.iso_to_region_UN_geoscheme_L1)
        df_with_areas['region_L2_L3'] = df_with_areas[cn.adm0_pattern].map(cn.iso_to_region_UN_geoscheme_L2_L3)

        # Because some rows for contextual layers may be blank
        df_with_areas[cn.adm0_pattern] = df_with_areas[cn.adm0_pattern].fillna("Unassigned")
        df_with_areas['country_name'] = df_with_areas['country_name'].fillna("Unassigned")
        df_with_areas['region_L1'] = df_with_areas['region_L1'].fillna("Unassigned")
        df_with_areas['region_L2_L3'] = df_with_areas['region_L2_L3'].fillna("Unassigned")

        # Renames some countries with long names
        df_with_areas["country_name"] = df_with_areas["country_name"].replace({
            "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
            "Russian Federation": "Russia",
            "Democratic Republic of the Congo": "DR Congo",
            "United States of America (the)": "USA"
        })

    # Maps cont_eco to continent, ecozone, ecozone-continent, and climate domain if the contextual layer is used
    # From https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/698a53aa-8674-832c-b734-4bd8afc6a6df
    # Tiles without any ecozone information at all, e.g., 00N_020W, were causing errors for continent and ecozone.
    # Fix per Claude session 'vegetation_zonal_stats performance'
    if cn.cont_eco_zstats_pattern in df_with_areas.columns:
        df_with_areas['continent'] = pd.Series(
            [cn.cont_eco_to_text.get(int(v), {}).get('continent') or 'Unassigned'
             for v in df_with_areas[cn.cont_eco_zstats_pattern]],
            index=df_with_areas.index, dtype=object
        )
        df_with_areas['ecozone'] = pd.Series(
            [cn.cont_eco_to_text.get(int(v), {}).get('ecozone') or 'Unassigned'
             for v in df_with_areas[cn.cont_eco_zstats_pattern]],
            index=df_with_areas.index, dtype=object
        )
        df_with_areas['continent_ecozone'] = df_with_areas['continent'] + "-" + df_with_areas['ecozone']

        # Assigns climate domain
        df_with_areas = assign_climate_domain(df_with_areas)

        # Because some rows for contextual layers may be blank
        df_with_areas["continent"] = df_with_areas["continent"].fillna("Unassigned")
        df_with_areas["ecozone"] = df_with_areas["ecozone"].fillna("Unassigned")
        df_with_areas["continent_ecozone"] = df_with_areas["continent_ecozone"].fillna("Unassigned")

    # Maps watershed codes to names if the contextual layer is used
    if cn.watersheds_pattern in df_with_areas.columns:
        df_with_areas['watershed_name'] = df_with_areas[cn.watersheds_pattern].map(cn.watershed_to_text)
        df_with_areas["watershed_name"] = df_with_areas["watershed_name"].fillna("Unassigned")

    # Maps WDPA codes to names if the contextual layer is used
    if cn.WDPA_pattern in df_with_areas.columns:
        df_with_areas['WDPA_type'] = df_with_areas[cn.WDPA_pattern].map(cn.WDPA_to_text)

        # Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69aee45e-ce6c-8325-b1d0-a6c6b0e7ae2e
        df_with_areas["WDPA_high_protection"] = "Other protection status"

        df_with_areas.loc[df_with_areas["WDPA_type"] == "NA", "WDPA_high_protection"] = "Not protected"
        df_with_areas.loc[df_with_areas["WDPA_type"].isin(["Category Ia", "Category Ib", "Category II", "Category III"]), "WDPA_high_protection"] = "High protection"

    # Maps driver of loss codes to names if the contextual layer is used
    if cn.drivers_of_loss_pattern in df_with_areas.columns:
        df_with_areas['driver_1km_text'] = df_with_areas[cn.drivers_of_loss_pattern].map(cn.drivers_to_text)
        df_with_areas['driver_1km_text'] = df_with_areas['driver_1km_text'].fillna("Unassigned")

    # Replaces managed land numeric values with managed/unmanaged if the contextual layer is used
    if cn.managed_land_CAN_pattern in df_with_areas.columns:
        df_with_areas[cn.managed_land_CAN_pattern] = df_with_areas[cn.managed_land_CAN_pattern].map(cn.managed_land_to_text)
    if cn.managed_land_USA_pattern in df_with_areas.columns:
        df_with_areas[cn.managed_land_USA_pattern] = df_with_areas[cn.managed_land_USA_pattern].map(cn.managed_land_to_text)
    if cn.BRA_biomes_pattern in df_with_areas.columns:
        df_with_areas[cn.BRA_biomes_pattern] = df_with_areas[cn.BRA_biomes_pattern].map(cn.BRA_biomes_to_text)
    if cn.forest_age_category_pattern in df_with_areas.columns:
        df_with_areas[cn.forest_age_category_pattern] = df_with_areas[cn.forest_age_category_pattern].map(cn.forest_age_category_to_text)
        df_with_areas[cn.forest_age_category_pattern] = df_with_areas[cn.forest_age_category_pattern].fillna("Unassigned")

    # Calculates flux density (Mg CO2(e)/ha) for each row
    df_with_areas['density__Mg_ha'] = df_with_areas['value'] / df_with_areas['pixel_area_ha'].replace(0, pd.NA)

    # Renames pixel_area_ha to area_ha
    df_with_areas = df_with_areas.rename(columns={'pixel_area_ha': 'area_ha'})

    return df_with_areas


# Converts long-format df to wide-format df
# Per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/69fe3161-edb8-832e-a90d-d9e75e4012d3
def create_wide_df(combined_df, main_logger):

    main_logger.info(f"Converting combined table from long to wide: {uu.timestr()}")

    # Columns to use and to not use as contextual layers. Drops gas because it's implicit in analysis_layer.
    id_cols = [
        c for c in combined_df.columns
        if c not in ["analysis_layer", "value", "density__Mg_ha", "area_ha", "gas", "LULUCF_component"]
    ]

    # Reshapes from long to wide, with value and area_ha for each analysis_layer.
    # ChatGPT says this is safer for giant tables than using pivot_table
    wide = (
        combined_df
        .groupby(id_cols + ["analysis_layer"], observed=True, sort=False, dropna=False)[["value", "area_ha"]]
        .sum()
        .unstack("analysis_layer")
        .fillna(0)
    )

    # Appends __value or __area_ha to each analysis layer
    wide.columns = [
        f"{analysis_layer}__{measure}"
        for measure, analysis_layer in wide.columns
    ]
    wide = wide.reset_index()

    # Reorders columns so that __value fields are before __area_ha fields
    value_cols = sorted([c for c in wide.columns if c.endswith("__value")])
    area_cols = sorted([c for c in wide.columns if c.endswith("__area_ha")])
    wide = wide[id_cols + value_cols + area_cols]
    combined_wide_df = wide.reset_index(drop=True)

    return combined_wide_df


# Uploads output tables (parquet and csv) if the run is large enough
def upload_zstats_to_s3(stage, local_zonal_stats_folder, s3_output_folder, main_logger, model_path_description, model_type,
                        model_version, tiles_processed):

    run_date = date.today().strftime("%Y%m%d")

    # Uploads output tables to s3 if it's a larger run where I might plausibly want to save the results
    if tiles_processed > 3:
        s3_zonal_stats_folder = s3_output_folder.replace(cn.model_version_type_description_placeholder,
                                                            f"version_{model_version}__{model_type}__{model_path_description}") + f"zonal_statistics/{run_date}_{stage}/"

        files_to_upload = [
            str(f) for f in local_zonal_stats_folder.iterdir()
            if f.suffix in ('.parquet', '.csv')
        ]

        main_logger.info(f"Uploading {len(files_to_upload)} files to {s3_zonal_stats_folder}")
        for local_file in files_to_upload:
            filename = os.path.basename(local_file)
            s3_dest = s3_zonal_stats_folder + filename
            main_logger.info(f"  Uploading {filename} to {s3_dest}")
            uu.upload_s3_file(s3_dest, local_file)

        main_logger.info("Upload complete")