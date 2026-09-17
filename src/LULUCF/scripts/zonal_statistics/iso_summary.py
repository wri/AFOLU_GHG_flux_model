"""
Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model/
python -m src.LULUCF.scripts.zonal_statistics.iso_summary

TODO: Option to pass in list of isos from the command line
TODO: update with ag data
TODO: Inputs/outputs in s3 instead of local copy
TODO: Update column names with units
"""

import os
import pandas as pd


# ---------------------------------------------------------------------
# PATHS / SETTINGS
# ---------------------------------------------------------------------

veg_parquet = ("/mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/LULUCF__v1_0_0__for_figures__wide__20260802_wdpa_fixed__from_Erin_Glen_via_Slack_20260803.parquet")

output_dir = ("/mnt/c/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/iso_summary")
output_file = os.path.join(output_dir, "LGMS_Horizon_Launch_QC_iso_summary.xlsx")

#isos = ["GMB", "IDN", "BOL", "SWE"]
isos =['AFG' 'ALB' 'DZA' 'AND' 'AGO' 'AIA' 'ATG' 'ARG' 'ARM' 'ABW' 'AUS' 'AUT'
       'AZE' 'BHS' 'BHR' 'BGD' 'BRB' 'BLR' 'BEL' 'BLZ' 'BEN' 'BMU' 'BTN' 'BOL'
       'BES' 'BIH' 'BWA' 'BRA' 'VGB' 'BRN' 'BGR' 'BFA' 'BDI' 'KHM' 'CMR' 'CAN'
       'CPV' 'CYM' 'CAF' 'TCD' 'CHL' 'CHN' 'COL' 'COM' 'CRI' 'HRV' 'CUB' 'CUW'
       'CYP' 'CZE' 'CIV' 'COD' 'DNK' 'DJI' 'DMA' 'DOM' 'TLS' 'ECU' 'EGY' 'SLV'
       'GNQ' 'ERI' 'EST' 'ETH' 'FLK' 'FRO' 'FJI' 'FIN' 'FRA' 'GUF' 'ATF' 'GAB'
       'GMB' 'GEO' 'DEU' 'GHA' 'GIB' 'GRC' 'GRL' 'GRD' 'GLP' 'GTM' 'GGY' 'GIN'
       'GNB' 'GUY' 'HTI' 'VAT' 'HND' 'HUN' 'ISL' 'IND' 'IDN' 'IRN' 'IRQ' 'IRL'
       'IMN' 'ISR' 'ITA' 'JAM' 'JPN' 'JEY' 'JOR' 'KAZ' 'KEN' 'KIR' 'KOR' 'PRK'
       'KWT' 'KGZ' 'LAO' 'LVA' 'LBN' 'LSO' 'LBR' 'LBY' 'LIE' 'LTU' 'LUX' 'MDG'
       'MWI' 'MYS' 'MDV' 'MLI' 'MLT' 'MTQ' 'MRT' 'MUS' 'MYT' 'MEX' 'FSM' 'MDA'
       'MCO' 'MNG' 'MNE' 'MSR' 'MAR' 'MOZ' 'MMR' 'NAM' 'NRU' 'NPL' 'NLD' 'NCL'
       'NZL' 'NIC' 'NER' 'NGA' 'NFK' 'MKD' 'NOR' 'OMN' 'PAK' 'PLW' 'PSE' 'PAN'
       'PNG' 'PRY' 'PER' 'PHL' 'POL' 'PRT' 'PRI' 'QAT' 'COG' 'ROU' 'RUS' 'RWA'
       'REU' 'BLM' 'KNA' 'LCA' 'MAF' 'SPM' 'VCT' 'SMR' 'STP' 'SAU' 'SEN' 'SRB'
       'SYC' 'SLE' 'SGP' 'SXM' 'SVK' 'SVN' 'SLB' 'SOM' 'ZAF' 'SSD' 'ESP' 'LKA'
       'SDN' 'SUR' 'SJM' 'SWZ' 'SWE' 'CHE' 'SYR' 'TWN' 'TJK' 'TZA' 'THA' 'TGO'
       'TTO' 'TUN' 'TUR' 'TKM' 'TCA' 'TUV' 'USA' 'UGA' 'UKR' 'ASM' 'ATA' 'COK'
       'GUM' 'MHL' 'PYF' 'TON' 'WLF' 'WSM' 'ARE' 'GBR' 'UMI' 'URY' 'UZB' 'VUT'
       'VEN' 'VNM' 'VIR' 'ESH' 'YEM' 'ZMB' 'ZWE' 'NA' 'ALA']
years = list(range(2016, 2025))


# ---------------------------------------------------------------------
# SOURCE COLUMNS
# ---------------------------------------------------------------------

#LULUCF
removals_col = "veg__gross_removals__all_C_pools__MgCO2_yr"
emissions_col = ( "veg__gross_emissions__all_C_pools__all_gases__MgCO2e_yr")
mineral_removals_col = ("SOC_gain__mineral_soil_extent__0_30cm_MgCO2_yr")
mineral_emissions_col = ("SOC_loss__mineral_soil_extent__0_30cm_MgCO2_yr")
organic_emissions_col = ("org_soil_drained_burned__all_gases__MgCO2e_yr")


# ---------------------------------------------------------------------
# LAND-STATE CLASSES
# ---------------------------------------------------------------------

tree_gain_classes = ["tree_gain"]
tree_loss_classes = ["tree_loss"]
trees_remaining_classes = ["tree_tree_undisturbed",
                           "tree_tree_disturbed_height_loss",
                           "tree_tree_disturbed_fire_only",]

# Everything else will be classified as
# non_trees
excluded_from_non_trees = (
    tree_gain_classes
    + tree_loss_classes
    + trees_remaining_classes
)


# ---------------------------------------------------------------------
# READ PARQUET
# ---------------------------------------------------------------------
df = pd.read_parquet(veg_parquet)


# ---------------------------------------------------------------------
# CLEAN GROUPING COLUMNS
# ---------------------------------------------------------------------
# Normalize ISO codes
df["adm0"] = (df["adm0"].astype(str).str.strip().str.upper())

# Make sure year is numeric
df["year"] = pd.to_numeric(df["year"], errors="coerce")


# ---------------------------------------------------------------------
# CHECK THAT REQUESTED DATA EXISTS
# ---------------------------------------------------------------------
print("Requested ISO codes found:")
print(sorted(set(isos) & set(df["adm0"].unique())))

print("\nYears found:")
print(sorted(df["year"].dropna().unique()))

print("\nRows before filtering:", len(df))


# ---------------------------------------------------------------------
# FILTER
# ---------------------------------------------------------------------
df = df[df["adm0"].isin(isos) & df["year"].isin(years)].copy()

# ---------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------

def summarize(data, value_col, classes=None, exclude_classes=None):
    x = data

    if classes is not None:
        x = x[
            x["land_state_detailed_class"].isin(classes)
        ]

    if exclude_classes is not None:
        x = x[
            ~x["land_state_detailed_class"].isin(exclude_classes)
        ]

    return (
        x.groupby(["adm0", "year"])[value_col]
        .sum(min_count=1)
    )


# ---------------------------------------------------------------------
# COMPLETE ISO x YEAR TABLE
#
# IMPORTANT:
# Use adm0 here so it matches the index returned by groupby().
# We'll rename it to iso at the end.
# ---------------------------------------------------------------------
index = pd.MultiIndex.from_product(
    [isos, years],
    names=["adm0", "year"]
)

out = pd.DataFrame(index=index)


# ---------------------------------------------------------------------
# TREE GAIN
# ---------------------------------------------------------------------
out["tree_gain_removals"] = summarize(df, removals_col, classes=tree_gain_classes)
out["tree_gain_emissions"] = summarize(df, emissions_col, classes=tree_gain_classes)


# ---------------------------------------------------------------------
# TREE LOSS
# ---------------------------------------------------------------------
out["tree_loss_removals"] = summarize(df, removals_col, classes=tree_loss_classes)
out["tree_loss_emissions"] = summarize(df, emissions_col, classes=tree_loss_classes)


# ---------------------------------------------------------------------
# TREES REMAINING TREES
# ---------------------------------------------------------------------
out["trees_remaining_trees_removals"] = summarize(df, removals_col, classes=trees_remaining_classes)
out["trees_remaining_trees_emissions"] = summarize(df, emissions_col, classes=trees_remaining_classes)


# ---------------------------------------------------------------------
# NON-TREES REMAINING NON-TREES
#
# Includes every land_state_detailed_class OTHER THAN:
#   tree_gain
#   tree_loss
#   tree_tree_undisturbed
#   tree_tree_disturbed_height_loss
#   tree_tree_disturbed_fire_only
# ---------------------------------------------------------------------
out["non_trees_remaining_non_trees_removals"] = summarize(df, removals_col, exclude_classes=excluded_from_non_trees)
out["non_trees_remaining_non_trees_emissions"] = summarize(df, emissions_col, exclude_classes=excluded_from_non_trees)


# ---------------------------------------------------------------------
# MINERAL SOIL
# ---------------------------------------------------------------------
out["mineral_soil_removals"] = summarize(df, mineral_removals_col)
out["mineral_soil_emissions"] = summarize(df, mineral_emissions_col)


# ---------------------------------------------------------------------
# ORGANIC SOIL
# ---------------------------------------------------------------------
out["organic_soil_emissions"] = summarize(df, organic_emissions_col)


# ---------------------------------------------------------------------
# TOTAL VEGETATION
# ---------------------------------------------------------------------
out["vegetation_emissions"] = summarize(df, emissions_col)
out["vegetation_removals"] = summarize(df, removals_col)


# ---------------------------------------------------------------------
# EMPTY COLUMNS
# ---------------------------------------------------------------------
out["cropland_emissions"] = pd.NA
out["livestock_emissions"] = pd.NA
out["net_flux"] = pd.NA


# ---------------------------------------------------------------------
# RESET INDEX / RENAME adm0 -> iso
# ---------------------------------------------------------------------
out = out.reset_index()
out = out.rename(columns={"adm0": "iso"})


# ---------------------------------------------------------------------
# FINAL COLUMN ORDER
# ---------------------------------------------------------------------
column_order = [
    "iso",
    "year",

    "cropland_emissions",

    "tree_gain_removals",
    "tree_loss_removals",

    "livestock_emissions",

    "tree_gain_emissions",
    "tree_loss_emissions",

    "mineral_soil_removals",
    "mineral_soil_emissions",
    "organic_soil_emissions",

    "trees_remaining_trees_removals",
    "trees_remaining_trees_emissions",

    "non_trees_remaining_non_trees_removals",
    "non_trees_remaining_non_trees_emissions",

    "net_flux",

    # Summary columns at end
    "vegetation_emissions",
    "vegetation_removals",
    "mineral_soil_emissions",
    "mineral_soil_removals",
    "organic_soil_emissions",
    "cropland_emissions",
    "livestock_emissions",
]

out = out[column_order]


# ---------------------------------------------------------------------
# CONVERT Mg -> million Mg
# ---------------------------------------------------------------------
# Divide every column except iso and year by 1e6.
# Use column positions because some column names intentionally occur twice.
for i, col in enumerate(out.columns):
    if col not in ["iso", "year"]:
        out.iloc[:, i] = pd.to_numeric(
            out.iloc[:, i],
            errors="coerce"
        ) / 1e6


# ---------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------
os.makedirs(output_dir, exist_ok=True)
out.to_excel(output_file, index=False)
print(f"\nSaved to:\n{output_file}")
print(out.head(20))