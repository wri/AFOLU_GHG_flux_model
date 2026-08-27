# Organic Soils Greenhouse Gas Emissions Model

Version 1.0.1

## Overview

This repository contains the core model used to estimate greenhouse gas emissions associated with drainage and fire on organic soils. It combines spatial information on organic-soil extent, land cover, climate domain, drainage indicators, land management, coastal wetland classes, and annual burned area with emission factors aligned with the IPCC Wetlands Supplement.

Organic soils store a disproportionate share of global soil carbon. Drainage lowers the water table and exposes that carbon to oxidation, while drained organic soils are also more susceptible to fire. The model follows the IPCC accounting definition of organic soils based on soil properties; peatlands are a subset of this broader, definition-dependent domain. Here, “disturbed organic soils” means pixels inferred as drained—including mapped extraction areas—and/or detected as burned.

The repository is intended primarily for external model review and model operations. It contains the model and its required runtime utilities, but it is not a standalone data distribution or a complete end-to-end processing system.

## Scope

The core model classifies organic-soil pixels, evaluates drainage and fire pathways, applies the selected emission-factor tables, and writes spatially explicit model results.

The model operates on a nominal 30 m geographic grid: 0.00025 degrees in EPSG:4326, or approximately 27–28 m at the equator with east–west cell width decreasing toward the poles. It supports the following canonical inventory periods:

- 2001–2005
- 2006–2010
- 2011–2015
- 2016–2020
- 2021–2024

Burned-area emissions are expressed as annual averages for each inventory period. The first four periods are divided by five years, and the final period is divided by four years.

The default configuration uses the OpenGeoHub (OGH) ensemble organic-soil probability surface, which combines soil-taxonomy and peat-depth probability evidence. Its mixed threshold profile is 0.16 for tropical (F2), 0.30 for boreal (F1), and 0.27 for temperate (F1.5) and fallback climate domains. The thresholds were selected from spatially held-out validation data to reflect domain-specific omission–commission tradeoffs; they were not calibrated to country areas, inventory totals, or modeled emissions. Thresholds stored on a 0–1 scale are converted to the native 0–100 probability scale at runtime. The default drainage-distance cutoff is 500 metres.

## Repository contents

| Component | Purpose |
| --- | --- |
| `src/scripts/core_model/0_drainage_emissions_model.py` | Organic-soil classification and drainage/fire emissions calculations |
| `src/scripts/core_model/state_codes.py` | Version 1.0.1 categorical-state registry and packed state encoding |
| `src/scripts/utilities/` | Runtime, cluster, Zarr, logging, geospatial, and emission-factor utilities |
| `config/thresholds/` | Runtime organic-soil threshold configuration |
| `environment.yml` | Pinned Python and geospatial runtime environment |

An aggregation utility is retained for compatibility, but output aggregation and all other postprocessing—including zonal statistics, uncertainty workflows, mapping, and downstream reporting—are outside the core scientific scope. Preprocessing is also out of scope.

## Inputs and access requirements

Input rasters are not included. Execution assumes access to model-ready inputs at the configured data endpoints and a compatible distributed runtime. The core model does not provision compute resources or prepare source datasets; separate cluster lifecycle helpers are included.

The principal input groups are:

| Input group | Role in the model |
| --- | --- |
| Organic-soil extent | Identifies organic-soil pixels using a selected dataset and climate-domain threshold profile |
| Land cover and management | Distinguishes forest, cropland, grassland, wetland, settlement, other land, plantations, and extraction areas |
| Climate domain | Routes pixels among boreal, temperate, tropical, and fallback factor classes |
| Drainage indicators | Represents proximity to canals and roads, or related drainage evidence |
| Coastal wetland classes | Distinguishes mangrove and tidal-marsh pathways |
| Annual burned area | Identifies fire occurrence and, when selected, the number of burned years in an inventory period |
| Emission factors | Supplies default, lower, or upper drainage and fire factor tables |

Principal source families include the OpenGeoHub/OpenLandMap organic-soil ensemble, GLCLUC land cover, MODIS MCD64A1 burned area, global and regional infrastructure datasets, plantation and extraction maps, coastal wetland masks, and IPCC climate domains. The underlying source datasets are publicly available, with full citations in the associated manuscript, but this repository neither includes nor generates the harmonized, tiled model inputs required for execution.

Land cover and climate domain are required wherever a pixel is modeled. Other input layers may be spatially sparse where a missing tile means no occurrence; any configured object that does exist must remain readable and compatible with the common grid. This repository does not reproject or prepare those inputs.

## Model behavior

At each modeled pixel, the core model:

1. applies the configured organic-soil threshold;
2. evaluates drainage evidence and land-use routing;
3. assigns drainage and fire state codes;
4. applies climate- and land-use-specific emission factors;
5. converts component gases to the configured reporting units; and
6. writes physical outputs and packed categorical state to the model Zarr store.

Drainage is inferred from observable proxies rather than measured directly. A fixed routing hierarchy assigns at most one drainage emission-factor stratum to each pixel inferred as drained, avoiding double counting. If a drained pixel is also detected as burned, fire emissions are added for the detected burned year or years; they do not replace drainage emissions.

Configured conversion assumptions include IPCC AR6 100-year global warming potentials of 27 for biogenic CH4 and 273 for N2O, a carbon-to-CO2 multiplier of 3.67, and an N2O-N-to-N2O multiplier of 1.571.

## Outputs

| Output group | Description and units |
| --- | --- |
| `organic_soil` | Binary organic-soil presence mask |
| `combined_state` | Packed drainage and fire classification code; this is categorical, not a physical magnitude |
| `burned_years_count` | Number of burned years represented in the inventory period |
| `drained_co2` | On-site and off-site drainage emissions in Mg CO2 ha⁻¹ yr⁻¹ |
| `drained_ch4_and_n2o` | Drainage emissions expressed in Mg CO2e ha⁻¹ yr⁻¹ |
| `drained_total` | Combined drainage emissions in Mg CO2e ha⁻¹ yr⁻¹ |
| `burned_co2` | Fire emissions in Mg CO2 ha⁻¹ yr⁻¹ |
| `burned_co` | Fire-related carbon monoxide in Mg CO ha⁻¹ yr⁻¹, reported separately rather than as CO2e |
| `burned_ch4` | Fire emissions expressed in Mg CO2e ha⁻¹ yr⁻¹ |
| `burned_total` | Combined CO2 and CH4 fire emissions in Mg CO2e ha⁻¹ yr⁻¹ |

Component variable names and data types are defined in `src/scripts/utilities/constants_and_names.py`.

The model store retains the per-hectare annual layers together with the categorical and count outputs. Per-hectare rates are not per-pixel totals; spatial totals require pixel-area weighting.

## Runtime environment

The pinned runtime is defined in `environment.yml`. It targets Python 3.12 and includes Dask, Distributed, Coiled, Xarray, Zarr, GDAL, Rasterio, GeoPandas, Numba, and supporting object-storage libraries. This environment was checked under Linux through WSL; an equivalent package set is required in other execution environments.

No installation or production command sequence is included here.

## Assumptions and limitations

- The outputs are spatial model estimates, not direct field observations or substitutes for national inventories. The baseline should be interpreted as an operational monitoring estimate rather than a definitive inventory of global organic-soil extent or emissions.
- Mapped organic-soil extent and the threshold used to convert probability to a binary mask are structural determinants of the modeled area and emissions.
- Results depend on the classification, completeness, resolution, and temporal alignment of the model-ready input rasters.
- The same configured OGH probability surface and threshold profile are used across all inventory periods. Many drainage, management, and coastal inputs are also fixed layers rather than period-specific historical reconstructions; configured land cover and annual burned area provide the principal temporal variation.
- Drainage onset and age are not modeled. Static proxies can represent legacy drainage throughout the analysis, while land-cover-based drainage is reevaluated by inventory period; rewetting and restoration are not represented explicitly.
- Global annual land-cover coverage is currently complete only for 2024; multi-year runs are restricted to the canonical inventory periods listed above.
- Execution is bounded to the fixed 266-tile reference land-cover domain; requested areas outside that domain are excluded.
- Invalid land-cover or climate-domain classes on modeled pixels cause the run to fail rather than being silently assigned to a fallback land class.
- Drainage and fire factors are based primarily on IPCC Tier 1 values. They represent broad, long-term averages and do not resolve site-specific water-table depth, drainage history, nutrient status, or management. Some classes use documented proxy factors where a dedicated value is unavailable.
- Drainage can be inferred from land use, extraction, and plantation classes as well as from canal and road evidence. The 500 m influence distance is a global proxy, narrow historical ditches are incompletely mapped, and one regional infrastructure input uses road density rather than direct distance.
- Land-cover labels such as settlement or forest identify routing strata for organic-soil emissions; they do not represent emissions from buildings or forest biomass.
- The MODIS MCD64A1 burned-area input has a native resolution of 500 m and can miss small, low-intensity, cloud-obscured, or subsurface smoldering fires. Fire estimates may therefore be conservative even though calculations are performed on the finer model grid.
- Peat-extraction coverage is geographically incomplete.
- Drained mangrove and tidal-marsh pixels use the applicable IPCC coastal CO2-C factor. Where the IPCC provides no Tier 1 default for coastal CH4, N2O, or off-site CO2 from dissolved organic carbon, the current implementation assigns zero; those zeroes indicate unavailable defaults, not evidence that real-world fluxes are absent.
- No Tier 1 fire-emissions estimate is assigned to tropical undrained organic-soil fires because the IPCC tables provide no fuel-consumption default for that stratum.
- Burned-area results for multi-year periods are annualized and should not be interpreted as emissions assigned to a single fire year.
- The model does not allocate emissions between land-use change and land management or implement a country-specific managed-land boundary.
- Preprocessing and downstream zonal statistics are not included, so this repository alone does not reproduce the complete data-to-reporting workflow.
- Input data and execution infrastructure are not distributed with the source code.
- Input licenses, immutable input checksums, public download routes for the harmonized model-ready inputs, test harnesses, and verification records are not included in this repository.

## Scientific references and citation

The emission-factor implementation follows the *2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands*, particularly the guidance and tables for managed organic soils and organic-soil fires.

- IPCC. 2014. [*2013 Supplement to the 2006 IPCC Guidelines for National Greenhouse Gas Inventories: Wetlands*](https://www.ipcc-nggip.iges.or.jp/public/wetlands/). Hiraishi, T., Krug, T., Tanabe, K., Srivastava, N., Baasansuren, J., Fukuda, M., and Troxler, T.G. (eds.). IPCC, Switzerland.
- IPCC. 2021. [*Climate Change 2021: The Physical Science Basis*, Chapter 7, Table 7.15](https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07.pdf). Cambridge University Press.
- Isik, M. S., and Hengl, T. 2026. [*Global 30 m resolution distribution of organic soils (Histosols) based on soil taxonomy and peat depth training points*](https://doi.org/10.5281/zenodo.20121128). Zenodo, version 1.
- **Associated manuscript (accepted):** [*Global disturbance and emissions from organic soils: leveraging Earth observation–based geospatial data within an IPCC framework*](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2026.1761716/abstract). *Frontiers in Environmental Science* (2026). Accepted 3 July 2026; final formatted version forthcoming.

## Contact

For scientific or technical questions about the model, contact [Erin Glen](mailto:erin.glen@wri.org), Land and Carbon Lab, World Resources Institute.

## License

This project's source code is licensed under the [GNU General Public License v3.0 only](LICENSE) (`GPL-3.0-only`).

Input data and third-party datasets are not redistributed with this repository and remain subject to their respective access and license terms.
