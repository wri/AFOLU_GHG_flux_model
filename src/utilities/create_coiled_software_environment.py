"""
Creates a Coiled software environment
per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_coiled_software_environment

# This may take a few minutes.

Currently, this software environment is used only for zonal stats scripts, which require the zarr library be pinned to a specifi version (v3.1.3).
"""

import coiled

coiled.create_software_environment(
    name="afolu-env_coiled_20251119",    # Software environment name

    # Made manually, with Claude's help. Has the main/key Python packages and specifies versions
    conda="/mnt/c/GIS/git/AFOLU_GHG_flux_model/AFOLU_vegetation_20251119.yml"
)
