"""
Creates a Coiled software environment
per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_coiled_software_environment

# This may take a few minutes.
"""


import coiled

coiled.create_software_environment(
    name="afolu-env_coiled_20251119",               # Pick a short, descriptive name
    conda="/mnt/c/GIS/git/AFOLU_GHG_flux_model/AFOLU_vegetation_20251222.yml"
)
