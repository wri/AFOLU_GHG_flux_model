"""
Creates specified Coiled software environment from yml package list
per https://chatgpt.com/g/g-p-69399a7fcc808191b337d3fac695447c-afolu-flux-model/c/6949a74e-1388-832d-8f8e-5e9bf084ecb8

Currently two pre-set software environments: one for zonal stats software and one for non-zonal stats software.

The zonal stats environment only has about 28 packages loosely pinned.

The non-zonal stats environment has ~387 packages pinned-- not just the basic ones I care about but their dependencies, too.
This prevents Coiled from resolving the dependencies differently each time I create a cluster;
if there was an update on conda-forge, Coiled would update dependency packages I hadn't specified and then there would
be a package mismatch.
So, the non-zonal stats one now has very specific package version information.

The workflow for creating the yml I used to create the non-zonal stats software environment here was:

1. Install conda-lock (in base environment):
   conda install -n base -c conda-forge conda-lock
2. Create a lock-file (run from Coiled_20251119 env but using conda-lock in base environment-- took about 10 minutes):
   conda run -n base conda-lock lock --platform linux-aarch64 -f AFOLU_not_zonal_stats_20251119.yml --lockfile AFOLU_not_zonal_stats_20251119.conda-lock.yml
3. Convert from conda-lock file to regular yml:
   conda run -n base conda-lock render --kind env -p linux-aarch64 --filename-template AFOLU_not_zonal_stats_20251119.locked.{platform} AFOLU_not_zonal_stats_20251119.conda-lock.yml
4. Run this script to create the software environment. It first strips the architecture-specific build strings, then hands to Coiled.
Note that when a non-zonal stats cluster is created, it still needs to have the src code zipped and uploaded to it.
Per Claude session 'Coiled cluster creation error'

Run from /mnt/c/GIS/git/AFOLU_GHG_flux_model

python -m src.utilities.create_coiled_software_environment

This may take a few minutes.
"""

import coiled
import re

# # To create software environment for cluster performing zonal stats
# coiled.create_software_environment(
#     name="afolu_zonal_stats_20251222",    # Software environment name
#
#     # Made manually, with Claude's help. Has the main/key Python packages and specifies versions
#     conda="/mnt/c/GIS/git/AFOLU_GHG_flux_model/conda_envs/AFOLU_zonal_stats_20251222.yml"
# )

# # To create software for cluster not performing zonal stats
# # Made with Claude session 'Coiled cluster creation error'. Uses conda-lock file, converted to regular yml

# Strips build strings from yml
with open('conda_envs/AFOLU_not_zonal_stats_20251119.locked.linux-aarch64.yml') as f:
    lines = f.readlines()

result = []
for line in lines:
    # Match conda dep lines: "  - name=version=build_string" -> "  - name=version"
    m = re.match(r'^(\s+- [a-zA-Z0-9_\-\.]+=[0-9][^=\s]*)=[^\s]+(.*)$', line.rstrip())
    if m:
        result.append(m.group(1) + m.group(2) + '\n')
    else:
        result.append(line)

with open('conda_envs/AFOLU_not_zonal_stats_20251119_no_build_string.yml', 'w') as f:
    f.writelines(result)
print('Done stripping build strings')

# Creates software environment
coiled.create_software_environment(
    name="afolu_not_zonal_stats_20251119",
    conda="conda_envs/AFOLU_not_zonal_stats_20251119_no_build_string.yml"
)
