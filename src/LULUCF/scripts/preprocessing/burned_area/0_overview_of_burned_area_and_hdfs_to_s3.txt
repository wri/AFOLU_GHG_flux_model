"""

This document describes the steps used to process MODIS burned area monthly h-v hdf stacks into annual burned area geotif composites.
There are three steps to the workflow, handled in different scripts.

---Step 0: Transfer hdfs for relevant years from DAAC to s3. Download to computer first, then upload to s3.
I couldn't figure out a way to directly transfer from DAAC to s3, so I downloaded the hdfs locally first.
I also was having trouble getting the hdf processing code to use the hdfs directly on DAAC,
so I decided to copy them to s3 for simplicity.
I also couldn't figure out an easy way to do the download/upload in Python programmatically, so I did it using the command line.
Did this step in Ubuntu 22.04 LTS in Windows System for Linux 2 (WSL2).

MODIS burned area v6.1 data landing page: https://lpdaac.usgs.gov/products/mcd64a1v061/
Site to download hdfs from: https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
These are hdf4, not hdf5. This affects what Python libraries to use with them.

Instructions on using DAAC and the command line:
https://lpdaac.usgs.gov/resources/e-learning/how-access-lp-daac-data-command-line/

To set up wget downloading of hdfs in home directory:
nano .wgetrc | chmod og-rw .wgetrc   # touch didn't work in Ubuntu
echo http-user=REPLACEWITHUSERNAME >> .wgetrc | echo http-password=REPLACEWITHPASSWORD >> .wgetrc

https://chatgpt.com/g/g-vK4oPfjfp-coding-assistant/c/67b13996-a308-800a-98a1-dba578305b8d

To download all hdfs in the upper folder (after it iterates through the index.htmls):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/

To download all hdfs for a specific year (all months):
wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/2001*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
2001 alone took 18 minutes

2000-2009:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/200*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
To upload hdfs to s3 for processing:
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
321 minutes to upload everything in the 2000s (29472 files)
Deleted local 2000-2009 downloaded hdf.

2010-2019:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/201*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
344 minutes to download everything in the 2010s (32162 files)
aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
Deleted local 2010-2019 downloaded hdfs.

2020-2024:
time wget -r -np -nH --cut-dirs=2 -R "index.html*" -P MCD64A1_data -e robots=off -A "*.hdf" -nd -I "MOTA/MCD64A1.061/202*" https://e4ftl01.cr.usgs.gov/MOTA/MCD64A1.061/
267 minutes to download 2020-2024 (16078 fies)
(base) dagibbs22@USAWDC7F81Q74:~/MODIS/MCD64A1_data$ time aws s3 cp . s3://gfw2-data/fires/MODIS_burned_area/MCD64A1.061/raw_hdfs/ --recursive
170 minutes to upload
Deleted local 2020-2024 downloaded hdfs.


---Step 1:
1_burned_area_hdf_to_raw_raster.py
This script converts monthly stacks of burned area hdfs into annual geotifs of the original extent, projection, and resolution.
Each hdf represents burned area for a given month in a given year, for a given horizontal-vertical (h-v) area.
Annual output rasters show everywhere that was burned in that year (1 for burned).

hdf processing based on https://chatgpt.com/c/67b0d477-1fc0-800a-b41e-44d954cb9b3e
I have not made this code align with other model components for the most part, e.g., no logs, no output stats, etc.


---Step 2: Convert annual burned area rasters to final Hansen rasters (10x10 deg, 0.00025x0.00025 deg, WGS84).
2_reproject_resample_Hansenize.py

The vrt-based general Hansenize script we have didn't work on MODIS burned area rasters for unclear reasons.
I was able to make vrts of the MODIS BA rasters but then the vrts couldn't be read, perhaps because the
alignment of the rasters inside them were off. I would've liked to be able to use the Hansenize script on
the outputs of Step 1 (raw annual rasters) rather than making a separate burned area script but just couldn't get that working.
"""




