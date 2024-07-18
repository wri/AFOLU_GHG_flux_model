"""
*** run get osm index first then this

TODO: automate s3 uploads

This script filters OSM (OpenStreetMap) data for highways and canals from regional PBF files using the osmium tool.
The filtered data is then saved to separate directories for highways and canals.

Functions:
- filter_highways: Filters highways from an input PBF file and saves to an output PBF file.
- filter_canals: Filters canals from an input PBF file and saves to an output PBF file.
- main: Main function that iterates through a dictionary of PBF files and processes each one.

The script sets up logging to track its progress and handle any errors during execution. It checks if the output
files already exist to avoid redundant processing.
"""

import os
import subprocess
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Dictionary of PBF files
pbf_files = {
    "north-america-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\north-america-latest.osm.pbf",
    "africa-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\africa-latest.osm.pbf",
    "antarctica-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\antarctica-latest.osm.pbf",
    "asia-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\asia-latest.osm.pbf",
    "australia-oceania-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\australia-oceania-latest.osm.pbf",
    "central-america-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\central-america-latest.osm.pbf",
    "europe-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\europe-latest.osm.pbf",
    "south-america-latest.osm.pbf": r"C:\GIS\Data\Global\OSM\raw_by_region\south-america-latest.osm.pbf"
}

# Output directories
output_dir_highways = r"C:\GIS\Data\Global\OSM\filtered_highways"
output_dir_canals = r"C:\GIS\Data\Global\OSM\filtered_canals"
os.makedirs(output_dir_highways, exist_ok=True)
os.makedirs(output_dir_canals, exist_ok=True)

def filter_highways(input_pbf, output_pbf):
    """
    Filters highways from an input PBF file and saves to an output PBF file.

    Args:
        input_pbf (str): Path to the input PBF file.
        output_pbf (str): Path to the output PBF file.
    """
    if os.path.exists(output_pbf):
        logging.info(f"Output file {output_pbf} already exists. Skipping.")
        return
    cmd = ['osmium', 'tags-filter', input_pbf, 'w/highway', '-o', output_pbf]
    try:
        subprocess.check_call(cmd)
        logging.info(f"Successfully filtered highways for {input_pbf} and saved to {output_pbf}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running osmium tags-filter on {input_pbf}: {e}")

def filter_canals(input_pbf, output_pbf):
    """
    Filters canals from an input PBF file and saves to an output PBF file.

    Args:
        input_pbf (str): Path to the input PBF file.
        output_pbf (str): Path to the output PBF file.
    """
    if os.path.exists(output_pbf):
        logging.info(f"Output file {output_pbf} already exists. Skipping.")
        return
    cmd = ['osmium', 'tags-filter', input_pbf, 'w/waterway=ditch,canal', '-o', output_pbf]
    try:
        subprocess.check_call(cmd)
        logging.info(f"Successfully filtered canals for {input_pbf} and saved to {output_pbf}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running osmium tags-filter on {input_pbf}: {e}")

def main():
    """
    Main function that iterates through a dictionary of PBF files and processes each one.
    """
    for output_name, input_pbf in pbf_files.items():
        if not os.path.exists(input_pbf):
            logging.warning(f"Input PBF file {input_pbf} does not exist. Skipping.")
            continue

        output_pbf_highways = os.path.join(output_dir_highways, f"highways_{output_name}")
        output_pbf_canals = os.path.join(output_dir_canals, f"canals_{output_name}")

        filter_highways(input_pbf, output_pbf_highways)
        filter_canals(input_pbf, output_pbf_canals)

if __name__ == '__main__':
    main()
