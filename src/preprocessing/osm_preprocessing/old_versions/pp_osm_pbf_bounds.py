import subprocess
import shapely.geometry

"""
The purpose of this script is to get the bounds of the osm pbfs for easier implementation in processing
"""

regional_pbf_files = [
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\north-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\africa-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\antarctica-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\asia-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\australia-oceania-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\central-america-latest.osm.pbf",
    r"C:\GIS\Data\Global\OSM\OSM_roads_by_region\OSM_Roads\europe-latest.osm.pbf"
]

def calculate_bounds(pbf_files):
    bounds_dict = {}
    for pbf_file in pbf_files:
        bounds = subprocess.check_output(['ogrinfo', '-ro', '-so', pbf_file, 'lines']).decode()
        bounds = bounds.split("Extent: ")[1].split(") - (")
        min_bounds = [float(b) for b in bounds[0].split("(")[1].split(",")]
        max_bounds = [float(b) for b in bounds[1].split(")")[0].split(",")]
        bounds_dict[pbf_file] = (min_bounds[0], min_bounds[1], max_bounds[0], max_bounds[1])
    return bounds_dict

bounds_dict = calculate_bounds(regional_pbf_files)
print(bounds_dict)
