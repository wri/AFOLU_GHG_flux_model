import geopandas as gpd

# Path to the shapefile
roads_shapefile_path = r'C:\GIS\Data\Global\GRIP\byRegion\GRIP4_Region1_vector_shp\GRIP4_region1.shp'

# Function to test reading the shapefile
def test_read_shapefile(shapefile_path):
    try:
        roads_gdf = gpd.read_file(shapefile_path)
        print(f"Successfully read shapefile with {len(roads_gdf)} features")
        return roads_gdf
    except Exception as e:
        print(f"Error reading shapefile: {e}")

# Test reading the shapefile
test_read_shapefile(roads_shapefile_path)
