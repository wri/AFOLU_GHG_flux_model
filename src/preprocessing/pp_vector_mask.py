"""
This script will loop through the peat mask directory and create a binary 1 km raster set
If a single 30 m pixel falls within a 1 km grid cell, it will be set = 1
This will be used as the input to pp_roads_canals
"""