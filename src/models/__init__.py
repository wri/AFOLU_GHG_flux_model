# utils/__init__.py

from .logging_utils import (
    setup_logging,
    print_and_log,
    compile_and_upload_log,
    timestr
)
from .chunk_utils import (
    get_10x10_tile_bounds,
    get_chunk_bounds,
    prepare_to_download_chunk,
    check_for_tile,
    check_chunk_for_data,
    boundstr,
    calc_chunk_length_pixels,
    flatten_list
)
from .raster_utils import (
    save_and_upload_raster_10x10,
    create_list_for_aggregation,
    merge_small_tiles_gdal,
    save_and_upload_small_raster_set,
    accrete_node,
    make_tile_footprint_shp
)
from .s3_utils import (
    list_rasters_in_folder,
    upload_shp
)
from .processing_utils import (
    accrete_node,  # Consider removing duplicate if it's already in raster_utils
    create_typed_dicts,
    convert_lookup_table_to_array,
    complete_inputs,
    calculate_stats,
    calculate_chunk_stats,
    process_soil
)
from .cluster_utils import (
    setup_coiled_cluster,
    setup_test_coiled_cluster,
    setup_local_single_process_cluster,
    setup_local_multi_process_cluster,
    shutdown_cluster
)

__all__ = [
    # Logging Utils
    'setup_logging',
    'print_and_log',
    'compile_and_upload_log',
    'timestr',

    # Chunk Utils
    'get_10x10_tile_bounds',
    'get_chunk_bounds',
    'prepare_to_download_chunk',
    'check_for_tile',
    'check_chunk_for_data',
    'boundstr',
    'calc_chunk_length_pixels',
    'flatten_list',

    # Raster Utils
    'save_and_upload_raster_10x10',
    'create_list_for_aggregation',
    'merge_small_tiles_gdal',
    'save_and_upload_small_raster_set',
    'accrete_node',
    'make_tile_footprint_shp',

    # S3 Utils
    'list_rasters_in_folder',
    'upload_shp',

    # Processing Utils
    'create_typed_dicts',
    'convert_lookup_table_to_array',
    'complete_inputs',
    'calculate_stats',
    'calculate_chunk_stats',
    'process_soil',

    # Cluster Utils
    'setup_coiled_cluster',
    'setup_test_coiled_cluster',
    'setup_local_single_process_cluster',
    'setup_local_multi_process_cluster',
    'shutdown_cluster'
]
