import coiled
from dask.distributed import Client
import dask.array as da

def run_test_operation(cluster_name):
    # Connect to the existing cluster using the cluster's name to get the scheduler address
    cluster = coiled.Cluster(name=cluster_name)
    client = Client(cluster)

    # Perform a test operation (e.g., compute the mean of a random Dask array)
    x = da.random.random((10000, 10000), chunks=(1000, 1000))
    result = x.mean().compute()

    # Close the client connection
    client.close()

    print(f"Test operation result: {result}")
    return result

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        cluster_name = sys.argv[1]
        run_test_operation(cluster_name)
    else:
        print("Please provide the cluster name as an argument.")
