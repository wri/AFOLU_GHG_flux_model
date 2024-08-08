import coiled
from dask.distributed import Client

def test_function(x):
    return x + 1

def main():
    # Set up the Coiled cluster
    coiled_cluster = coiled.Cluster(
        n_workers=1,
        use_best_zone=True,
        compute_purchase_option="spot_with_fallback",
        idle_timeout="15 minutes",
        region="us-east-1",
        name="test_py_connection",
        account='wri-forest-research',
        worker_memory="32GiB"
    )

    # Get the Dask client connected to the Coiled cluster
    coiled_client = coiled_cluster.get_client()
    print("Connected to Coiled cluster")

    # Run a test function on the cluster
    future = coiled_client.submit(test_function, 10)
    result = future.result()
    print(f"Result from the test function: {result}")

    # Close the cluster
    # coiled_cluster.close()
    # print("Cluster closed")

if __name__ == "__main__":
    main()
