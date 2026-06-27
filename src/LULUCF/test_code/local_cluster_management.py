import os

# dask/parallelization libraries
import coiled
import dask
from dask.distributed import Client, LocalCluster
from dask.distributed import print as dask_print
import dask.config
import distributed
import bokeh

def make_local_cluster():

    print("Making local cluster and client")

    # Local cluster with multiple workers
    local_cluster = LocalCluster()
    local_client = Client(local_cluster)
    print(local_client)
    print(local_cluster.dashboard_link)

    print("Local cluster made")
    return local_client


def delete_local_cluster(local_client):

    print("Deleting local cluster and client")

    local_client.shutdown()

    print("Local cluster deleted")


if __name__ == '__main__':

    print("test")
    local_client = make_local_cluster()
    delete_local_cluster(local_client)



