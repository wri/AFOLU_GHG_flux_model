"""
Standalone script for closing local dask clients
"""

from dask.distributed import Client

# List of clients to shutdown
clients = Client._instances.copy()

# Shutdown each client
for client in clients:
    try:
        client.shutdown()
    except Exception as e:
        print(f"Error shutting down client: {e}")

print("All Dask clients have been shut down.")
