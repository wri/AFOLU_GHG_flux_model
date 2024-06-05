"""
another script for closing dask
"""

rom dask.distributed import Client
import psutil

def close_dask_clusters():
    """
    Closes all active Dask clusters on the local machine.
    """
    for proc in psutil.process_iter(attrs=['pid', 'name']):
        if 'dask-scheduler' in proc.info['name'] or 'dask-worker' in proc.info['name']:
            try:
                p = psutil.Process(proc.info['pid'])
                p.terminate()
                p.wait(timeout=5)
                print(f"Terminated {proc.info['name']} with PID {proc.info['pid']}")
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                print(f"Failed to terminate {proc.info['name']} with PID {proc.info['pid']}: {e}")

if __name__ == "__main__":
    close_dask_clusters()
