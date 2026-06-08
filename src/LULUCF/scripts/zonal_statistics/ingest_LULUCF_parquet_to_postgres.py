r"""
Ingests wide-format LULUCF parquet table with all contextual rows and all columns into Postgres.
It creates the table schema directly from the parquet file, so there's no need to write out a long Postgres command
with all the columns and their datatypes.
Imports in batches to monitor progress.
Takes about 25 minutes to run locally.

Run in Windows command prompt, not WSL, because Postgres is in Windows, not WSL

Run from:
cd C:\GIS\git\AFOLU_GHG_flux_model\src\LULUCF\scripts\zonal_statistics

Need to activate the base conda environment, which has the necessary packages (psycopg2, duckdb, pyarrow, tqdm):
conda activate base

Run with:
python ingest_LULUCF_parquet_to_postgres

Per Claude session 'LULUCF and component table creation'
"""

import time
import psycopg2
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

parquet_path = 'C:/GIS/AFOLU_flux_model/LULUCF/zonal_statistics/LULUCF_v1_0_0__veg_v1_0_5__minsoil_v1_0_1__orgsoil_v1_0_1/LULUCF__v1_0_0__LULUCF_summative_vars__wide__20260606.parquet'
tbl         = 'lulucf100_wide__veg105__soc101__orgsoil101__20260606'
batch_size  = 500_000
# batch_size  = 50_000  # For testing
db_params   = dict(dbname='AFOLU_GHG_flux_model', host='localhost', port=5432)  # No username or password needed on my computer

def arrow_to_pg(t):
    if   pa.types.is_uint8(t)  or pa.types.is_int8(t) \
      or pa.types.is_uint16(t) or pa.types.is_int16(t): return 'SMALLINT'
    elif pa.types.is_uint32(t) or pa.types.is_int32(t): return 'INTEGER'
    elif pa.types.is_uint64(t) or pa.types.is_int64(t): return 'BIGINT'
    elif pa.types.is_float32(t):                        return 'REAL'
    elif pa.types.is_float64(t):                        return 'DOUBLE PRECISION'
    elif pa.types.is_boolean(t):                        return 'BOOLEAN'
    else:                                               return 'TEXT'

# ── Read parquet schema ──────────────────────────────────────────────────────
pf     = pq.ParquetFile(parquet_path)
schema = pf.schema_arrow
print(f"schema: \n {schema}")

# ── Step 1: Create table ─────────────────────────────────────────────────────
col_defs   = ',\n    '.join([f'"{f.name}" {arrow_to_pg(f.type)}' for f in schema])
create_sql = f'CREATE TABLE IF NOT EXISTS {tbl} (\n    {col_defs}\n);'

print("Creating table...")
pg = psycopg2.connect(**db_params)
pg.autocommit = True
with pg.cursor() as cur:
    cur.execute(create_sql)
print(f"  {tbl} created")

# ── Step 2: Ingest ───────────────────────────────────────────────────────────
total_rows = pf.metadata.num_rows
print(f"\nIngesting {total_rows:,} rows in batches of {batch_size:,}...")

con = duckdb.connect()
con.execute("INSTALL postgres; LOAD postgres;")
con.execute(f"ATTACH 'host=localhost port=5432 dbname=AFOLU_GHG_flux_model' AS pg (TYPE POSTGRES);")

t_start = time.time()
batches = list(pf.iter_batches(batch_size=batch_size))
# for i, batch in enumerate(tqdm(batches[0:2])): # for testing
for i, batch in enumerate(tqdm(batches)):
    t0 = time.time()
    con.register('batch', batch)
    con.execute(f'INSERT INTO pg.public."{tbl}" SELECT * FROM batch')
    con.unregister('batch')
    elapsed = time.time() - t0
    tqdm.write(f"  Batch {i+1}/{len(batches)}: {len(batch):,} rows in {elapsed:.1f}s ({len(batch)/elapsed:,.0f} rows/s)")

con.close()
print(f"\nIngestion complete in {(time.time() - t_start)/60:.1f} min")

# ── Step 3: Analyze and index to improve query speeds ────────────────────────────────────────────────
print("\nRunning ANALYZE and creating indexes...")
t0 = time.time()
with pg.cursor() as cur:
    print("  ANALYZE...")
    cur.execute(f"ANALYZE {tbl};")

    print("  BRIN index on year...")
    cur.execute(f"CREATE INDEX ON {tbl} USING BRIN (year);")

    print("  Indexes on specific columns...")
    cur.execute(f'CREATE INDEX ON {tbl} (adm0);')
    cur.execute(f'CREATE INDEX ON {tbl} ("region_L1");')
    cur.execute(f'CREATE INDEX ON {tbl} (land_state_meaning);')
    cur.execute(f'CREATE INDEX ON {tbl} (land_state_detailed_class);')

    print("  Composite index on (year, adm0)...")
    cur.execute(f'CREATE INDEX ON {tbl} (year, adm0);')

    print("  Setting parallel_workers = 4...")
    cur.execute(f'ALTER TABLE {tbl} SET (parallel_workers = 4);')

print(f"Indexes created in {(time.time() - t0)/60:.1f} min")

# ── Step 4: Report table size ────────────────────────────────────────────────
with pg.cursor() as cur:
    cur.execute(f"SELECT pg_size_pretty(pg_total_relation_size('{tbl}'));")
    size = cur.fetchone()[0]

pg.close()
print(f"\nTable size in Postgres: {size}")