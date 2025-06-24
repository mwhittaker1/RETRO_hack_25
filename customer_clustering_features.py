import pandas as pd
import duckdb
import numpy as np
from pathlib import Path
import logging
from typing import Union, Optional
import gc
from datetime import datetime

class PrependFileHandler(logging.Handler):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def emit(self, record):
        log_entry = self.format(record)
        try:
            with open(self.filename, 'r', encoding='utf-8') as f:
                old_content = f.read()
        except FileNotFoundError:
            old_content = ''
        with open(self.filename, 'w', encoding='utf-8') as f:
            f.write(log_entry + '\n' + old_content)

class CustomFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = self.converter(record.created)
        return f"{ct.tm_hour:02}:{ct.tm_min:02}:{ct.tm_mday:02}::{ct.tm_mon:02}:{ct.tm_year % 100:02}"

logger = logging.getLogger("customer_clustering_features")

def create_customer_clustering_features(
    file_path: str,
    table_name: str = 'customer_data',
    features_table_name: str = 'customer_features',
    chunk_size: int = 100_000,
    db_file: str = 'customer_features.db',
    force_recreate: bool = False
) -> duckdb.DuckDBPyConnection:
    """
    Loads base data and creates customer clustering features in DuckDB.
    Returns DuckDB connection for further analytics.
    """
    conn = duckdb.connect(db_file)
    logger.info(f"Connected to DuckDB: {db_file}")

    # --- Load base data (CSV/Excel) ---
    _load_raw_data(conn, file_path, table_name, chunk_size, force_recreate)

    # --- Create feature tables ---
    _create_intermediate_tables(conn, table_name)
    _create_customer_features(conn, features_table_name, force_recreate, table_name=table_name)
    _feature_validation(conn, features_table_name)
    logger.info(f"Feature engineering complete. Features in table: {features_table_name}")
    return conn

def _load_raw_data(conn, file_path, table_name, chunk_size, force_recreate):
    file_path = Path(file_path)
    table_exists = conn.execute(f"""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = '{table_name}'
    """).fetchone()[0] > 0
    if table_exists and not force_recreate:
        logger.info(f"Table {table_name} already exists, skipping data load")
        return
    if table_exists and force_recreate:
        conn.execute(f"DROP TABLE IF EXISTS {table_name}")
    if file_path.suffix.lower() == '.xlsx':
        _load_excel_chunked(conn, file_path, table_name, chunk_size)
    elif file_path.suffix.lower() == '.csv':
        _load_csv_direct(conn, file_path, table_name)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")
    logger.info("Creating indexes on raw data table")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_customer ON {table_name}(CUSTOMER_EMAILID)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_order ON {table_name}(SALES_ORDER_NO)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(ORDER_DATE)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_return ON {table_name}(UNITS_RETURNED_FLAG)")

def _load_excel_chunked(conn, file_path, table_name, chunk_size):
    logger.info(f"Reading Excel file in chunks of {chunk_size:,} rows")
    df_sample = pd.read_excel(file_path, nrows=1)
    chunk_num = 0
    table_created = False
    for chunk in pd.read_excel(file_path, chunksize=chunk_size):
        chunk_num += 1
        logger.info(f"Processing chunk {chunk_num} ({len(chunk):,} rows)")
        chunk = _clean_raw_data_chunk(chunk)
        if not table_created:
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM chunk")
            table_created = True
        else:
            conn.execute(f"INSERT INTO {table_name} SELECT * FROM chunk")
        del chunk
        gc.collect()

def _load_csv_direct(conn, file_path, table_name):
    logger.info("Loading CSV using DuckDB's native CSV reader")
    conn.execute(f"""
        CREATE TABLE {table_name} AS 
        SELECT 
            CUSTOMER_EMAILID,
            SALES_ORDER_NO,
            Q_GMM_ID,
            Q_CLS_ID,
            SKU,
            Q_SKU_DESC,
            SALES_QTY,
            UNITS_RETURNED_FLAG,
            RETURN_NO,
            RETURN_QTY,
            CAST(ORDER_DATE AS TIMESTAMP) AS ORDER_DATE,
            CASE 
                WHEN RETURN_DATE = '-' OR RETURN_DATE IS NULL THEN NULL
                ELSE CAST(RETURN_DATE AS TIMESTAMP)
            END AS RETURN_DATE
        FROM read_csv_auto('{file_path}')
    """)

def _clean_raw_data_chunk(df):
    # No renaming, use original column names
    df['RETURN_DATE'] = df['RETURN_DATE'].replace('-', None)
    df['ORDER_DATE'] = pd.to_datetime(df['ORDER_DATE'])
    df['RETURN_DATE'] = pd.to_datetime(df['RETURN_DATE'], errors='coerce')
    df['SALES_QTY'] = pd.to_numeric(df['SALES_QTY'], errors='coerce').fillna(0)
    df['RETURN_QTY'] = pd.to_numeric(df['RETURN_QTY'], errors='coerce').fillna(0)
    df['UNITS_RETURNED_FLAG'] = df['UNITS_RETURNED_FLAG'].str.upper().eq('YES')
    return df

def _create_intermediate_tables(conn, table_name):
    reference_date = conn.execute(f"SELECT MAX(ORDER_DATE) FROM {table_name}").fetchone()[0]
    logger.info(f"Using reference date: {reference_date}")
    logger.info("Creating customer order summary")
    conn.execute(f"""
        CREATE OR REPLACE TABLE customer_order_summary AS
        SELECT 
            CUSTOMER_EMAILID,
            SALES_ORDER_NO,
            ORDER_DATE,
            COUNT(*) as items_in_order,
            SUM(SALES_QTY) as total_qty_ordered,
            COUNT(DISTINCT SKU) as unique_skus_in_order,
            COUNT(DISTINCT Q_CLS_ID) as unique_categories_in_order,
            SUM(CASE WHEN UNITS_RETURNED_FLAG THEN 1 ELSE 0 END) as items_returned_in_order,
            SUM(CASE WHEN UNITS_RETURNED_FLAG THEN RETURN_QTY ELSE 0 END) as qty_returned_in_order
        FROM {table_name}
        GROUP BY CUSTOMER_EMAILID, SALES_ORDER_NO, ORDER_DATE
    """)
    logger.info("Creating customer item summary")
    conn.execute(f"""
        CREATE OR REPLACE TABLE customer_item_summary AS
        SELECT 
            CUSTOMER_EMAILID,
            SKU,
            Q_CLS_ID,
            COUNT(*) as purchase_frequency,
            SUM(SALES_QTY) as total_qty_purchased,
            SUM(CASE WHEN UNITS_RETURNED_FLAG THEN RETURN_QTY ELSE 0 END) as total_qty_returned,
            COUNT(CASE WHEN UNITS_RETURNED_FLAG THEN 1 END) as return_frequency,
            MIN(ORDER_DATE) as first_purchase_date,
            MAX(ORDER_DATE) as last_purchase_date,
            AVG(CASE WHEN UNITS_RETURNED_FLAG AND RETURN_DATE IS NOT NULL 
                     THEN EXTRACT(DAY FROM (RETURN_DATE - ORDER_DATE)) END) as avg_days_to_return
        FROM {table_name}
        GROUP BY CUSTOMER_EMAILID, SKU, Q_CLS_ID
    """)
    logger.info("Creating return timing analysis")
    conn.execute(f"""
        CREATE OR REPLACE TABLE return_timing_analysis AS
        SELECT 
            CUSTOMER_EMAILID,
            SALES_ORDER_NO,
            SKU,
            ORDER_DATE,
            RETURN_DATE,
            EXTRACT(DAY FROM (RETURN_DATE - ORDER_DATE)) as days_to_return,
            EXTRACT(MONTH FROM ORDER_DATE) as order_month,
            EXTRACT(MONTH FROM RETURN_DATE) as return_month
        FROM {table_name}
        WHERE UNITS_RETURNED_FLAG AND RETURN_DATE IS NOT NULL
    """)
    conn.execute(f"""
        CREATE OR REPLACE TABLE reference_metadata AS
        SELECT 
            '{reference_date}'::TIMESTAMP as reference_date,
            '{reference_date}'::TIMESTAMP - INTERVAL '90 days' as recent_cutoff_date
    """)

def _create_customer_features(conn, features_table_name, force_recreate, table_name=None):
    table_exists = conn.execute(f"""
        SELECT COUNT(*) FROM information_schema.tables 
        WHERE table_name = '{features_table_name}'
    """).fetchone()[0] > 0
    if table_exists and not force_recreate:
        logger.info(f"Features table {features_table_name} already exists, skipping creation")
        return
    if table_exists and force_recreate:
        conn.execute(f"DROP TABLE IF EXISTS {features_table_name}")
    logger.info("Creating comprehensive customer features")
    conn.execute(f"""
        CREATE TABLE {features_table_name} AS
        WITH customer_base AS (
            SELECT DISTINCT CUSTOMER_EMAILID
            FROM customer_order_summary
        ),
        basic_metrics AS (
            SELECT 
                cb.CUSTOMER_EMAILID,
                COALESCE(COUNT(DISTINCT cos.SALES_ORDER_NO), 0) as SALES_ORDER_NO_nunique,
                COALESCE(COUNT(DISTINCT cis.SKU), 0) as SKU_nunique,
                COALESCE(SUM(cis.return_frequency), 0) as ITEMS_RETURNED_COUNT,
                COALESCE(AVG(cis.total_qty_purchased), 0) as SALES_QTY_mean,
                COALESCE(AVG(cos.items_in_order), 0) as AVG_ORDER_SIZE
            FROM customer_base cb
            LEFT JOIN customer_order_summary cos ON cb.CUSTOMER_EMAILID = cos.CUSTOMER_EMAILID
            LEFT JOIN customer_item_summary cis ON cb.CUSTOMER_EMAILID = cis.CUSTOMER_EMAILID
            GROUP BY cb.CUSTOMER_EMAILID
        ),
        return_behavior AS (
            SELECT 
                cb.CUSTOMER_EMAILID,
                COALESCE(
                    SUM(cis.return_frequency)::FLOAT / NULLIF(COUNT(cis.SKU), 0), 0
                ) as RETURN_RATE,
                COALESCE(
                    SUM(cis.total_qty_returned)::FLOAT / NULLIF(SUM(cis.total_qty_purchased), 0), 0
                ) as RETURN_RATIO,
                COALESCE(COUNT(DISTINCT CASE WHEN cis.return_frequency > 0 THEN cis.SKU END), 0) as RETURN_PRODUCT_VARIETY,
                COALESCE(
                    SUM(cos.items_returned_in_order)::FLOAT / NULLIF(COUNT(DISTINCT cos.SALES_ORDER_NO), 0), 0
                ) as AVG_RETURNS_PER_ORDER,
                COALESCE(
                    SUM(CASE WHEN cos.items_returned_in_order > 0 THEN 1 ELSE 0 END)::FLOAT / 
                    NULLIF(COUNT(DISTINCT cos.SALES_ORDER_NO), 0), 0
                ) as RETURN_FREQUENCY_RATIO,
                COALESCE(
                    AVG(CASE WHEN cis.total_qty_purchased > 0 
                        THEN cis.total_qty_returned::FLOAT / cis.total_qty_purchased END), 0
                ) as RETURN_INTENSITY,
                COALESCE(cr.CONSECUTIVE_RETURNS, 0) as CONSECUTIVE_RETURNS,
                COALESCE(cr.AVG_CONSECUTIVE_RETURNS, 0) as AVG_CONSECUTIVE_RETURNS
            FROM customer_base cb
            LEFT JOIN customer_order_summary cos ON cb.CUSTOMER_EMAILID = cos.CUSTOMER_EMAILID
            LEFT JOIN customer_item_summary cis ON cb.CUSTOMER_EMAILID = cis.CUSTOMER_EMAILID
            LEFT JOIN (
                WITH ordered_returns AS (
                    SELECT
                        CUSTOMER_EMAILID,
                        SALES_ORDER_NO,
                        ORDER_DATE,
                        UNITS_RETURNED_FLAG,
                        CASE WHEN UNITS_RETURNED_FLAG THEN 1 ELSE 0 END AS is_return,
                        ROW_NUMBER() OVER (PARTITION BY CUSTOMER_EMAILID ORDER BY ORDER_DATE) AS rn
                    FROM {table_name}
                ),
                return_groups AS (
                    SELECT *,
                        SUM(CASE WHEN is_return = 0 THEN 1 ELSE 0 END) OVER (
                            PARTITION BY CUSTOMER_EMAILID ORDER BY rn
                            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                        ) AS return_group
                    FROM ordered_returns
                ),
                return_streaks AS (
                    SELECT
                        CUSTOMER_EMAILID,
                        return_group,
                        COUNT(*) AS streak_length
                    FROM return_groups
                    WHERE is_return = 1
                    GROUP BY CUSTOMER_EMAILID, return_group
                )
                SELECT
                    CUSTOMER_EMAILID,
                    COUNT(*) AS CONSECUTIVE_RETURNS,
                    AVG(streak_length) AS AVG_CONSECUTIVE_RETURNS
                FROM return_streaks
                GROUP BY CUSTOMER_EMAILID
            ) cr ON cb.CUSTOMER_EMAILID = cr.CUSTOMER_EMAILID
            GROUP BY cb.CUSTOMER_EMAILID, cr.CONSECUTIVE_RETURNS, cr.AVG_CONSECUTIVE_RETURNS
        ),
        temporal_features AS (
            SELECT 
                cos.CUSTOMER_EMAILID,
                MIN(cos.ORDER_DATE) as first_order_date,
                MAX(cos.ORDER_DATE) as last_order_date,
                DATEDIFF('day', MIN(cos.ORDER_DATE), MAX(cos.ORDER_DATE)) as CUSTOMER_LIFETIME_DAYS,
                CASE 
                    WHEN DATEDIFF('day', MIN(cos.ORDER_DATE), MAX(cos.ORDER_DATE)) < 90 THEN 'New'
                    WHEN DATEDIFF('day', MIN(cos.ORDER_DATE), MAX(cos.ORDER_DATE)) < 180 THEN 'Growing'
                    ELSE 'Mature'
                END as CUSTOMER_TENURE_STAGE
            FROM customer_order_summary cos
            GROUP BY cos.CUSTOMER_EMAILID
        ),
        return_timing AS (
            SELECT 
                CUSTOMER_EMAILID,
                AVG(days_to_return) as AVG_DAYS_TO_RETURN,
                STDDEV(days_to_return) as RETURN_TIMING_SPREAD
            FROM return_timing_analysis
            GROUP BY CUSTOMER_EMAILID
        ),
        recent_activity AS (
            SELECT 
                cos.CUSTOMER_EMAILID,
                COUNT(DISTINCT CASE WHEN cos.ORDER_DATE >= (SELECT recent_cutoff_date FROM reference_metadata) THEN cos.SALES_ORDER_NO END) as RECENT_ORDERS,
                SUM(CASE WHEN cos.ORDER_DATE >= (SELECT recent_cutoff_date FROM reference_metadata) THEN cos.items_returned_in_order END) as RECENT_RETURNS
            FROM customer_order_summary cos
            GROUP BY cos.CUSTOMER_EMAILID
        )
        SELECT 
            bm.CUSTOMER_EMAILID,
            bm.SALES_ORDER_NO_nunique,
            bm.SKU_nunique,
            bm.ITEMS_RETURNED_COUNT,
            bm.SALES_QTY_mean,
            bm.AVG_ORDER_SIZE,
            rb.RETURN_RATE,
            rb.RETURN_RATIO,
            rb.RETURN_PRODUCT_VARIETY,
            rb.AVG_RETURNS_PER_ORDER,
            rb.RETURN_FREQUENCY_RATIO,
            rb.RETURN_INTENSITY,
            rb.CONSECUTIVE_RETURNS,
            rb.AVG_CONSECUTIVE_RETURNS,
            tf.CUSTOMER_LIFETIME_DAYS,
            tf.CUSTOMER_TENURE_STAGE,
            tf.first_order_date,
            tf.last_order_date,
            rt.AVG_DAYS_TO_RETURN,
            rt.RETURN_TIMING_SPREAD,
            ra.RECENT_ORDERS,
            ra.RECENT_RETURNS,
            NULL AS SKU_ADJACENCY_ORDERS,
            NULL AS SKU_ADJACENCY_RETURNS,
            NULL AS SKU_ADJACENCY_TIMING,
            NULL AS SKU_ADJACENCY_RETURN_TIMING,
            cis.Q_CLS_ID
        FROM basic_metrics bm
        JOIN return_behavior rb ON bm.CUSTOMER_EMAILID = rb.CUSTOMER_EMAILID
        LEFT JOIN temporal_features tf ON bm.CUSTOMER_EMAILID = tf.CUSTOMER_EMAILID
        LEFT JOIN return_timing rt ON bm.CUSTOMER_EMAILID = rt.CUSTOMER_EMAILID
        LEFT JOIN recent_activity ra ON bm.CUSTOMER_EMAILID = ra.CUSTOMER_EMAILID
        LEFT JOIN customer_item_summary cis ON bm.CUSTOMER_EMAILID = cis.CUSTOMER_EMAILID
    """)

def _feature_validation(conn, features_table_name):
    """
    Basic feature validation: checks row count, nulls, and prints schema for the features table.
    """
    logger.info(f"Validating features table: {features_table_name}")
    # Row count
    row_count = conn.execute(f"SELECT COUNT(*) FROM {features_table_name}").fetchone()[0]
    logger.info(f"Feature table row count: {row_count}")
    print(f"Feature table row count: {row_count}")
    # Null summary for each column (Python loop, not SQL string concat)
    columns = [row[0] for row in conn.execute(f"PRAGMA table_info('{features_table_name}')").fetchall()]
    nulls = []
    for col in columns:
        null_count = conn.execute(f"SELECT COUNT(*) FROM {features_table_name} WHERE {col} IS NULL").fetchone()[0]
        nulls.append({'column': col, 'null_count': null_count})
    import pandas as pd
    nulls_df = pd.DataFrame(nulls)
    logger.info(f"Nulls per column:\n{nulls_df}")
    print("Nulls per column:")
    print(nulls_df)
    # Print schema
    schema = conn.execute(f"PRAGMA table_info('{features_table_name}')").fetchdf()
    logger.info(f"Feature table schema:\n{schema}")
    print("Feature table schema:")
    print(schema)
