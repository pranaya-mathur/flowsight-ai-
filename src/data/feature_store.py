from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

from config import settings
from src.exceptions import DataLoadError, FeatureStoreError
from src.logging_utils import get_logger

logger = get_logger(__name__)


class FeatureStore:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        try:
            self.db_path = db_path or (settings.FEATURE_STORE_DIR / "flowsight.duckdb")
            settings.FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
            self.con = duckdb.connect(str(self.db_path))
            logger.info("Connected to DuckDB at %s", self.db_path)
        except Exception as exc:
            logger.exception("Failed to initialize feature store")
            raise FeatureStoreError("Could not initialize DuckDB feature store") from exc

    def load_raw_tables(self) -> None:
        """Load three raw CSVs into DuckDB tables."""
        try:
            primary_df = pd.read_csv(settings.PRIMARY_DATA_PATH)
            vendor_df = pd.read_csv(settings.VENDOR_DATA_PATH)
            route_df = pd.read_csv(settings.ROUTE_DATA_PATH)
        except Exception as exc:
            logger.exception("Error reading raw CSV files")
            raise DataLoadError("Failed to read one or more raw CSV files") from exc

        try:
            self.con.register("primary_df", primary_df)
            self.con.register("vendor_df", vendor_df)
            self.con.register("route_df", route_df)

            self.con.execute('CREATE OR REPLACE TABLE primary_shipments AS SELECT * FROM primary_df;')
            self.con.execute('CREATE OR REPLACE TABLE vendor_stats AS SELECT * FROM vendor_df;')
            self.con.execute('CREATE OR REPLACE TABLE route_stats AS SELECT * FROM route_df;')

            logger.info(
                "Loaded raw tables: primary(%d), vendors(%d), routes(%d)",
                primary_df.shape[0],
                vendor_df.shape[0],
                route_df.shape[0],
            )
        except Exception as exc:
            logger.exception("Error creating DuckDB tables from raw data")
            raise FeatureStoreError("Failed to create tables in feature store") from exc

    def basic_health_check(self) -> bool:
        """Simple sanity checks on table row counts."""
        try:
            counts = self.con.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM primary_shipments) AS primary_rows,
                    (SELECT COUNT(*) FROM vendor_stats)      AS vendor_rows,
                    (SELECT COUNT(*) FROM route_stats)       AS route_rows
                """
            ).fetchone()
            logger.info(
                "Feature store health: primary=%s, vendors=%s, routes=%s",
                counts[0],
                counts[1],
                counts[2],
            )
            return all(c > 0 for c in counts)
        except Exception as exc:
            logger.exception("Health check failed")
            raise FeatureStoreError("Health check on feature store failed") from exc
