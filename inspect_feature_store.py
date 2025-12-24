from pathlib import Path
import duckdb

from config import settings
from src.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> None:
    db_path = settings.FEATURE_STORE_DIR / "flowsight.duckdb"
    con = duckdb.connect(str(db_path))
    logger.info("Connected to %s", db_path)

    for table in ["primary_shipments", "vendor_stats", "route_stats"]:
        print(f"\n=== {table} ===")
        # Show first 3 rows for quick sanity
        print(con.execute(f"SELECT * FROM {table} LIMIT 3;").fetch_df())
        # Show schema
        print("\nColumns:")
        print(con.execute(f"DESCRIBE {table};").fetch_df())


if __name__ == "__main__":
    main()
