# test_db.py
import os
import duckdb

from src.logging_utils import get_logger
from src.exceptions import DataLoadError

logger = get_logger(__name__)


def main(db_path: str = "data/feature_store/flowsight.duckdb") -> None:
    try:
        logger.info("Connecting to DuckDB at %s", db_path)

        if not os.path.exists(db_path):
            raise DataLoadError(f"DuckDB file not found at {db_path}")

        con = duckdb.connect(db_path, read_only=False)

        tables = con.execute("SHOW TABLES").df()
        print("\n=== TABLES ===")
        print(tables)

        schema = con.execute("DESCRIBE training_frame").df()
        print("\n=== training_frame SCHEMA ===")
        print(schema)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to inspect DuckDB.")
        raise DataLoadError("DuckDB inspection failed.") from exc


if __name__ == "__main__":
    main()
