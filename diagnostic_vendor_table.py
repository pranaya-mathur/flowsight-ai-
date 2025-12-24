# diagnostic_vendor_table.py
import duckdb
from config import settings

db_path = settings.FEATURE_STORE_DIR / "flowsight.duckdb"
con = duckdb.connect(str(db_path), read_only=True)

print("Vendor Stats Table Schema:")
print(con.execute("DESCRIBE vendor_stats;").fetch_df())

print("\nSample Data:")
print(con.execute("SELECT * FROM vendor_stats LIMIT 3;").fetch_df())
