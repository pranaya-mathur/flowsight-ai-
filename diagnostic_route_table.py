# diagnostic_route_table.py
import duckdb
from config import settings

db_path = settings.FEATURE_STORE_DIR / "flowsight.duckdb"
con = duckdb.connect(str(db_path), read_only=True)

print("Route Stats Table Schema:")
print(con.execute("DESCRIBE route_stats;").fetch_df())

print("\nSample Data:")
print(con.execute("SELECT * FROM route_stats LIMIT 3;").fetch_df())
