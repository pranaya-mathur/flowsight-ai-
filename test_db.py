from src.data.feature_store import FeatureStore

fs = FeatureStore()
fs.load_raw_tables()
fs.basic_health_check()
