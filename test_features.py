from src.data.features import FeatureBuilder

fb = FeatureBuilder()
df = fb.build_training_frame()
print(df[["will_delay", "delay_days", "delay_reason"]].head())
print("Shape:", df.shape)
