from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import duckdb
import pandas as pd

from config import settings
from src.exceptions import FeatureStoreError
from src.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    categorical_cols: List[str]
    numeric_cols: List[str]


DEFAULT_FEATURE_CONFIG = FeatureConfig(
    categorical_cols=[
        "origin_city",
        "destination_city",
        "origin_state",
        "destination_state",
        "product_category",
        "supplier_name",
        "carrier_name",
        "truck_type",
        "month",
    ],
    numeric_cols=[
        "quantity",
        "weight_kg",
        "value_inr",
        "risk_score",
        "vendor_avg_delay_days",
        "vendor_on_time_rate",
        "route_distance_km",
        "route_vendor_reliability_sim",
        "route_delay_probability_sim",
    ],
)


class FeatureBuilder:
    def __init__(self, db_path: Optional[str] = None) -> None:
        try:
            self.con = duckdb.connect(
                str(db_path or (settings.FEATURE_STORE_DIR / "flowsight.duckdb"))
            )
            logger.info("Connected to DuckDB for feature building")
        except Exception as exc:
            logger.exception("Failed to connect to feature store for features")
            raise FeatureStoreError("Could not connect to feature store") from exc

    def build_training_frame(self) -> pd.DataFrame:
        """
        Build model-ready dataframe from primary_shipments + vendor_stats + route_stats.

        Targets:
          - will_delay (binary 0/1 from on_time)
          - delay_days (regression)
          - delay_reason (multi-class)
        """
        try:
            query = """
            SELECT
                p.shipment_id,
                p.order_date,
                p.expected_delivery_date,
                p.actual_delivery_date,
                p.origin_city,
                p.destination_city,
                p.origin_state,
                p.destination_state,
                p.product_category,
                p.supplier_name,
                p.carrier_name,
                p.truck_type,
                p.quantity,
                p.weight_kg,
                p.value_inr,
                p.gst_rate,
                p.on_time,
                p.delay_days,
                p.delay_reason,
                p.weather_impact,
                p.festival_impact,
                p.risk_score,
                p.status,
                p.notes,
                p.month,

                -- Targets
                CASE WHEN p.on_time = 'Yes' THEN 0 ELSE 1 END AS will_delay,

                -- Vendor enrichment
                v.avg_delay_days   AS vendor_avg_delay_days,
                v.on_time_rate     AS vendor_on_time_rate,

                -- Route enrichment (join by shipment_id and vendor_id)
                r.distance_km                AS route_distance_km,
                r.vendor_reliability_sim     AS route_vendor_reliability_sim,
                r.delay_probability_sim      AS route_delay_probability_sim

            FROM primary_shipments p
            LEFT JOIN vendor_stats v
              ON p.supplier_name = v.vendor_id
            LEFT JOIN route_stats r
              ON p.shipment_id = r.shipment_id
        """
            df = self.con.execute(query).fetch_df()
            logger.info("Built raw training frame with %d rows, %d columns", *df.shape)
        except Exception as exc:
            logger.exception("Failed to build training frame from DuckDB")
            raise FeatureStoreError("Failed to build training dataframe") from exc

        try:
            df["value_per_kg"] = df["value_inr"] / df["weight_kg"].clip(lower=1)
            df["route_pair"] = df["origin_city"] + "->" + df["destination_city"]
        except Exception as exc:
            logger.exception("Failed to compute derived features")
            raise FeatureStoreError("Failed to compute derived features") from exc

        return df
