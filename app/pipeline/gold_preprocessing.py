import re
import logging
from io import BytesIO

import pandas as pd

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

SILVER_BUCKET = "silver"
GOLD_BUCKET = "gold"
SILVER_PREFIX = "pipeline/"


def preprocess_all(storage: StorageBackend) -> dict[str, pd.DataFrame]:
    storage.ensure_bucket(GOLD_BUCKET)

    silver_files = storage.list_objects(SILVER_BUCKET, SILVER_PREFIX)
    gold_tables = {}

    for obj_name in silver_files:
        table_name = obj_name.replace(SILVER_PREFIX, "").replace(".csv", "")
        df = _read_csv(storage, obj_name)
        if df.empty:
            continue

        processor = PROCESSORS.get(table_name, _default_processor)
        cleaned = processor(df)
        cleaned = _common_cleanup(cleaned)

        _save_to_gold(storage, table_name, cleaned)
        gold_tables[table_name] = cleaned

    return gold_tables


def _read_csv(storage: StorageBackend, object_name: str) -> pd.DataFrame:
    try:
        data = storage.get_object(SILVER_BUCKET, object_name)
        return pd.read_csv(data, encoding="utf-8")
    except Exception as e:
        logger.error(f"[GOLD] Error reading {object_name}: {e}")
        return pd.DataFrame()


def _common_cleanup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(how="all", axis=0)
    df = df.dropna(how="all", axis=1)
    df.columns = [_to_snake_case(c) for c in df.columns]
    df = df.reset_index(drop=True)
    return df


def _to_snake_case(name: str) -> str:
    s = re.sub(r"[^\w\s]", "", str(name))
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.lower()


def _parse_number(val) -> float | None:
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    s = s.replace(",", "")
    s = s.replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def _process_annual_consumption(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["kwh_per_household", "kwh_per_m2", "kwh_per_person", "avg_max_power_demand_w"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_number)
    return df


def _process_appliance_consumption(df: pd.DataFrame) -> pd.DataFrame:
    if "avg_annual_consumption_kwh" in df.columns:
        df["avg_annual_consumption_kwh"] = df["avg_annual_consumption_kwh"].apply(_parse_number)
    return df


def _process_potential_savings(df: pd.DataFrame) -> pd.DataFrame:
    if "avg_annual_savings_kwh" in df.columns:
        df["avg_annual_savings_kwh"] = df["avg_annual_savings_kwh"].apply(_parse_number)
    return df


def _process_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    pct_cols = [c for c in df.columns if c not in ("household_type", "category")]
    for col in pct_cols:
        df[col] = df[col].apply(_parse_number)
    return df


def _process_detailed_consumption(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["kwh_per_household", "kwh_per_m2", "kwh_per_person"]:
        if col in df.columns:
            df[col] = df[col].apply(_parse_number)
    return df


def _default_processor(df: pd.DataFrame) -> pd.DataFrame:
    return df


PROCESSORS = {
    "annual_consumption_by_household": _process_annual_consumption,
    "appliance_avg_consumption": _process_appliance_consumption,
    "potential_savings": _process_potential_savings,
    "consumption_breakdown_by_category": _process_breakdown,
    "detailed_consumption_by_house_type": _process_detailed_consumption,
}


def _save_to_gold(storage: StorageBackend, table_name: str, df: pd.DataFrame):
    csv_buffer = BytesIO()
    df.to_csv(csv_buffer, index=False, encoding="utf-8")
    csv_buffer.seek(0)
    size = csv_buffer.getbuffer().nbytes

    object_name = f"pipeline/{table_name}.csv"
    storage.put_object(GOLD_BUCKET, object_name, csv_buffer, "text/csv")
    logger.info(f"[GOLD] Saved '{object_name}' ({len(df)} rows, {size} bytes)")
