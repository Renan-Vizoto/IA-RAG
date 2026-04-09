import logging
from io import BytesIO

import pandas as pd
import pdfplumber

from app.pipeline.storage import StorageBackend

logger = logging.getLogger(__name__)

SILVER_BUCKET = "silver"


def extract_all_tables(pdf_path: str, storage: StorageBackend) -> dict[str, pd.DataFrame]:
    tables = {}

    with pdfplumber.open(pdf_path) as pdf:
        tables["annual_consumption_by_household"] = _extract_table1(pdf)
        tables["appliance_avg_consumption"] = _extract_table3(pdf)
        tables["potential_savings"] = _extract_table4(pdf)
        tables["consumption_breakdown_by_category"] = _extract_breakdown_tables(pdf)
        tables["detailed_consumption_by_house_type"] = _extract_table9(pdf)

    _save_all(tables, storage)
    return tables


def _extract_table1(pdf: pdfplumber.PDF) -> pd.DataFrame:
    """Table 1 (p26): Annual consumption by household type."""
    page = pdf.pages[25]
    raw_tables = page.extract_tables()
    if not raw_tables:
        return pd.DataFrame()

    raw = raw_tables[0]
    rows = []
    for row in raw:
        cells = [c for c in row if c is not None]
        non_empty = [c for c in cells if c and c.strip()]
        if not non_empty or len(non_empty) < 3:
            continue
        text = non_empty[0].strip().lower()
        # Skip header rows (single-word headers or known keywords)
        if text in ("household", "maximum", "power", "demand", "(w)"):
            continue
        if any(kw in text for kw in ["annual consumption", "household type", "(kwh)"]):
            continue
        if all(w in ("per", "household", "m²", "person", "average", "maximum") for w in text.split()):
            continue
        rows.append(non_empty)

    data = []
    for row in rows:
        name = row[0].replace("\n", " ").strip()
        if len(row) == 5:
            # Household type rows: name, per_household, per_m2, per_person, max_power
            data.append({
                "household_type": name,
                "kwh_per_household": row[1],
                "kwh_per_m2": row[2],
                "kwh_per_person": row[3],
                "avg_max_power_demand_w": row[4],
            })
        elif len(row) == 4:
            # House type rows: name, per_household, per_m2, max_power (no per_person)
            data.append({
                "household_type": name,
                "kwh_per_household": row[1],
                "kwh_per_m2": row[2],
                "kwh_per_person": None,
                "avg_max_power_demand_w": row[3],
            })

    df = pd.DataFrame(data)
    logger.info(f"[SILVER] Table 1 - Annual consumption: {len(df)} rows")
    return df


def _extract_table3(pdf: pdfplumber.PDF) -> pd.DataFrame:
    """Table 3 (p31): Average consumption per appliance type."""
    page = pdf.pages[30]
    raw_tables = page.extract_tables()
    if not raw_tables:
        return pd.DataFrame()

    raw = raw_tables[0]
    data = []
    for row in raw:
        cells = [c for c in row if c is not None]
        non_empty = [c for c in cells if c and c.strip()]
        if not non_empty:
            continue
        if any(kw in non_empty[0].lower() for kw in ["appliance type", "average annual", "consumption", "(kwh)"]):
            continue
        if len(non_empty) >= 2:
            data.append({
                "appliance_type": non_empty[0].replace("\n", " ").strip(),
                "avg_annual_consumption_kwh": non_empty[1],
            })

    df = pd.DataFrame(data)
    logger.info(f"[SILVER] Table 3 - Appliance consumption: {len(df)} rows")
    return df


def _extract_table4(pdf: pdfplumber.PDF) -> pd.DataFrame:
    """Table 4 (p32): Potential savings per appliance type."""
    page = pdf.pages[31]
    raw_tables = page.extract_tables()
    if not raw_tables:
        return pd.DataFrame()

    raw = raw_tables[0]
    data = []
    for row in raw:
        cells = [c for c in row if c is not None]
        non_empty = [c for c in cells if c and c.strip()]
        if not non_empty:
            continue
        if any(kw in non_empty[0].lower() for kw in ["type of appliance", "average annual", "savings", "(kwh)"]):
            continue
        if len(non_empty) >= 2:
            data.append({
                "appliance_type": non_empty[0].replace("\n", " ").strip(),
                "avg_annual_savings_kwh": non_empty[1],
            })

    df = pd.DataFrame(data)
    logger.info(f"[SILVER] Table 4 - Potential savings: {len(df)} rows")
    return df


def _extract_breakdown_tables(pdf: pdfplumber.PDF) -> pd.DataFrame:
    """Pages 29-30: Consumption breakdown by category and household type."""
    all_data = []
    for page_idx in [28, 29]:
        page = pdf.pages[page_idx]
        raw_tables = page.extract_tables()

        for table in raw_tables:
            if not table or len(table) < 4:
                continue

            header_cells = [c for c in table[0] if c is not None and c.strip()]
            household_type = header_cells[0] if header_cells else "Unknown"

            data_start = 0
            for i, row in enumerate(table):
                cells = [c for c in row if c is not None]
                if any("All days" in str(c) for c in cells):
                    data_start = i + 1
                    break

            if data_start == 0:
                continue

            for row in table[data_start:]:
                cells = [c for c in row if c is not None]
                non_empty = [c for c in cells if c and c.strip()]
                if not non_empty:
                    continue

                category = non_empty[0].strip()
                values = non_empty[1:] if len(non_empty) > 1 else []

                record = {
                    "household_type": household_type.replace("\n", " ").strip(),
                    "category": category,
                }

                col_map = [
                    "without_heating_all_days",
                    "without_heating_holidays",
                    "without_heating_workdays",
                    "with_heating_all_days",
                    "with_heating_holidays",
                    "with_heating_workdays",
                ]
                for j, val in enumerate(values):
                    if j < len(col_map):
                        record[col_map[j]] = val

                all_data.append(record)

    df = pd.DataFrame(all_data)
    logger.info(f"[SILVER] Breakdown tables: {len(df)} rows")
    return df


def _extract_table9(pdf: pdfplumber.PDF) -> pd.DataFrame:
    """Table 9 (p73-75): Detailed annual consumption by house type."""
    all_data = []

    sections = {
        72: "with_and_without_electric_heating",
        73: "without_electric_heating",
        74: "with_primary_electric_heating",
    }

    for page_idx, section_label in sections.items():
        page = pdf.pages[page_idx]
        raw_tables = page.extract_tables()
        if not raw_tables:
            continue

        raw = raw_tables[0]
        for row in raw:
            cells = [c for c in row if c is not None]
            non_empty = [c for c in cells if c and c.strip()]
            if not non_empty:
                continue

            first = non_empty[0]
            if any(kw in first.lower() for kw in [
                "annual consumption", "per household", "per m", "per person",
                "gnitaeh", "kwh"
            ]):
                continue

            if len(non_empty) >= 3:
                name = first.replace("\n", " ").replace("  ", " ").strip()
                all_data.append({
                    "heating_category": section_label,
                    "house_type": name,
                    "kwh_per_household": non_empty[1],
                    "kwh_per_m2": non_empty[2],
                    "kwh_per_person": non_empty[3] if len(non_empty) > 3 and non_empty[3] else None,
                })

    df = pd.DataFrame(all_data)
    logger.info(f"[SILVER] Table 9 - Detailed consumption: {len(df)} rows")
    return df


def _save_all(tables: dict[str, pd.DataFrame], storage: StorageBackend):
    storage.ensure_bucket(SILVER_BUCKET)

    for name, df in tables.items():
        if df.empty:
            logger.warning(f"[SILVER] Skipping empty table: {name}")
            continue

        csv_buffer = BytesIO()
        df.to_csv(csv_buffer, index=False, encoding="utf-8")
        csv_buffer.seek(0)
        size = csv_buffer.getbuffer().nbytes

        object_name = f"pipeline/{name}.csv"
        storage.put_object(SILVER_BUCKET, object_name, csv_buffer, "text/csv")
        logger.info(f"[SILVER] Saved '{object_name}' ({len(df)} rows, {size} bytes)")
