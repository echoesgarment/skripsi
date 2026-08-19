# ==============================================================================
# FILE: import_database.py
# Import Excel Greenline ke SQLite
#
# Dapat dijalankan dari:
# 1. Terminal: python import_database.py
# 2. Jupyter Notebook: %run import_database.py
# ==============================================================================

import re
import sqlite3
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------------------------
# 1. KONFIGURASI NAMA FILE
# ------------------------------------------------------------------------------
EXCEL_FILENAME = "data greenline 2024 sd juni 2026.xlsx"
DATABASE_FILENAME = "greenline.db"
TABLE_NAME = "passenger_daily"


# ------------------------------------------------------------------------------
# 2. MAPPING NAMA STASIUN
# ------------------------------------------------------------------------------
STATION_NAMES = [
    "CICAYUR",
    "CIKOYA",
    "CILEJIT",
    "CISAUK",
    "CITERAS",
    "DARU",
    "JATAKE",
    "JURANGMANGU",
    "KEBAYORAN",
    "MAJA",
    "PALMERAH",
    "PARUNGPANJANG",
    "PONDOKRANJI",
    "RANGKASBITUNG",
    "RAWA BUNTU",
    "SERPONG",
    "SUDIMARA",
    "TANAHABANG",
    "TENJO",
    "TIGARAKSA",
]

COMPACT_STATION_MAPPING = {
    re.sub(r"[^A-Z0-9]", "", station): station
    for station in STATION_NAMES
}


def canonical_station_name(value):
    compact_name = re.sub(
        r"[^A-Z0-9]",
        "",
        str(value).strip().upper(),
    )

    return COMPACT_STATION_MAPPING.get(
        compact_name,
        str(value).strip().upper(),
    )


# ------------------------------------------------------------------------------
# 3. PENCARIAN FILE EXCEL
# ------------------------------------------------------------------------------
try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

SEARCH_ROOTS = [
    SCRIPT_DIR,
    SCRIPT_DIR.parent,
    Path.cwd(),
    Path.cwd().parent,
]


def unique_paths(paths):
    result = []
    seen = set()

    for path in paths:
        resolved = path.resolve()

        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)

    return result


SEARCH_ROOTS = unique_paths(SEARCH_ROOTS)


def find_excel_file():
    checked_paths = []

    for root in SEARCH_ROOTS:
        candidates = [
            root / EXCEL_FILENAME,
            root / "DATASET" / EXCEL_FILENAME,
        ]

        for candidate in candidates:
            checked_paths.append(candidate)

            if candidate.exists() and candidate.is_file():
                return candidate.resolve(), checked_paths

    return None, checked_paths


excel_path, checked_paths = find_excel_file()

if excel_path is None:
    locations = "\n".join(
        f"- {path.resolve()}"
        for path in checked_paths
    )

    raise FileNotFoundError(
        f"File Excel `{EXCEL_FILENAME}` tidak ditemukan.\n\n"
        f"Lokasi yang diperiksa:\n{locations}"
    )

database_path = excel_path.parent / DATABASE_FILENAME


# ------------------------------------------------------------------------------
# 4. FUNGSI CLEANING
# ------------------------------------------------------------------------------
def normalize_column_name(column_name):
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return normalized.strip("_")


def clean_integer_series(series):
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors="coerce",
        ).round()

    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace(r"[^\d\-]", "", regex=True)
        .replace("", pd.NA)
    )

    return pd.to_numeric(
        cleaned,
        errors="coerce",
    ).round()


# ------------------------------------------------------------------------------
# 5. LOAD EXCEL
# ------------------------------------------------------------------------------
print("=" * 78)
print("IMPORT EXCEL KE SQLITE")
print("=" * 78)
print(f"File Excel     : {excel_path}")
print(f"Database tujuan: {database_path}")
print()

dataframe = pd.read_excel(excel_path)

raw_row_count = len(dataframe)

dataframe.columns = [
    normalize_column_name(column)
    for column in dataframe.columns
]

required_columns = {
    "tanggal",
    "nama_stasiun",
    "penumpang_berangkat_komuter",
    "penumpang_datang_komuter",
}

missing_columns = required_columns.difference(dataframe.columns)

if missing_columns:
    raise ValueError(
        "Kolom berikut tidak ditemukan pada file Excel: "
        + ", ".join(sorted(missing_columns))
        + "\n\nKolom yang tersedia: "
        + ", ".join(dataframe.columns)
    )


# ------------------------------------------------------------------------------
# 6. CLEANING DATA
# ------------------------------------------------------------------------------
dataframe["tanggal"] = pd.to_datetime(
    dataframe["tanggal"],
    dayfirst=True,
    errors="coerce",
)

dataframe["nama_stasiun"] = (
    dataframe["nama_stasiun"]
    .astype("string")
    .str.strip()
    .map(canonical_station_name)
)

passenger_columns = [
    "penumpang_berangkat_komuter",
    "penumpang_datang_komuter",
]

for column in passenger_columns:
    dataframe[column] = clean_integer_series(
        dataframe[column]
    )

validation_columns = [
    "tanggal",
    "nama_stasiun",
    "penumpang_berangkat_komuter",
    "penumpang_datang_komuter",
]

invalid_mask = dataframe[
    validation_columns
].isna().any(axis=1)

invalid_rows = dataframe.loc[
    invalid_mask,
    validation_columns,
].copy()

if not invalid_rows.empty:
    invalid_output_path = (
        excel_path.parent
        / "baris_invalid_import.csv"
    )

    invalid_rows.to_csv(
        invalid_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    raise ValueError(
        f"Terdapat {len(invalid_rows)} baris tidak valid. "
        f"Daftar baris telah disimpan di:\n{invalid_output_path}"
    )

dataframe["penumpang_berangkat_komuter"] = (
    dataframe["penumpang_berangkat_komuter"]
    .astype("int64")
)

dataframe["penumpang_datang_komuter"] = (
    dataframe["penumpang_datang_komuter"]
    .astype("int64")
)

dataframe["volume_penumpang"] = (
    dataframe["penumpang_berangkat_komuter"]
    + dataframe["penumpang_datang_komuter"]
)

dataframe["tanggal"] = (
    dataframe["tanggal"]
    .dt.normalize()
    .dt.strftime("%Y-%m-%d")
)

database_dataframe = dataframe[
    [
        "tanggal",
        "nama_stasiun",
        "penumpang_berangkat_komuter",
        "penumpang_datang_komuter",
        "volume_penumpang",
    ]
].copy()


# ------------------------------------------------------------------------------
# 7. VALIDASI DUPLIKAT
# ------------------------------------------------------------------------------
duplicate_mask = database_dataframe.duplicated(
    subset=["nama_stasiun", "tanggal"],
    keep=False,
)

duplicate_row_count = int(duplicate_mask.sum())

if duplicate_row_count > 0:
    duplicate_output_path = (
        excel_path.parent
        / "baris_duplikat_import.csv"
    )

    database_dataframe.loc[
        duplicate_mask
    ].sort_values(
        ["nama_stasiun", "tanggal"]
    ).to_csv(
        duplicate_output_path,
        index=False,
        encoding="utf-8-sig",
    )

    # Satu kombinasi stasiun-tanggal hanya boleh memiliki satu baris.
    # Jika ada duplikat, baris terakhir pada Excel dipakai.
    database_dataframe = (
        database_dataframe
        .drop_duplicates(
            subset=["nama_stasiun", "tanggal"],
            keep="last",
        )
    )

    print(
        f"Peringatan      : {duplicate_row_count} baris duplikat ditemukan."
    )
    print(
        f"Daftar duplikat : {duplicate_output_path}"
    )
    print(
        "Tindakan        : data terakhir untuk setiap stasiun-tanggal dipakai."
    )
    print()

database_dataframe = (
    database_dataframe
    .sort_values(
        ["nama_stasiun", "tanggal"]
    )
    .reset_index(drop=True)
)


# ------------------------------------------------------------------------------
# 8. SIMPAN KE SQLITE
# ------------------------------------------------------------------------------
with sqlite3.connect(database_path) as connection:
    connection.execute(
        f"DROP TABLE IF EXISTS {TABLE_NAME}"
    )

    database_dataframe.to_sql(
        name=TABLE_NAME,
        con=connection,
        if_exists="replace",
        index=False,
    )

    connection.execute(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_station_date
        ON {TABLE_NAME} (nama_stasiun, tanggal)
        """
    )

    connection.execute(
        f"""
        CREATE INDEX IF NOT EXISTS idx_date
        ON {TABLE_NAME} (tanggal)
        """
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS import_metadata (
            imported_at TEXT NOT NULL,
            source_file TEXT NOT NULL,
            raw_rows INTEGER NOT NULL,
            stored_rows INTEGER NOT NULL,
            minimum_date TEXT,
            maximum_date TEXT
        )
        """
    )

    connection.execute(
        "DELETE FROM import_metadata"
    )

    connection.execute(
        """
        INSERT INTO import_metadata (
            imported_at,
            source_file,
            raw_rows,
            stored_rows,
            minimum_date,
            maximum_date
        )
        VALUES (
            datetime('now', 'localtime'),
            ?,
            ?,
            ?,
            ?,
            ?
        )
        """,
        (
            str(excel_path),
            raw_row_count,
            len(database_dataframe),
            database_dataframe["tanggal"].min(),
            database_dataframe["tanggal"].max(),
        ),
    )

    connection.commit()


# ------------------------------------------------------------------------------
# 9. VERIFIKASI HASIL
# ------------------------------------------------------------------------------
with sqlite3.connect(database_path) as connection:
    verification = pd.read_sql_query(
        f"""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT nama_stasiun) AS total_stations,
            MIN(tanggal) AS minimum_date,
            MAX(tanggal) AS maximum_date
        FROM {TABLE_NAME}
        """,
        connection,
    )

print("=" * 78)
print("IMPORT BERHASIL")
print("=" * 78)
print(f"Baris Excel awal : {raw_row_count:,}")
print(f"Baris database   : {int(verification.loc[0, 'total_rows']):,}")
print(f"Jumlah stasiun   : {int(verification.loc[0, 'total_stations'])}")
print(f"Periode awal     : {verification.loc[0, 'minimum_date']}")
print(f"Periode akhir    : {verification.loc[0, 'maximum_date']}")
print(f"Database tersimpan di:\n{database_path}")
