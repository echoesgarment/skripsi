# ==============================================================================
# FILE: app.py
# Dashboard Prediksi Volume Penumpang KRL Commuter Line
# ==============================================================================

import datetime
import re
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ------------------------------------------------------------------------------
# 1. KONFIGURASI HALAMAN
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Greenline Predict - Dashboard Prediksi Volume",
    page_icon="🚉",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------------
# 2. KONFIGURASI PATH FILE
# ------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent

SEARCH_ROOTS = [
    APP_DIR,
    APP_DIR.parent,
]


def find_existing_file(filename, subfolders=("", "DATASET", "MODEL", "MODELS")):
    """Mencari file pada folder aplikasi dan satu tingkat di atasnya."""
    checked_paths = []

    for root in SEARCH_ROOTS:
        for subfolder in subfolders:
            candidate = root / subfolder / filename if subfolder else root / filename
            checked_paths.append(candidate)

            if candidate.exists() and candidate.is_file():
                return candidate.resolve(), checked_paths

    return None, checked_paths


DATABASE_PATH, database_checked_paths = find_existing_file(
    "greenline.db",
    subfolders=("", "DATASET"),
)

MODEL_PATH, model_checked_paths = find_existing_file(
    "model_rf_greenline.joblib"
)

FEATURES_PATH, features_checked_paths = find_existing_file(
    "model_features.joblib"
)

STATIONS_PATH, stations_checked_paths = find_existing_file(
    "stations_list.joblib"
)


def format_checked_paths(paths):
    return "\n".join(f"- {path.resolve()}" for path in paths)


if DATABASE_PATH is None:
    st.error(
        "Database `greenline.db` tidak ditemukan.\n\n"
        "Lokasi yang diperiksa:\n"
        f"{format_checked_paths(database_checked_paths)}\n\n"
        "Jalankan file `import_database.py` terlebih dahulu."
    )
    st.stop()

if MODEL_PATH is None:
    st.error(
        "File `model_rf_greenline.joblib` tidak ditemukan.\n\n"
        "Lokasi yang diperiksa:\n"
        f"{format_checked_paths(model_checked_paths)}"
    )
    st.stop()


# ------------------------------------------------------------------------------
# 3. CUSTOM CSS
# ------------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@24,400,0,0');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.stApp {
    background-color: #F4F7FC !important;
}

[data-testid="stSidebar"] {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E5E7EB !important;
    padding-top: 1rem !important;
}

.stButton > button {
    background-color: #00322D !important;
    color: #FFFFFF !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: 42px !important;
    border: none !important;
    box-shadow: 0 4px 10px rgba(0, 50, 45, 0.2) !important;
    width: 100% !important;
}

.nav-item-active {
    background-color: #A7F3D0;
    color: #065F46;
    font-weight: 700;
    padding: 10px 16px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 14px;
}

.nav-item {
    color: #4B5563;
    padding: 10px 16px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
    font-size: 14px;
    font-weight: 500;
}

.kpi-card-box {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 20px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
    border: 1px solid #F1F5F9;
    position: relative;
    height: 100%;
}

.kpi-border-green {
    border-left: 5px solid #006E2A !important;
}

.kpi-border-red {
    border-left: 5px solid #B91C1C !important;
}

.kpi-border-orange {
    border-left: 5px solid #C2410C !important;
}

.kpi-icon-box {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    background: #F1F5F9;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #334155;
}

.kpi-badge-positive {
    background-color: #DCFCE7;
    color: #15803D;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
}

.kpi-badge-negative {
    background-color: #FEE2E2;
    color: #B91C1C;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
}

.kpi-badge-neutral {
    background-color: #E2E8F0;
    color: #475569;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
}

.kpi-title {
    font-size: 11px;
    font-weight: 600;
    color: #64748B;
    margin-top: 12px;
    text-transform: capitalize;
}

.kpi-value-main {
    font-size: 26px;
    font-weight: 800;
    color: #0F172A;
    margin: 2px 0;
}

.kpi-sub-text {
    font-size: 11px;
    color: #94A3B8;
}

.recom-card {
    background: #EAF4F4;
    border-radius: 20px;
    padding: 20px;
    margin-bottom: 16px;
    border: 1px solid #D1E7E5;
}

.perf-card {
    background: #FFFFFF;
    border-radius: 18px;
    padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    display: flex;
    align-items: center;
    gap: 14px;
}

.auto-data-box {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 12px;
    margin-top: 4px;
    margin-bottom: 8px;
}

.auto-data-label {
    color: #64748B;
    font-size: 11px;
    font-weight: 600;
    margin-bottom: 3px;
}

.auto-data-value {
    color: #0F172A;
    font-size: 16px;
    font-weight: 800;
}

.auto-data-date {
    color: #94A3B8;
    font-size: 10px;
    margin-top: 2px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------------------
# 4. MAPPING STASIUN
# ------------------------------------------------------------------------------
STATION_LABEL_MAPPING = {
    "CICAYUR": 0,
    "CIKOYA": 1,
    "CILEJIT": 2,
    "CISAUK": 3,
    "CITERAS": 4,
    "DARU": 5,
    "JATAKE": 6,
    "JURANGMANGU": 7,
    "KEBAYORAN": 8,
    "MAJA": 9,
    "PALMERAH": 10,
    "PARUNGPANJANG": 11,
    "PONDOKRANJI": 12,
    "RANGKASBITUNG": 13,
    "RAWA BUNTU": 14,
    "SERPONG": 15,
    "SUDIMARA": 16,
    "TANAHABANG": 17,
    "TENJO": 18,
    "TIGARAKSA": 19,
}

COMPACT_STATION_MAPPING = {
    re.sub(r"[^A-Z0-9]", "", station): station
    for station in STATION_LABEL_MAPPING
}


def canonical_station_name(value):
    """Menyamakan variasi penulisan nama stasiun."""
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
# 5. LOAD MODEL DAN DAFTAR FITUR
# ------------------------------------------------------------------------------
@st.cache_resource
def load_machine_learning_assets(
    model_path,
    model_modified_time,
    features_path=None,
    features_modified_time=None,
    stations_path=None,
    stations_modified_time=None,
):
    del model_modified_time
    del features_modified_time
    del stations_modified_time

    model = joblib.load(model_path)

    if features_path is not None:
        features = joblib.load(features_path)
    elif hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    else:
        raise FileNotFoundError(
            "File `model_features.joblib` tidak ditemukan dan model tidak "
            "menyimpan atribut `feature_names_in_`."
        )

    if stations_path is not None:
        stations = joblib.load(stations_path)
    else:
        stations = list(STATION_LABEL_MAPPING.keys())

    return model, list(features), list(stations)


try:
    rf_model, model_features, model_stations = load_machine_learning_assets(
        model_path=str(MODEL_PATH),
        model_modified_time=MODEL_PATH.stat().st_mtime,
        features_path=str(FEATURES_PATH) if FEATURES_PATH else None,
        features_modified_time=(
            FEATURES_PATH.stat().st_mtime if FEATURES_PATH else None
        ),
        stations_path=str(STATIONS_PATH) if STATIONS_PATH else None,
        stations_modified_time=(
            STATIONS_PATH.stat().st_mtime if STATIONS_PATH else None
        ),
    )
except Exception as error:
    st.error(f"Gagal memuat aset model: {error}")
    st.stop()


# ------------------------------------------------------------------------------
# 6. FUNGSI DATABASE
# ------------------------------------------------------------------------------
REQUIRED_DATABASE_COLUMNS = {
    "tanggal",
    "nama_stasiun",
    "volume_penumpang",
}


@st.cache_data
def load_database_information(database_path, database_modified_time):
    del database_modified_time

    with sqlite3.connect(database_path) as connection:
        table_check = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
              AND name = 'passenger_daily'
            """
        ).fetchone()

        if table_check is None:
            raise ValueError(
                "Tabel `passenger_daily` tidak ditemukan dalam database."
            )

        table_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(passenger_daily)"
            ).fetchall()
        }

        missing_columns = REQUIRED_DATABASE_COLUMNS.difference(table_columns)

        if missing_columns:
            raise ValueError(
                "Kolom database tidak lengkap: "
                + ", ".join(sorted(missing_columns))
            )

        station_dataframe = pd.read_sql_query(
            """
            SELECT DISTINCT nama_stasiun
            FROM passenger_daily
            WHERE nama_stasiun IS NOT NULL
            ORDER BY nama_stasiun
            """,
            connection,
        )

        date_information = pd.read_sql_query(
            """
            SELECT
                MIN(tanggal) AS minimum_date,
                MAX(tanggal) AS maximum_date,
                COUNT(*) AS total_rows
            FROM passenger_daily
            """,
            connection,
        )

    if station_dataframe.empty:
        raise ValueError("Database tidak memiliki data stasiun.")

    if (
        date_information.empty
        or pd.isna(date_information.loc[0, "minimum_date"])
        or pd.isna(date_information.loc[0, "maximum_date"])
    ):
        raise ValueError("Database tidak memiliki rentang tanggal yang valid.")

    station_lookup = {}

    for raw_station in station_dataframe["nama_stasiun"].astype(str):
        canonical_name = canonical_station_name(raw_station)

        if canonical_name in STATION_LABEL_MAPPING:
            station_lookup[canonical_name] = raw_station

    minimum_date = pd.to_datetime(
        date_information.loc[0, "minimum_date"],
        errors="raise",
    ).date()

    maximum_date = pd.to_datetime(
        date_information.loc[0, "maximum_date"],
        errors="raise",
    ).date()

    total_rows = int(date_information.loc[0, "total_rows"])

    return station_lookup, minimum_date, maximum_date, total_rows


@st.cache_data
def get_passenger_volume(
    database_path,
    database_modified_time,
    database_station,
    target_date_text,
):
    del database_modified_time

    query = """
        SELECT volume_penumpang
        FROM passenger_daily
        WHERE nama_stasiun = ?
          AND tanggal = ?
        LIMIT 1
    """

    with sqlite3.connect(database_path) as connection:
        result = connection.execute(
            query,
            (database_station, target_date_text),
        ).fetchone()

    if result is None or result[0] is None:
        return None

    return int(round(float(result[0])))


@st.cache_data
def load_station_history(
    database_path,
    database_modified_time,
    database_station,
):
    del database_modified_time

    query = """
        SELECT
            tanggal,
            volume_penumpang
        FROM passenger_daily
        WHERE nama_stasiun = ?
        ORDER BY tanggal
    """

    with sqlite3.connect(database_path) as connection:
        dataframe = pd.read_sql_query(
            query,
            connection,
            params=(database_station,),
        )

    if dataframe.empty:
        return dataframe

    dataframe["tanggal"] = pd.to_datetime(
        dataframe["tanggal"],
        errors="coerce",
    )

    dataframe["volume_penumpang"] = pd.to_numeric(
        dataframe["volume_penumpang"],
        errors="coerce",
    )

    return dataframe.dropna(
        subset=["tanggal", "volume_penumpang"]
    ).reset_index(drop=True)


try:
    station_lookup, minimum_data_date, maximum_data_date, total_database_rows = (
        load_database_information(
            database_path=str(DATABASE_PATH),
            database_modified_time=DATABASE_PATH.stat().st_mtime,
        )
    )
except Exception as error:
    st.error(f"Gagal membaca database: {error}")
    st.stop()


# Hanya tampilkan stasiun yang tersedia di database, mapping, dan aset model.
model_station_set = {
    canonical_station_name(station)
    for station in model_stations
}

recognized_model_station_set = (
    model_station_set
    .intersection(STATION_LABEL_MAPPING)
)

available_stations = sorted(
    station
    for station in station_lookup
    if station in STATION_LABEL_MAPPING
    and (
        not recognized_model_station_set
        or station in recognized_model_station_set
    )
)

if not available_stations:
    st.error(
        "Tidak ada stasiun database yang cocok dengan mapping dan daftar "
        "stasiun pada model."
    )
    st.stop()


# ------------------------------------------------------------------------------
# 7. FUNGSI UTILITAS PREDIKSI DAN FORMAT
# ------------------------------------------------------------------------------
def format_integer_indonesia(value):
    if value is None:
        return "Data tidak tersedia"

    return f"{int(round(value)):,}".replace(",", ".")


def format_signed_integer_indonesia(value):
    rounded_value = int(round(value))
    sign = "+" if rounded_value > 0 else ""

    return f"{sign}{rounded_value:,}".replace(",", ".")


def format_percentage(value):
    if value is None:
        return "N/A"

    sign = "+" if value > 0 else ""

    return f"{sign}{value:.2f}%".replace(".", ",")


def format_absolute_percentage(value):
    if value is None:
        return "N/A"

    return f"{abs(value):.2f}%".replace(".", ",")


def build_model_input(
    features,
    selected_station,
    selected_date,
    lag_1,
    lag_7,
):
    input_dataframe = pd.DataFrame(
        0.0,
        index=[0],
        columns=features,
    )

    day_of_week = selected_date.weekday()
    iso_calendar = selected_date.isocalendar()

    feature_values = {
        "day_of_week": day_of_week,
        "dayofweek": day_of_week,
        "weekday": day_of_week,
        "month": selected_date.month,
        "year": selected_date.year,
        "day": selected_date.day,
        "day_of_month": selected_date.day,
        "week_of_year": int(iso_calendar.week),
        "weekofyear": int(iso_calendar.week),
        "quarter": ((selected_date.month - 1) // 3) + 1,
        "is_weekend": int(day_of_week >= 5),
        "lag_1": lag_1,
        "lag1": lag_1,
        "lag_7": lag_7,
        "lag7": lag_7,
        "stasiun_encoded": STATION_LABEL_MAPPING[selected_station],
        "station_encoded": STATION_LABEL_MAPPING[selected_station],
    }

    for feature_name, feature_value in feature_values.items():
        if feature_name in input_dataframe.columns:
            input_dataframe.loc[0, feature_name] = feature_value

    # Dukungan dasar apabila model memakai one-hot encoding nama stasiun.
    compact_selected_station = re.sub(
        r"[^A-Z0-9]",
        "",
        selected_station.upper(),
    )

    for column in input_dataframe.columns:
        compact_column = re.sub(
            r"[^A-Z0-9]",
            "",
            str(column).upper(),
        )

        if (
            compact_selected_station in compact_column
            and (
                "STASIUN" in compact_column
                or "STATION" in compact_column
            )
        ):
            input_dataframe.loc[0, column] = 1.0

    return input_dataframe


def classify_density(predicted_volume, station_history):
    if station_history.empty:
        return {
            "label": "TIDAK TERKLASIFIKASI",
            "color": "#475569",
            "border_class": "",
            "description": "Data historis tidak cukup untuk klasifikasi.",
        }

    quantiles = station_history["volume_penumpang"].quantile(
        [0.50, 0.75, 0.90]
    )

    q50 = float(quantiles.loc[0.50])
    q75 = float(quantiles.loc[0.75])
    q90 = float(quantiles.loc[0.90])

    if predicted_volume <= q50:
        return {
            "label": "NORMAL",
            "color": "#15803D",
            "border_class": "kpi-border-green",
            "description": "Di bawah atau sama dengan median historis.",
        }

    if predicted_volume <= q75:
        return {
            "label": "PADAT",
            "color": "#C2410C",
            "border_class": "kpi-border-orange",
            "description": "Di atas median historis stasiun.",
        }

    if predicted_volume <= q90:
        return {
            "label": "SANGAT PADAT",
            "color": "#B91C1C",
            "border_class": "kpi-border-red",
            "description": "Masuk 25% volume historis tertinggi.",
        }

    return {
        "label": "EKSTREM",
        "color": "#7F1D1D",
        "border_class": "kpi-border-red",
        "description": "Melebihi persentil ke-90 historis.",
    }


def recommendation_items(density_label, trend_percentage):
    recommendations = []

    if density_label == "NORMAL":
        recommendations.append(
            "Pertahankan pola operasi reguler dan lakukan monitoring rutin."
        )
    elif density_label == "PADAT":
        recommendations.extend(
            [
                "Tambahkan petugas pengaturan alur penumpang pada jam sibuk.",
                "Pantau kepadatan peron dan akses masuk secara berkala.",
            ]
        )
    elif density_label == "SANGAT PADAT":
        recommendations.extend(
            [
                "Siapkan penguatan petugas peron dan pengaturan antrean.",
                "Evaluasi penambahan kapasitas perjalanan pada jam sibuk.",
                "Aktifkan pemantauan operasional dengan interval lebih rapat.",
            ]
        )
    elif density_label == "EKSTREM":
        recommendations.extend(
            [
                "Aktifkan skenario pengendalian kepadatan penumpang.",
                "Siapkan kapasitas perjalanan tambahan bila tersedia.",
                "Koordinasikan petugas stasiun, pengamanan, dan pusat kendali.",
            ]
        )
    else:
        recommendations.append(
            "Lakukan validasi data historis sebelum menetapkan tindakan operasi."
        )

    if trend_percentage is not None and trend_percentage >= 10:
        recommendations.append(
            "Kenaikan lebih dari 10% terhadap H-1 memerlukan perhatian tambahan."
        )
    elif trend_percentage is not None and trend_percentage <= -10:
        recommendations.append(
            "Penurunan lebih dari 10% terhadap H-1 perlu diperiksa terhadap pola layanan."
        )

    return recommendations[:4]


# ------------------------------------------------------------------------------
# 8. SIDEBAR
# ------------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px;">
    <div style="background: #00322D; color: white; width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center;">
        <span class="material-symbols-outlined" style="font-size: 22px;">directions_subway</span>
    </div>
    <div>
        <h3 style="margin: 0; font-size: 16px; font-weight: 800; color: #00322D;">Greenline Predict</h3>
        <p style="margin: 0; font-size: 11px; color: #94A3B8;">v2.5.0 • Emerald Transit</p>
    </div>
</div>
<div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 13px; color: #1E293B; margin-bottom: 12px;">
    <span class="material-symbols-outlined" style="font-size: 18px;">filter_alt</span>
    Saringan Prediksi
</div>
""",
        unsafe_allow_html=True,
    )

    default_station_index = (
        available_stations.index("SERPONG")
        if "SERPONG" in available_stations
        else 0
    )

    selected_station = st.selectbox(
        "Pilih Stasiun",
        options=available_stations,
        index=default_station_index,
    )

    minimum_prediction_date = (
        minimum_data_date
        + datetime.timedelta(days=7)
    )

    maximum_prediction_date = (
        maximum_data_date
        + datetime.timedelta(days=1)
    )

    selected_date = st.date_input(
        "Tanggal Prediksi",
        value=maximum_prediction_date,
        min_value=minimum_prediction_date,
        max_value=maximum_prediction_date,
        format="YYYY/MM/DD",
    )

    database_station = station_lookup[selected_station]

    lag_1_date = (
        selected_date
        - datetime.timedelta(days=1)
    )

    lag_7_date = (
        selected_date
        - datetime.timedelta(days=7)
    )

    input_lag_1 = get_passenger_volume(
        database_path=str(DATABASE_PATH),
        database_modified_time=DATABASE_PATH.stat().st_mtime,
        database_station=database_station,
        target_date_text=lag_1_date.strftime("%Y-%m-%d"),
    )

    input_lag_7 = get_passenger_volume(
        database_path=str(DATABASE_PATH),
        database_modified_time=DATABASE_PATH.stat().st_mtime,
        database_station=database_station,
        target_date_text=lag_7_date.strftime("%Y-%m-%d"),
    )

    col_lag1, col_lag7 = st.columns(2)

    with col_lag1:
        st.markdown(
            f"""
<div class="auto-data-box">
    <div class="auto-data-label">Lag-1 (H-1)</div>
    <div class="auto-data-value">{format_integer_indonesia(input_lag_1)}</div>
    <div class="auto-data-date">{lag_1_date.strftime("%d/%m/%Y")}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    with col_lag7:
        st.markdown(
            f"""
<div class="auto-data-box">
    <div class="auto-data-label">Lag-7 (H-7)</div>
    <div class="auto-data-value">{format_integer_indonesia(input_lag_7)}</div>
    <div class="auto-data-date">{lag_7_date.strftime("%d/%m/%Y")}</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.caption(
        "Nilai H-1 dan H-7 diambil otomatis dari database."
    )

    st.markdown("<br><br>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="nav-item-active">
    <span class="material-symbols-outlined">grid_view</span>
    Overview
</div>
<div class="nav-item">
    <span class="material-symbols-outlined">history</span>
    Riwayat Data
</div>
<div class="nav-item">
    <span class="material-symbols-outlined">settings</span>
    Konfigurasi
</div>
<br><br><br>
<div class="nav-item" style="color: #64748B;">
    <span class="material-symbols-outlined">logout</span>
    Keluar Sesi
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 9. VALIDASI DATA LAG
# ------------------------------------------------------------------------------
missing_lag_labels = []

if input_lag_1 is None:
    missing_lag_labels.append(
        f"H-1 ({lag_1_date.strftime('%d/%m/%Y')})"
    )

if input_lag_7 is None:
    missing_lag_labels.append(
        f"H-7 ({lag_7_date.strftime('%d/%m/%Y')})"
    )

if missing_lag_labels:
    st.error(
        "Prediksi tidak dapat dilakukan karena data "
        + " dan ".join(missing_lag_labels)
        + f" untuk Stasiun {selected_station} tidak tersedia."
    )
    st.info(
        "Pilih tanggal lain yang memiliki data H-1 dan H-7 lengkap, "
        "atau perbarui database melalui `import_database.py`."
    )
    st.stop()


# ------------------------------------------------------------------------------
# 10. PREDIKSI OTOMATIS
# ------------------------------------------------------------------------------
input_dataframe = build_model_input(
    features=model_features,
    selected_station=selected_station,
    selected_date=selected_date,
    lag_1=input_lag_1,
    lag_7=input_lag_7,
)

try:
    predicted_volume = float(
        rf_model.predict(input_dataframe)[0]
    )
except Exception as error:
    st.error(
        "Model gagal melakukan prediksi. "
        f"Detail: {error}"
    )
    st.write("Fitur yang dikirim ke model:")
    st.dataframe(input_dataframe)
    st.stop()

predicted_volume = max(0.0, predicted_volume)

day_of_week = selected_date.weekday()
is_weekend = int(day_of_week >= 5)

delta_lag1 = predicted_volume - input_lag_1
delta_lag7 = predicted_volume - input_lag_7

percentage_lag1 = (
    (delta_lag1 / input_lag_1) * 100
    if input_lag_1 != 0
    else None
)

percentage_lag7 = (
    (delta_lag7 / input_lag_7) * 100
    if input_lag_7 != 0
    else None
)

station_history = load_station_history(
    database_path=str(DATABASE_PATH),
    database_modified_time=DATABASE_PATH.stat().st_mtime,
    database_station=database_station,
)

density_status = classify_density(
    predicted_volume,
    station_history,
)

recommendations = recommendation_items(
    density_status["label"],
    percentage_lag1,
)


# ------------------------------------------------------------------------------
# 11. HEADER DASHBOARD
# ------------------------------------------------------------------------------
col_sub, col_user = st.columns([3, 1])

with col_sub:
    st.caption(
        f"Lintas Green Line • KAI Commuter • "
        f"{selected_station} • {selected_date.strftime('%d %B %Y')}"
    )

with col_user:
    st.markdown(
        """
<div style="display: flex; align-items: center; justify-content: flex-end; gap: 16px;">
    <span class="material-symbols-outlined" style="color: #64748B; cursor: pointer;">
        notifications
    </span>
    <div style="display: flex; align-items: center; gap: 8px;">
        <div style="width: 32px; height: 32px; border-radius: 50%; background: #DCFCE7; color: #065F46; display: flex; align-items: center; justify-content: center; font-weight: 800;">
            AT
        </div>
        <span style="font-size: 13px; font-weight: 700; color: #1E293B;">
            Admin Transit
        </span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<h1 style="font-size: 24px; font-weight: 800; color: #0F172A; margin-top: 10px; margin-bottom: 4px;">
    Dashboard Prediksi Volume Penumpang Harian KRL Lintas Green Line
</h1>
<p style="font-size: 13px; color: #64748B; margin-bottom: 8px;">
    Prediksi diperbarui otomatis berdasarkan stasiun, tanggal, data H-1, dan data H-7 dari database.
</p>
""",
    unsafe_allow_html=True,
)

st.caption(
    f"Database: {total_database_rows:,} baris | "
    f"Periode: {minimum_data_date.strftime('%d/%m/%Y')}–"
    f"{maximum_data_date.strftime('%d/%m/%Y')}"
)


# ------------------------------------------------------------------------------
# 12. KARTU KPI
# ------------------------------------------------------------------------------
if percentage_lag1 is None:
    badge_class = "kpi-badge-neutral"
elif percentage_lag1 > 0:
    badge_class = "kpi-badge-positive"
elif percentage_lag1 < 0:
    badge_class = "kpi-badge-negative"
else:
    badge_class = "kpi-badge-neutral"

weekly_direction = (
    "naik"
    if delta_lag7 > 0
    else "turun"
    if delta_lag7 < 0
    else "tetap"
)

weekly_color = (
    "#16A34A"
    if delta_lag7 > 0
    else "#B91C1C"
    if delta_lag7 < 0
    else "#64748B"
)

status_hari_text = (
    "Weekend"
    if is_weekend
    else "Weekday"
)

status_hari_description = (
    "Hari akhir pekan (Sabtu–Minggu)"
    if is_weekend
    else "Hari kerja (Senin–Jumat)"
)

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
<div class="kpi-card-box kpi-border-green">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="kpi-icon-box">
            <span class="material-symbols-outlined">groups</span>
        </div>
        <div class="{badge_class}">
            {format_percentage(percentage_lag1)} vs H-1
        </div>
    </div>
    <div class="kpi-title">Estimasi Volume Total</div>
    <div class="kpi-value-main">{format_integer_indonesia(predicted_volume)}</div>
    <div class="kpi-sub-text">Penumpang pada tanggal terpilih</div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""
<div class="kpi-card-box">
    <div class="kpi-icon-box">
        <span class="material-symbols-outlined">event_repeat</span>
    </div>
    <div class="kpi-title">Selisih Mingguan (H-7)</div>
    <div class="kpi-value-main">{format_signed_integer_indonesia(delta_lag7)}</div>
    <div class="kpi-sub-text" style="color: {weekly_color}; font-weight: 600;">
        Tren {weekly_direction} {format_absolute_percentage(percentage_lag7)}
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""
<div class="kpi-card-box">
    <div class="kpi-icon-box">
        <span class="material-symbols-outlined">calendar_today</span>
    </div>
    <div class="kpi-title">Karakteristik Hari</div>
    <div class="kpi-value-main" style="font-size: 22px;">{status_hari_text}</div>
    <div class="kpi-sub-text">{status_hari_description}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with kpi4:
    st.markdown(
        f"""
<div class="kpi-card-box {density_status['border_class']}">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div class="kpi-icon-box" style="color: {density_status['color']};">
            <span class="material-symbols-outlined">warning</span>
        </div>
        <div style="width: 8px; height: 8px; background: {density_status['color']}; border-radius: 50%;"></div>
    </div>
    <div class="kpi-title">Status Kepadatan</div>
    <div class="kpi-value-main" style="color: {density_status['color']}; font-size: 20px;">
        {density_status['label']}
    </div>
    <div class="kpi-sub-text">{density_status['description']}</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 13. GRAFIK RIWAYAT DAN PREDIKSI
# ------------------------------------------------------------------------------
col_chart, col_right = st.columns([2.2, 1])

with col_chart:
    st.markdown(
        """
<div style="background: #FFFFFF; border-radius: 20px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.03);">
    <div>
        <h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #0F172A;">
            Riwayat 7 Hari dan Estimasi Tanggal Terpilih
        </h3>
        <p style="margin: 0; font-size: 12px; color: #64748B;">
            Data aktual dari database dan satu titik hasil prediksi Random Forest
        </p>
    </div>
""",
        unsafe_allow_html=True,
    )

    history_start_date = (
        pd.Timestamp(selected_date)
        - pd.Timedelta(days=7)
    )

    history_end_date = pd.Timestamp(selected_date)

    recent_history = station_history[
        (station_history["tanggal"] >= history_start_date)
        & (station_history["tanggal"] <= history_end_date)
    ].copy()

    figure = go.Figure()

    if not recent_history.empty:
        figure.add_trace(
            go.Scatter(
                x=recent_history["tanggal"],
                y=recent_history["volume_penumpang"],
                mode="lines+markers",
                name="Aktual",
                line=dict(
                    color="#64748B",
                    width=2,
                ),
                marker=dict(
                    size=6,
                    color="#64748B",
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[pd.Timestamp(selected_date)],
            y=[predicted_volume],
            mode="markers",
            name="Prediksi",
            marker=dict(
                size=12,
                color="#006E2A",
                line=dict(
                    width=2,
                    color="white",
                ),
            ),
        )
    )

    figure.update_layout(
        xaxis=dict(
            showgrid=False,
            tickfont=dict(
                size=11,
                color="#64748B",
            ),
            tickformat="%d/%m",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(
                size=11,
                color="#64748B",
            ),
            tickformat=",",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=10,
        ),
        height=280,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    st.plotly_chart(
        figure,
        use_container_width=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    recommendation_list_html = "".join(
        f"<li style='margin-bottom: 8px;'>{item}</li>"
        for item in recommendations
    )

    st.markdown(
        f"""
<div class="recom-card">
    <div style="display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 14px; color: #0F172A; margin-bottom: 12px;">
        <span class="material-symbols-outlined" style="color: #0F172A;">lightbulb</span>
        Rekomendasi Operasional
    </div>
    <ul style="margin: 0; padding-left: 18px; font-size: 12px; color: #334155; line-height: 1.6;">
        {recommendation_list_html}
    </ul>
</div>
<div class="perf-card">
    <div style="width: 38px; height: 38px; border-radius: 12px; background: #DCFCE7; color: #16A34A; display: flex; align-items: center; justify-content: center;">
        <span class="material-symbols-outlined">database</span>
    </div>
    <div>
        <p style="margin: 0; font-size: 11px; color: #64748B; font-weight: 500;">
            Sumber Lag
        </p>
        <h4 style="margin: 0; font-size: 14px; font-weight: 800; color: #0F172A;">
            SQLite Database
        </h4>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 14. DETAIL INPUT MODEL
# ------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Lihat detail input prediksi"):
    detail_data = pd.DataFrame(
        {
            "Parameter": [
                "Stasiun",
                "Tanggal prediksi",
                "Tanggal H-1",
                "Volume H-1",
                "Tanggal H-7",
                "Volume H-7",
                "Hari ke-",
                "Bulan",
                "Weekend",
                "Hasil prediksi",
            ],
            "Nilai": [
                selected_station,
                selected_date.strftime("%d/%m/%Y"),
                lag_1_date.strftime("%d/%m/%Y"),
                format_integer_indonesia(input_lag_1),
                lag_7_date.strftime("%d/%m/%Y"),
                format_integer_indonesia(input_lag_7),
                day_of_week,
                selected_date.month,
                "Ya" if is_weekend else "Tidak",
                format_integer_indonesia(predicted_volume),
            ],
        }
    )

    st.dataframe(
        detail_data,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Data H-1 dan H-7 tidak dapat diubah manual karena diambil langsung "
        "dari tabel `passenger_daily`."
    )


# ------------------------------------------------------------------------------
# 15. RINGKASAN AKURASI MODEL
# ------------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)

col_title, col_badge = st.columns([3, 1])

with col_title:
    st.markdown(
        "### 📊 Ringkasan Akurasi & Performa Model "
        "(Tabel 4.10 Skripsi)"
    )
    st.caption(
        "Evaluasi performa model Machine Learning terhadap testing set"
    )

with col_badge:
    st.success("✅ **R-Squared: 93,25%**")

evaluation_data = {
    "Metrik Evaluasi": [
        "Mean Absolute Error (MAE)",
        "Root Mean Squared Error (RMSE)",
        "Koefisien Determinasi (R²)",
    ],
    "Data Latih (Training Set)": [
        "1.338,55 Penumpang",
        "2.924,40 Penumpang",
        "0,9777 (97,77%)",
    ],
    "Data Uji (Testing Set)": [
        "2.565,25 Penumpang",
        "5.050,89 Penumpang",
        "0,9325 (93,25%)",
    ],
    "Interpretasi Performa": [
        "Kesalahan rata-rata harian relatif rendah",
        "Sensitivitas terhadap lonjakan ekstrem memadai",
        "Model mampu menjelaskan 93,25% variansi data",
    ],
}

evaluation_dataframe = pd.DataFrame(evaluation_data)

st.table(evaluation_dataframe)

