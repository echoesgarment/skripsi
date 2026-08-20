# ==============================================================================
# FILE: app.py
# Dashboard Prediksi Volume Penumpang KRL Commuter Line
# ==============================================================================

import datetime
import hmac
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

# Jika database belum ada, tentukan lokasi pembuatan database baru.
if DATABASE_PATH is None:
    DEFAULT_DATASET_DIR = (APP_DIR.parent / "DATASET").resolve()
    DEFAULT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    DATABASE_PATH = DEFAULT_DATASET_DIR / "greenline.db"

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


MODEL_LOAD_ERROR = None

if MODEL_PATH is None:
    MODEL_LOAD_ERROR = (
        "File `model_rf_greenline.joblib` tidak ditemukan.\n\n"
        "Lokasi yang diperiksa:\n"
        f"{format_checked_paths(model_checked_paths)}"
    )


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

/* ================================================================
   WARNA TEKS AREA UTAMA
   - Judul di luar card menggunakan putih murni.
   - Deskripsi dan label menggunakan putih lembut.
   - Caption menggunakan putih transparan agar hierarki visual lebih jelas.
   - Tautan menggunakan biru sangat muda yang selaras dengan gradasi.
   Seluruh teks di dalam card putih dikembalikan menjadi warna navy-slate
   melalui selector card yang lebih spesifik pada bagian berikutnya.
   ================================================================ */
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4,
[data-testid="stMain"] h5,
[data-testid="stMain"] h6 {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: 0 1px 3px rgba(15, 23, 42, 0.45) !important;
}

[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: 0 1px 2px rgba(15, 23, 42, 0.35) !important;
}

[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
[data-testid="stMain"] small {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: 0 1px 2px rgba(15, 23, 42, 0.35) !important;
}

[data-testid="stMain"] a {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    font-weight: 600 !important;
    text-shadow: 0 1px 2px rgba(15, 23, 42, 0.35) !important;
}

.stApp {
    background: linear-gradient(
        135deg,
        #1B4EF5 0%,
        #3874FF 32%,
        #5996FF 66%,
        #F4CEFF 100%
    ) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #1B4EF5 0%,
        #3874FF 36%,
        #5996FF 70%,
        #F4CEFF 100%
    ) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.45) !important;
    padding-top: 1rem !important;
}

/* Semua teks sidebar dibuat hitam */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #000000 !important;
}

.stButton > button {
    background: linear-gradient(90deg, #1B4EF5 0%, #3874FF 55%, #5996FF 100%) !important;
    color: #000000 !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    height: 42px !important;
    border: 1px solid rgba(255, 255, 255, 0.35) !important;
    box-shadow: 0 4px 12px rgba(27, 78, 245, 0.28) !important;
    width: 100% !important;
}

.stButton > button:hover {
    filter: brightness(1.06) !important;
    border-color: rgba(255, 255, 255, 0.65) !important;
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

/* Navigasi halaman pada sidebar */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 6px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border: 0;
    border-radius: 12px;
    padding: 10px 12px;
    margin: 0;
    transition: background-color 0.15s ease;
}

[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.20);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: rgba(255, 255, 255, 0.92);
    color: #1B4EF5 !important;
    font-weight: 700;
    box-shadow: 0 5px 14px rgba(27, 78, 245, 0.18);
}

[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) div {
    color: #1B4EF5 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none;
}

.page-header-card {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 18px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.03);
}

/* Header utama khusus halaman Dashboard */
.dashboard-header-card {
    background: #FFFFFF;
    border: 1px solid rgba(255, 255, 255, 0.75);
    border-radius: 20px;
    padding: 22px 24px;
    margin-bottom: 18px;
    box-shadow: 0 12px 30px rgba(15, 23, 42, 0.14);
}

.dashboard-header-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 16px;
}

.dashboard-header-kicker {
    color: #2563EB !important;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.dashboard-admin-box {
    display: flex;
    align-items: center;
    gap: 10px;
}

.dashboard-admin-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: #DBEAFE;
    color: #1D4ED8 !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
}

.dashboard-admin-name {
    color: #1E293B !important;
    font-size: 13px;
    font-weight: 700;
}

.dashboard-header-title {
    color: #0F172A !important;
    font-size: 25px;
    line-height: 1.25;
    font-weight: 800;
    margin: 0 0 7px 0;
}

.dashboard-header-description {
    color: #64748B !important;
    font-size: 13px;
    line-height: 1.6;
    margin: 0;
}

.dashboard-database-info {
    color: #475569 !important;
    font-size: 11px;
    font-weight: 600;
    margin-top: 13px;
    padding-top: 12px;
    border-top: 1px solid #E2E8F0;
}

/* Semua teks di dalam card putih harus gelap dan tanpa text-shadow */
.page-header-card,
.page-header-card *,
.dashboard-header-card,
.dashboard-header-card *,
.kpi-card-box,
.kpi-card-box *,
.auto-data-box,
.auto-data-box *,
.recom-card,
.recom-card *,
.evaluation-card,
.evaluation-card *,
.model-conclusion-card,
.model-conclusion-card * {
    color: inherit;
    text-shadow: none !important;
}

.page-header-card h1,
.page-header-card h2,
.page-header-card h3,
.page-header-card p {
    color: #0F172A !important;
}

.page-header-card p {
    color: #64748B !important;
}

/* Card formulir login dibuat putih dan seluruh teksnya berwarna hitam */
[data-testid="stForm"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 24px !important;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.12) !important;
}

[data-testid="stForm"] label,
[data-testid="stForm"] p,
[data-testid="stForm"] span,
[data-testid="stForm"] div {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[data-testid="stForm"] input {
    background: #FFFFFF !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

[data-testid="stForm"] input::placeholder {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    opacity: 1 !important;
}

/* Tombol submit login tetap biru dengan teks hitam */
[data-testid="stForm"] button,
[data-testid="stForm"] button * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Card container Streamlit dibuat putih solid, termasuk Seleksi Data Prediksi */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    opacity: 1 !important;
    overflow: hidden !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
    isolation: isolate !important;
}

/* Lapisan internal container juga dipaksa putih agar tidak mengikuti gradasi */
div[data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    opacity: 1 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3,
div[data-testid="stVerticalBlockBorderWrapper"] h4,
div[data-testid="stVerticalBlockBorderWrapper"] label {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    text-shadow: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] p {
    color: #334155 !important;
    -webkit-text-fill-color: #334155 !important;
    text-shadow: none !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"],
div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stCaptionContainer"] * {
    color: #64748B !important;
    -webkit-text-fill-color: #64748B !important;
    text-shadow: none !important;
}

/* Dropdown/selectbox putih */
div[data-baseweb="select"] > div {
    background: #FFFFFF !important;
    border-color: #CBD5E1 !important;
    color: #0F172A !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06) !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] svg {
    color: #0F172A !important;
    fill: #475569 !important;
}

/* Kalender/date input putih */
[data-testid="stDateInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: #FFFFFF !important;
    color: #0F172A !important;
    border-color: #CBD5E1 !important;
}

[data-testid="stDateInput"] button,
[data-testid="stTextInput"] button,
[data-testid="stNumberInput"] button {
    color: #475569 !important;
}

/* Daftar pilihan yang terbuka */
div[role="listbox"],
ul[role="listbox"] {
    background: #FFFFFF !important;
}

div[role="option"],
div[role="option"] *,
li[role="option"],
li[role="option"] * {
    color: #0F172A !important;
}

/* Card metrik ringkasan prediksi semua stasiun */
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 16px !important;
    padding: 16px 18px !important;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.10) !important;
    min-height: 128px;
}

[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricValue"],
[data-testid="stMetric"] [data-testid="stMetricDelta"],
[data-testid="stMetric"] div,
[data-testid="stMetric"] span {
    color: #0F172A !important;
}

/* Tabel/dataframe tetap nyaman dibaca */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    background: #FFFFFF !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* Notifikasi sistem mempertahankan kontras teks */
[data-testid="stAlert"] p,
[data-testid="stAlert"] div,
[data-testid="stAlert"] span {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Perbaikan kontras pada teks proses/progress dan status.
   Caption tidak dipaksa gelap secara global karena caption di luar card
   harus tetap putih lembut. */
[data-testid="stProgress"] *,
[data-testid="stStatusWidget"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    text-shadow: none !important;
}

/* Pastikan seluruh isi metric card tampil hitam, termasuk label, nilai, dan delta */
[data-testid="stMetric"],
[data-testid="stMetric"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

[data-testid="stMetric"] svg,
[data-testid="stMetric"] svg * {
    fill: #0F172A !important;
    color: #0F172A !important;
}

/* Teks tabel HTML dibuat hitam */
[data-testid="stTable"],
[data-testid="stTable"] *,
[data-testid="stDataFrame"] {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

[data-testid="stTable"] th,
[data-testid="stTable"] td {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

/* Teks tombol dibuat hitam */
[data-testid="stMain"] .stButton button,
[data-testid="stMain"] .stButton button * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
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
    border-left: 5px solid #EAB308 !important;
}

.kpi-border-yellow {
    border-left: 5px solid #EAB308 !important;
}

.kpi-border-light-orange {
    border-left: 5px solid #FDBA74 !important;
}

.kpi-border-orange {
    border-left: 5px solid #F97316 !important;
}

.kpi-border-red {
    border-left: 5px solid #B91C1C !important;
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
    background-color: #FFEDD5;
    color: #C2410C;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
}

.kpi-badge-negative {
    background-color: #FFF7CC;
    color: #A16207;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 8px;
    border-radius: 20px;
}

.kpi-badge-neutral {
    background-color: #FEF3C7;
    color: #92400E;
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
    color: #64748B;
}

.recom-card {
    background: #FFF7ED;
    border-radius: 20px;
    padding: 24px;
    border: 1px solid #FED7AA;
}

.dashboard-recom-card {
    height: 390px;
    min-height: 390px;
    margin-bottom: 0;
    display: flex;
    flex-direction: column;
    box-sizing: border-box;
    overflow: hidden;
}

.dashboard-recom-card ul {
    margin-bottom: 0;
    overflow-y: auto;
    padding-right: 6px;
}

.evaluation-card {
    border-radius: 18px;
    padding: 22px;
    min-height: 190px;
    border: 1px solid;
    box-shadow: 0 3px 12px rgba(15, 23, 42, 0.04);
    box-sizing: border-box;
}

.evaluation-card-mae {
    background: #EFF6FF;
    border-color: #BFDBFE;
    border-left: 6px solid #2563EB;
}

.evaluation-card-rmse {
    background: #FFF7ED;
    border-color: #FED7AA;
    border-left: 6px solid #EA580C;
}

.evaluation-card-r2 {
    background: #ECFDF5;
    border-color: #A7F3D0;
    border-left: 6px solid #059669;
}

.evaluation-icon {
    width: 42px;
    height: 42px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
}

.evaluation-icon-mae {
    background: #DBEAFE;
    color: #1D4ED8;
}

.evaluation-icon-rmse {
    background: #FFEDD5;
    color: #C2410C;
}

.evaluation-icon-r2 {
    background: #D1FAE5;
    color: #047857;
}

.evaluation-label {
    font-size: 12px;
    font-weight: 700;
    color: #475569;
    margin-bottom: 6px;
}

.evaluation-value {
    font-size: 30px;
    line-height: 1.1;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 8px;
}

.evaluation-description {
    font-size: 12px;
    line-height: 1.5;
    color: #64748B;
}


.model-conclusion-card {
    background: #F8FAFC;
    border: 1px solid #CBD5E1;
    border-left: 6px solid #0F766E;
    border-radius: 18px;
    padding: 20px 22px;
    margin: 18px 0;
}

.model-conclusion-title {
    font-size: 15px;
    font-weight: 800;
    color: #0F172A;
    margin-bottom: 8px;
}

.model-conclusion-text {
    font-size: 13px;
    line-height: 1.7;
    color: #475569;
}

.auto-data-source {
    display: inline-block;
    margin-top: 6px;
    padding: 2px 7px;
    border-radius: 999px;
    background: #E2E8F0;
    color: #475569;
    font-size: 9px;
    font-weight: 700;
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
    color: #64748B;
    font-size: 10px;
    margin-top: 2px;
}

/* ================================================================
   CARD PUTIH UNTUK LOADING, CHART, EXPANDER, TABEL, DAN IMPORT EXCEL
   ================================================================ */

/* Loading/progress dibuat sebagai card putih solid */
div[data-testid="stProgress"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 14px 16px !important;
    margin: 6px 0 14px 0 !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    opacity: 1 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

div[data-testid="stProgress"] p,
div[data-testid="stProgress"] span,
div[data-testid="stProgress"] div {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    text-shadow: none !important;
}

/* Chart Plotly memiliki card dan kanvas putih, tidak transparan */
div[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    padding: 14px 14px 10px 14px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    overflow: visible !important;
    opacity: 1 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

div[data-testid="stPlotlyChart"] > div,
div[data-testid="stPlotlyChart"] .js-plotly-plot,
div[data-testid="stPlotlyChart"] .plot-container,
div[data-testid="stPlotlyChart"] .svg-container {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
}

/* Expander digunakan pada detail input prediksi dan ketentuan Excel */
div[data-testid="stExpander"],
div[data-testid="stExpander"] details,
div[data-testid="stExpanderDetails"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border-color: #E2E8F0 !important;
    border-radius: 18px !important;
    opacity: 1 !important;
    backdrop-filter: none !important;
    -webkit-backdrop-filter: none !important;
}

div[data-testid="stExpander"] {
    border: 1px solid #E2E8F0 !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    overflow: hidden !important;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary *,
div[data-testid="stExpander"] p,
div[data-testid="stExpander"] label,
div[data-testid="stExpander"] span,
div[data-testid="stExpanderDetails"],
div[data-testid="stExpanderDetails"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
    text-shadow: none !important;
}

/* Hilangkan text-shadow pada seluruh komponen putih agar tampil bersih */
div[data-testid="stVerticalBlockBorderWrapper"] *,
div[data-testid="stProgress"] *,
div[data-testid="stPlotlyChart"] *,
div[data-testid="stExpander"] *,
div[data-testid="stDataFrame"] *,
div[data-testid="stTable"] *,
div[data-testid="stFileUploader"] *,
[data-testid="stMetric"] *,
[data-testid="stAlert"] * {
    text-shadow: none !important;
}

/* Dataframe/tabel diberi card putih lengkap, bukan hanya area grid */
div[data-testid="stDataFrame"],
div[data-testid="stTable"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.12) !important;
    overflow: hidden !important;
    opacity: 1 !important;
}

/* Area upload Excel tetap putih dan teksnya gelap */
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] *,
div[data-testid="stFileUploaderDropzone"],
div[data-testid="stFileUploaderDropzone"] * {
    color: #0F172A !important;
    -webkit-text-fill-color: #0F172A !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background: #F8FAFC !important;
    background-color: #F8FAFC !important;
    border: 1px dashed #94A3B8 !important;
    border-radius: 14px !important;
}

/* Seluruh teks dalam bordered container/card Streamlit harus gelap */
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] * {
    color: #0F172A;
    -webkit-text-fill-color: #0F172A;
}

/* Tombol di dalam card menggunakan teks hitam */
div[data-testid="stVerticalBlockBorderWrapper"] .stButton button,
div[data-testid="stVerticalBlockBorderWrapper"] .stButton button *,
div[data-testid="stFileUploader"] button,
div[data-testid="stFileUploader"] button * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* ================================================================
   OVERRIDE FINAL: SELURUH FONT BERWARNA HITAM
   Aturan ini ditempatkan paling akhir agar menimpa warna teks sebelumnya,
   termasuk sidebar, tombol, form, card, caption, tabel, status, dan chart.
   ================================================================ */

/* Semua teks pada aplikasi */
.stApp,
.stApp * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: none !important;
}

/* Teks pada seluruh input dan placeholder */
.stApp input,
.stApp textarea,
.stApp input::placeholder,
.stApp textarea::placeholder {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    opacity: 1 !important;
}

/* Teks dan ikon pada dropdown */
.stApp div[data-baseweb="select"] span,
.stApp div[data-baseweb="select"] input,
.stApp div[role="option"],
.stApp div[role="option"] *,
.stApp li[role="option"],
.stApp li[role="option"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Seluruh teks SVG Plotly, termasuk sumbu, legenda, anotasi, dan heatmap */
.stApp div[data-testid="stPlotlyChart"] svg text,
.stApp div[data-testid="stPlotlyChart"] .plotly text,
.stApp div[data-testid="stPlotlyChart"] .xtick text,
.stApp div[data-testid="stPlotlyChart"] .ytick text,
.stApp div[data-testid="stPlotlyChart"] .legendtext,
.stApp div[data-testid="stPlotlyChart"] .annotation-text,
.stApp div[data-testid="stPlotlyChart"] .gtitle,
.stApp div[data-testid="stPlotlyChart"] .cbtitle,
.stApp div[data-testid="stPlotlyChart"] .cbaxis text {
    fill: #000000 !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}

/* Tooltip Plotly */
.stApp div[data-testid="stPlotlyChart"] .hovertext text,
.stApp div[data-testid="stPlotlyChart"] .hoverlayer text {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Ikon berbasis SVG juga dibuat hitam */
.stApp svg,
.stApp svg path {
    color: #000000 !important;
}

/* OVERRIDE KHUSUS: SEMUA TEKS DI DALAM CARDBOX / KOMPONEN PUTIH HARUS HITAM */
.page-header-card,
.page-header-card *,
.dashboard-header-card,
.dashboard-header-card *,
.kpi-card-box,
.kpi-card-box *,
.auto-data-box,
.auto-data-box *,
.recom-card,
.recom-card *,
.evaluation-card,
.evaluation-card *,
.model-conclusion-card,
.model-conclusion-card *,
[data-testid="stMetric"],
[data-testid="stMetric"] *,
[data-testid="stExpander"],
[data-testid="stExpander"] *,
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] *,
[data-testid="stTable"],
[data-testid="stTable"] *,
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stVerticalBlockBorderWrapper"] *,
div[data-testid="stFileUploader"],
div[data-testid="stFileUploader"] *,
[data-testid="stProgress"],
[data-testid="stProgress"] * {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
    text-shadow: none !important;
}

.page-header-card svg,
.page-header-card svg *,
.dashboard-header-card svg,
.dashboard-header-card svg *,
.kpi-card-box svg,
.kpi-card-box svg *,
.auto-data-box svg,
.auto-data-box svg *,
.recom-card svg,
.recom-card svg *,
.evaluation-card svg,
.evaluation-card svg *,
.model-conclusion-card svg,
.model-conclusion-card svg *,
[data-testid="stMetric"] svg,
[data-testid="stMetric"] svg *,
[data-testid="stExpander"] svg,
[data-testid="stExpander"] svg *,
[data-testid="stDataFrame"] svg,
[data-testid="stDataFrame"] svg *,
[data-testid="stTable"] svg,
[data-testid="stTable"] svg *,
div[data-testid="stVerticalBlockBorderWrapper"] svg,
div[data-testid="stVerticalBlockBorderWrapper"] svg * {
    fill: #000000 !important;
    color: #000000 !important;
}

/* Pastikan badge dan subtext pada KPI juga hitam */
.kpi-badge-positive,
.kpi-badge-negative,
.kpi-badge-neutral,
.kpi-title,
.kpi-value-main,
.kpi-sub-text,
.auto-data-label,
.auto-data-value,
.auto-data-date,
.auto-data-source,
.evaluation-label,
.evaluation-value,
.evaluation-description,
.model-conclusion-title,
.model-conclusion-text,
.dashboard-header-title,
.dashboard-header-description,
.dashboard-database-info,
.dashboard-admin-name,
.dashboard-header-kicker {
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
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


rf_model = None
model_features = None
model_stations = list(STATION_LABEL_MAPPING.keys())

if MODEL_PATH is not None:
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
        MODEL_LOAD_ERROR = f"Gagal memuat aset model: {error}"


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


# ------------------------------------------------------------------------------
# 7. IMPORT DATASET EXCEL MELALUI WEBSITE
# ------------------------------------------------------------------------------
UPLOAD_REQUIRED_COLUMNS = {
    "tanggal",
    "nama_stasiun",
    "penumpang_berangkat_komuter",
    "penumpang_datang_komuter",
}

COLUMN_ALIASES = {
    "tanggal": {
        "tanggal",
        "tgl",
        "date",
        "tanggal_data",
    },
    "nama_stasiun": {
        "nama_stasiun",
        "stasiun",
        "station",
        "nama_station",
    },
    "penumpang_berangkat_komuter": {
        "penumpang_berangkat_komuter",
        "penumpang_berangkat",
        "berangkat",
        "tap_in",
        "tapin",
    },
    "penumpang_datang_komuter": {
        "penumpang_datang_komuter",
        "penumpang_datang",
        "datang",
        "tap_out",
        "tapout",
    },
}


def normalize_column_name(value):
    """Menormalkan nama kolom Excel menjadi huruf kecil dan underscore."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    )

    return normalized.strip("_")


def resolve_uploaded_columns(dataframe):
    """Mencocokkan variasi nama kolom Excel dengan kolom wajib sistem."""
    normalized_to_original = {
        normalize_column_name(column): column
        for column in dataframe.columns
    }

    rename_mapping = {}
    missing_columns = []

    for canonical_column, aliases in COLUMN_ALIASES.items():
        matched_original = None

        for alias in aliases:
            if alias in normalized_to_original:
                matched_original = normalized_to_original[alias]
                break

        if matched_original is None:
            missing_columns.append(canonical_column)
        else:
            rename_mapping[matched_original] = canonical_column

    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: "
            + ", ".join(sorted(missing_columns))
            + ". Kolom Excel yang diterima: tanggal, nama_stasiun, "
            "penumpang_berangkat_komuter, dan "
            "penumpang_datang_komuter."
        )

    renamed_dataframe = dataframe.rename(columns=rename_mapping)

    duplicated_columns = renamed_dataframe.columns[
        renamed_dataframe.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Terdapat kolom ganda setelah normalisasi: "
            + ", ".join(map(str, duplicated_columns))
        )

    return renamed_dataframe


def parse_passenger_value(value):
    """Mengubah angka Excel atau angka bertanda pemisah ribuan menjadi integer."""
    if pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return None

        return int(round(float(value)))

    text = str(value).strip()

    if not text:
        return None

    if re.fullmatch(r"-?\d+(?:[.,]0+)?", text):
        return int(round(float(text.replace(",", "."))))

    compact_text = re.sub(r"[^\d-]", "", text)

    if not compact_text or compact_text == "-":
        return None

    return int(compact_text)


def parse_uploaded_dates(series):
    """Mengubah tanggal teks, datetime, atau serial Excel menjadi datetime."""
    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]",
    )

    numeric_series = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric_mask = (
        numeric_series.notna()
        & ~series.map(
            lambda value: isinstance(
                value,
                (datetime.date, datetime.datetime, pd.Timestamp),
            )
        )
    )

    if numeric_mask.any():
        result.loc[numeric_mask] = pd.to_datetime(
            numeric_series.loc[numeric_mask],
            unit="D",
            origin="1899-12-30",
            errors="coerce",
        )

    non_numeric_mask = ~numeric_mask

    if non_numeric_mask.any():
        result.loc[non_numeric_mask] = pd.to_datetime(
            series.loc[non_numeric_mask],
            dayfirst=True,
            errors="coerce",
        )

    return result


def clean_uploaded_dataset(raw_dataframe):
    """Validasi dan membersihkan dataset sebelum disimpan ke SQLite."""
    if raw_dataframe is None or raw_dataframe.empty:
        raise ValueError("File Excel tidak memiliki data.")

    dataframe = resolve_uploaded_columns(raw_dataframe.copy())

    dataframe = dataframe[
        [
            "tanggal",
            "nama_stasiun",
            "penumpang_berangkat_komuter",
            "penumpang_datang_komuter",
        ]
    ].copy()

    dataframe.insert(
        0,
        "baris_excel",
        range(2, len(dataframe) + 2),
    )

    dataframe["tanggal"] = parse_uploaded_dates(
        dataframe["tanggal"]
    )

    dataframe["nama_stasiun_asli"] = (
        dataframe["nama_stasiun"]
        .astype(str)
        .str.strip()
    )

    dataframe["nama_stasiun"] = (
        dataframe["nama_stasiun_asli"]
        .map(canonical_station_name)
    )

    for column in [
        "penumpang_berangkat_komuter",
        "penumpang_datang_komuter",
    ]:
        dataframe[column] = dataframe[column].map(
            parse_passenger_value
        )

    invalid_reasons = pd.Series(
        "",
        index=dataframe.index,
        dtype="object",
    )

    invalid_reasons = invalid_reasons.mask(
        dataframe["tanggal"].isna(),
        invalid_reasons + "Tanggal tidak valid; ",
    )

    invalid_reasons = invalid_reasons.mask(
        ~dataframe["nama_stasiun"].isin(STATION_LABEL_MAPPING),
        invalid_reasons + "Nama stasiun tidak dikenal; ",
    )

    for column, label in [
        (
            "penumpang_berangkat_komuter",
            "Penumpang berangkat tidak valid",
        ),
        (
            "penumpang_datang_komuter",
            "Penumpang datang tidak valid",
        ),
    ]:
        invalid_reasons = invalid_reasons.mask(
            dataframe[column].isna(),
            invalid_reasons + f"{label}; ",
        )

        invalid_reasons = invalid_reasons.mask(
            dataframe[column].fillna(0) < 0,
            invalid_reasons + f"{label} bernilai negatif; ",
        )

    dataframe["alasan_tidak_valid"] = (
        invalid_reasons
        .str.strip()
        .str.rstrip(";")
    )

    invalid_dataframe = dataframe[
        dataframe["alasan_tidak_valid"] != ""
    ].copy()

    if not invalid_dataframe.empty:
        return None, invalid_dataframe, {
            "raw_rows": len(raw_dataframe),
            "valid_rows": 0,
            "invalid_rows": len(invalid_dataframe),
            "duplicate_rows": 0,
        }

    dataframe["penumpang_berangkat_komuter"] = (
        dataframe["penumpang_berangkat_komuter"]
        .astype(int)
    )

    dataframe["penumpang_datang_komuter"] = (
        dataframe["penumpang_datang_komuter"]
        .astype(int)
    )

    dataframe["volume_penumpang"] = (
        dataframe["penumpang_berangkat_komuter"]
        + dataframe["penumpang_datang_komuter"]
    )

    dataframe["tanggal"] = (
        dataframe["tanggal"]
        .dt.strftime("%Y-%m-%d")
    )

    duplicate_mask = dataframe.duplicated(
        subset=["nama_stasiun", "tanggal"],
        keep="last",
    )

    duplicate_rows = int(duplicate_mask.sum())

    cleaned_dataframe = (
        dataframe.loc[
            ~duplicate_mask,
            [
                "tanggal",
                "nama_stasiun",
                "penumpang_berangkat_komuter",
                "penumpang_datang_komuter",
                "volume_penumpang",
            ],
        ]
        .sort_values(["nama_stasiun", "tanggal"])
        .reset_index(drop=True)
    )

    summary = {
        "raw_rows": len(raw_dataframe),
        "valid_rows": len(cleaned_dataframe),
        "invalid_rows": 0,
        "duplicate_rows": duplicate_rows,
        "minimum_date": cleaned_dataframe["tanggal"].min(),
        "maximum_date": cleaned_dataframe["tanggal"].max(),
        "station_count": cleaned_dataframe["nama_stasiun"].nunique(),
    }

    return cleaned_dataframe, invalid_dataframe, summary


def ensure_passenger_table(connection):
    """Membuat atau memvalidasi tabel passenger_daily."""
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS passenger_daily (
            tanggal TEXT NOT NULL,
            nama_stasiun TEXT NOT NULL,
            penumpang_berangkat_komuter INTEGER NOT NULL,
            penumpang_datang_komuter INTEGER NOT NULL,
            volume_penumpang INTEGER NOT NULL
        )
        """
    )

    table_columns = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(passenger_daily)"
        ).fetchall()
    }

    required_columns = {
        "tanggal",
        "nama_stasiun",
        "penumpang_berangkat_komuter",
        "penumpang_datang_komuter",
        "volume_penumpang",
    }

    missing_columns = required_columns.difference(table_columns)

    if missing_columns:
        raise ValueError(
            "Struktur tabel passenger_daily tidak lengkap. Kolom hilang: "
            + ", ".join(sorted(missing_columns))
        )

    connection.execute(
        """
        DELETE FROM passenger_daily
        WHERE rowid NOT IN (
            SELECT MAX(rowid)
            FROM passenger_daily
            GROUP BY nama_stasiun, tanggal
        )
        """
    )

    connection.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_passenger_station_date
        ON passenger_daily (nama_stasiun, tanggal)
        """
    )


def create_database_backup(database_path):
    """Membuat backup database sebelum import."""
    database_path = Path(database_path)

    if not database_path.exists():
        return None

    backup_directory = (
        database_path.parent
        / "BACKUP"
    )

    backup_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        backup_directory
        / f"greenline_backup_{timestamp}.db"
    )

    shutil.copy2(
        database_path,
        backup_path,
    )

    return backup_path


def save_uploaded_dataset(
    dataframe,
    database_path,
):
    """Menambahkan data baru dan memperbarui data lama menggunakan upsert."""
    database_path = Path(database_path)
    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        (
            row.tanggal,
            row.nama_stasiun,
            int(row.penumpang_berangkat_komuter),
            int(row.penumpang_datang_komuter),
            int(row.volume_penumpang),
        )
        for row in dataframe.itertuples(index=False)
    ]

    with sqlite3.connect(database_path) as connection:
        ensure_passenger_table(connection)

        existing_keys = {
            (row[0], row[1])
            for row in connection.execute(
                """
                SELECT nama_stasiun, tanggal
                FROM passenger_daily
                """
            ).fetchall()
        }

        uploaded_keys = {
            (row[1], row[0])
            for row in rows
        }

        updated_count = len(
            uploaded_keys.intersection(existing_keys)
        )

        inserted_count = len(
            uploaded_keys.difference(existing_keys)
        )

        connection.executemany(
            """
            INSERT INTO passenger_daily (
                tanggal,
                nama_stasiun,
                penumpang_berangkat_komuter,
                penumpang_datang_komuter,
                volume_penumpang
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(nama_stasiun, tanggal)
            DO UPDATE SET
                penumpang_berangkat_komuter =
                    excluded.penumpang_berangkat_komuter,
                penumpang_datang_komuter =
                    excluded.penumpang_datang_komuter,
                volume_penumpang =
                    excluded.volume_penumpang
            """,
            rows,
        )

        connection.commit()

        total_rows = connection.execute(
            "SELECT COUNT(*) FROM passenger_daily"
        ).fetchone()[0]

    return {
        "inserted_count": inserted_count,
        "updated_count": updated_count,
        "total_rows": int(total_rows),
    }


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



def render_excel_importer():
    """
    Import dataset sebagai TESTING DATASET.
    Dataset hanya diproses sementara:
    upload -> preprocessing -> feature engineering -> prediksi -> evaluasi.
    Tidak menyimpan data ke greenline.db.
    """

    st.markdown("### Import Dataset Testing Model")

    st.caption(
        "Upload dataset baru untuk menguji performa model tanpa mengubah "
        "database utama."
    )

    st.info(
        "Dataset yang diupload hanya digunakan pada sesi ini. "
        "Tidak ada INSERT, UPDATE, atau perubahan pada greenline.db."
    )

    uploaded_file = st.file_uploader(
        "Pilih file Excel Testing",
        type=["xlsx", "xls"],
        accept_multiple_files=False,
        key="greenline_testing_uploader",
        help=(
            "Kolom wajib: tanggal, nama_stasiun, "
            "penumpang_berangkat_komuter, penumpang_datang_komuter."
        ),
    )

    if uploaded_file is None:
        st.info("Upload file Excel untuk menjalankan preprocessing dan evaluasi.")
        return

    try:
        raw_dataframe = pd.read_excel(uploaded_file, sheet_name=0)

        (
            cleaned_dataframe,
            invalid_dataframe,
            summary,
        ) = clean_uploaded_dataset(raw_dataframe)

    except Exception as error:
        st.error(f"Preprocessing gagal: {error}")
        return

    if not invalid_dataframe.empty:
        st.error("Dataset memiliki data tidak valid.")
        st.dataframe(invalid_dataframe.head(100), use_container_width=True)
        return

    # Format tanggal hasil import agar hanya menampilkan tanggal tanpa jam 00:00:00
    # Data tetap dapat dikonversi kembali ke datetime saat proses prediksi.
    cleaned_dataframe["tanggal"] = pd.to_datetime(
        cleaned_dataframe["tanggal"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    st.success("Preprocessing berhasil.")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Data", f"{summary['valid_rows']:,}")
    c2.metric("Jumlah Stasiun", summary["station_count"])
    c3.metric("Tanggal Awal", summary["minimum_date"])
    c4.metric("Tanggal Akhir", summary["maximum_date"])

    with st.expander("Preview Dataset Testing"):
        st.dataframe(
            cleaned_dataframe.head(100),
            use_container_width=True,
            hide_index=True,
        )

    if rf_model is None:
        st.error("Model belum tersedia.")
        return

    testing_dataframe = cleaned_dataframe.copy()

    testing_dataframe["tanggal"] = pd.to_datetime(
        testing_dataframe["tanggal"]
    )

    predictions = []
    actual_values = []

    progress = st.progress(0)

    total_rows = len(testing_dataframe)

    for index, row in testing_dataframe.iterrows():

        station = row["nama_stasiun"]
        date = row["tanggal"].date()

        history = testing_dataframe[
            (testing_dataframe["nama_stasiun"] == station)
            & (testing_dataframe["tanggal"] < row["tanggal"])
        ]

        lag_1_data = history[
            history["tanggal"] == row["tanggal"] - pd.Timedelta(days=1)
        ]

        lag_7_data = history[
            history["tanggal"] == row["tanggal"] - pd.Timedelta(days=7)
        ]

        if lag_1_data.empty or lag_7_data.empty:
            predictions.append(None)
            actual_values.append(row["volume_penumpang"])
            continue

        lag_1 = float(
            lag_1_data.iloc[-1]["volume_penumpang"]
        )

        lag_7 = float(
            lag_7_data.iloc[-1]["volume_penumpang"]
        )

        model_input = build_model_input(
            features=model_features,
            selected_station=station,
            selected_date=date,
            lag_1=lag_1,
            lag_7=lag_7,
        )

        prediction = float(
            rf_model.predict(model_input)[0]
        )

        predictions.append(max(0, prediction))
        actual_values.append(row["volume_penumpang"])

        progress.progress(
            min((index + 1) / total_rows, 1.0)
        )

    result_dataframe = testing_dataframe.copy()

    result_dataframe["actual"] = actual_values
    result_dataframe["prediction"] = predictions

    result_dataframe = result_dataframe.dropna(
        subset=["prediction"]
    )

    # Format tanggal hasil evaluasi agar tidak menampilkan jam 00:00:00
    result_dataframe["tanggal"] = pd.to_datetime(
        result_dataframe["tanggal"],
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")

    # Membulatkan nilai prediksi agar tabel lebih mudah dibaca
    result_dataframe["prediction"] = (
        result_dataframe["prediction"]
        .round(0)
        .astype(int)
    )

    if result_dataframe.empty:
        st.warning(
            "Tidak ada data yang memiliki lag H-1 dan H-7 lengkap "
            "untuk evaluasi."
        )
        return

    mae = mean_absolute_error(
        result_dataframe["actual"],
        result_dataframe["prediction"],
    )

    mse = mean_squared_error(
        result_dataframe["actual"],
        result_dataframe["prediction"],
    )
    rmse = mse ** 0.5

    r2 = r2_score(
        result_dataframe["actual"],
        result_dataframe["prediction"],
    )

    st.markdown("## Hasil Evaluasi Dataset Testing")

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE",
        f"{mae:,.2f}"
    )

    m2.metric(
        "RMSE",
        f"{rmse:,.2f}"
    )

    m3.metric(
        "R²",
        f"{r2:.4f}"
    )

    # Interpretasi otomatis hasil evaluasi model
    if r2 >= 0.90:
        model_quality = "sangat baik"
    elif r2 >= 0.75:
        model_quality = "baik"
    elif r2 >= 0.50:
        model_quality = "cukup"
    else:
        model_quality = "perlu perbaikan"

    st.markdown(
        f"""
<div class="model-conclusion-card">
    <div class="model-conclusion-title">
        Interpretasi Hasil Evaluasi Dataset Testing
    </div>


        Berdasarkan hasil pengujian menggunakan dataset testing,
        model menghasilkan nilai MAE sebesar {mae:,.2f}
        Nilai tersebut menunjukkan bahwa rata-rata selisih antara hasil
        prediksi dengan data aktual berada pada kisaran
        {mae:,.2f} penumpang

        Nilai RMSE sebesar {rmse:,.2f}menunjukkan tingkat kesalahan
        prediksi dengan memberikan penalti lebih besar terhadap kesalahan
        yang bernilai ekstrem. Perbedaan nilai antara MAE dan RMSE
        menunjukkan bahwa terdapat beberapa data dengan selisih prediksi
        yang lebih besar.

        Model memperoleh nilai R² sebesar {r2:.4f}, yang berarti model
        mampu menjelaskan sekitar {r2*100:.2f}% variasi perubahan
        volume penumpang berdasarkan pola data historis yang digunakan.

        Secara keseluruhan, performa model dapat dikategorikan
        {model_quality}. Model mampu menangkap pola historis data
        dengan baik dan dapat digunakan sebagai alat bantu prediksi
        volume penumpang KRL.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Actual vs Prediction")

    st.dataframe(
        result_dataframe[
            [
                "tanggal",
                "nama_stasiun",
                "actual",
                "prediction",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    csv = result_dataframe.to_csv(index=False)

    st.download_button(
        "Download Hasil Evaluasi",
        csv,
        file_name="hasil_testing_model.csv",
        mime="text/csv",
    )


# ------------------------------------------------------------------------------
# 8. AUTENTIKASI ADMIN DAN NAVIGASI HALAMAN SIDEBAR
# ------------------------------------------------------------------------------
def get_admin_credentials():
    """
    Mengambil kredensial admin dari Streamlit Secrets atau environment variable.

    Prioritas konfigurasi:
    1. .streamlit/secrets.toml pada bagian [admin]
    2. Environment variable GREENLINE_ADMIN_EMAIL dan GREENLINE_ADMIN_PASSWORD
    3. Kredensial bawaan untuk pengujian lokal
    """
    secret_admin = {}

    try:
        secret_admin = dict(st.secrets.get("admin", {}))
    except Exception:
        # Aplikasi tetap dapat berjalan ketika secrets.toml belum dibuat.
        secret_admin = {}

    admin_email = str(
        secret_admin.get("email")
        or os.getenv("GREENLINE_ADMIN_EMAIL")
        or "admin@greenline.com"
    ).strip()

    admin_password = str(
        secret_admin.get("password")
        or os.getenv("GREENLINE_ADMIN_PASSWORD")
        or "admin123"
    )

    return admin_email, admin_password


ADMIN_EMAIL, ADMIN_PASSWORD = get_admin_credentials()

if "admin_logged_in" not in st.session_state:
    st.session_state["admin_logged_in"] = False

if "admin_login_page_open" not in st.session_state:
    st.session_state["admin_login_page_open"] = False


def validate_admin_login(email, password):
    """Memvalidasi email dan password admin secara aman."""
    normalized_email = str(email).strip().lower()
    expected_email = ADMIN_EMAIL.strip().lower()

    email_valid = hmac.compare_digest(
        normalized_email,
        expected_email,
    )

    password_valid = hmac.compare_digest(
        str(password),
        ADMIN_PASSWORD,
    )

    return email_valid and password_valid


def close_admin_login_page():
    """Menutup halaman login ketika pengguna memilih menu utama."""
    st.session_state["admin_login_page_open"] = False


def render_sidebar_navigation():
    with st.sidebar:
        st.markdown(
            """
<div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
    <div style="background: rgba(255,255,255,0.20); color: #000000; width: 38px; height: 38px; border: 1px solid rgba(255,255,255,0.45); border-radius: 10px; display: flex; align-items: center; justify-content: center; box-shadow: 0 5px 14px rgba(27,78,245,0.20);">
        <span class="material-symbols-outlined" style="font-size: 22px;">directions_subway</span>
    </div>
    <div>
        <h3 style="margin: 0; font-size: 16px; font-weight: 800; color: #000000;">Greenline Predict</h3>
        <p style="margin: 0; font-size: 11px; color: #000000;">v2.11.2 • Multi Page</p>
    </div>
</div>
<div style="font-size: 12px; font-weight: 700; color: #000000; margin: 0 0 8px 4px; text-transform: uppercase; letter-spacing: 0.04em;">
    Menu Utama
</div>
""",
            unsafe_allow_html=True,
        )

        navigation_options = [
            "Dashboard",
            "Riwayat Data",
            "Evaluasi Model",
            "Import Excel",
        ]

        current_page = st.session_state.get(
            "greenline_main_page",
            "Dashboard",
        )

        if current_page not in navigation_options:
            st.session_state["greenline_main_page"] = "Dashboard"

        page = st.radio(
            "Navigasi",
            options=navigation_options,
            key="greenline_main_page",
            label_visibility="collapsed",
            on_change=close_admin_login_page,
        )

        st.markdown(
            "<hr style='margin: 14px 0 18px 0; border: 0; border-top: 1px solid rgba(255,255,255,0.40);'>",
            unsafe_allow_html=True,
        )

        if st.session_state["admin_logged_in"]:
            if st.button(
                "Logout Admin",
                key="admin_logout_button",
                use_container_width=True,
            ):
                st.session_state["admin_logged_in"] = False
                st.session_state["admin_login_page_open"] = False
                st.session_state["greenline_main_page"] = "Dashboard"
                st.rerun()
        else:
            if st.button(
                "Login Admin",
                key="open_admin_login_page_button",
                use_container_width=True,
            ):
                st.session_state["admin_login_page_open"] = True

        if st.session_state.get("admin_login_page_open", False):
            return "Login Admin"

    return page


selected_page = render_sidebar_navigation()


# ------------------------------------------------------------------------------
# 9. HALAMAN LOGIN ADMIN
# ------------------------------------------------------------------------------
if selected_page == "Login Admin":
    st.markdown(
        """

""",
        unsafe_allow_html=True,
    )

    login_left_column, login_form_column, login_right_column = st.columns(
        [1, 1.15, 1]
    )

    with login_form_column:
        with st.form("greenline_admin_login_form"):
            admin_email_input = st.text_input(
                "Email",
                placeholder="Masukkan email admin",
                key="admin_email_input",
            )

            admin_password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Masukkan password",
                key="admin_password_input",
            )

            login_submitted = st.form_submit_button(
                "Login",
                use_container_width=True,
            )

        if login_submitted:
            if validate_admin_login(
                admin_email_input,
                admin_password_input,
            ):
                st.session_state["admin_logged_in"] = True
                st.session_state["admin_login_page_open"] = False
                st.session_state["greenline_main_page"] = "Dashboard"
                st.rerun()
            else:
                st.error("Email atau password salah.")

    st.stop()


# ------------------------------------------------------------------------------
# 10. HALAMAN KHUSUS IMPORT EXCEL
# ------------------------------------------------------------------------------
if selected_page == "Import Excel":
    st.markdown(
        """
<div class="page-header-card">
    <div style="font-size: 12px; font-weight: 700; color: #047857; margin-bottom: 6px;">ADMINISTRASI DATASET</div>
    <h1 style="font-size: 26px; font-weight: 800; color: #0F172A; margin: 0 0 6px 0;">Import Dataset Excel</h1>
    <p style="font-size: 13px; color: #64748B; margin: 0;">
        Tambahkan data baru atau perbarui data penumpang yang sudah ada melalui file Excel.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    if "database_import_success" in st.session_state:
        import_success = st.session_state.pop(
            "database_import_success"
        )

        st.success(
            "Import berhasil. "
            f"File: {import_success['filename']} | "
            f"Data baru: {import_success['inserted_count']:,} | "
            f"Data diperbarui: {import_success['updated_count']:,} | "
            f"Total database: {import_success['total_rows']:,} baris."
        )

        if import_success["backup_path"]:
            st.caption(
                "Backup database lama: "
                + import_success["backup_path"]
            )

    # Form import ditempatkan di dalam card putih solid.
    with st.container(border=True):
        render_excel_importer()

    with st.expander("Ketentuan format file Excel"):
        st.markdown(
            """
Kolom wajib:

- `tanggal`
- `nama_stasiun`
- `penumpang_berangkat_komuter`
- `penumpang_datang_komuter`

Kolom `volume_penumpang` dihitung otomatis oleh sistem. Kombinasi nama stasiun dan tanggal menjadi kunci: data baru ditambahkan dan data yang sudah ada diperbarui otomatis.
"""
        )

    st.stop()


# ------------------------------------------------------------------------------
# 11. HALAMAN EVALUASI MODEL
# ------------------------------------------------------------------------------
if selected_page == "Evaluasi Model":
    st.markdown(
        """
<div class="page-header-card">
    <div style="font-size: 12px; font-weight: 700; color: #047857; margin-bottom: 6px;">EVALUASI MACHINE LEARNING</div>
    <h1 style="font-size: 26px; font-weight: 800; color: #0F172A; margin: 0 0 6px 0;">Evaluasi & Performa Model</h1>
    <p style="font-size: 13px; color: #64748B; margin: 0;">
        Ringkasan hasil pengujian Random Forest pada data latih dan data uji.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    evaluation_mae_col, evaluation_rmse_col, evaluation_r2_col = st.columns(3)

    with evaluation_mae_col:
        st.markdown(
            """
<div class="evaluation-card evaluation-card-mae">
    <div class="evaluation-icon evaluation-icon-mae">
        <span class="material-symbols-outlined">straighten</span>
    </div>
    <div class="evaluation-label">Mean Absolute Error (MAE)</div>
    <div class="evaluation-value">2.565,25</div>
    <div class="evaluation-description">
        Rata-rata kesalahan absolut prediksi pada data uji, dalam satuan penumpang.
        Nilai yang lebih kecil menunjukkan kesalahan yang lebih rendah.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with evaluation_rmse_col:
        st.markdown(
            """
<div class="evaluation-card evaluation-card-rmse">
    <div class="evaluation-icon evaluation-icon-rmse">
        <span class="material-symbols-outlined">monitoring</span>
    </div>
    <div class="evaluation-label">Root Mean Squared Error (RMSE)</div>
    <div class="evaluation-value">5.050,89</div>
    <div class="evaluation-description">
        Kesalahan yang memberi penalti lebih besar pada penyimpangan ekstrem.
        Nilai yang lebih kecil menunjukkan performa yang lebih baik.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    with evaluation_r2_col:
        st.markdown(
            """
<div class="evaluation-card evaluation-card-r2">
    <div class="evaluation-icon evaluation-icon-r2">
        <span class="material-symbols-outlined">verified</span>
    </div>
    <div class="evaluation-label">Koefisien Determinasi (R²)</div>
    <div class="evaluation-value">93,25%</div>
    <div class="evaluation-description">
        Model menjelaskan 93,25% variasi volume penumpang pada data pengujian.
        Nilai yang semakin mendekati 100% menunjukkan daya jelas yang semakin kuat.
    </div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
<div class="model-conclusion-card">
    <div class="model-conclusion-title">Kesimpulan Kinerja Model</div>
    <div class="model-conclusion-text">
        Nilai R² data uji sebesar <b>93,25%</b> menunjukkan bahwa Random Forest
        memiliki kemampuan penjelasan yang kuat pada data pengujian. Namun,
        angka 90% bukan batas universal yang otomatis membuktikan bahwa model
        adalah metode terbaik. R² juga bukan persentase prediksi yang selalu
        tepat. Model baru dapat dinyatakan lebih unggul daripada metode lain
        apabila seluruh metode diuji pada pembagian data dan periode pengujian
        yang sama, lalu Random Forest menghasilkan <b>R² lebih tinggi</b> serta
        <b>MAE dan RMSE lebih rendah</b>.
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    comparison_data = pd.DataFrame(
        {
            "Kriteria Perbandingan": [
                "Koefisien Determinasi (R²)",
                "Mean Absolute Error (MAE)",
                "Root Mean Squared Error (RMSE)",
                "Skema pengujian",
            ],
            "Hasil Random Forest": [
                "0,9325 (93,25%)",
                "2.565,25 penumpang",
                "5.050,89 penumpang",
                "Testing set yang sama",
            ],
            "Syarat untuk Menyatakan Random Forest Lebih Baik": [
                "R² harus lebih tinggi daripada model pembanding",
                "MAE harus lebih rendah daripada model pembanding",
                "RMSE harus lebih rendah daripada model pembanding",
                "Semua model memakai fitur, periode, dan data uji yang sama",
            ],
        }
    )

    st.markdown("### Cara Membandingkan dengan Metode Lain")
    st.caption(
        "Perbandingan yang adil harus menggunakan data uji dan prosedur validasi yang sama."
    )
    st.dataframe(
        comparison_data,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        "Dashboard belum memuat hasil metode pembanding. Karena itu, halaman "
        "ini dapat menyimpulkan bahwa performa Random Forest pada data uji "
        "tergolong kuat, tetapi belum dapat menyatakan bahwa Random Forest "
        "pasti lebih baik daripada seluruh metode lain."
    )

    st.markdown("### Ringkasan Akurasi & Performa Model")
    st.caption(
        "Tabel 4.10 Skripsi — evaluasi model terhadap training set dan testing set"
    )

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
            "Rata-rata kesalahan absolut pada data uji",
            "Model masih sensitif terhadap kesalahan yang sangat besar",
            "Daya jelas model pada data uji tergolong kuat",
        ],
    }

    evaluation_dataframe = pd.DataFrame(evaluation_data)
    st.table(evaluation_dataframe)

    st.info(
        "Selisih R² data latih dan data uji adalah 4,52 poin persentase "
        "(97,77% − 93,25%). Penurunan ini menunjukkan performa testing "
        "lebih rendah daripada training, sehingga generalisasi tetap perlu "
        "dipantau pada data baru."
    )

    st.stop()


# ------------------------------------------------------------------------------
# 12. PEMERIKSAAN KETERSEDIAAN MODEL DAN DATABASE
# ------------------------------------------------------------------------------
if MODEL_LOAD_ERROR is not None:
    st.error(MODEL_LOAD_ERROR)
    st.stop()


if not DATABASE_PATH.exists():
    st.warning(
        "Database belum tersedia. Buka menu `Import Excel` pada sidebar "
        "untuk mengunggah dataset pertama."
    )
    st.stop()


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
# 13. FUNGSI UTILITAS PREDIKSI DAN FORMAT
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


def resolve_prediction_lags(
    station_history,
    selected_station,
    target_date,
    model,
    features,
):
    """
    Mengambil lag aktual jika tersedia dan membuat prediksi bertahap untuk
    menjangkau tanggal setelah data aktual terakhir.

    Fungsi juga mengembalikan predicted_values agar riwayat tujuh hari sebelum
    tanggal target dapat ditampilkan ketika target berada pada masa depan.
    """
    if station_history.empty:
        return {
            "lag_1": None,
            "lag_7": None,
            "lag_1_source": "Tidak tersedia",
            "lag_7_source": "Tidak tersedia",
            "projection_days": 0,
            "last_actual_date": None,
            "predicted_values": {},
            "error": "Riwayat stasiun kosong.",
        }

    actual_values = {
        row.tanggal.date(): float(row.volume_penumpang)
        for row in station_history.itertuples(index=False)
    }

    last_actual_date = max(actual_values)
    predicted_values = {}

    def get_available_value(date_value):
        if date_value in actual_values:
            return actual_values[date_value], "Data aktual"

        if date_value in predicted_values:
            return predicted_values[date_value], "Prediksi bertahap"

        return None, "Tidak tersedia"

    forecast_date = last_actual_date + datetime.timedelta(days=1)
    forecast_end_date = target_date - datetime.timedelta(days=1)

    projection_days = 0

    while forecast_date <= forecast_end_date:
        lag_1_value, _ = get_available_value(
            forecast_date - datetime.timedelta(days=1)
        )
        lag_7_value, _ = get_available_value(
            forecast_date - datetime.timedelta(days=7)
        )

        if lag_1_value is None or lag_7_value is None:
            return {
                "lag_1": None,
                "lag_7": None,
                "lag_1_source": "Tidak tersedia",
                "lag_7_source": "Tidak tersedia",
                "projection_days": projection_days,
                "last_actual_date": last_actual_date,
                "predicted_values": predicted_values,
                "error": (
                    "Data historis tidak lengkap untuk membangun prediksi "
                    f"bertahap pada {forecast_date.strftime('%d/%m/%Y')}."
                ),
            }

        recursive_input = build_model_input(
            features=features,
            selected_station=selected_station,
            selected_date=forecast_date,
            lag_1=lag_1_value,
            lag_7=lag_7_value,
        )

        recursive_prediction = float(
            model.predict(recursive_input)[0]
        )

        predicted_values[forecast_date] = max(
            0.0,
            recursive_prediction,
        )

        projection_days += 1
        forecast_date += datetime.timedelta(days=1)

    lag_1_date = target_date - datetime.timedelta(days=1)
    lag_7_date = target_date - datetime.timedelta(days=7)

    lag_1_value, lag_1_source = get_available_value(lag_1_date)
    lag_7_value, lag_7_source = get_available_value(lag_7_date)

    return {
        "lag_1": (
            int(round(lag_1_value))
            if lag_1_value is not None
            else None
        ),
        "lag_7": (
            int(round(lag_7_value))
            if lag_7_value is not None
            else None
        ),
        "lag_1_source": lag_1_source,
        "lag_7_source": lag_7_source,
        "projection_days": projection_days,
        "last_actual_date": last_actual_date,
        "predicted_values": predicted_values,
        "error": None,
    }


def predict_station_for_date(
    station_history,
    selected_station,
    target_date,
    model,
    features,
):
    """Menghasilkan satu prediksi lengkap untuk satu stasiun dan satu tanggal."""
    lag_resolution = resolve_prediction_lags(
        station_history=station_history,
        selected_station=selected_station,
        target_date=target_date,
        model=model,
        features=features,
    )

    if lag_resolution["error"] is not None:
        return {
            "error": lag_resolution["error"],
            "station": selected_station,
        }

    lag_1 = lag_resolution["lag_1"]
    lag_7 = lag_resolution["lag_7"]

    if lag_1 is None or lag_7 is None:
        return {
            "error": "Nilai H-1 atau H-7 tidak tersedia.",
            "station": selected_station,
        }

    input_dataframe = build_model_input(
        features=features,
        selected_station=selected_station,
        selected_date=target_date,
        lag_1=lag_1,
        lag_7=lag_7,
    )

    predicted_volume = max(
        0.0,
        float(model.predict(input_dataframe)[0]),
    )

    actual_target_rows = station_history[
        station_history["tanggal"].dt.date == target_date
    ]

    actual_target = (
        float(actual_target_rows.iloc[-1]["volume_penumpang"])
        if not actual_target_rows.empty
        else None
    )

    density_status = classify_density(
        predicted_volume,
        station_history,
    )

    return {
        "error": None,
        "station": selected_station,
        "target_date": target_date,
        "predicted_volume": predicted_volume,
        "actual_target": actual_target,
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_1_source": lag_resolution["lag_1_source"],
        "lag_7_source": lag_resolution["lag_7_source"],
        "projection_days": lag_resolution["projection_days"],
        "last_actual_date": lag_resolution["last_actual_date"],
        "predicted_values": lag_resolution["predicted_values"],
        "density_status": density_status,
        "input_dataframe": input_dataframe,
    }


def build_extended_station_history(
    station_history,
    selected_station,
    start_date,
    end_date,
    model,
    features,
    progress_callback=None,
):
    """
    Menggabungkan data aktual dengan prediksi bertahap sampai tanggal akhir.

    Data aktual tetap diberi label Data Aktual. Tanggal setelah data aktual
    terakhir diberi label Prediksi Bertahap.
    """
    if station_history.empty:
        return pd.DataFrame(), "Riwayat stasiun kosong."

    station_history = station_history.copy()
    station_history["sumber"] = "Data Aktual"

    actual_values = {
        row.tanggal.date(): float(row.volume_penumpang)
        for row in station_history.itertuples(index=False)
    }

    last_actual_date = max(actual_values)
    predicted_values = {}

    forecast_start = last_actual_date + datetime.timedelta(days=1)
    total_forecast_days = max(
        0,
        (end_date - forecast_start).days + 1,
    )

    forecast_date = forecast_start
    completed_days = 0

    while forecast_date <= end_date:
        lag_1_date = forecast_date - datetime.timedelta(days=1)
        lag_7_date = forecast_date - datetime.timedelta(days=7)

        lag_1_value = actual_values.get(
            lag_1_date,
            predicted_values.get(lag_1_date),
        )
        lag_7_value = actual_values.get(
            lag_7_date,
            predicted_values.get(lag_7_date),
        )

        if lag_1_value is None or lag_7_value is None:
            return pd.DataFrame(), (
                "Data H-1 atau H-7 tidak tersedia untuk membangun proyeksi "
                f"pada {forecast_date.strftime('%d/%m/%Y')}."
            )

        forecast_input = build_model_input(
            features=features,
            selected_station=selected_station,
            selected_date=forecast_date,
            lag_1=lag_1_value,
            lag_7=lag_7_value,
        )

        forecast_value = max(
            0.0,
            float(model.predict(forecast_input)[0]),
        )

        predicted_values[forecast_date] = forecast_value
        completed_days += 1

        if progress_callback is not None:
            progress_callback(
                completed_days,
                total_forecast_days,
                forecast_date,
            )

        forecast_date += datetime.timedelta(days=1)

    predicted_dataframe = pd.DataFrame(
        [
            {
                "tanggal": pd.Timestamp(date_value),
                "volume_penumpang": volume_value,
                "sumber": "Prediksi Bertahap",
            }
            for date_value, volume_value in predicted_values.items()
        ]
    )

    combined_dataframe = pd.concat(
        [
            station_history[
                [
                    "tanggal",
                    "volume_penumpang",
                    "sumber",
                ]
            ],
            predicted_dataframe,
        ],
        ignore_index=True,
    )

    combined_dataframe = combined_dataframe[
        (combined_dataframe["tanggal"].dt.date >= start_date)
        & (combined_dataframe["tanggal"].dt.date <= end_date)
    ].copy()

    combined_dataframe = (
        combined_dataframe
        .sort_values("tanggal")
        .reset_index(drop=True)
    )

    return combined_dataframe, None


def format_elapsed_time(seconds):
    """Memformat durasi proses agar mudah dibaca pengguna."""
    if seconds < 60:
        return f"{seconds:.2f} detik"

    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60

    return f"{minutes} menit {remaining_seconds:.1f} detik"


DASHBOARD_DENSITY_COLORS = {
    "NORMAL": "#FFF7CC",
    "PADAT": "#FDBA74",
    "SANGAT PADAT": "#F97316",
    "EKSTREM": "#B91C1C",
    "TIDAK TERKLASIFIKASI": "#CBD5E1",
}


def render_dashboard_urgency_legend():
    """Menampilkan legenda kepadatan dengan palet yang sama seperti heatmap."""
    dashboard_palette_legend_html = (
        '<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
        'border-radius:14px;padding:14px 16px;margin:8px 0 14px 0;">'
        '<div style="font-size:13px;font-weight:700;color:#0F172A;'
        'margin-bottom:10px;">Interpretasi Tingkat Kepadatan</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:12px 22px;'
        'align-items:center;">'
        '<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
        'color:#475569;"><span style="width:18px;height:18px;border-radius:5px;'
        'background:#FFF7CC;border:1px solid #E5E7EB;display:inline-block;">'
        '</span><span><b>Normal</b> — volume relatif rendah</span></div>'
        '<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
        'color:#475569;"><span style="width:18px;height:18px;border-radius:5px;'
        'background:#FDBA74;display:inline-block;"></span>'
        '<span><b>Padat</b> — volume mulai meningkat</span></div>'
        '<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
        'color:#475569;"><span style="width:18px;height:18px;border-radius:5px;'
        'background:#F97316;display:inline-block;"></span>'
        '<span><b>Sangat Padat</b> — volume tinggi</span></div>'
        '<div style="display:flex;align-items:center;gap:8px;font-size:12px;'
        'color:#475569;"><span style="width:18px;height:18px;border-radius:5px;'
        'background:#B91C1C;display:inline-block;"></span>'
        '<span><b>Ekstrem</b> — prioritas perhatian</span></div>'
        '</div></div>'
    )

    st.markdown(
        dashboard_palette_legend_html,
        unsafe_allow_html=True,
    )


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
            "color": "#A16207",
            "border_class": "kpi-border-yellow",
            "description": "Volume relatif rendah dalam pola historis.",
        }

    if predicted_volume <= q75:
        return {
            "label": "PADAT",
            "color": "#C2410C",
            "border_class": "kpi-border-light-orange",
            "description": "Volume mulai meningkat di atas median historis.",
        }

    if predicted_volume <= q90:
        return {
            "label": "SANGAT PADAT",
            "color": "#F97316",
            "border_class": "kpi-border-orange",
            "description": "Masuk kelompok volume historis tinggi.",
        }

    return {
        "label": "EKSTREM",
        "color": "#B91C1C",
        "border_class": "kpi-border-red",
        "description": "Melebihi persentil ke-90 historis dan perlu perhatian.",
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
# 13. HALAMAN RIWAYAT DATA DAN PROYEKSI SAMPAI 2027
# ------------------------------------------------------------------------------
if selected_page == "Riwayat Data":
    st.markdown(
        """
<div class="page-header-card">
    <div style="font-size: 12px; font-weight: 700; color: #047857; margin-bottom: 6px;">
        RIWAYAT DATA & PROYEKSI
    </div>
    <h1 style="font-size: 26px; font-weight: 800; color: #0F172A; margin: 0 0 6px 0;">
        Riwayat dan Proyeksi Volume Penumpang
    </h1>
    <p style="font-size: 13px; color: #64748B; margin: 0;">
        Tampilkan data aktual sampai periode database dan proyeksi bertahap sampai akhir 2027.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Saringan Riwayat dan Proyeksi")

    maximum_history_date = max(
        datetime.date(2027, 12, 31),
        maximum_data_date,
    )

    default_history_start = max(
        minimum_data_date,
        maximum_data_date - datetime.timedelta(days=30),
    )

    with st.container(border=True):
        history_filter_station_col, history_filter_date_col = st.columns(
            [1.0, 1.8]
        )

        with history_filter_station_col:
            history_station = st.selectbox(
                "Pilih Stasiun",
                options=available_stations,
                index=(
                    available_stations.index("SERPONG")
                    if "SERPONG" in available_stations
                    else 0
                ),
                key="history_station",
            )

        with history_filter_date_col:
            history_date_range = st.date_input(
                "Rentang Tanggal",
                value=(default_history_start, maximum_data_date),
                min_value=minimum_data_date,
                max_value=maximum_history_date,
                format="YYYY/MM/DD",
                key="history_date_range",
            )

        st.caption(
            "Tanggal sampai data terakhir menggunakan data aktual. "
            "Tanggal setelah data terakhir menggunakan prediksi bertahap."
        )

    if (
        isinstance(history_date_range, (tuple, list))
        and len(history_date_range) == 2
    ):
        history_start_date, history_end_date = history_date_range
    else:
        history_start_date = history_date_range
        history_end_date = history_date_range

    if history_start_date > history_end_date:
        st.error("Tanggal awal tidak boleh lebih besar daripada tanggal akhir.")
        st.stop()

    history_database_station = station_lookup[history_station]
    history_station_dataframe = load_station_history(
        database_path=str(DATABASE_PATH),
        database_modified_time=DATABASE_PATH.stat().st_mtime,
        database_station=history_database_station,
    )

    history_process_start = time.perf_counter()
    history_progress = st.progress(
        0,
        text="Menyiapkan data riwayat...",
    )

    def update_history_progress(completed, total, current_date):
        if total <= 0:
            progress_value = 1.0
        else:
            progress_value = min(
                1.0,
                completed / total,
            )

        elapsed_so_far = (
            time.perf_counter()
            - history_process_start
        )

        estimated_remaining = (
            (elapsed_so_far / completed)
            * (total - completed)
            if completed > 0
            else 0.0
        )

        history_progress.progress(
            progress_value,
            text=(
                "Membangun proyeksi "
                f"{current_date.strftime('%d/%m/%Y')} "
                f"({completed:,}/{total:,} hari) • "
                f"Estimasi sisa {format_elapsed_time(estimated_remaining)}"
            ),
        )

    history_filtered, history_forecast_error = build_extended_station_history(
        station_history=history_station_dataframe,
        selected_station=history_station,
        start_date=history_start_date,
        end_date=history_end_date,
        model=rf_model,
        features=model_features,
        progress_callback=update_history_progress,
    )

    history_elapsed = time.perf_counter() - history_process_start

    if history_forecast_error is not None:
        history_progress.empty()
        st.error(history_forecast_error)
        st.stop()

    history_progress.progress(
        1.0,
        text=(
            "Data riwayat dan proyeksi siap dalam "
            f"{format_elapsed_time(history_elapsed)}"
        ),
    )

    st.caption(
        f"Stasiun {history_station} • "
        f"{history_start_date.strftime('%d/%m/%Y')}–"
        f"{history_end_date.strftime('%d/%m/%Y')} • "
        f"Waktu proses: {format_elapsed_time(history_elapsed)}"
    )

    if history_filtered.empty:
        st.warning(
            "Tidak ada data aktual maupun prediksi pada stasiun dan "
            "rentang tanggal yang dipilih."
        )
        st.stop()

    predicted_history_count = int(
        (history_filtered["sumber"] == "Prediksi Bertahap").sum()
    )

    if predicted_history_count > 0:
        st.warning(
            f"Rentang yang dipilih memuat {predicted_history_count:,} hari "
            "prediksi bertahap. Semakin jauh tanggal dari data aktual "
            "terakhir."
        )

    history_metric_1, history_metric_2, history_metric_3, history_metric_4 = (
        st.columns(4)
    )

    history_metric_1.metric(
        "Jumlah Hari",
        f"{len(history_filtered):,}",
    )
    history_metric_2.metric(
        "Rata-rata Volume",
        format_integer_indonesia(
            history_filtered["volume_penumpang"].mean()
        ),
    )
    history_metric_3.metric(
        "Volume Tertinggi",
        format_integer_indonesia(
            history_filtered["volume_penumpang"].max()
        ),
    )
    history_metric_4.metric(
        "Jumlah Hari Proyeksi",
        f"{predicted_history_count:,}",
        help=(
            "Jumlah hari setelah data aktual terakhir yang dihitung "
            "menggunakan prediksi bertahap."
        ),
    )

    history_figure = go.Figure()

    actual_history = history_filtered[
        history_filtered["sumber"] == "Data Aktual"
    ]

    predicted_history = history_filtered[
        history_filtered["sumber"] == "Prediksi Bertahap"
    ]

    if not actual_history.empty:
        history_figure.add_trace(
            go.Scatter(
                x=actual_history["tanggal"],
                y=actual_history["volume_penumpang"],
                mode="lines+markers",
                name="Data Aktual",
                line=dict(
                    color="#006E2A",
                    width=3,
                ),
                marker=dict(
                    size=5,
                    color="#006E2A",
                ),
                fill="tozeroy",
                fillcolor="rgba(0, 110, 42, 0.05)",
            )
        )

    if not predicted_history.empty:
        history_figure.add_trace(
            go.Scatter(
                x=predicted_history["tanggal"],
                y=predicted_history["volume_penumpang"],
                mode="lines+markers",
                name="Prediksi Bertahap",
                line=dict(
                    color="#F97316",
                    width=3,
                    dash="dash",
                ),
                marker=dict(
                    size=5,
                    color="#F97316",
                ),
            )
        )

    history_figure.update_layout(
        font=dict(
            color="#000000",
        ),
        xaxis=dict(
            showgrid=False,
            tickformat="%d/%m/%Y",
            tickfont=dict(
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2E8F0",
            tickformat=",",
            tickfont=dict(
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(
                color="#000000",
                size=12,
            ),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(
            l=10,
            r=10,
            t=25,
            b=10,
        ),
        height=380,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                color="#000000",
            ),
        ),
    )

    st.plotly_chart(
        history_figure,
        use_container_width=True,
    )

    # --------------------------------------------------------------------------
    # HEATMAP VOLUME PENUMPANG: MINGGU × HARI
    # --------------------------------------------------------------------------
    st.markdown("### Heatmap Volume Penumpang")
    st.caption(
        "Heatmap mencakup data aktual dan prediksi bertahap pada rentang "
        "tanggal yang dipilih."
    )

    heatmap_dataframe = history_filtered.copy()
    day_order = ["Sen", "Sel", "Rab", "Kam", "Jum", "Sab", "Min"]
    day_mapping = {
        0: "Sen",
        1: "Sel",
        2: "Rab",
        3: "Kam",
        4: "Jum",
        5: "Sab",
        6: "Min",
    }

    heatmap_dataframe["hari"] = (
        heatmap_dataframe["tanggal"]
        .dt.dayofweek
        .map(day_mapping)
    )

    heatmap_dataframe["awal_minggu"] = (
        heatmap_dataframe["tanggal"]
        - pd.to_timedelta(
            heatmap_dataframe["tanggal"].dt.dayofweek,
            unit="D",
        )
    )

    heatmap_pivot = heatmap_dataframe.pivot_table(
        index="awal_minggu",
        columns="hari",
        values="volume_penumpang",
        aggfunc="mean",
    ).reindex(columns=day_order)

    heatmap_pivot = heatmap_pivot.sort_index()

    heatmap_week_labels = [
        f"Minggu {week_date.strftime('%d/%m/%Y')}"
        for week_date in heatmap_pivot.index
    ]

    heatmap_colorscale = [
        [0.00, "#FFF7CC"],
        [0.35, "#FDBA74"],
        [0.65, "#F97316"],
        [1.00, "#B91C1C"],
    ]

    heatmap_minimum = float(
        heatmap_pivot.min().min()
    )

    heatmap_maximum = float(
        heatmap_pivot.max().max()
    )

    heatmap_range = (
        heatmap_maximum
        - heatmap_minimum
    )

    if heatmap_range > 0:
        heatmap_tick_values = [
            heatmap_minimum,
            heatmap_minimum + (0.35 * heatmap_range),
            heatmap_minimum + (0.65 * heatmap_range),
            heatmap_maximum,
        ]

        heatmap_tick_text = [
            "Rendah",
            "Sedang",
            "Tinggi",
            "Sangat Tinggi",
        ]
    else:
        heatmap_tick_values = [
            heatmap_minimum
        ]

        heatmap_tick_text = [
            "Volume Sama"
        ]

    heatmap_legend_html = (
        '<div style="background:#FFFFFF;border:1px solid #E2E8F0;'
        'border-radius:14px;padding:16px 18px;margin:8px 0 16px 0;">'
        '<div style="font-size:13px;font-weight:700;color:#0F172A;'
        'margin-bottom:12px;">Interpretasi Warna Heatmap</div>'
        '<div style="display:flex;flex-wrap:wrap;gap:16px 24px;'
        'align-items:center;">'
        '<div style="display:flex;align-items:center;gap:8px;'
        'font-size:12px;color:#475569;">'
        '<span style="width:18px;height:18px;border-radius:5px;'
        'background:#FFF7CC;border:1px solid #E5E7EB;'
        'display:inline-block;flex-shrink:0;"></span>'
        '<span><b>Rendah</b> — volume relatif lebih kecil</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:8px;'
        'font-size:12px;color:#475569;">'
        '<span style="width:18px;height:18px;border-radius:5px;'
        'background:#FDBA74;display:inline-block;flex-shrink:0;"></span>'
        '<span><b>Sedang</b> — volume mulai meningkat</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:8px;'
        'font-size:12px;color:#475569;">'
        '<span style="width:18px;height:18px;border-radius:5px;'
        'background:#F97316;display:inline-block;flex-shrink:0;"></span>'
        '<span><b>Tinggi</b> — volume relatif besar</span>'
        '</div>'
        '<div style="display:flex;align-items:center;gap:8px;'
        'font-size:12px;color:#475569;">'
        '<span style="width:18px;height:18px;border-radius:5px;'
        'background:#B91C1C;display:inline-block;flex-shrink:0;"></span>'
        '<span><b>Sangat Tinggi</b> — volume tertinggi</span>'
        '</div>'
        '</div>'
        '<div style="margin-top:12px;padding-top:10px;'
        'border-top:1px solid #E2E8F0;font-size:11px;line-height:1.6;'
        'color:#64748B;">'
        'Warna dibandingkan secara relatif dalam rentang yang dipilih. '
        'Warna tidak menyatakan kapasitas operasional resmi. Sumber aktual '
        'atau prediksi dapat dilihat pada grafik dan tabel.'
        '</div>'
        '</div>'
    )

    st.markdown(
        heatmap_legend_html,
        unsafe_allow_html=True,
    )

    heatmap_figure = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=day_order,
            y=heatmap_week_labels,
            colorscale=heatmap_colorscale,
            zmin=heatmap_minimum,
            zmax=heatmap_maximum,
            colorbar=dict(
                title=dict(
                    text="Tingkat<br>Volume",
                    side="top",
                    font=dict(
                        color="#000000",
                    ),
                ),
                tickvals=heatmap_tick_values,
                ticktext=heatmap_tick_text,
                thickness=18,
                len=0.85,
                outlinewidth=0,
                tickfont=dict(
                    size=11,
                    color="#000000",
                ),
            ),
            hoverongaps=False,
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Hari: %{x}<br>"
                "Volume penumpang: %{z:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    heatmap_height = min(
        700,
        max(
            300,
            90 + (
                len(heatmap_pivot.index)
                * 42
            ),
        ),
    )

    heatmap_figure.update_layout(
        font=dict(
            color="#0F172A",
        ),
        xaxis=dict(
            title="Hari",
            side="top",
            showgrid=False,
            tickfont=dict(
                size=12,
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        yaxis=dict(
            title="Awal Minggu",
            autorange="reversed",
            showgrid=False,
            tickfont=dict(
                size=11,
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(
                color="#000000",
                size=12,
            ),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(
            l=10,
            r=20,
            t=30,
            b=10,
        ),
        height=heatmap_height,
    )

    st.plotly_chart(
        heatmap_figure,
        use_container_width=True,
    )

    history_table = history_filtered.sort_values(
        "tanggal",
        ascending=False,
    ).copy()

    history_table["tanggal"] = (
        history_table["tanggal"]
        .dt.strftime("%d/%m/%Y")
    )

    history_table["volume_penumpang"] = (
        history_table["volume_penumpang"]
        .round()
        .astype(int)
    )

    history_table = history_table.rename(
        columns={
            "tanggal": "Tanggal",
            "volume_penumpang": "Volume Penumpang",
            "sumber": "Sumber Data",
        }
    )

    st.markdown("### Tabel Riwayat dan Proyeksi")
    st.dataframe(
        history_table[
            [
                "Tanggal",
                "Volume Penumpang",
                "Sumber Data",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    st.stop()


# ------------------------------------------------------------------------------
# 14. HALAMAN DASHBOARD (SEMUA STASIUN ATAU SATU STASIUN)
# ------------------------------------------------------------------------------
st.markdown(
    f"""
<div class="dashboard-header-card">
    <div class="dashboard-header-top">
        <div class="dashboard-header-kicker">
            Lintas Green Line • KAI Commuter
        </div>
        <div class="dashboard-admin-box">
            <span class="material-symbols-outlined" style="color:#475569 !important;">
                notifications
            </span>
            <div class="dashboard-admin-avatar">AT</div>
            <div class="dashboard-admin-name">Admin Transit</div>
        </div>
    </div>
    <h1 class="dashboard-header-title">
        Dashboard Prediksi Volume Penumpang Harian KRL Lintas Green Line
    </h1>
    <p class="dashboard-header-description">
        Tampilan awal memproses seluruh stasiun untuk satu tanggal. Pilih satu
        stasiun untuk melihat analisis yang lebih terperinci.
    </p>
    <div class="dashboard-database-info">
        Database: {total_database_rows:,} baris &nbsp;•&nbsp;
        Periode aktual: {minimum_data_date.strftime('%d/%m/%Y')}–{maximum_data_date.strftime('%d/%m/%Y')}
    </div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("### Seleksi Data Prediksi")

ALL_STATIONS_OPTION = "SEMUA STASIUN"

minimum_prediction_date = (
    minimum_data_date
    + datetime.timedelta(days=7)
)

maximum_prediction_date = max(
    datetime.date(2027, 12, 31),
    maximum_data_date + datetime.timedelta(days=1),
)

default_prediction_date = min(
    maximum_data_date + datetime.timedelta(days=1),
    maximum_prediction_date,
)

with st.container(border=True):
    filter_station_col, filter_date_col = st.columns(
        [1.25, 1.0]
    )

    with filter_station_col:
        selected_station_scope = st.selectbox(
            "Pilih Cakupan Stasiun",
            options=[
                ALL_STATIONS_OPTION,
                *available_stations,
            ],
            index=0,
            key="overview_station_scope",
        )

    with filter_date_col:
        selected_date = st.date_input(
            "Tanggal Prediksi",
            value=default_prediction_date,
            min_value=minimum_prediction_date,
            max_value=maximum_prediction_date,
            format="YYYY/MM/DD",
            key="overview_prediction_date",
        )

    if selected_station_scope == ALL_STATIONS_OPTION:
        st.caption(
            f"Sistem akan menghitung seluruh {len(available_stations)} stasiun "
            "untuk satu tanggal yang dipilih."
        )
    else:
        st.caption(
            "Sistem akan menampilkan detail H-1, H-7, grafik, status "
            "kepadatan, dan rekomendasi untuk satu stasiun."
        )

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# 15. MODE SEMUA STASIUN
# ------------------------------------------------------------------------------
if selected_station_scope == ALL_STATIONS_OPTION:
    batch_start_time = time.perf_counter()

    batch_progress = st.progress(
        0,
        text="Menyiapkan prediksi seluruh stasiun...",
    )

    all_station_results = []
    all_station_errors = []

    total_stations = len(available_stations)

    for station_index, station_name in enumerate(
        available_stations,
        start=1,
    ):
        batch_progress.progress(
            (station_index - 1) / total_stations,
            text=(
                f"Memproses {station_name} "
                f"({station_index}/{total_stations})"
            ),
        )

        database_station_name = station_lookup[station_name]

        station_history = load_station_history(
            database_path=str(DATABASE_PATH),
            database_modified_time=DATABASE_PATH.stat().st_mtime,
            database_station=database_station_name,
        )

        try:
            station_result = predict_station_for_date(
                station_history=station_history,
                selected_station=station_name,
                target_date=selected_date,
                model=rf_model,
                features=model_features,
            )
        except Exception as error:
            station_result = {
                "error": str(error),
                "station": station_name,
            }

        if station_result["error"] is not None:
            all_station_errors.append(
                {
                    "Stasiun": station_name,
                    "Kesalahan": station_result["error"],
                }
            )
        else:
            all_station_results.append(
                station_result
            )

        elapsed_so_far = (
            time.perf_counter()
            - batch_start_time
        )

        estimated_remaining = (
            (elapsed_so_far / station_index)
            * (total_stations - station_index)
        )

        batch_progress.progress(
            station_index / total_stations,
            text=(
                f"Selesai {station_index}/{total_stations} stasiun • "
                f"Estimasi sisa {format_elapsed_time(estimated_remaining)}"
            ),
        )

    batch_elapsed = (
        time.perf_counter()
        - batch_start_time
    )

    batch_progress.progress(
        1.0,
        text=(
            f"Seluruh stasiun selesai dalam "
            f"{format_elapsed_time(batch_elapsed)}"
        ),
    )

    st.caption(
        f"Waktu pemrosesan {len(available_stations)} stasiun: "
        f"{format_elapsed_time(batch_elapsed)}"
    )

    if not all_station_results:
        st.error(
            "Tidak ada prediksi stasiun yang berhasil dihitung."
        )

        if all_station_errors:
            st.dataframe(
                pd.DataFrame(all_station_errors),
                use_container_width=True,
                hide_index=True,
            )

        st.stop()

    all_station_dataframe = pd.DataFrame(
        [
            {
                "Stasiun": result["station"],
                "Tanggal": result["target_date"].strftime("%d/%m/%Y"),
                "Prediksi Volume": int(
                    round(result["predicted_volume"])
                ),
                "Volume Aktual": (
                    int(round(result["actual_target"]))
                    if result["actual_target"] is not None
                    else None
                ),
                "H-1": int(result["lag_1"]),
                "H-7": int(result["lag_7"]),
                "Sumber H-1": result["lag_1_source"],
                "Sumber H-7": result["lag_7_source"],
                "Jumlah Hari Proyeksi": int(result["projection_days"]),
                "Status Kepadatan": result["density_status"]["label"],
            }
            for result in all_station_results
        ]
    )

    all_station_dataframe["Selisih Prediksi-Aktual"] = (
        all_station_dataframe["Prediksi Volume"]
        - all_station_dataframe["Volume Aktual"]
    )

    all_station_dataframe = (
        all_station_dataframe
        .sort_values(
            "Prediksi Volume",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    total_prediction = int(
        all_station_dataframe["Prediksi Volume"].sum()
    )

    average_prediction = float(
        all_station_dataframe["Prediksi Volume"].mean()
    )

    highest_station_row = (
        all_station_dataframe.iloc[0]
    )

    high_density_count = int(
        all_station_dataframe[
            "Status Kepadatan"
        ].isin(
            [
                "SANGAT PADAT",
                "EKSTREM",
            ]
        ).sum()
    )

    summary_1, summary_2, summary_3, summary_4 = st.columns(4)

    summary_1.metric(
        "Total Prediksi Semua Stasiun",
        format_integer_indonesia(total_prediction),
        help="Jumlah prediksi volume seluruh stasiun pada tanggal terpilih.",
    )

    summary_2.metric(
        "Rata-rata per Stasiun",
        format_integer_indonesia(average_prediction),
    )

    summary_3.metric(
        "Stasiun Tertinggi",
        highest_station_row["Stasiun"],
        format_integer_indonesia(
            highest_station_row["Prediksi Volume"]
        ),
    )

    summary_4.metric(
        "Sangat Padat / Ekstrem",
        f"{high_density_count} stasiun",
    )

    if selected_date > maximum_data_date:
        maximum_projection_days = int(
            all_station_dataframe["Jumlah Hari Proyeksi"].max()
        )

        if maximum_projection_days > 0:
            st.warning(
                f"Tanggal berada setelah data aktual terakhir. Prediksi semua "
                f"stasiun menggunakan proses bertahap hingga "
                f"{maximum_projection_days:,} hari. Ketidakpastian meningkat "
                "pada horizon yang semakin panjang."
            )
        else:
            st.info(
                "Tanggal yang dipilih berada tepat setelah data aktual "
                "terakhir. Prediksi masih menggunakan H-1 dan H-7 aktual "
                "tanpa rangkaian prediksi bertahap sebelumnya."
            )

    st.markdown("### Perbandingan Prediksi Antarstasiun")

    render_dashboard_urgency_legend()

    station_bar_colors = [
        DASHBOARD_DENSITY_COLORS.get(
            status,
            "#CBD5E1",
        )
        for status in all_station_dataframe[
            "Status Kepadatan"
        ]
    ]

    station_bar_figure = go.Figure(
        data=go.Bar(
            x=all_station_dataframe["Stasiun"],
            y=all_station_dataframe["Prediksi Volume"],
            marker=dict(
                color=station_bar_colors,
                line=dict(
                    color="#CBD5E1",
                    width=0.8,
                ),
            ),
            customdata=all_station_dataframe[
                [
                    "Status Kepadatan",
                    "H-1",
                    "H-7",
                ]
            ].values,
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Prediksi: %{y:,.0f}<br>"
                "Status: %{customdata[0]}<br>"
                "H-1: %{customdata[1]:,.0f}<br>"
                "H-7: %{customdata[2]:,.0f}"
                "<extra></extra>"
            ),
        )
    )

    station_bar_figure.update_layout(
        font=dict(
            color="#000000",
        ),
        xaxis=dict(
            title="Stasiun",
            showgrid=False,
            tickangle=-35,
            tickfont=dict(
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        yaxis=dict(
            title="Prediksi Volume Penumpang",
            showgrid=True,
            gridcolor="#E2E8F0",
            tickformat=",",
            tickfont=dict(
                color="#000000",
            ),
            title_font=dict(
                color="#000000",
            ),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor="#CBD5E1",
            font=dict(
                color="#000000",
                size=12,
            ),
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        margin=dict(
            l=10,
            r=10,
            t=20,
            b=80,
        ),
        height=460,
        showlegend=False,
    )

    st.plotly_chart(
        station_bar_figure,
        use_container_width=True,
    )

    st.markdown("### Data Seluruh Stasiun pada Satu Hari")

    display_all_station_dataframe = (
        all_station_dataframe.copy()
    )

    integer_columns = [
        "Prediksi Volume",
        "Volume Aktual",
        "Selisih Prediksi-Aktual",
        "H-1",
        "H-7",
        "Jumlah Hari Proyeksi",
    ]

    for integer_column in integer_columns:
        if integer_column in display_all_station_dataframe:
            display_all_station_dataframe[integer_column] = (
                pd.to_numeric(
                    display_all_station_dataframe[integer_column],
                    errors="coerce",
                )
                .astype("Int64")
            )

    if display_all_station_dataframe["Volume Aktual"].isna().all():
        display_all_station_dataframe = (
            display_all_station_dataframe.drop(
                columns=[
                    "Volume Aktual",
                    "Selisih Prediksi-Aktual",
                ]
            )
        )

    st.dataframe(
        display_all_station_dataframe,
        use_container_width=True,
        hide_index=True,
    )

    if all_station_errors:
        with st.expander(
            f"Lihat {len(all_station_errors)} stasiun yang gagal diproses"
        ):
            st.dataframe(
                pd.DataFrame(all_station_errors),
                use_container_width=True,
                hide_index=True,
            )

    st.stop()


# ------------------------------------------------------------------------------
# 16. MODE DETAIL SATU STASIUN
# ------------------------------------------------------------------------------
selected_station = selected_station_scope
database_station = station_lookup[selected_station]

single_process_start = time.perf_counter()

single_progress = st.progress(
    0,
    text=f"Membaca riwayat Stasiun {selected_station}...",
)

station_history_for_lag = load_station_history(
    database_path=str(DATABASE_PATH),
    database_modified_time=DATABASE_PATH.stat().st_mtime,
    database_station=database_station,
)

single_progress.progress(
    0.35,
    text="Menyiapkan nilai H-1 dan H-7...",
)

try:
    station_prediction_result = predict_station_for_date(
        station_history=station_history_for_lag,
        selected_station=selected_station,
        target_date=selected_date,
        model=rf_model,
        features=model_features,
    )
except Exception as error:
    station_prediction_result = {
        "error": str(error),
        "station": selected_station,
    }

single_progress.progress(
    0.85,
    text="Menyusun hasil prediksi dan visualisasi...",
)

if station_prediction_result["error"] is not None:
    single_progress.empty()
    st.error(
        station_prediction_result["error"]
    )
    st.stop()

input_lag_1 = station_prediction_result["lag_1"]
input_lag_7 = station_prediction_result["lag_7"]
lag_1_source = station_prediction_result["lag_1_source"]
lag_7_source = station_prediction_result["lag_7_source"]
projection_forecast_days = station_prediction_result["projection_days"]
predicted_values = station_prediction_result["predicted_values"]
predicted_volume = station_prediction_result["predicted_volume"]
input_dataframe = station_prediction_result["input_dataframe"]
station_history = station_history_for_lag

lag_1_date = (
    selected_date
    - datetime.timedelta(days=1)
)

lag_7_date = (
    selected_date
    - datetime.timedelta(days=7)
)

single_elapsed = (
    time.perf_counter()
    - single_process_start
)

single_progress.progress(
    1.0,
    text=(
        f"Prediksi {selected_station} selesai dalam "
        f"{format_elapsed_time(single_elapsed)}"
    ),
)

st.caption(
    f"Waktu pemrosesan: {format_elapsed_time(single_elapsed)}"
)

lag_col_1, lag_col_2 = st.columns(2)

with lag_col_1:
    st.markdown(
        f"""
<div class="auto-data-box">
    <div class="auto-data-label">Lag-1 (H-1)</div>
    <div class="auto-data-value">{format_integer_indonesia(input_lag_1)}</div>
    <div class="auto-data-date">{lag_1_date.strftime("%d/%m/%Y")}</div>
    <div class="auto-data-source">{lag_1_source}</div>
</div>
""",
        unsafe_allow_html=True,
    )

with lag_col_2:
    st.markdown(
        f"""
<div class="auto-data-box">
    <div class="auto-data-label">Lag-7 (H-7)</div>
    <div class="auto-data-value">{format_integer_indonesia(input_lag_7)}</div>
    <div class="auto-data-date">{lag_7_date.strftime("%d/%m/%Y")}</div>
    <div class="auto-data-source">{lag_7_source}</div>
</div>
""",
        unsafe_allow_html=True,
    )

if projection_forecast_days > 0:
    st.warning(
        f"Tanggal yang dipilih berada setelah data aktual terakhir. "
        f"Sistem membentuk {projection_forecast_days:,} prediksi harian "
        "secara bertahap untuk memperoleh nilai H-1 dan H-7. "
        "Semakin jauh horizon prediksi, ketidakpastian dapat meningkat."
    )
else:
    st.caption(
        "Nilai H-1 dan H-7 berasal dari data aktual database berdasarkan "
        "stasiun dan tanggal prediksi yang dipilih."
    )

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

density_status = station_prediction_result["density_status"]

recommendations = recommendation_items(
    density_status["label"],
    percentage_lag1,
)

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
    "#B91C1C"
    if delta_lag7 > 0
    else "#A16207"
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

render_dashboard_urgency_legend()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(
        f"""
<div class="kpi-card-box {density_status['border_class']}">
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
# 17. GRAFIK TUJUH HARI DAN REKOMENDASI
# ------------------------------------------------------------------------------
col_chart, col_right = st.columns([2.2, 1])

history_start_timestamp = (
    pd.Timestamp(selected_date)
    - pd.Timedelta(days=7)
)

history_end_timestamp = pd.Timestamp(selected_date)

recent_actual_history = station_history[
    (station_history["tanggal"] >= history_start_timestamp)
    & (station_history["tanggal"] < history_end_timestamp)
].copy()

recent_predicted_rows = [
    {
        "tanggal": pd.Timestamp(date_value),
        "volume_penumpang": volume_value,
    }
    for date_value, volume_value in predicted_values.items()
    if (
        history_start_timestamp.date()
        <= date_value
        < selected_date
    )
]

recent_predicted_history = pd.DataFrame(
    recent_predicted_rows
)

figure = go.Figure()

if not recent_actual_history.empty:
    figure.add_trace(
        go.Scatter(
            x=recent_actual_history["tanggal"],
            y=recent_actual_history["volume_penumpang"],
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

if not recent_predicted_history.empty:
    figure.add_trace(
        go.Scatter(
            x=recent_predicted_history["tanggal"],
            y=recent_predicted_history["volume_penumpang"],
            mode="lines+markers",
            name="Prediksi Bertahap",
            line=dict(
                color="#F97316",
                width=2,
                dash="dash",
            ),
            marker=dict(
                size=6,
                color="#F97316",
            ),
        )
    )

figure.add_trace(
    go.Scatter(
        x=[pd.Timestamp(selected_date)],
        y=[predicted_volume],
        mode="markers",
        name="Prediksi Tanggal Terpilih",
        marker=dict(
            size=12,
            color=DASHBOARD_DENSITY_COLORS.get(
                density_status["label"],
                "#CBD5E1",
            ),
            line=dict(
                width=2,
                color=density_status["color"],
            ),
        ),
    )
)

figure.update_layout(
    font=dict(
        color="#0F172A",
    ),
    xaxis=dict(
        showgrid=False,
        tickfont=dict(
            size=11,
            color="#000000",
        ),
        tickformat="%d/%m",
        ticklabelposition="outside bottom",
        automargin=True,
        showticklabels=True,
        ticks="outside",
        ticklen=5,
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="#F1F5F9",
        tickfont=dict(
            size=11,
            color="#0F172A",
        ),
        tickformat=",",
    ),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    margin=dict(
        l=55,
        r=20,
        t=45,
        b=65,
    ),
    height=340,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
        font=dict(
            color="#000000",
        ),
    ),
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#CBD5E1",
        font=dict(
            color="#000000",
            size=12,
        ),
    ),
    hovermode="x unified",
)

with col_chart:
    with st.container(border=True):
        st.markdown(
            """
<h3 style="margin: 0; font-size: 16px; font-weight: 700; color: #0F172A;">
    Riwayat 7 Hari dan Estimasi Tanggal Terpilih
</h3>
<p style="margin: 2px 0 0 0; font-size: 12px; color: #64748B;">
    Garis abu-abu menunjukkan data aktual, garis oranye menunjukkan prediksi bertahap, dan titik berwarna mengikuti tingkat kepadatan pada tanggal terpilih.
</p>
""",
            unsafe_allow_html=True,
        )

        st.plotly_chart(
            figure,
            use_container_width=True,
        )

with col_right:
    recommendation_list_html = "".join(
        f"<li style='margin-bottom: 12px;'>{item}</li>"
        for item in recommendations
    )

    st.markdown(
        f"""
<div class="recom-card dashboard-recom-card">
    <div style="display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 15px; color: #0F172A; margin-bottom: 18px;">
        <span class="material-symbols-outlined" style="color: #0F172A;">lightbulb</span>
        Rekomendasi Operasional
    </div>
    <ul style="padding-left: 20px; font-size: 12px; color: #334155; line-height: 1.7;">
        {recommendation_list_html}
    </ul>
</div>
""",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------------------
# 18. DETAIL INPUT MODEL
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
                "Sumber H-1",
                "Tanggal H-7",
                "Volume H-7",
                "Sumber H-7",
                "Hari ke-",
                "Bulan",
                "Weekend",
                "Jumlah hari proyeksi",
                "Hasil prediksi",
            ],
            "Nilai": [
                selected_station,
                selected_date.strftime("%d/%m/%Y"),
                lag_1_date.strftime("%d/%m/%Y"),
                format_integer_indonesia(input_lag_1),
                lag_1_source,
                lag_7_date.strftime("%d/%m/%Y"),
                format_integer_indonesia(input_lag_7),
                lag_7_source,
                day_of_week,
                selected_date.month,
                "Ya" if is_weekend else "Tidak",
                projection_forecast_days,
                format_integer_indonesia(predicted_volume),
            ],
        }
    )

    st.table(
        detail_data.set_index(detail_data.columns[0])
    )

    st.caption(
        
        "Data aktual digunakan selama tersedia. Untuk tanggal setelah data "
        "aktual terakhir, H-1 dan H-7 dapat berasal dari prediksi bertahap."
    )
