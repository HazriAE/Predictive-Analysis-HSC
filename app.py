"""
Aplikasi Prediksi Days Until Reorder
Model: XGBoost Regressor
Tujuan: Memprediksi jumlah hari tersisa sebelum pemesanan ulang (reorder)
        untuk barang habis pakai (consumable) medis non-obat di rumah sakit.
"""

import streamlit as st
import pandas as pd
import joblib
from datetime import date

# ============================================
# KONFIGURASI HALAMAN
# ============================================
st.set_page_config(
    page_title="Prediksi Reorder Point - Consumable RS",
    page_icon="🏥",
    layout="centered"
)

# ============================================
# LOAD MODEL (cache agar tidak load ulang setiap interaksi)
# ============================================
@st.cache_resource
def load_model():
    return joblib.load("xgb_model.pkl")

model = load_model()

# Urutan fitur ini WAJIB sama persis dengan saat training
FEATURE_ORDER = [
    'Current_Stock', 'Min_Required', 'Max_Capacity', 'Avg_Usage_Per_Day',
    'Restock_Lead_Time', 'Stock_to_Min_Ratio', 'Stock_to_Max_Ratio',
    'Usage_Intensity', 'Buffer_Days', 'Stock_Lead_Ratio',
    'Month', 'Day_of_Week', 'Is_Weekend', 'Quarter',
    'item_Gloves', 'item_Surgical Mask',
    'vendor_V001', 'vendor_V002', 'vendor_V003'
]

ITEM_OPTIONS = ["Gloves", "Surgical Mask"]
VENDOR_OPTIONS = ["V001", "V002", "V003"]

# ============================================
# HEADER
# ============================================
st.title("🏥 Prediksi Reorder Point")
st.markdown(
    "Aplikasi ini memprediksi **jumlah hari tersisa sebelum barang habis pakai medis "
    "perlu dipesan ulang** (*Days Until Reorder*), menggunakan model **XGBoost Regressor** "
    "(R² = 0,9960 pada data testing)."
)
st.divider()

# ============================================
# FORM INPUT
# ============================================
st.subheader("📋 Input Data Persediaan")

col1, col2 = st.columns(2)

with col1:
    item_name = st.selectbox("Nama Barang", ITEM_OPTIONS)
    vendor_id = st.selectbox("Vendor", VENDOR_OPTIONS)
    current_stock = st.number_input(
        "Current Stock (jumlah stok saat ini)",
        min_value=0, max_value=10000, value=1000, step=1
    )
    min_required = st.number_input(
        "Min Required (stok minimum)",
        min_value=1, max_value=5000, value=400, step=1
    )

with col2:
    max_capacity = st.number_input(
        "Max Capacity (kapasitas gudang maksimum)",
        min_value=1, max_value=10000, value=3000, step=1
    )
    avg_usage = st.number_input(
        "Avg Usage Per Day (rata-rata pemakaian/hari)",
        min_value=1, max_value=1000, value=250, step=1
    )
    lead_time = st.number_input(
        "Restock Lead Time (hari pengiriman ulang)",
        min_value=1, max_value=60, value=15, step=1
    )

st.caption(
    "💡 Nilai default di atas merepresentasikan rata-rata data historis yang digunakan saat training model."
)

predict_btn = st.button("🔍 Prediksi Sekarang", type="primary", use_container_width=True)

# ============================================
# VALIDASI SEDERHANA
# ============================================
def validate_input(current_stock, min_required, max_capacity):
    warnings = []
    if current_stock > max_capacity:
        warnings.append("⚠️ Current Stock melebihi Max Capacity — periksa kembali input Anda.")
    if min_required > max_capacity:
        warnings.append("⚠️ Min Required melebihi Max Capacity — periksa kembali input Anda.")
    return warnings

# ============================================
# FEATURE ENGINEERING (harus identik dengan proses training)
# ============================================
def build_feature_row(current_stock, min_required, max_capacity, avg_usage,
                       lead_time, item_name, vendor_id, input_date):
    stock_to_min_ratio = current_stock / max(min_required, 1)
    stock_to_max_ratio = current_stock / max(max_capacity, 1)
    usage_intensity = avg_usage / max(current_stock, 1)
    buffer_days = current_stock / max(avg_usage, 1)
    stock_lead_ratio = current_stock / max(lead_time, 1)

    month = input_date.month
    day_of_week = input_date.weekday()
    is_weekend = int(day_of_week >= 5)
    quarter = (month - 1) // 3 + 1

    row = {
        'Current_Stock': current_stock,
        'Min_Required': min_required,
        'Max_Capacity': max_capacity,
        'Avg_Usage_Per_Day': avg_usage,
        'Restock_Lead_Time': lead_time,
        'Stock_to_Min_Ratio': stock_to_min_ratio,
        'Stock_to_Max_Ratio': stock_to_max_ratio,
        'Usage_Intensity': usage_intensity,
        'Buffer_Days': buffer_days,
        'Stock_Lead_Ratio': stock_lead_ratio,
        'Month': month,
        'Day_of_Week': day_of_week,
        'Is_Weekend': is_weekend,
        'Quarter': quarter,
        'item_Gloves': 1 if item_name == "Gloves" else 0,
        'item_Surgical Mask': 1 if item_name == "Surgical Mask" else 0,
        'vendor_V001': 1 if vendor_id == "V001" else 0,
        'vendor_V002': 1 if vendor_id == "V002" else 0,
        'vendor_V003': 1 if vendor_id == "V003" else 0,
    }
    return pd.DataFrame([row])[FEATURE_ORDER]

# ============================================
# PREDIKSI & OUTPUT
# ============================================
if predict_btn:
    warnings = validate_input(current_stock, min_required, max_capacity)
    for w in warnings:
        st.warning(w)

    input_date = date.today()
    X_input = build_feature_row(
        current_stock, min_required, max_capacity, avg_usage,
        lead_time, item_name, vendor_id, input_date
    )

    prediction = model.predict(X_input)[0]
    prediction = max(prediction, 0)  # tidak boleh negatif

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    if prediction <= 3:
        status = "🔴 KRITIS — Segera Pesan Ulang"
        color = "red"
    elif prediction <= 7:
        status = "🟠 WASPADA — Rencanakan Pemesanan"
        color = "orange"
    else:
        status = "🟢 AMAN — Stok Masih Memadai"
        color = "green"

    st.metric(
        label="Perkiraan Hari Tersisa Sebelum Reorder",
        value=f"{prediction:.1f} hari"
    )
    st.markdown(f"**Status:** :{color}[{status}]")

    with st.expander("Lihat detail fitur yang digunakan untuk prediksi"):
        st.dataframe(X_input.T.rename(columns={0: "Nilai"}))

    st.caption(
        f"Prediksi dihitung berdasarkan tanggal {input_date.strftime('%d %B %Y')} "
        f"menggunakan model XGBoost (R² = 0,9960, MAE = 0,77 hari pada data testing)."
    )

# ============================================
# FOOTER
# ============================================
st.divider()
st.caption(
    "Model dikembangkan menggunakan metodologi CRISP-DM untuk mendukung "
    "Sistem Informasi Manajemen Rumah Sakit (SIMRS) dalam optimalisasi "
    "persediaan barang habis pakai medis."
)
