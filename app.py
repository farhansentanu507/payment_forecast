import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Pembayaran Klaim", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model_pipeline_final.pkl')
    metadata = joblib.load('model_metadata_final.pkl')
    return model, metadata

model, metadata = load_artifacts()
numerical_cols = metadata['numerical_cols']
editable_numerical_cols = metadata['editable_numerical_cols']
fixed_numerical_defaults = metadata['fixed_numerical_defaults']
categorical_cols = metadata['categorical_cols']
category_values = metadata['category_values']
target_col = metadata['target_col']

st.title("Prediksi Total Pembayaran Klaim Bulan Depan")
st.markdown(
    "Aplikasi ini memprediksi nilai `%s` berdasarkan variabel `payments_lag1`, "
    "`total_pembayaran_lag1`, `jumlah_klaim_median`, serta karakteristik klaim." % target_col
)

# --- Sidebar input ---
st.sidebar.header("Input Fitur")

input_data = {}

st.sidebar.subheader("Variabel Numerik")
for col in editable_numerical_cols:
    input_data[col] = st.sidebar.number_input(
        label=col, value=0.0, format="%.4f"
    )

st.sidebar.subheader("Variabel Kategorikal")
for col in categorical_cols:
    options = category_values.get(col, [])
    input_data[col] = st.sidebar.selectbox(label=col, options=options)

# Lengkapi kolom numerik yang tidak ditampilkan dengan nilai fixed (median)
for col, default_val in fixed_numerical_defaults.items():
    input_data[col] = default_val

# Susun ulang kolom sesuai urutan asli saat training
input_df = pd.DataFrame([input_data])[numerical_cols + categorical_cols]

st.subheader("Ringkasan Input")
st.dataframe(input_df)

with st.expander("Lihat nilai fixed (median) untuk variabel numerik lainnya"):
    st.json(fixed_numerical_defaults)

# --- Prediksi ---
if st.sidebar.button("Jalankan Prediksi"):
    try:
        prediction = model.predict(input_df)[0]
        st.subheader("Hasil Prediksi")
        st.metric(
            label=target_col,
            value=f"Rp {prediction:,.0f}".replace(",", ".")
        )
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {e}")
