import streamlit as st


st.set_page_config(page_title="Prediksi Pembayaran Klaim", layout="wide")

@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model_pipeline.pkl')
    metadata = joblib.load('model_metadata.pkl')
    return model, metadata

model, metadata = load_artifacts()
numerical_cols = metadata['numerical_cols']
categorical_cols = metadata['categorical_cols']
category_values = metadata['category_values']
target_col = metadata['target_col']

st.title("Prediksi Total Pembayaran Klaim Bulan Depan")
st.markdown(
    "Aplikasi ini memprediksi nilai `%s` berdasarkan variabel historis "
    "dan karakteristik klaim." % target_col
)

# --- Sidebar input ---
st.sidebar.header("Input Fitur")

input_data = {}

st.sidebar.subheader("Variabel Numerik")
for col in numerical_cols:
    input_data[col] = st.sidebar.number_input(
        label=col, value=0.0, format="%.4f"
    )

st.sidebar.subheader("Variabel Kategorikal")
for col in categorical_cols:
    options = category_values.get(col, [])
    input_data[col] = st.sidebar.selectbox(label=col, options=options)

input_df = pd.DataFrame([input_data])

st.subheader("Ringkasan Input")
st.dataframe(input_df)

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

# --- Feature Importance ---
st.subheader("Feature Importance (Top 20)")
try:
    importances = model.named_steps['model'].feature_importances_
    fi_df = pd.DataFrame({
        'feature': metadata['feature_names'],
        'importance': importances
    }).sort_values('importance', ascending=False).head(20)
    st.bar_chart(fi_df.set_index('feature'))
except Exception as e:
    st.warning(f"Feature importance tidak tersedia: {e}")
