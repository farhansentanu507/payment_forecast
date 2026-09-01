from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Prediksi Pembayaran Klaim",
    page_icon="📈",
    layout="wide",
)


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    BASE_DIR
    / "best_model_pipeline.pkl"
)

METADATA_FILE = (
    BASE_DIR
    / "model_metadata.pkl"
)


# ============================================================
# VALIDASI FILE
# ============================================================

if not MODEL_FILE.exists():
    st.error(
        f"File model tidak ditemukan: {MODEL_FILE}"
    )
    st.stop()


if not METADATA_FILE.exists():
    st.error(
        f"File metadata tidak ditemukan: {METADATA_FILE}"
    )
    st.stop()


# ============================================================
# LOAD MODEL & METADATA
# ============================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(
        MODEL_FILE
    )

    metadata = joblib.load(
        METADATA_FILE
    )

    return model, metadata


model, metadata = load_artifacts()


# ============================================================
# READ METADATA
# ============================================================

numerical_cols = (
    metadata[
        "numerical_cols"
    ]
)

categorical_cols = (
    metadata[
        "categorical_cols"
    ]
)

category_values = (
    metadata[
        "category_values"
    ]
)

target_col = (
    metadata[
        "target_col"
    ]
)

feature_names = (
    metadata.get(
        "feature_names",
        []
    )
)


# ============================================================
# HEADER
# ============================================================

st.title(
    "Prediksi Total Pembayaran Klaim Bulan Depan"
)

st.markdown(
    f"""
    Aplikasi ini memprediksi nilai **{target_col}**
    berdasarkan variabel historis dan karakteristik
    pembayaran, klaim, korban, kecelakaan,
    kendaraan, dan wilayah.
    """
)


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander(
    "Model Information"
):

    try:

        model_name = (
            type(
                model.named_steps[
                    "model"
                ]
            ).__name__
        )

    except Exception:

        model_name = (
            type(model).__name__
        )

    st.write(
        "Model:",
        model_name
    )

    st.write(
        "Jumlah numerical features:",
        len(
            numerical_cols
        )
    )

    st.write(
        "Jumlah categorical features:",
        len(
            categorical_cols
        )
    )

    st.write(
        "Target:",
        target_col
    )


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header(
    "Input Fitur"
)

input_data = {}


# ============================================================
# NUMERICAL INPUT
# ============================================================

st.sidebar.subheader(
    "Variabel Numerik"
)

for col in numerical_cols:

    input_data[
        col
    ] = (
        st.sidebar.number_input(
            label=col,
            value=0.0,
            format="%.4f",
        )
    )


# ============================================================
# CATEGORICAL INPUT
# ============================================================

st.sidebar.subheader(
    "Variabel Kategorikal"
)

for col in categorical_cols:

    options = (
        category_values
        .get(
            col,
            []
        )
    )

    if len(options) > 0:

        input_data[
            col
        ] = (
            st.sidebar.selectbox(
                label=col,
                options=options,
            )
        )

    else:

        input_data[
            col
        ] = (
            st.sidebar.text_input(
                label=col,
                value="",
            )
        )


# ============================================================
# INPUT DATAFRAME
# ============================================================

input_df = pd.DataFrame(
    [
        input_data
    ]
)


# ============================================================
# RINGKASAN INPUT
# ============================================================

st.subheader(
    "Ringkasan Input"
)

st.dataframe(
    input_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# PREDICTION
# ============================================================

if st.sidebar.button(
    "Jalankan Prediksi",
    use_container_width=True,
):

    try:

        prediction = (
            model
            .predict(
                input_df
            )[0]
        )

        # Forecast pembayaran tidak boleh negatif
        prediction = max(
            float(
                prediction
            ),
            0.0,
        )

        st.subheader(
            "Hasil Prediksi"
        )

        st.metric(
            label=(
                "Prediksi Total Pembayaran "
                "Bulan Depan"
            ),
            value=(
                f"Rp "
                f"{prediction:,.0f}"
                .replace(
                    ",",
                    "."
                )
            ),
        )

    except Exception as e:

        st.error(
            "Terjadi error saat prediksi: "
            f"{e}"
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Feature Importance"
)

try:

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    if hasattr(
        estimator,
        "feature_importances_"
    ):

        importances = (
            estimator
            .feature_importances_
        )

        if (
            len(feature_names) > 0
            and
            len(importances)
            == len(feature_names)
        ):

            fi_df = (
                pd.DataFrame({
                    "feature":
                        feature_names,

                    "importance":
                        importances,
                })
                .sort_values(
                    "importance",
                    ascending=False
                )
                .head(20)
            )

            st.bar_chart(
                fi_df
                .set_index(
                    "feature"
                )
            )

            st.dataframe(
                fi_df,
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.info(
                "Feature importance tersedia, "
                "tetapi feature_names pada metadata "
                "tidak sesuai dengan jumlah feature model."
            )

    else:

        st.info(
            "Model tidak memiliki native "
            "feature importance."
        )

except Exception as e:

    st.warning(
        "Feature importance "
        "tidak tersedia: "
        f"{e}"
    )
