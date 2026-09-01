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

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

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
        "File model tidak ditemukan."
    )

    st.write(
        "Path yang dicari:"
    )

    st.code(
        str(MODEL_FILE)
    )

    st.stop()


if not METADATA_FILE.exists():

    st.error(
        "File metadata tidak ditemukan."
    )

    st.write(
        "Path yang dicari:"
    )

    st.code(
        str(METADATA_FILE)
    )

    st.stop()


# ============================================================
# LOAD ARTIFACTS
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


try:

    model, metadata = (
        load_artifacts()
    )

except Exception as e:

    st.error(
        "Gagal memuat model atau metadata."
    )

    st.exception(
        e
    )

    st.stop()


# ============================================================
# VALIDASI METADATA
# ============================================================

required_metadata = [
    "numerical_cols",
    "categorical_cols",
    "category_values",
    "target_col",
]

missing_metadata = [
    key
    for key in required_metadata
    if key not in metadata
]

if missing_metadata:

    st.error(
        "Metadata tidak lengkap."
    )

    st.write(
        "Key yang tidak ditemukan:"
    )

    st.write(
        missing_metadata
    )

    st.stop()


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
# MODEL INFO
# ============================================================

try:

    estimator = (
        model.named_steps[
            "model"
        ]
    )

    model_name = (
        type(estimator)
        .__name__
    )

except Exception:

    estimator = model

    model_name = (
        type(model)
        .__name__
    )


# ============================================================
# HEADER
# ============================================================

st.title(
    "Prediksi Total Pembayaran Klaim Bulan Depan"
)

st.markdown(
    f"""
    Aplikasi ini memprediksi **{target_col}**
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

    col1, col2, col3, col4 = (
        st.columns(4)
    )

    col1.metric(
        "Model",
        model_name
    )

    col2.metric(
        "Numerical Features",
        len(
            numerical_cols
        )
    )

    col3.metric(
        "Categorical Features",
        len(
            categorical_cols
        )
    )

    col4.metric(
        "Total Features",
        (
            len(
                numerical_cols
            )
            +
            len(
                categorical_cols
            )
        )
    )

    st.write(
        "Target:"
    )

    st.code(
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
# CREATE INPUT DATAFRAME
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
            "Terjadi error saat prediksi."
        )

        st.exception(
            e
        )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader(
    "Feature Importance"
)

try:

    if hasattr(
        estimator,
        "feature_importances_"
    ):

        importances = (
            estimator
            .feature_importances_
        )

        if (
            len(
                feature_names
            ) > 0
            and
            len(
                importances
            )
            ==
            len(
                feature_names
            )
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
                "tetapi feature_names di metadata "
                "tidak sesuai dengan jumlah feature model."
            )

    else:

        st.info(
            "Model tidak memiliki "
            "native feature importance."
        )

except Exception as e:

    st.warning(
        "Feature importance "
        "tidak dapat ditampilkan."
    )

    st.exception(
        e
    )
