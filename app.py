import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# KONFIGURASI APLIKASI
st.set_page_config(
    page_title="SPK Prioritas Penerima Bantuan",
    layout="wide"
)

st.title(" Sistem Pendukung Keputusan Prioritas Penerima Bantuan")
st.write("""
Aplikasi ini menggunakan **K-Means Clustering**
untuk menentukan **prioritas verifikasi penerima bantuan**.
""")

# UPLOAD DATA
uploaded_file = st.file_uploader(
    " Upload File Excel Data Penerima Bantuan",
    type=["xlsx"]
)

if uploaded_file:
    # LOAD & PREPROCESS DATA
    df = pd.read_excel(uploaded_file, sheet_name="Penerima Bantuan")

    df = df[
        (df["id"].notna()) &
        (df["nama"].notna()) &
        (df["id"] < 900)
    ].reset_index(drop=True)

    numeric_cols = df.columns[4:22]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    X = df[numeric_cols].values

    # SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-MEANS CLUSTERING
    kmeans = KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    )
    df["cluster"] = kmeans.fit_predict(X_scaled)

  
    # PRIORITAS
    df["skor_utama"] = df.iloc[:, 21]
    df["prioritas"] = df["cluster"].map({
        0: "Tinggi",
        1: "Sedang",
        2: "Rendah"
    })

    # TAMPIL DATA
    st.subheader("Hasil Clustering")
    st.dataframe(
        df[["id", "nama", "kecamatan", "skor_utama", "prioritas"]],
        use_container_width=True
    )

    # VISUALISASI
    st.subheader("Visualisasi")

    # --- HEATMAP ---
    pivot = df.pivot_table(
            values="id",
            index="kecamatan",
            columns="prioritas",
            aggfunc="count",
            fill_value=0
        )

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlOrRd", ax=ax2)
    ax2.set_title("Jumlah Penerima per Prioritas")
    st.pyplot(fig2)


    # RINGKASAN CLUSTER
    st.subheader("Ringkasan Cluster")

    summary = df.groupby(
        ["cluster", "prioritas"]
    ).agg(
        jumlah=("id", "count"),
        rata_skor=("skor_utama", "mean"),
        skor_min=("skor_utama", "min"),
        skor_max=("skor_utama", "max")
    ).round(2)

    st.dataframe(summary, use_container_width=True)

else:
    st.info("Silakan upload file Excel terlebih dahulu")
