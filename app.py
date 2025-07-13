# Nama  : Muhammad Daffa Alfharijy
# NIM   : 22146023

import streamlit as st

st.set_page_config(page_title="UAS Machine Learning", layout="wide")
st.sidebar.title("Menu")
option = st.sidebar.selectbox(
    "Pilih Fitur",
    ("Clustering", "Klasifikasi")
)

if option == "Clustering":
    st.markdown("## Clustering Lokasi Gerai Kopi")
    st.info("Nama: **Muhammad Daffa Alfharijy**  \nNIM: **22146023**")
    st.markdown("""
    ### Deskripsi Proyek
    Segmentasi lokasi gerai kopi menggunakan algoritma K-Means untuk menentukan cluster optimal.
    """)

    import pandas as pd
    import pickle
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.preprocessing import StandardScaler

    # Load data dan model
    df = pd.read_csv("lokasi_gerai_kopi_clean.csv")
    features = ['x', 'y', 'population_density', 'traffic_flow', 'competitor_count', 'is_commercial']
    X = df[features]

    # Load scaler dan model
    with open("kmeans_model.pkl", "rb") as f:
        kmeans = pickle.load(f)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Prediksi cluster pada data testing
    cluster_labels = kmeans.predict(X_scaled)
    df['Cluster'] = cluster_labels

    # Visualisasi hasil clustering (menggunakan dua fitur utama: x dan y)
    st.markdown("### Visualisasi Hasil Clustering Pada Data Testing")
    fig, ax = plt.subplots(figsize=(6,4))  # ukuran lebih kecil
    sns.scatterplot(x='x', y='y', hue='Cluster', data=df, palette='Set1', ax=ax)
    ax.set_title('Clustering Lokasi Gerai Kopi')
    st.pyplot(fig)

    # Input Data Baru
    st.markdown("### Input Data Baru")
    input_data = []
    col1, col2 = st.columns(2)
    for idx, col in enumerate(features):
        if idx % 2 == 0:
            val = col1.number_input(col, value=0.0)
        else:
            val = col2.number_input(col, value=0.0)
        input_data.append(val)

    if st.button("Prediksi Cluster"):
        # Scaling input data baru
        input_scaled = scaler.transform([input_data])
        cluster_pred = kmeans.predict(input_scaled)[0]
        st.success(f"Data baru masuk ke **Cluster: {cluster_pred}**")

elif option == "Klasifikasi":
    st.markdown("## Klasifikasi Diabetes")
    st.info("Nama: **Muhammad Daffa Alfharijy**  \nNIM: **22146023**")
    st.markdown("""
    ### Deskripsi Proyek
    Prediksi apakah seseorang menderita diabetes berdasarkan data kesehatan menggunakan model klasifikasi.
    """)

    import pandas as pd
    import pickle
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Load data dan model
    df = pd.read_csv("diabetes.csv")
    with open("model_klasifikasi.pkl", "rb") as f:
        model = pickle.load(f)

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]
    y_pred = model.predict(X)

    # Metrik Klasifikasi
    st.markdown("### Metrik Klasifikasi")
    report = classification_report(y, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

    # Confusion Matrix
    st.markdown("### Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5,3))  # ukuran lebih kecil
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    st.pyplot(fig)

    # Input Data Baru
    st.markdown("### Input Data Baru")
    cols = X.columns.tolist()
    input_data = []
    col1, col2 = st.columns(2)
    for idx, col in enumerate(cols):
        if idx % 2 == 0:
            val = col1.number_input(col, value=0.0)
        else:
            val = col2.number_input(col, value=0.0)
        input_data.append(val)

    if st.button("Prediksi Diabetes"):
        new_df = pd.DataFrame([input_data], columns=cols)
        pred = model.predict(new_df)[0]
        hasil = "Diabetes" if pred == 1 else "Tidak Diabetes"
        st.success(f"Hasil Prediksi: **{hasil}**")