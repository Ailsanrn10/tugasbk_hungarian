import itertools
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score
import streamlit as st
import time
import pickle

# Menampilkan accuracy dan nilai min max

model = pickle.load(open("model_rf1.pickle", "rb"))
model_info = pickle.load(open("model_info.pickle", "rb"))

accuracy = model_info["accuracy"]
df_final = model_info["df4MinMax"]

scaler_ = pickle.load(open("scaler.pkl", "rb"))

# ===================================================================== #

# STREAMLIT
st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon=":ekg:")

st.title("Prediksi Penyakit Jantung: Hungarian Dataset")
st.write(f"**Akurasi Model** :  :sacramento[**{accuracy}**]%")
st.write("")

tab1, tab2 = st.tabs(["Prediksi Individu", "Prediksi Massal"])

with tab1:
    st.sidebar.header("**Data Masukan** Sidebar")

    age = st.sidebar.number_input(
        label=":magenta[**Usia**]",
        min_value=df_final["age"].min(),
        max_value=df_final["age"].max(),
    )
    st.sidebar.write(
        f"Nilai :yellow[Minimum]: :yellow[**{df_final['age'].min()}**], Nilai :red[Maximum]: :red[**{df_final['age'].max()}**]"
    )
    st.sidebar.write("")

    sex_sb = st.sidebar.selectbox(
        label=":magenta[**Jenis Kelamin**]", options=["Pria", "Wanita"]
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if sex_sb == "Pria":
        sex = 1
    elif sex_sb == "Wanita":
        sex = 0
    # -- Nilai 0: Wanita
    # -- Nilai 1: Pria

    cp_sb = st.sidebar.selectbox(
        label=":magenta[**Tipe Nyeri Dada**]",
        options=[
            "Typical angina",
            "Atypical angina",
            "Non-anginal pain",
            "Asymptomatic",
        ],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if cp_sb == "Typical angina":
        cp = 1
    elif cp_sb == "Atypical angina":
        cp = 2
    elif cp_sb == "Non-anginal pain":
        cp = 3
    elif cp_sb == "Asymptomatic":
        cp = 4
    # -- Nilai 1: typical angina
    # -- Nilai 2: atypical angina
    # -- Nilai 3: non-anginal pain
    # -- Nilai 4: asymptomatic

    trestbps = st.sidebar.number_input(
        label=":magenta[**Tekanan Darah saat Istirahat** (dalam mm Hg saat pertama masuk ke RS)]",
        min_value=df_final["trestbps"].min(),
        max_value=df_final["trestbps"].max(),
    )
    st.sidebar.write(
        f"Nilai :yellow[Minimum]: :yellow[**{df_final['trestbps'].min()}**], Nilai :red[Maximum]: :red[**{df_final['trestbps'].max()}**]"
    )
    st.sidebar.write("")

    chol = st.sidebar.number_input(
        label=":magenta[**Jumlah Kolesterol dalam Darah** (dalam mg/dl)]",
        min_value=df_final["chol"].min(),
        max_value=df_final["chol"].max(),
    )
    st.sidebar.write(
        f"Nilai :yellow[Minimum]: :yellow[**{df_final['chol'].min()}**], Nilai :red[Maximum]: :red[**{df_final['chol'].max()}**]"
    )
    st.sidebar.write("")

    fbs_sb = st.sidebar.selectbox(
        label=":magenta[**Kadar Gula Darah Puasa > 120 mg/dl?**]",
        options=["False", "True"],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if fbs_sb == "False":
        fbs = 0
    elif fbs_sb == "True":
        fbs = 1
    # -- Value 0: false
    # -- Value 1: true

    restecg_sb = st.sidebar.selectbox(
        label=":magenta[**Hasil EKG Istirahat**]",
        options=[
            "Normal",
            "Mengalami kelainan gelombang ST-T",
            "Menunjukkan hipertrofi ventrikel kiri",
        ],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if restecg_sb == "Normal":
        restecg = 0
    elif restecg_sb == "Mengalami kelainan gelombang ST-T":
        restecg = 1
    elif restecg_sb == "Menunjukkan hipertrofi ventrikel kiri":
        restecg = 2
    # -- Nilai 0: normal
    # -- Nilai 1: mengalami kelainan gelombang ST-T (Inversi gelombang T > 0.05 mV)
    # -- Nilai 2: menunjukkan hipertrofi ventrikel kiri

    thalach = st.sidebar.number_input(
        label=":magenta[**Denyut Jantung Maksimum**]",
        min_value=df_final["thalach"].min(),
        max_value=df_final["thalach"].max(),
    )
    st.sidebar.write(
        f"Nilai :yellow[Minimum]: :yellow[**{df_final['thalach'].min()}**], Nilai :red[Maximum]: :red[**{df_final['thalach'].max()}**]"
    )
    st.sidebar.write("")

    exang_sb = st.sidebar.selectbox(
        label=":magenta[**Nyeri Dada saat Beraktivitas Fisik?**]",
        options=["Tidak", "Iya"],
    )
    st.sidebar.write("")
    st.sidebar.write("")
    if exang_sb == "Tidak":
        exang = 0
    elif exang_sb == "Iya":
        exang = 1
    # -- Nilai 0: Tidak
    # -- Nilai 1: Iya

    oldpeak = st.sidebar.number_input(
        label=":magenta[**Penurunan Segmen ST pada EKG saat Latihan Fisik**]",
        min_value=df_final["oldpeak"].min(),
        max_value=df_final["oldpeak"].max(),
    )
    st.sidebar.write(
        f"Nilai :yellow[Minimum]: :yellow[**{df_final['oldpeak'].min()}**], Nilai :red[Maximum]: :red[**{df_final['oldpeak'].max()}**]"
    )
    st.sidebar.write("")

    data = {
        "Age": age,
        "Sex": sex_sb,
        "Chest pain type": cp_sb,
        "RPB": f"{trestbps} mm Hg",
        "Serum Cholestoral": f"{chol} mg/dl",
        "FBS > 120 mg/dl?": fbs_sb,
        "Resting ECG": restecg_sb,
        "Maximum heart rate": thalach,
        "Exercise induced angina?": exang_sb,
        "ST depression": oldpeak,
    }

    preview_df = pd.DataFrame(data, index=["input"])

    st.header("Data Masukan sebagai DataFrame")
    st.write("")
    st.dataframe(preview_df.iloc[:, :6])
    st.write("")
    st.dataframe(preview_df.iloc[:, 6:])
    st.write("")

    result = ":magenta[-]"

    predict_btn = st.button("**Prediksi**", type="primary")

    st.write("")
    if predict_btn:
        inputs = [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak]]
        prediction = model.predict(inputs)[0]

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        if prediction == 0:
            result = ":green[**Sehat (Tidak Sakit Jantung)**]"
        elif prediction == 1:
            result = ":yellow[**Sakit Jantung level 1**]"
        elif prediction == 2:
            result = ":orange[**Sakit Jantung level 2**]"
        elif prediction == 3:
            result = ":red[**Sakit Jantung level 3**]"
        elif prediction == 4:
            result = ":darkred[**Sakit Jantung level 4**]"

    st.write("")
    st.write("")
    st.subheader("Hasil Prediksi:")
    st.subheader(result)

with tab2:
    st.header("Memprediksi beberapa data:")

    sample_csv = df_final.iloc[:5, :-1].to_csv(index=False).encode("utf-8")

    st.write("")
    st.download_button(
        "Download Contoh CSV",
        data=sample_csv,
        file_name="sample_heart_disease_parameters.csv",
        mime="text/csv",
    )

    st.write("")
    st.write("")
    file_uploaded = st.file_uploader("Upload file CSV", type="csv")

    if file_uploaded:
        uploaded_df = pd.read_csv(file_uploaded)
        prediction_arr = model.predict(uploaded_df)

        bar = st.progress(0)
        status_text = st.empty()

        for i in range(1, 70):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)

        result_arr = []

        for prediction in prediction_arr:
            if prediction == 0:
                result = "Sehat (Tidak Sakit Jantung)"
            elif prediction == 1:
                result = "Sakit Jantung level 1"
            elif prediction == 2:
                result = "Sakit Jantung level 2"
            elif prediction == 3:
                result = "Sakit Jantung level 3"
            elif prediction == 4:
                result = "Sakit Jantung level 4"
            result_arr.append(result)

        uploaded_result = pd.DataFrame({"Hasil Prediksi": result_arr})

        for i in range(70, 101):
            status_text.text(f"{i}% complete")
            bar.progress(i)
            time.sleep(0.01)
            if i == 100:
                time.sleep(1)
                status_text.empty()
                bar.empty()

        col1, col2 = st.columns([1, 2])

        with col1:
            st.dataframe(uploaded_result)
        with col2:
            st.dataframe(uploaded_df)
