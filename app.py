import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === 1. Load Model, Encoder, Scaler ===
model = joblib.load('model_terbaik.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

# === 2. Deskripsi Project ===
st.title("Customer Segmentation Classification")
st.write("""
## Deskripsi Proyek
Sebuah perusahaan otomotif ingin mengklasifikasikan calon pelanggan baru ke dalam 4 segmen: A, B, C, D. 
Model klasifikasi ini dilatih dari data pelanggan eksisting yang telah dikelompokkan sebelumnya oleh tim sales.
""")

# === 3. Data Training Overview ===
st.subheader("Contoh Data Training & Penjelasan Fitur")

# Url Github
train_url = "https://github.com/wilywho/customer-segmentation-multiclass-classification/blob/main/Train.csv"
df_train = pd.read_csv(train_url)
st.dataframe(df_train.head())

with st.expander("Penjelasan Fitur"):
    st.markdown("""
    - `Gender`: Jenis kelamin pelanggan  
    - `Ever_Married`: Status pernikahan  
    - `Age`: Usia pelanggan  
    - `Graduated`: Status pendidikan  
    - `Profession`: Profesi pelanggan  
    - `Work_Experience`: Pengalaman kerja dalam tahun  
    - `Spending_Score`: Skor pengeluaran dari perusahaan  
    - `Family_Size`: Jumlah anggota keluarga  
    - `Segmentation`: Target kelas (A, B, C, D)
    """)

# === 4. Visualisasi Data Umum ===
st.subheader("Visualisasi Umum Data")
fig, ax = plt.subplots()
sns.countplot(data=df_train, x='Segmentation', palette='pastel', ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(df_train['Age'], kde=True, bins=20, color='skyblue', ax=ax2)
ax2.set_title("Distribusi Usia")
st.pyplot(fig2)

# === 5. Penjelasan Model ===
st.subheader("Model Klasifikasi yang Digunakan")
st.write("Model klasifikasi yang digunakan dalam proyek ini adalah: **(tulis nama model di sini)**.")
st.write("Model ini dipilih karena memberikan performa terbaik berdasarkan uji evaluasi dan cross-validation.")

st.subheader("Pilih Metode Input Data")
input_mode = st.radio("Pilih metode input:", ("File Testing dari GitHub", "Input Manual"))

if input_mode == "File Testing dari GitHub":
    # Url Github
    test_url = "https://github.com/wilywho/customer-segmentation-multiclass-classification/blob/main/Test.csv"

    try:
        df_test = pd.read_csv(test_url)
        st.success("File test.csv berhasil diambil dari GitHub!")
        st.dataframe(df_test.head())

        # Encoding per kolom
        for col in encoder:
            if col in df_test.columns:
                df_test[col] = encoder[col].transform(df_test[col])
            else:
                st.warning(f"Kolom '{col}' tidak ditemukan di data test.")

        # Scaling jika ada
        if scaler is not None:
            df_scaled = scaler.transform(df_test)
        else:
            df_scaled = df_test

        # Prediksi
        y_pred = model.predict(df_scaled)
        df_test['Predicted_Segment'] = y_pred

        st.subheader("Hasil Prediksi")
        st.dataframe(df_test)

        # Evaluasi jika tersedia label asli
        if 'Segmentation' in df_test.columns:
            acc = accuracy_score(df_test['Segmentation'], y_pred)
            st.metric("Akurasi Model", f"{acc:.2%}")

            st.subheader("Classification Report")
            report = classification_report(df_test['Segmentation'], y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(df_test['Segmentation'], y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=np.unique(y_pred),
                        yticklabels=np.unique(y_pred), ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        csv_download = df_test.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh Hasil Prediksi", data=csv_download,
                           file_name="hasil_prediksi.csv", mime='text/csv')

    except Exception as e:
        st.error(f"Gagal membaca test.csv dari GitHub: {e}")

elif input_mode == "Input Manual":
    st.write("Masukkan data pelanggan secara manual:")

    gender = st.selectbox("Gender", ["Male", "Female"])
    ever_married = st.selectbox("Pernah Menikah?", ["Yes", "No"])
    graduated = st.selectbox("Lulusan Perguruan Tinggi?", ["Yes", "No"])
    profession = st.selectbox("Profesi", [
        "Healthcare", "Engineer", "Lawyer", "Marketing", "Executive",
        "Artist", "Doctor", "Entertainment", "Homemaker"
    ])
    spending_score = st.selectbox("Skor Pengeluaran", ["Low", "Average", "High"])
    var_1 = st.selectbox("Var_1", ["Cat_1", "Cat_2", "Cat_3", "Cat_4", "Cat_5", "Cat_6"])

    if st.button("Prediksi Segmentasi"):
        df_input = pd.DataFrame({
            "Gender": [gender],
            "Ever_Married": [ever_married],
            "Graduated": [graduated],
            "Profession": [profession],
            "Spending_Score": [spending_score],
            "Var_1": [var_1]
        })

        try:
            for col in encoder:
                if col in df_input.columns:
                    df_input[col] = encoder[col].transform(df_input[col])
                else:
                    st.warning(f"Kolom '{col}' tidak ditemukan di input manual.")

            if scaler is not None:
                df_scaled = scaler.transform(df_input)
            else:
                df_scaled = df_input

            prediction = model.predict(df_scaled)[0]
            st.success(f"Segmentasi Pelanggan yang Diprediksi: **{prediction}**")

            df_result = df_input.copy()
            df_result['Segment_Prediction'] = prediction

            csv_result = df_result.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Unduh Hasil Prediksi",
                data=csv_result,
                file_name='hasil_prediksi_manual.csv',
                mime='text/csv')

        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi input manual: {e}")
