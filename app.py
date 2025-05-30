import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === 1. Load Model ===
model = joblib.load('model_terbaik.pkl')  # Ganti dengan nama file model kamu

# === 2. Deskripsi Project ===
st.title("Customer Segmentation Classification")
st.write("""
## Deskripsi Proyek
Sebuah perusahaan otomotif ingin mengklasifikasikan calon pelanggan baru ke dalam 4 segmen: A, B, C, D. 
Model klasifikasi ini dilatih dari data pelanggan eksisting yang telah dikelompokkan sebelumnya oleh tim sales.
""")

# === 3. Data Training Overview ===
st.subheader("Contoh Data Training & Penjelasan Fitur")
df_train = pd.read_csv("data_train.csv")  # Data yang digunakan untuk pelatihan
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
input_mode = st.radio("Pilih metode input:", ("Upload File CSV", "Input Manual"))

if input_mode == "Upload File CSV":
    uploaded_file = st.file_uploader("Unggah file CSV berisi data pelanggan baru", type=["csv"])
    
    if uploaded_file is not None:
        df_test = pd.read_csv(uploaded_file)
        st.success("File berhasil diunggah!")
        st.dataframe(df_test.head())

        # --- Prediksi ---
        df_test_pred = model.predict(df_test)
        df_test['Segment_Prediction'] = df_test_pred
        st.subheader("Hasil Prediksi")
        st.dataframe(df_test.head())

        # Unduh hasil
        csv = df_test.to_csv(index=False).encode('utf-8')
        st.download_button("Unduh Hasil Prediksi", csv, "prediksi_segmentasi.csv", "text/csv")

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

    # Buat DataFrame dari input
    if st.button("Prediksi Segmentasi"):
        df_input = pd.DataFrame({
            "Gender": [gender],
            "Ever_Married": [ever_married],
            "Graduated": [graduated],
            "Profession": [profession],
            "Spending_Score": [spending_score],
            "Var_1": [var_1]
        })

        # Jika kamu menggunakan pipeline yang sudah meng-handle preprocessing, langsung prediksi
        prediction = model.predict(df_input)[0]
        st.success(f"Segmentasi Pelanggan yang Diprediksi: **{prediction}**")

    y_pred = model.predict(df_input)
    df_test['Predicted_Segment'] = y_pred

    # === 8. Tampilkan Head Hasil Prediksi ===
    st.dataframe(df_test[['Predicted_Segment']].head())

    # === 9. Evaluasi Model (opsional untuk test set jika ada label)
    if 'Segmentation' in df_test.columns:
        acc = accuracy_score(df_test['Segmentation'], y_pred)
        st.metric("Akurasi Model", f"{acc:.2%}")

        st.subheader("Classification Report")
        report = classification_report(df_test['Segmentation'], y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

        st.subheader("Confusion Matrix")
        cm = confusion_matrix(df_test['Segmentation'], y_pred)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_pred), yticklabels=np.unique(y_pred), ax=ax_cm)
        ax_cm.set_xlabel("Predicted")
        ax_cm.set_ylabel("Actual")
        st.pyplot(fig_cm)

    # === 10. Unduh File Hasil Prediksi ===
    st.subheader("Unduh File Hasil Prediksi")
    csv_download = df_test.to_csv(index=False).encode('utf-8')
    st.download_button("Unduh Hasil", data=csv_download, file_name="hasil_prediksi.csv", mime='text/csv')