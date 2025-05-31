import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# === Load Model, Encoder, Scaler ===
model = joblib.load('LightGBM.pkl')
encoder = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# === Deskripsi Proyek ===
st.title("Customer Segmentation Classification")
st.write("""Author: Kelompok 4 DataBender's  
* Farhan Wily  
* Ghazy Shidqy  
* Naufal Hafizh Dhiya Ulhaq  
* Yosef Sony Koesprasetyo""")

st.write("""
## Deskripsi Proyek  
Sebuah perusahaan otomotif ingin mengklasifikasikan calon pelanggan baru ke dalam 4 segmen: A, B, C, D.  
Model klasifikasi ini dilatih dari data pelanggan eksisting yang telah dikelompokkan sebelumnya oleh tim sales.
""")

# === Tampilkan Data Training & Penjelasan Fitur ===
st.subheader("Data Training & Penjelasan Fitur")
train_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Train.csv"
df_train = pd.read_csv(train_url)
st.dataframe(df_train.head())

with st.expander("Penjelasan Fitur"):
    st.markdown("""
    - Gender: Jenis kelamin pelanggan  
    - Ever_Married: Status pernikahan  
    - Age: Usia pelanggan  
    - Graduated: Status pendidikan  
    - Profession: Profesi pelanggan  
    - Work_Experience: Pengalaman kerja dalam tahun  
    - Spending_Score: Skor pengeluaran dari perusahaan  
    - Family_Size: Jumlah anggota keluarga  
    - Segmentation: Target kelas (A, B, C, D)
    """)

# === Visualisasi Data Training ===
st.subheader("Visualisasi Umum Data Training")
fig, ax = plt.subplots()
sns.countplot(data=df_train, x='Segmentation', palette='pastel', ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(df_train['Age'], kde=True, bins=20, color='skyblue', ax=ax2)
ax2.set_title("Distribusi Usia")
st.pyplot(fig2)

# === Penjelasan Model ===
st.subheader("Model Klasifikasi yang Digunakan")
st.write("Model klasifikasi yang digunakan dalam proyek ini adalah: **LightGBM**.")
st.write("Model ini dipilih karena memberikan performa terbaik berdasarkan uji evaluasi dan cross-validation.")

# === Pilih Opsi Input Data ===
st.subheader("Input Data Testing untuk Prediksi")
input_option = st.radio("Pilih metode input data:", ["Gunakan data dari GitHub", "Upload file CSV sendiri"])

# === Siapkan Data Testing ===
df_test = None
df_test_original = None

if input_option == "Gunakan data dari GitHub":
    test_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test.csv"
    test_enc_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test_encoding.csv"

    try:
        df_test_original = pd.read_csv(test_url)
        df_test = pd.read_csv(test_enc_url)
        st.success("Berhasil memuat data dari GitHub")
        st.dataframe(df_test_original.head())
    except Exception as e:
        st.error(f"Gagal membaca data dari GitHub: {e}")

else:
    uploaded_file = st.file_uploader("Unggah file CSV data testing", type=["csv"])
    if uploaded_file is not None:
        try:
            df_test = pd.read_csv(uploaded_file)
            df_test_original = df_test.copy()
            st.success("File berhasil diunggah")
            st.dataframe(df_test.head())
        except Exception as e:
            st.error(f"Gagal membaca file CSV: {e}")

# === Lakukan Prediksi jika Data Tersedia ===
if df_test is not None:
    try:
        # Drop kolom tidak perlu
        df_model = df_test.drop(columns=[col for col in ['ID', 'Segmentation'] if col in df_test.columns])

        # Scaling
        if scaler is not None:
            df_scaled = scaler.transform(df_model)
        else:
            df_scaled = df_model

        # Prediksi
        y_pred = model.predict(df_scaled)
        df_test['Predicted_Segment_Num'] = y_pred

        # Mapping angka ke huruf
        inv_seg_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        df_test['Predicted_Segment'] = df_test['Predicted_Segment_Num'].map(inv_seg_map)

        # Tampilkan hasil prediksi
        st.subheader("Hasil Prediksi Segmentasi")
        st.dataframe(df_test[['Predicted_Segment']])

        # Tambahkan evaluasi jika ada label asli
        if 'Segmentation' in df_test_original.columns:
            df_test['Original_Segment'] = df_test_original['Segmentation'].values
            acc = accuracy_score(df_test['Original_Segment'], df_test['Predicted_Segment'])
            st.subheader("Akurasi Prediksi")
            st.write(f"Akurasi model terhadap data input: **{acc:.2%}**")
            st.dataframe(df_test[['Original_Segment', 'Predicted_Segment']])
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")
