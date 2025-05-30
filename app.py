import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# === Load Model, Encoder, Scaler ===
model = joblib.load('LightGBM.pkl')
encoder = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# === Deskripsi Proyek ===
st.title("Customer Segmentation Classification")
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

# === Load dan Prediksi Data Testing dari GitHub ===
st.subheader("Preview Data Testing Asli (Test.csv)")

test_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test.csv"
try:
    df_test_original = pd.read_csv(test_url)
    st.dataframe(df_test_original.head())
except Exception as e:
    st.error(f"Gagal membaca test.csv dari GitHub: {e}")

st.subheader("Proses Prediksi Menggunakan Data Testing yang Sudah Encoding (Test_enc.csv)")

test_enc_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test_encoding.csv"
try:
    df_test = pd.read_csv(test_enc_url)
    # Drop kolom yang tidak digunakan model
    if 'ID' in df_test.columns:
        df_test = df_test.drop(columns=['ID'])
    if 'Segmentation' in df_test_enc.columns:
        df_test = df_test.drop(columns=['Segmentation'])

    # Scaling
    if scaler is not None:
        df_scaled = scaler.transform(df_test)
    else:
        df_scaled = df_test

    # Prediksi
    y_pred = model.predict(df_scaled)
    df_test['Predicted_Segment'] = y_pred

    # Tampilkan hasil prediksi
    st.subheader("Hasil Prediksi")
    st.dataframe(df_test)

except Exception as e:
    st.error(f"Gagal membaca atau memproses Test_enc.csv dari GitHub: {e}")
