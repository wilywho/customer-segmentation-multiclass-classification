import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report

# === Load Model, Encoder, Scaler ===
model = joblib.load('LightGBM.pkl')
encoder = joblib.load('encoders.pkl')
scaler = joblib.load('scaler.pkl')

# === Deskripsi Proyek ===
st.title("Customer Segmentation Classification")
st.write("""Author: Kelompok 4 DataBender's
* Nama :
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
ax.set_title("Distribusi Segmentation")
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(df_train['Age'], kde=True, bins=20, color='skyblue', ax=ax2)
ax2.set_title("Distribusi Usia")
st.pyplot(fig2)

# Tambahan visualisasi kolom fitur lain supaya lebih ramai
fig3, axs = plt.subplots(2, 2, figsize=(12, 8))
sns.countplot(data=df_train, x='Gender', palette='Set2', ax=axs[0,0])
axs[0,0].set_title("Distribusi Gender")

sns.countplot(data=df_train, x='Ever_Married', palette='Set3', ax=axs[0,1])
axs[0,1].set_title("Distribusi Ever Married")

sns.countplot(data=df_train, x='Graduated', palette='pastel', ax=axs[1,0])
axs[1,0].set_title("Distribusi Graduated")

sns.countplot(data=df_train, x='Spending_Score', palette='bright', ax=axs[1,1])
axs[1,1].set_title("Distribusi Spending Score")

plt.tight_layout()
st.pyplot(fig3)

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

st.subheader("Proses Prediksi Menggunakan Data Testing yang Sudah Encoding (Test_encoding.csv)")

test_enc_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test_encoding.csv"
try:
    df_test_enc = pd.read_csv(test_enc_url)

    # Simpan label asli Segmentation sebelum di-drop untuk evaluasi
    if 'Segmentation' in df_test_enc.columns:
        y_true = df_test_enc['Segmentation']
    else:
        y_true = None

    # Drop kolom ID dan Segmentation sebelum prediksi
    drop_cols = [col for col in ['ID', 'Segmentation'] if col in df_test_enc.columns]
    df_test_enc = df_test_enc.drop(columns=drop_cols)

    # Pastikan urutan kolom sesuai fitur training jika tersedia
    feature_cols = encoder['feature_names'] if 'feature_names' in encoder else df_test_enc.columns.tolist()
    df_test_enc = df_test_enc.reindex(columns=feature_cols)

    # Scaling
    if scaler is not None:
        df_scaled = scaler.transform(df_test_enc)
    else:
        df_scaled = df_test_enc

    # Prediksi
    y_pred = model.predict(df_scaled)

    # Gabungkan hasil prediksi ke data asli agar bisa bandingkan
    df_test_original['Predicted_Segment'] = y_pred
    if y_true is not None:
        df_test_original['Actual_Segment'] = y_true.values

    # Tampilkan hasil perbandingan
    st.subheader("Perbandingan Actual vs Predicted Segmentation (Data Test Asli + Prediksi)")
    st.dataframe(df_test_original[['Actual_Segment', 'Predicted_Segment']].head())

    # Tampilkan hasil prediksi lengkap dengan fitur encoded (optional)
    st.subheader("Hasil Prediksi Lengkap (Data Encoded dengan Prediksi)")
    df_pred_full = df_test_enc.copy()
    df_pred_full['Predicted_Segment'] = y_pred
    if y_true is not None:
        df_pred_full['Actual_Segment'] = y_true.values
    st.dataframe(df_pred_full.head())

    # Evaluasi model
    if y_true is not None:
        st.subheader("Evaluasi Model")
        acc = accuracy_score(y_true, y_pred)
        st.write(f"Accuracy: {acc:.4f}")

        report = classification_report(y_true, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        st.dataframe(df_report)

except Exception as e:
    st.error(f"Gagal membaca atau memproses Test_encoding.csv dari GitHub: {e}")
