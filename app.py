import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === 1. Load Model, Encoder, Scaler ===
model = joblib.load('LightGBM.pkl')
encoder = joblib.load('encoders.pkl')
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

# === Input Data via File Testing dari GitHub saja ===
st.subheader("Ambil Data Testing dari GitHub")

test_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test.csv"

try:
    df_test = pd.read_csv(test_url)
    st.success("File test.csv berhasil diambil dari GitHub!")
    st.dataframe(df_test.head())

    # --- Proses data test --- #
    if 'Segmentation' in df_test.columns:
        df_test = df_test.drop(columns=['Segmentation'])

    df_test['Gender'] = df_test['Gender'].str.strip().map({'Male': 0, 'Female': 1})
    df_test['Ever_Married'] = df_test['Ever_Married'].str.strip().map({'No': 0, 'Yes': 1})
    df_test['Graduated'] = df_test['Graduated'].str.strip().map({'No': 0, 'Yes': 1})
    df_test['Spending_Score'] = df_test['Spending_Score'].str.strip().map({'Low': 0, 'Average': 1, 'High': 2})

    for col in encoder:
        if col in df_test.columns:
            df_test[col] = df_test[col].astype(str)
            df_test[col] = encoder[col].transform(df_test[col])
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan di data test.")

    feature_cols = encoder['feature_names'] if 'feature_names' in encoder else df_test.columns.tolist()
    df_test = df_test.reindex(columns=feature_cols, fill_value=0)

    if scaler is not None:
        df_scaled = scaler.transform(df_test)
    else:
        df_scaled = df_test

    y_pred = model.predict(df_scaled)
    df_test['Predicted_Segment'] = y_pred

    st.subheader("Hasil Prediksi")
    st.dataframe(df_test)

except Exception as e:
    st.error(f"Gagal membaca test.csv dari GitHub: {e}")
