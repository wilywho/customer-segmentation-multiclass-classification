import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# === 1. Load Model, Encoder, Scaler ===
model = joblib.load('LightGBM.pkl')
encoder = joblib.load('encoders.pkl')  # dictionary of LabelEncoders per column
scaler = joblib.load('scaler.pkl')

# === Fungsi preprocessing data ===
def preprocess_df(df):
    # Strip & map manual categorical ke numerik (agar aman)
    df['Gender'] = df['Gender'].str.strip().map({'Male': 0, 'Female': 1})
    df['Ever_Married'] = df['Ever_Married'].str.strip().map({'No': 0, 'Yes': 1})
    df['Graduated'] = df['Graduated'].str.strip().map({'No': 0, 'Yes': 1})
    df['Spending_Score'] = df['Spending_Score'].str.strip().map({'Low': 0, 'Average': 1, 'High': 2})

    # Apply LabelEncoder yang sudah diload untuk kolom lain
    for col in ['Profession', 'Var_1']:
        if col in df.columns:
            if col in encoder:
                # Jika nilai baru yang tidak ada di encoder, handle dengan fillna -1
                df[col] = df[col].map(lambda x: x.strip() if isinstance(x, str) else x)
                try:
                    df[col] = encoder[col].transform(df[col])
                except:
                    # Kalau ada nilai yang belum pernah dilihat, ubah jadi -1 (atau pilihan lain)
                    df[col] = df[col].apply(lambda x: encoder[col].transform([x])[0] if x in encoder[col].classes_ else -1)
        else:
            st.warning(f"Kolom '{col}' tidak ditemukan dalam data input.")

    return df

def align_features(df, train_columns):
    # Pastikan kolom di df lengkap dan urut sesuai train_columns
    missing_cols = set(train_columns) - set(df.columns)
    for col in missing_cols:
        df[col] = 0
    # Urutkan kolom
    df = df[train_columns]
    return df

# === 2. Deskripsi Project ===
st.title("Customer Segmentation Classification")
st.write("""
## Deskripsi Proyek
Sebuah perusahaan otomotif ingin mengklasifikasikan calon pelanggan baru ke dalam 4 segmen: A, B, C, D. 
Model klasifikasi ini dilatih dari data pelanggan eksisting yang telah dikelompokkan sebelumnya oleh tim sales.
""")

# === 3. Data Training Overview ===
st.subheader("Contoh Data Training & Penjelasan Fitur")

train_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Train.csv"
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

# Visualisasi data umum
st.subheader("Visualisasi Umum Data")
fig, ax = plt.subplots()
sns.countplot(data=df_train, x='Segmentation', palette='pastel', ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(df_train['Age'], kde=True, bins=20, color='skyblue', ax=ax2)
ax2.set_title("Distribusi Usia")
st.pyplot(fig2)

# === 4. Preprocessing training data untuk referensi kolom ===
df_train_processed = preprocess_df(df_train.copy())
# Hilangkan kolom target untuk input ke model
train_features = df_train_processed.drop(columns=['Segmentation', 'Age', 'Work_Experience', 'Family_Size']).copy()

# Jika Age, Work_Experience, Family_Size termasuk fitur yang digunakan model,
# sesuaikan sesuai kebutuhan. Jika fitur ini ada di model, jangan drop.

# Scaling train features supaya kita bisa tahu kolom lengkap setelah scaling
if scaler is not None:
    train_scaled = scaler.transform(train_features)
else:
    train_scaled = train_features.values

train_scaled_df = pd.DataFrame(train_scaled, columns=train_features.columns)

# Simpan list kolom fitur hasil preprocessing dan scaling sebagai acuan input
feature_columns = train_features.columns.tolist()

# === 5. Penjelasan Model ===
st.subheader("Model Klasifikasi yang Digunakan")
st.write("Model klasifikasi yang digunakan dalam proyek ini adalah: **LightGBM**.")
st.write("Model ini dipilih karena memberikan performa terbaik berdasarkan uji evaluasi dan cross-validation.")

st.subheader("Pilih Metode Input Data")
input_mode = st.radio("Pilih metode input:", ("File Testing dari GitHub", "Input Manual"))

if input_mode == "File Testing dari GitHub":
    test_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test.csv"
    try:
        df_test = pd.read_csv(test_url)
        st.success("File test.csv berhasil diambil dari GitHub!")
        st.dataframe(df_test.head())

        df_test_processed = preprocess_df(df_test.copy())
        # Drop kolom yang tidak dipakai model jika ada
        if 'Segmentation' in df_test_processed.columns:
            df_test_features = df_test_processed.drop(columns=['Segmentation', 'Age', 'Work_Experience', 'Family_Size'])
        else:
            df_test_features = df_test_processed.drop(columns=['Age', 'Work_Experience', 'Family_Size'])

        # Align fitur test sesuai fitur train
        df_test_features = align_features(df_test_features, feature_columns)

        # Scaling test features
        if scaler is not None:
            df_test_scaled = scaler.transform(df_test_features)
        else:
            df_test_scaled = df_test_features.values

        # Prediksi
        y_pred = model.predict(df_test_scaled)
        df_test['Predicted_Segment'] = y_pred

        st.subheader("Hasil Prediksi")
        st.dataframe(df_test)

        # Evaluasi jika label asli ada
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
            df_input_processed = preprocess_df(df_input.copy())

            # Align fitur input manual sesuai fitur train
            df_input_features = align_features(df_input_processed, feature_columns)

            # Scaling
            if scaler is not None:
                df_scaled = scaler.transform(df_input_features)
            else:
                df_scaled = df_input_features.values

            prediction = model.predict(df_scaled)[0]
            st.success(f"Segmentasi Pelanggan yang Diprediksi: **{prediction}**")

        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses input: {e}")
