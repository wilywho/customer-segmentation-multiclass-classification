import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# === Load Model, Encoder, dan Scaler ===
model = joblib.load("LightGBM.pkl")
encoder = joblib.load("encoders.pkl")  # dictionary LabelEncoders
scaler = joblib.load("scaler.pkl")     # StandardScaler atau MinMaxScaler

# === Judul dan Deskripsi Proyek ===
st.title("Customer Segmentation Classification")
st.write("""
### Author: Kelompok 4 DataBender's
- Farhan Wily
- Ghazy Shidqy
- Naufal Hafizh Dhiya Ulhaq
- Yosef Sony Koesprasetyo
""")

st.write("""
## Deskripsi Proyek
Sebuah perusahaan otomotif ingin mengklasifikasikan calon pelanggan baru ke dalam 4 segmen: A, B, C, D.  
Model klasifikasi ini dilatih dari data pelanggan eksisting yang telah dikelompokkan sebelumnya oleh tim sales.
""")

# === Load dan Tampilkan Data Training ===
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

# === Visualisasi Umum ===
st.subheader("Visualisasi Umum Data Training")
fig, ax = plt.subplots()
sns.countplot(data=df_train, x='Segmentation', palette='Set2', ax=ax)
st.pyplot(fig)

fig2, ax2 = plt.subplots()
sns.histplot(df_train['Age'], bins=20, kde=True, color='skyblue', ax=ax2)
ax2.set_title("Distribusi Umur")
st.pyplot(fig2)

# === Deskripsi Model ===
st.subheader("Model yang Digunakan")
st.write("""
Model klasifikasi terbaik dari hasil evaluasi adalah: **LightGBM**  
Model ini memiliki akurasi tinggi dan efisiensi yang baik.
""")

# === Load Data Testing Asli dari GitHub ===
st.subheader("Preview Data Testing Asli")
test_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test.csv"
try:
    df_test_original = pd.read_csv(test_url)
    st.dataframe(df_test_original.head())
except Exception as e:
    st.error(f"Gagal membaca test.csv dari GitHub: {e}")

# === Proses Prediksi dari Test_encoding.csv ===
st.subheader("Hasil Prediksi pada Data Testing")
try:
    test_enc_url = "https://raw.githubusercontent.com/wilywho/customer-segmentation-multiclass-classification/refs/heads/main/Test_encoding.csv"
    df_test = pd.read_csv(test_enc_url)

    if 'ID' in df_test.columns:
        df_test = df_test.drop(columns=['ID'])
    if 'Segmentation' in df_test.columns:
        df_test = df_test.drop(columns=['Segmentation'])

    df_scaled = scaler.transform(df_test)
    y_pred = model.predict(df_scaled)

    df_test['Predicted_Segment_Num'] = y_pred
    seg_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
    df_test['Predicted_Segment'] = df_test['Predicted_Segment_Num'].map(seg_map)

    if 'Segmentation' in df_test_original.columns:
        df_test['Original_Segment'] = df_test_original['Segmentation']

    # Hitung akurasi
    if 'Original_Segment' in df_test.columns:
        acc = accuracy_score(df_test['Original_Segment'], df_test['Predicted_Segment'])
        st.write(f"**Akurasi model pada data testing:** {acc:.2%}")

    st.dataframe(df_test[['Original_Segment', 'Predicted_Segment']])

except Exception as e:
    st.error(f"Gagal membaca atau memproses Test_encoding.csv: {e}")

# === Input Manual untuk Prediksi ===
st.subheader("Prediksi Segmentasi Pelanggan Baru (Input Manual)")
with st.form("manual_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Ever Married", ["Yes", "No"])
    age = st.slider("Age", 18, 90, 30)
    graduated = st.selectbox("Graduated", ["Yes", "No"])
    profession = st.selectbox("Profession", [
        "Healthcare", "Engineer", "Lawyer", "Artist", "Doctor",
        "Entertainment", "Executive", "Marketing", "Homemaker"
    ])
    work_exp = st.slider("Work Experience", 0, 20, 1)
    spending = st.selectbox("Spending Score", ["Low", "Average", "High"])
    family_size = st.slider("Family Size", 1, 10, 3)

    submit = st.form_submit_button("Prediksi")

    if submit:
        user_df = pd.DataFrame([{
            "Gender": gender,
            "Ever_Married": married,
            "Age": age,
            "Graduated": graduated,
            "Profession": profession,
            "Work_Experience": work_exp,
            "Spending_Score": spending,
            "Family_Size": family_size
        }])

        # Encoding fitur kategori
        for col in encoder:
            if col in user_df.columns:
                user_df[col] = encoder[col].transform(user_df[col])

        # Scaling
        user_scaled = scaler.transform(user_df)

        # Prediksi
        pred_class = model.predict(user_scaled)[0]
        pred_segment = seg_map.get(pred_class, "Unknown")

        st.success(f"Segmentasi Pelanggan yang Diprediksi: **{pred_segment}**")
