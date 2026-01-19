import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import pickle
import os

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Food Detective AI",
    page_icon="🍔",
    layout="centered"
)

st.title("🍔 Food Detective AI")
st.write("Aplikasi Pintar Pendeteksi Makanan & Penghitung Kalori")

# ==========================================
# 2. LOAD MODEL & LABEL (OTOMATIS)
# ==========================================
@st.cache_resource
def load_resources():
    # A. LOAD LABEL (Kamus Data)
    try:
        with open('class_names_fixed.pkl', 'rb') as f:
            class_names = pickle.load(f)
    except FileNotFoundError:
        st.error("⚠️ File 'class_names_fixed.pkl' tidak ditemukan! Pastikan sudah download dari hasil training.")
        return None, None

    # B. LOAD MODEL
    try:
        model = models.resnet18(weights=None)
        # Sesuaikan kepala model dengan jumlah kelas yang ada di label
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, len(class_names))
        )
        
        device = torch.device('cpu')
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
        model.eval()
        return model, class_names
    except FileNotFoundError:
        st.error("⚠️ File 'best_model.pth' tidak ditemukan! Pastikan file model ada di folder ini.")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

@st.cache_data
def load_nutrition():
    try:
        df = pd.read_csv('Training_Food_Kalori.csv')
        df.columns = df.columns.str.strip() # Bersihkan nama kolom dari spasi
        return df
    except Exception as e:
        st.error(f"⚠️ Gagal membaca 'Training_Food_Kalori.csv': {e}")
        return None

# Load semua resource di awal
model, class_names = load_resources()
df_nutrition = load_nutrition()

# ==========================================
# 3. PREPROCESSING GAMBAR
# ==========================================
def process_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

# ==========================================
# 4. FUNGSI UTAMA (PREDIKSI & GUI)
# ==========================================
def predict_food(image_input):
    # Tampilkan gambar input
    st.image(image_input, caption='Gambar yang discan', use_container_width=True)
    
    if model is None or class_names is None:
        st.error("Sistem belum siap (File model/label hilang).")
        return

    with st.spinner('AI sedang berpikir... 🤖'):
        # --- TAHAP 1: AI MENERKA GAMBAR ---
        img_tensor = process_image(image_input)
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        
        conf, pred_idx = torch.max(probs, 1)
        ai_category = class_names[pred_idx.item()] # Nama Kategori (misal: Rice)
        confidence = conf.item() * 100

        st.divider()
        st.markdown(f"### 🤖 AI Mendeteksi: **{ai_category}**")
        st.caption(f"Tingkat Keyakinan: {confidence:.2f}%")

        # --- TAHAP 2: INTERAKSI USER & DATABASE GIZI ---
        if df_nutrition is not None:
            # Fitur Koreksi Manual (Jaga-jaga kalau AI salah)
            st.write("---")
            col_check, col_text = st.columns([1, 3])
            with col_check:
                manual_mode = st.checkbox("⚠️ AI Salah?")
            with col_text:
                st.caption("Centang jika tebakan AI meleset, lalu pilih makanan yang benar di bawah.")

            # Logika Filter Dropdown
            if manual_mode:
                # Jika Manual: Tampilkan SEMUA makanan (Urut Abjad)
                food_options = sorted(df_nutrition['Nama_Makanan'].unique().tolist())
                st.info("Mode Manual Aktif: Silakan cari nama makanan yang benar.")
            else:
                # Jika Auto: Filter berdasarkan kategori tebakan AI
                # (Cari baris di CSV yang kolom Category_AI-nya cocok)
                filtered_df = df_nutrition[
                    df_nutrition['Category_AI'].str.contains(ai_category, case=False, na=False)
                ]
                
                if not filtered_df.empty:
                    food_options = filtered_df['Nama_Makanan'].tolist()
                else:
                    st.warning(f"Kategori '{ai_category}' terdeteksi, tapi tidak ada detail di database CSV.")
                    food_options = []

            # Tampilkan Dropdown jika ada pilihan
            if food_options:
                selected_food = st.selectbox("👉 Pilih Detail Makanan:", food_options)
                
                # Tampilkan Kartu Informasi Gizi
                if selected_food:
                    data = df_nutrition[df_nutrition['Nama_Makanan'] == selected_food].iloc[0]
                    
                    st.success(f"📊 Nutrisi: **{selected_food}**")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🔥 Kalori", f"{data['Kalori']} kkal")
                    c2.metric("🥩 Protein", f"{data['Protein']} g")
                    c3.metric("aaa Lemak", f"{data['Lemak']} g")
                    c4.metric("🍚 Porsi", f"{data['Porsi_Std_g']} g")
                    
                    st.caption(f"Keterangan Porsi: {data['Ket_Porsi']}")

# ==========================================
# 5. INPUT USER (KAMERA / UPLOAD)
# ==========================================
st.divider()
st.subheader("Mulai Scan Makanan")
option = st.radio("Pilih Metode:", ("📸 Kamera HP/Laptop", "📂 Upload File"), horizontal=True)

input_image = None

if option == "📸 Kamera HP/Laptop":
    picture = st.camera_input("Ambil foto makananmu!")
    if picture:
        input_image = Image.open(picture).convert('RGB')
        
elif option == "📂 Upload File":
    uploaded_file = st.file_uploader("Pilih gambar JPG/PNG...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        input_image = Image.open(uploaded_file).convert('RGB')

# Eksekusi jika ada gambar
if input_image is not None:
    predict_food(input_image)