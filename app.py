import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import pickle
import torch.nn.functional as F

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Food Detective AI (Final)",
    page_icon="🍱",
    layout="centered"
)

st.title("🍱 Food Detective AI")
st.write("Deteksi Multi-Item + Pilih Varian Spesifik (Nasi Goreng, dll)")

# ==========================================
# 2. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    try:
        with open('class_names_fixed.pkl', 'rb') as f:
            class_names = pickle.load(f)
            
        model = models.resnet18(weights=None)
        model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(model.fc.in_features, len(class_names))
        )
        model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
        model.eval()
        return model, class_names
    except Exception as e:
        st.error(f"Gagal memuat sistem: {e}")
        return None, None

@st.cache_data
def load_nutrition():
    try:
        df = pd.read_csv('Training_Food_Kalori.csv')
        df.columns = df.columns.str.strip()
        return df
    except:
        return None

model, class_names = load_resources()
df_nutrition = load_nutrition()

# ==========================================
# 3. PREPROCESSING
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
# 4. FUNGSI UTAMA
# ==========================================
def predict_food(image_input, temperature=1.0):
    st.image(image_input, caption='Gambar Input', use_container_width=True)
    
    if model is None: return

    with st.spinner('Menganalisis... 🤖'):
        img_tensor = process_image(image_input)
        with torch.no_grad():
            outputs = model(img_tensor)
            outputs = outputs / temperature 
            probs = F.softmax(outputs, dim=1)
        
        top5_prob, top5_idx = torch.topk(probs, 5)
        
        st.divider()
        st.subheader("🔍 Deteksi Awal")
        st.caption(f"Sensitivitas: {temperature}x")
        
        # --- TAHAP 1: PILIH KATEGORI (RICE, EGG, DLL) ---
        st.write("1️⃣ Centang kategori yang terdeteksi:")
        selected_categories = []
        found_any = False
        
        for i in range(5):
            idx = top5_idx[0][i].item()
            prob = top5_prob[0][i].item() * 100
            category_name = class_names[idx]
            
            # Tampilkan jika probabilitas > 5%
            if prob > 5.0:
                found_any = True
                # Juara 1 otomatis dicentang
                is_checked = st.checkbox(
                    f"**{category_name}** ({prob:.1f}%)",
                    value=(i==0), 
                    key=f"chk_cat_{i}"
                )
                if is_checked:
                    selected_categories.append(category_name)
        
        if not found_any:
            st.warning("Tidak ada item yang yakin. Coba naikkan sensitivitas!")

        # --- TAHAP 2: PILIH VARIAN (NASI GORENG / NASI PUTIH) ---
        if selected_categories and df_nutrition is not None:
            st.divider()
            st.write("2️⃣ Tentukan spesifik makanannya:")
            
            final_selected_foods = []
            
            # Loop setiap kategori yang dipilih user
            for category in selected_categories:
                # Cari semua makanan di database yang masuk kategori ini
                # Misal Category="Rice" -> Dapat [Nasi Putih, Nasi Goreng, Nasi Uduk]
                variants = df_nutrition[
                    df_nutrition['Category_AI'].str.contains(category, case=False, na=False)
                ]
                
                if not variants.empty:
                    # Buat Dropdown untuk memilih varian
                    variant_list = variants['Nama_Makanan'].tolist()
                    
                    # Layout kolom biar rapi
                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.info(f"Kategori: {category}")
                    with c2:
                        # Dropdown pilihan
                        chosen_food = st.selectbox(
                            f"Jenis {category} apa?", 
                            variant_list,
                            key=f"sel_{category}"
                        )
                        final_selected_foods.append(chosen_food)
                else:
                    st.warning(f"Data detail untuk {category} tidak ditemukan.")

            # --- TAHAP 3: HITUNG TOTAL GIZI ---
            if final_selected_foods:
                st.divider()
                st.subheader("📊 Rincian Gizi Final")
                
                totals = {'Kalori': 0, 'Protein': 0, 'Lemak': 0, 'Karbo': 0}
                detail_data = []

                for food_name in final_selected_foods:
                    # Ambil data berdasarkan nama makanan yang SUDAH DIPILIH (Nasi Goreng)
                    data = df_nutrition[df_nutrition['Nama_Makanan'] == food_name].iloc[0]
                    
                    totals['Kalori'] += data['Kalori']
                    totals['Protein'] += data['Protein']
                    totals['Lemak'] += data['Lemak']
                    totals['Karbo'] += data['Karbo']
                    
                    detail_data.append([
                        food_name, 
                        f"{data['Kalori']} kkal",
                        f"{data['Protein']} g",
                        f"{data['Lemak']} g"
                    ])

                # Tampilkan Metrics Total
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("🔥 Total Kalori", f"{totals['Kalori']}")
                c2.metric("🥩 Protein", f"{totals['Protein']:.1f}g")
                c3.metric("aaa Lemak", f"{totals['Lemak']:.1f}g")
                c4.metric("🍞 Karbo", f"{totals['Karbo']:.1f}g")
                
                st.table(pd.DataFrame(detail_data, columns=["Menu Pilihan", "Energi", "Protein", "Lemak"]))

# ==========================================
# 5. UI INPUT
# ==========================================
st.sidebar.header("⚙️ Pengaturan")
sensitivity = st.sidebar.slider("Sensitivitas AI", 1.0, 5.0, 2.5, 0.5)

st.divider()
option = st.radio("Metode Input:", ("📸 Kamera", "📂 Upload"), horizontal=True)

input_image = None
if option == "📸 Kamera":
    picture = st.camera_input("Jepret makananmu!")
    if picture: input_image = Image.open(picture).convert('RGB')
elif option == "📂 Upload":
    upl = st.file_uploader("Upload Foto", type=["jpg", "png"])
    if upl: input_image = Image.open(upl).convert('RGB')

if input_image:
    predict_food(input_image, temperature=sensitivity)