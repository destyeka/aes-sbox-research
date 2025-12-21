import streamlit as st
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io

# ==========================================
# 1. CONFIGURATION & MATH CONSTANTS
# ==========================================
AES_MODULUS = 0x11B
C_AES = 0x63
SIZE = 256
N = 8

# [cite_start]Affine Matrices [cite: 1120-1184]
K_MATRICES = {
    "AES Standard": [0b10001111, 0b11000111, 0b11100011, 0b11110001, 0b11111000, 0b01111100, 0b00111110, 0b00011111],
    "K-4":          [0b00000111, 0b10000011, 0b11000001, 0b11100000, 0b01110000, 0b00111000, 0b00011100, 0b00001110],
    "K-44 (Prop)":  [0b01010111, 0b10101011, 0b11010101, 0b11101010, 0b01110101, 0b10111010, 0b01011101, 0b10101110],
    "K-81":         [0b10100001, 0b11010000, 0b01101000, 0b00110100, 0b00011010, 0b00001101, 0b10000110, 0b01000011],
    "K-111":        [0b11011100, 0b01101110, 0b00110111, 0b10011011, 0b11001101, 0b11100110, 0b01110011, 0b10111001],
    "K-128":        [0b01111111, 0b10111111, 0b11011111, 0b11011111, 0b11101111, 0b11110111, 0b11111011, 0b11111101]
}

# ==========================================
# 2. CORE CRYPTOGRAPHIC FUNCTIONS
# ==========================================
@st.cache_data
def gf_multiply(a, b):
    p = 0
    for i in range(8):
        if b & 1: p ^= a
        hi_bit_set = a & 0x80
        a <<= 1
        if hi_bit_set: a ^= AES_MODULUS
        b >>= 1
    return p & 0xFF

@st.cache_data
def gf_inverse_table():
    inv = [0] * 256
    for i in range(256):
        if i == 0: continue
        for j in range(1, 256):
            if gf_multiply(i, j) == 1:
                inv[i] = j
                break
    return inv

def affine_transform(byte_val, matrix):
    result = 0
    for i in range(8):
        row_val = matrix[i]
        parity = 0
        temp = byte_val & row_val
        while temp:
            parity ^= (temp & 1)
            temp >>= 1
        if parity: result |= (1 << i)
    return result ^ C_AES

@st.cache_data
def construct_sbox(matrix_name):
    matrix = K_MATRICES[matrix_name]
    inv_table = gf_inverse_table()
    sbox = [affine_transform(inv_table[i], matrix) for i in range(SIZE)]
    inv_sbox = [0] * SIZE
    for i in range(SIZE): inv_sbox[sbox[i]] = i
    return sbox, inv_sbox

# ==========================================
# 3. METRIC CALCULATIONS
# ==========================================
def fast_walsh_transform(f):
    fwht = list(f)
    h = 1
    while h < SIZE:
        for i in range(0, SIZE, h * 2):
            for j in range(i, i + h):
                x = fwht[j]; y = fwht[j + h]
                fwht[j] = x + y; fwht[j + h] = x - y
        h *= 2
    return fwht

def calculate_algebraic_degree(sbox):
    max_degree = 0
    for mask in [1, 2, 4, 8, 16, 32, 64, 128]:
        f = [(sbox[x] & mask) > 0 for x in range(SIZE)]
        anf = list(f)
        h = 1
        while h < SIZE:
            for i in range(0, SIZE, h * 2):
                for j in range(i, i + h):
                    anf[j + h] = anf[j + h] ^ anf[j]
            h *= 2
        current_max = 0
        for i in range(SIZE):
            if anf[i]:
                deg = bin(i).count('1')
                if deg > current_max: current_max = deg
        if current_max > max_degree: max_degree = current_max
    return max_degree

def calculate_transparency_order(sbox):
    sbox_np = np.array(sbox)
    total_beta_vals = []
    for beta in range(1, SIZE):
        g = np.array([bin(sbox[x] & beta).count('1') % 2 for x in range(SIZE)])
        g_polar = 1 - 2*g 
        wht = fast_walsh_transform(g_polar)
        wht_sq = [w * w for w in wht]
        ac_spectrum = fast_walsh_transform(wht_sq) 
        ac_spectrum = [x // SIZE for x in ac_spectrum]
        sum_abs_ac = sum(abs(ac) for i, ac in enumerate(ac_spectrum) if i != 0)
        val = N - (sum_abs_ac / (SIZE - 1))
        total_beta_vals.append(val)
    return max(total_beta_vals)

@st.cache_data
def calculate_metrics(sbox_tuple):
    sbox = sbox_tuple[0] 
    sbox_np = np.array(sbox, dtype=int)
    
    # NL
    nl_scores = []
    for v in range(1, SIZE):
        f = [1 if bin(sbox[x] & v).count('1') % 2 == 0 else -1 for x in range(SIZE)]
        fwht = fast_walsh_transform(f)
        nl = 128 - (max(abs(x) for x in fwht) / 2)
        nl_scores.append(nl)
    nl_val = min(nl_scores)

    # SAC & BIC
    sac_matrix = np.zeros((N, N))
    bic_sac_vals = []
    for j in range(N):
        alpha = (1 << j)
        diff = sbox_np ^ sbox_np[np.arange(SIZE) ^ alpha]
        for k in range(N):
            sac_matrix[j, k] = np.mean((diff >> k) & 1)
        for k1 in range(N):
            for k2 in range(k1 + 1, N):
                bic_sac_vals.append(np.mean(((diff >> k1) & 1) ^ ((diff >> k2) & 1)))
                
    sac_val = np.mean(sac_matrix)
    bic_sac = np.mean(bic_sac_vals) if bic_sac_vals else 0
    bic_nl = np.mean([nl_scores[v-1] for v in range(1, SIZE) if bin(v).count('1') == 2])

    # DAP, DU, LAP
    ddt = np.zeros((SIZE, SIZE), dtype=int)
    for dx in range(1, SIZE):
        dy = sbox_np ^ sbox_np[np.arange(SIZE) ^ dx]
        unique, counts = np.unique(dy, return_counts=True)
        ddt[dx, unique] = counts
    du_val = np.max(ddt)
    dap_val = du_val / SIZE
    lap_val = (128 - nl_val) / 256
    
    ad_val = calculate_algebraic_degree(sbox)
    to_val = calculate_transparency_order(sbox)

    return {
        "NL": nl_val, "SAC": sac_val, "BIC-NL": bic_nl, "BIC-SAC": bic_sac,
        "LAP": lap_val, "DAP": dap_val, "DU": int(du_val), 
        "AD": int(ad_val), "TO": round(to_val, 4)
    }

# ==========================================
# 4. ENCRYPTION HELPER FUNCTIONS
# ==========================================
def encrypt_bytes(data_bytes, key_string, sbox):
    """Generic Encryption Logic (Key Mixing + S-box)"""
    key_bytes = [ord(k) for k in key_string]
    if len(key_bytes) == 0: return data_bytes # Safety
    
    enc_bytes = []
    # Using a rolling IV concept (CBC-like) for better diffusion in both text and image
    iv = sum(key_bytes) % 256
    prev = iv
    
    for i, b in enumerate(data_bytes):
        k = key_bytes[i % len(key_bytes)]
        # Mix: (Byte XOR Key XOR Previous_Cipher) -> Sbox
        # This ensures high diffusion
        mixed = b ^ k ^ prev
        c = sbox[mixed]
        enc_bytes.append(c)
        prev = c # Chain the ciphertext
        
    return enc_bytes

def decrypt_bytes(enc_bytes, key_string, inv_sbox):
    """Generic Decryption Logic"""
    key_bytes = [ord(k) for k in key_string]
    if len(key_bytes) == 0: return enc_bytes
    
    dec_bytes = []
    iv = sum(key_bytes) % 256
    prev = iv
    
    for i, c in enumerate(enc_bytes):
        k = key_bytes[i % len(key_bytes)]
        # Reverse: InvSbox[Cipher] XOR Key XOR Previous_Cipher
        p_mixed = inv_sbox[c]
        p = p_mixed ^ k ^ prev
        dec_bytes.append(p)
        prev = c # Use the CIPHERTEXT from this round as prev for next
        
    return dec_bytes

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="AES S-Box Research Tool", layout="wide", page_icon="🔐")

st.title("🔐 AES S-Box Research & Deployment Tool")
st.markdown("""
Explore the custom affine matrices, validate their cryptographic strength, and test **Real-World Encryption**.
""")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
selected_matrix_name = st.sidebar.selectbox("Select K-Matrix (Affine)", list(K_MATRICES.keys()), index=2)

# Calculation (Cached)
all_metrics = []
sbox_db = {}
for name in K_MATRICES.keys():
    sb = construct_sbox(name)
    sbox_db[name] = sb
    m = calculate_metrics(sb)
    m["Name"] = name
    all_metrics.append(m)

df_all = pd.DataFrame(all_metrics).set_index("Name")
current_sbox, current_inv = sbox_db[selected_matrix_name]

# --- SECTION 1: S-BOX VISUALIZATION ---
with st.expander("Show S-Box Matrix Details", expanded=False):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Decimal Table")
        grid_dec = pd.DataFrame(np.array(current_sbox).reshape(16, 16))
        st.dataframe(grid_dec, height=250, use_container_width=True)
    with col2:
        st.subheader("Hexadecimal Table")
        hex_grid = [[f"{val:02X}" for val in row] for row in np.array(current_sbox).reshape(16, 16)]
        grid_hex = pd.DataFrame(hex_grid)
        st.dataframe(grid_hex, height=250, use_container_width=True)

# --- SECTION 2: METRICS & COMPARISON ---
st.header("1. Cryptographic Strength")

# S-Value Calculation
df_all["S-Value"] = (abs(df_all["SAC"] - 0.5) + abs(df_all["BIC-SAC"] - 0.5)) / 2
best_name = df_all.drop("AES Standard")["S-Value"].idxmin()

# Comparison Table
comp_cols = ["AES Standard", best_name]
if selected_matrix_name not in comp_cols: comp_cols.append(selected_matrix_name)

st.table(df_all.loc[comp_cols][["NL", "SAC", "BIC-SAC", "LAP", "DAP", "S-Value"]].T)
st.caption(f"*Selected Matrix: {selected_matrix_name}. Best Proposed: {best_name}.*")


# --- SECTION 3: DEPLOYMENT DEMOS ---
st.header("2. Deployment: Encryption & Decryption")

tab_text, tab_image = st.tabs(["🔤 Text Tool", "🖼️ Image Tool (Color Support)"])

# --- TEXT TOOL ---
with tab_text:
    st.subheader("Text Cryptography")
    
    op_mode = st.radio("Operation Mode", ["Encrypt Text", "Decrypt Text"], horizontal=True)
    
    if op_mode == "Encrypt Text":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            txt_input = st.text_area("Enter Plaintext", "Research Project: AES S-box 2025")
            key_enc = st.text_input("Secret Key", "MYSECRETKEY", type="password", key="k_enc")
            
            if st.button("Encrypt"):
                if not key_enc:
                    st.error("Key is required!")
                else:
                    data = [ord(c) for c in txt_input]
                    res = encrypt_bytes(data, key_enc, current_sbox)
                    b64_res = base64.b64encode(bytes(res)).decode()
                    st.session_state['last_cipher'] = b64_res
                    
        with col_t2:
            st.info("Result (Base64)")
            if 'last_cipher' in st.session_state:
                st.code(st.session_state['last_cipher'], language="text")

    elif op_mode == "Decrypt Text":
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            cipher_input = st.text_area("Enter Encrypted Text (Base64)", "")
            key_dec = st.text_input("Secret Key", "MYSECRETKEY", type="password", key="k_dec")
            
            if st.button("Decrypt"):
                if not key_dec:
                    st.error("Key is required!")
                else:
                    try:
                        raw_bytes = base64.b64decode(cipher_input)
                        res_bytes = decrypt_bytes(raw_bytes, key_dec, current_inv)
                        res_text = "".join([chr(c) for c in res_bytes])
                        st.session_state['last_plain'] = res_text
                    except Exception as e:
                        st.error(f"Decryption error: {e}")
                        
        with col_t2:
            st.success("Decrypted Message")
            if 'last_plain' in st.session_state:
                st.write(st.session_state['last_plain'])

# --- IMAGE TOOL ---
with tab_image:
    st.subheader("Image Cryptography (Full Color Support)")
    img_op = st.radio("Image Operation", ["Encrypt Image", "Decrypt Image"], horizontal=True)
    
    if img_op == "Encrypt Image":
        img_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
        key_img_enc = st.text_input("Encryption Key", "MYSECRETKEY", type="password", key="k_img_enc")
        
        if img_file and key_img_enc:
            image = Image.open(img_file).convert("RGB") # Ensure RGB
            st.image(image, caption="Original", width=300)
            
            if st.button("Encrypt Image"):
                # Process
                img_arr = np.array(image) # (H, W, 3)
                shape = img_arr.shape
                flat_pixels = img_arr.flatten() # Flatten R,G,B sequence
                
                # Encrypt flattened bytes
                enc_pixels = encrypt_bytes(flat_pixels, key_img_enc, current_sbox)
                
                enc_arr = np.array(enc_pixels, dtype=np.uint8).reshape(shape)
                enc_img = Image.fromarray(enc_arr, mode="RGB")
                
                st.image(enc_img, caption="Encrypted Image (Noise)", width=300)
                
                # Download
                buf = io.BytesIO()
                enc_img.save(buf, format="PNG") # PNG required to save exact pixel values
                st.download_button("Download Encrypted Image", buf.getvalue(), "encrypted_image.png", "image/png")
                
    elif img_op == "Decrypt Image":
        enc_file = st.file_uploader("Upload Encrypted Image (PNG)", type=["png"])
        key_img_dec = st.text_input("Decryption Key", "MYSECRETKEY", type="password", key="k_img_dec")
        
        if enc_file and key_img_dec:
            # Load as is
            image_enc = Image.open(enc_file).convert("RGB")
            st.image(image_enc, caption="Encrypted Input", width=300)
            
            if st.button("Decrypt Image"):
                img_arr = np.array(image_enc)
                shape = img_arr.shape
                flat_pixels = img_arr.flatten()
                
                # Decrypt
                dec_pixels = decrypt_bytes(flat_pixels, key_img_dec, current_inv)
                
                dec_arr = np.array(dec_pixels, dtype=np.uint8).reshape(shape)
                dec_img = Image.fromarray(dec_arr, mode="RGB")
                
                st.image(dec_img, caption="Decrypted Result", width=300)
