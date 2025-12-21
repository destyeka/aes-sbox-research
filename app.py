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

# Affine Matrices
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
    if len(key_bytes) == 0: return data_bytes 
    
    enc_bytes = []
    # Rolling IV for diffusion (CBC-like)
    iv = sum(key_bytes) % 256
    prev = iv
    
    for i, b in enumerate(data_bytes):
        k = key_bytes[i % len(key_bytes)]
        # Mix: (Byte XOR Key XOR Previous_Cipher) -> Sbox
        mixed = b ^ k ^ prev
        c = sbox[mixed]
        enc_bytes.append(c)
        prev = c # Chain
        
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
        prev = c # Use current cipher as prev for next
        
    return dec_bytes

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="AES S-Box Research Tool", layout="wide", page_icon="🔐")

st.title("🔐 AES S-Box Research & Deployment Tool")
st.markdown("""
Implementation of **"AES S-box modification uses affine matrices exploration"**.
Use the tabs below to analyze S-box strength or deploy encryption algorithms.
""")

# --- SIDEBAR CONFIG ---
st.sidebar.header("⚙️ Configuration")
selected_matrix_name = st.sidebar.selectbox("Select K-Matrix (Affine)", list(K_MATRICES.keys()), index=2)

# Calculate Metrics (Cached)
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

# Identify Best Matrix (K-44 Proposed) by S-Value
df_all["S-Value"] = (abs(df_all["SAC"] - 0.5) + abs(df_all["BIC-SAC"] - 0.5)) / 2
best_name = df_all.drop("AES Standard")["S-Value"].idxmin()

# --- TABS LAYOUT ---
tab_analysis, tab_demo = st.tabs(["📊 S-Box Analysis", "🔐 Encryption Demo"])

# =========================================================
# TAB 1: S-BOX VISUALIZATION & COMPARISON
# =========================================================
with tab_analysis:
    st.header(f"1. S-Box Matrix: {selected_matrix_name}")
    
    col_v1, col_v2 = st.columns(2)
    with col_v1:
        st.subheader("Decimal Table")
        grid_dec = pd.DataFrame(np.array(current_sbox).reshape(16, 16))
        st.dataframe(grid_dec, height=250, use_container_width=True)
    with col_v2:
        st.subheader("Hexadecimal Table")
        hex_grid = [[f"{val:02X}" for val in row] for row in np.array(current_sbox).reshape(16, 16)]
        grid_hex = pd.DataFrame(hex_grid)
        st.dataframe(grid_hex, height=250, use_container_width=True)

    st.divider()

    st.header("2. Performance Comparison")
    
    # Constructing the Comparison Table
    # We want columns: [AES Standard, Best Proposed, Selected, Target]
    
    # Define Targets (Ideal Values)
    targets = {
        "NL": "Max (112)",
        "SAC": "0.5",
        "BIC-NL": "High",
        "BIC-SAC": "0.5",
        "LAP": "Min (0)",
        "DAP": "Min (0)",
        "DU": "Min (4)",
        "AD": "7",
        "TO": "Min (0)",
        "S-Value": "0"
    }
    
    # Extract Columns
    cols_to_show = ["AES Standard", best_name]
    if selected_matrix_name not in cols_to_show:
        cols_to_show.append(selected_matrix_name)
    
    # Create comparison dataframe (Transposed)
    comp_df = df_all.loc[cols_to_show].T
    
    # Add Target Column
    comp_df["Target (Ideal)"] = [targets.get(idx, "-") for idx in comp_df.index]
    
    # Formatting helper
    def highlight_target(s):
        return ['background-color: #f0f2f6; font-weight: bold' if s.name == "Target (Ideal)" else '' for _ in s]

    st.table(comp_df.style.format("{:.5f}", subset=cols_to_show).apply(highlight_target, axis=0))
    
    st.info(f"**Conclusion:** The matrix **{best_name}** is highlighted as the best proposed candidate based on the S-Value (Closeness to ideal SAC/BIC).")


# =========================================================
# TAB 2: ENCRYPTION / DECRYPTION DEMOS
# =========================================================
with tab_demo:
    st.header("Real-World Deployment")
    
    subtab_text, subtab_image = st.tabs(["🔤 Text Tool", "🖼️ Image Tool (RGB)"])

    # --- TEXT TOOL ---
    with subtab_text:
        st.markdown("### Text Cryptography")
        t_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True, key="t_mode")
        
        col_t1, col_t2 = st.columns(2)
        
        if t_mode == "Encrypt":
            with col_t1:
                txt_in = st.text_area("Plaintext", "Research Project: AES S-box 2025")
                key_in = st.text_input("Key", "MYKEY", type="password", key="tk1")
                if st.button("Encrypt Text"):
                    if not key_in: st.error("Key required")
                    else:
                        d = [ord(c) for c in txt_in]
                        enc = encrypt_bytes(d, key_in, current_sbox)
                        b64 = base64.b64encode(bytes(enc)).decode()
                        st.session_state['res_txt'] = b64
            with col_t2:
                st.info("Encrypted Output (Base64)")
                if 'res_txt' in st.session_state:
                    st.code(st.session_state['res_txt'], language="text")

        else: # Decrypt
            with col_t1:
                cipher_in = st.text_area("Ciphertext (Base64)", "")
                key_in_d = st.text_input("Key", "MYKEY", type="password", key="tk2")
                if st.button("Decrypt Text"):
                    if not key_in_d: st.error("Key required")
                    else:
                        try:
                            raw = base64.b64decode(cipher_in)
                            dec = decrypt_bytes(raw, key_in_d, current_inv)
                            res = "".join([chr(c) for c in dec])
                            st.session_state['res_plain'] = res
                        except: st.error("Invalid Base64 or Key")
            with col_t2:
                st.success("Decrypted Output")
                if 'res_plain' in st.session_state:
                    st.write(st.session_state['res_plain'])

    # --- IMAGE TOOL ---
    with subtab_image:
        st.markdown("### Image Cryptography (Full Color Support)")
        i_mode = st.radio("Mode", ["Encrypt Image", "Decrypt Image"], horizontal=True, key="i_mode")
        
        if i_mode == "Encrypt Image":
            img_file = st.file_uploader("Upload Original (JPG/PNG)", type=["png", "jpg", "jpeg"])
            k_img = st.text_input("Key", "MYKEY", type="password", key="ik1")
            
            if img_file and k_img:
                img = Image.open(img_file).convert("RGB")
                st.image(img, caption="Original", width=300)
                
                if st.button("Encrypt Image"):
                    arr = np.array(img)
                    shape = arr.shape
                    flat = arr.flatten()
                    
                    enc_flat = encrypt_bytes(flat, k_img, current_sbox)
                    enc_arr = np.array(enc_flat, dtype=np.uint8).reshape(shape)
                    enc_img = Image.fromarray(enc_arr, mode="RGB")
                    
                    st.image(enc_img, caption="Encrypted (Noise)", width=300)
                    
                    # Download
                    buf = io.BytesIO()
                    enc_img.save(buf, format="PNG")
                    st.download_button("Download Encrypted PNG", buf.getvalue(), "encrypted.png", "image/png")

        else: # Decrypt
            enc_file = st.file_uploader("Upload Encrypted (PNG)", type=["png"])
            k_img_d = st.text_input("Key", "MYKEY", type="password", key="ik2")
            
            if enc_file and k_img_d:
                img_enc = Image.open(enc_file).convert("RGB")
                st.image(img_enc, caption="Encrypted Input", width=300)
                
                if st.button("Decrypt Image"):
                    arr = np.array(img_enc)
                    shape = arr.shape
                    flat = arr.flatten()
                    
                    dec_flat = decrypt_bytes(flat, k_img_d, current_inv)
                    dec_arr = np.array(dec_flat, dtype=np.uint8).reshape(shape)
                    dec_img = Image.fromarray(dec_arr, mode="RGB")
                    
                    st.image(dec_img, caption="Decrypted Result", width=300)
