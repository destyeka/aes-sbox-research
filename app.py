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
# 4. HELPER FUNCTIONS (CRYPTO & PLOTTING)
# ==========================================
def encrypt_bytes(data_bytes, key_string, sbox):
    """Generic Encryption Logic"""
    key_bytes = [ord(k) for k in key_string]
    if len(key_bytes) == 0: return data_bytes 
    
    enc_bytes = []
    iv = sum(key_bytes) % 256
    prev = iv
    
    for i, b in enumerate(data_bytes):
        k = key_bytes[i % len(key_bytes)]
        mixed = b ^ k ^ prev
        c = sbox[mixed]
        enc_bytes.append(c)
        prev = c 
        
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
        p_mixed = inv_sbox[c]
        p = p_mixed ^ k ^ prev
        dec_bytes.append(p)
        prev = c 
        
    return dec_bytes

def plot_rgb_histogram(image_pil):
    """Generates a Matplotlib Figure for RGB Histogram"""
    img_arr = np.array(image_pil)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    
    if len(img_arr.shape) == 3: # RGB
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(img_arr[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=color.upper())
    else: # Grayscale
        ax.hist(img_arr.ravel(), bins=256, color='gray', alpha=0.7, label='Gray')
    
    ax.set_title("RGB Pixel Distribution")
    ax.set_xlim([0, 256])
    ax.legend(prop={'size': 8})
    plt.tight_layout()
    return fig

# ==========================================
# 5. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="AES S-Box Research Tool", layout="wide", page_icon="🔐")

st.title("🔐 AES S-Box Research & Deployment Tool")
st.markdown("""
Implementation of **"AES S-box modification uses affine matrices exploration"**.
Use the tabs below to analyze S-box strength or deploy encryption algorithms.
""")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")
selected_matrix_name = st.sidebar.selectbox("Select K-Matrix (Affine)", list(K_MATRICES.keys()), index=2)

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

# Best matrix
df_all["S-Value"] = (abs(df_all["SAC"] - 0.5) + abs(df_all["BIC-SAC"] - 0.5)) / 2
best_name = df_all.drop("AES Standard")["S-Value"].idxmin()

# --- TABS ---
tab_analysis, tab_demo = st.tabs(["📊 S-Box Analysis", "🔐 Encryption Demo"])

# =========================================================
# TAB 1: S-BOX VISUALIZATION & COMPARISON
# =========================================================
with tab_analysis:
    st.header(f"1. S-Box Matrix: {selected_matrix_name}")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Decimal")
        st.dataframe(pd.DataFrame(np.array(current_sbox).reshape(16, 16)), height=250)
    with c2:
        st.subheader("Hexadecimal")
        st.dataframe(pd.DataFrame([[f"{x:02X}" for x in r] for r in np.array(current_sbox).reshape(16, 16)]), height=250)

    st.divider()
    st.header("2. Performance Comparison")
    
    # Target Values
    targets = {
        "NL": "Max (112)", "SAC": "0.5", "BIC-NL": "High", "BIC-SAC": "0.5",
        "LAP": "Min (0)", "DAP": "Min (0)", "DU": "Min (4)", "AD": "7", "TO": "Min (0)", "S-Value": "0"
    }
    
    cols = ["AES Standard", best_name]
    if selected_matrix_name not in cols: cols.append(selected_matrix_name)
    
    comp_df = df_all.loc[cols].T
    comp_df["Target (Ideal)"] = [targets.get(i, "-") for i in comp_df.index]
    
    # Highlight Target Column
    def style_target(s):
        return ['background-color: #e6f3ff; font-weight: bold' if s.name == "Target (Ideal)" else '' for _ in s]

    st.table(comp_df.style.format("{:.5f}", subset=cols).apply(style_target, axis=0))

# =========================================================
# TAB 2: ENCRYPTION / DECRYPTION DEMOS
# =========================================================
with tab_demo:
    st.header("Real-World Deployment")
    subtab_text, subtab_image = st.tabs(["🔤 Text Tool", "🖼️ Image Tool (RGB + Histogram)"])

    # --- TEXT TOOL ---
    with subtab_text:
        st.markdown("### Text Cryptography")
        t_mode = st.radio("Mode", ["Encrypt", "Decrypt"], horizontal=True)
        
        c1, c2 = st.columns(2)
        if t_mode == "Encrypt":
            with c1:
                txt_in = st.text_area("Plaintext", "Research Project 2025")
                k_in = st.text_input("Key", "MYKEY", type="password")
                if st.button("Encrypt"):
                    if not k_in: st.error("Key required")
                    else:
                        enc = encrypt_bytes([ord(c) for c in txt_in], k_in, current_sbox)
                        st.session_state['res_txt'] = base64.b64encode(bytes(enc)).decode()
            with c2:
                st.info("Encrypted (Base64)")
                if 'res_txt' in st.session_state: st.code(st.session_state['res_txt'])
        else:
            with c1:
                c_in = st.text_area("Ciphertext", "")
                k_in = st.text_input("Key", "MYKEY", type="password", key="kd")
                if st.button("Decrypt"):
                    if not k_in: st.error("Key required")
                    else:
                        try:
                            dec = decrypt_bytes(base64.b64decode(c_in), k_in, current_inv)
                            st.session_state['res_pln'] = "".join([chr(c) for c in dec])
                        except: st.error("Error")
            with c2:
                st.success("Decrypted")
                if 'res_pln' in st.session_state: st.write(st.session_state['res_pln'])

    # --- IMAGE TOOL ---
    with subtab_image:
        st.markdown("### Image Cryptography")
        
        # We assume Encryption workflow first, which allows internal verification
        img_file = st.file_uploader("Upload Original Image", type=["png", "jpg", "jpeg"])
        k_img = st.text_input("Encryption Key", "MYSECRETKEY", type="password", key="kimg")
        
        if img_file and k_img:
            # Load Original
            img_orig = Image.open(img_file).convert("RGB")
            
            if st.button("Run Encryption & Analysis"):
                # 1. Encrypt
                arr = np.array(img_orig)
                shape = arr.shape
                enc_bytes = encrypt_bytes(arr.flatten(), k_img, current_sbox)
                img_enc = Image.fromarray(np.array(enc_bytes, dtype=np.uint8).reshape(shape), "RGB")
                
                # 2. Decrypt (Verification)
                dec_bytes = decrypt_bytes(enc_bytes, k_img, current_inv)
                img_dec = Image.fromarray(np.array(dec_bytes, dtype=np.uint8).reshape(shape), "RGB")
                
                # --- VISUALIZATION ROW 1: ORIGINAL vs ENCRYPTED ---
                st.subheader("1. Encryption Result (Confusion & Diffusion)")
                c1, c2 = st.columns(2)
                
                with c1:
                    st.image(img_orig, caption="Original Image", use_container_width=True)
                    st.pyplot(plot_rgb_histogram(img_orig))
                    
                with c2:
                    st.image(img_enc, caption=f"Encrypted ({selected_matrix_name})", use_container_width=True)
                    st.pyplot(plot_rgb_histogram(img_enc))
                
                st.divider()
                
                # --- VISUALIZATION ROW 2: ORIGINAL vs DECRYPTED ---
                st.subheader("2. Decryption Verification (Correctness)")
                c3, c4 = st.columns(2)
                
                with c3:
                    st.image(img_orig, caption="Original Input", use_container_width=True)
                    st.pyplot(plot_rgb_histogram(img_orig))
                    
                with c4:
                    st.image(img_dec, caption="Decrypted Result", use_container_width=True)
                    st.pyplot(plot_rgb_histogram(img_dec))

                # Download Button
                buf = io.BytesIO()
                img_enc.save(buf, format="PNG")
                st.download_button("Download Encrypted Image (PNG)", buf.getvalue(), "encrypted.png", "image/png")
