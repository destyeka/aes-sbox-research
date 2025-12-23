import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import base64
import io
import math
import os

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

def construct_sbox_dynamic(matrix):
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

def calculate_metrics(sbox):
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

    return {
        "NL": nl_val, "SAC": sac_val, "BIC-NL": bic_nl, "BIC-SAC": bic_sac,
        "LAP": lap_val, "DAP": dap_val, "DU": int(du_val), 
        "AD": int(ad_val), "TO": 0 # Placeholder for speed
    }

# ==========================================
# 4. ENCRYPTION HELPER FUNCTIONS
# ==========================================
def encrypt_bytes(data_bytes, key_string, sbox):
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

def calculate_entropy(image_pil):
    """Calculates Shannon Entropy of an image"""
    img_arr = np.array(image_pil)
    histogram = np.histogram(img_arr, bins=256, range=(0, 256))[0]
    histogram = histogram / histogram.sum()
    entropy = -np.sum([p * math.log2(p) for p in histogram if p > 0])
    return entropy

def plot_rgb_histogram(image_pil):
    img_arr = np.array(image_pil)
    fig, ax = plt.subplots(figsize=(6, 2.5))
    if len(img_arr.shape) == 3: 
        colors = ['red', 'green', 'blue']
        for i, color in enumerate(colors):
            ax.hist(img_arr[:, :, i].ravel(), bins=256, color=color, alpha=0.5, label=color.upper())
    else: 
        ax.hist(img_arr.ravel(), bins=256, color='gray', alpha=0.7, label='Gray')
    ax.set_title("Pixel Distribution")
    ax.set_xlim([0, 256])
    plt.tight_layout()
    return fig

# ==========================================
# 5. UI: PAGE FUNCTIONS
# ==========================================

def render_main_tool():
    st.title("🔐 AES S-Box Research Laboratory")
    st.markdown("""
    **Laboratory Mode:** Modify Affine Matrices, Analyze S-box Strength, and Test Encryption.
    Based on research: *AES S-box modification uses affine matrices exploration*.
    """)

    # --- SIDEBAR CONFIG (Only for Main Tool) ---
    st.sidebar.header("⚙️ Configuration")
    matrix_options = list(K_MATRICES.keys()) + ["🧪 Custom (Laboratory Mode)"]
    selected_option = st.sidebar.selectbox("Select K-Matrix", matrix_options, index=2)

    if selected_option == "🧪 Custom (Laboratory Mode)":
        st.sidebar.markdown("### Define Affine Matrix")
        custom_matrix = []
        default_vals = K_MATRICES["AES Standard"]
        for i in range(8):
            val = st.sidebar.number_input(f"Row {i} (0-255)", 0, 255, default_vals[i], key=f"row_{i}")
            custom_matrix.append(val)
        current_matrix = custom_matrix
        selected_matrix_name = "Custom Matrix"
    else:
        current_matrix = K_MATRICES[selected_option]
        selected_matrix_name = selected_option

    # Construct S-box
    current_sbox, current_inv = construct_sbox_dynamic(current_matrix)

    # --- PRE-COMPUTE COMPARISON ---
    all_metrics = []
    for name, mat in K_MATRICES.items():
        sb, _ = construct_sbox_dynamic(mat)
        m = calculate_metrics(sb)
        m["Name"] = name
        all_metrics.append(m)

    if selected_option == "🧪 Custom (Laboratory Mode)":
        m_cust = calculate_metrics(current_sbox)
        m_cust["Name"] = "Custom Matrix"
        all_metrics.append(m_cust)

    df_all = pd.DataFrame(all_metrics).set_index("Name")
    df_all["S-Value"] = (abs(df_all["SAC"] - 0.5) + abs(df_all["BIC-SAC"] - 0.5)) / 2

    # --- TABS ---
    tab_analysis, tab_demo = st.tabs(["📊 S-Box Analysis", "🔐 Encryption & Entropy"])

    # TAB 1: VISUALIZATION & COMPARISON
    with tab_analysis:
        st.header(f"1. S-Box Matrix: {selected_matrix_name}")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Decimal Representation")
            st.dataframe(pd.DataFrame(np.array(current_sbox).reshape(16, 16)), height=250)
        with c2:
            st.caption("Hexadecimal Representation")
            st.dataframe(pd.DataFrame([[f"{x:02X}" for x in r] for r in np.array(current_sbox).reshape(16, 16)]), height=250)

        st.divider()
        st.header("2. Strength Comparison")
        
        targets = {
            "NL": "Max (112)", "SAC": "0.5", "BIC-NL": "High", "BIC-SAC": "0.5",
            "LAP": "Min (0)", "DAP": "Min (0)", "DU": "Min (4)", "AD": "Max (7)", "S-Value": "Min (0)"
        }
        
        cols = ["AES Standard"]
        if "K-44 (Prop)" in df_all.index: cols.append("K-44 (Prop)")
        if selected_matrix_name not in cols: cols.append(selected_matrix_name)
        
        comp_df = df_all.loc[cols].T
        comp_df["Target (Ideal)"] = [targets.get(i, "-") for i in comp_df.index]
        
        def style_comparison(styler):
            styler.set_properties(subset=["Target (Ideal)"], **{'background-color': '#e6f3ff', 'font-weight': 'bold'})
            for idx in styler.index:
                data_cols = [c for c in styler.columns if c != "Target (Ideal)"]
                row_vals = styler.data.loc[idx, data_cols]
                best_val = None
                if idx in ["NL", "AD", "BIC-NL"]: best_val = row_vals.max()
                elif idx in ["SAC", "BIC-SAC"]:   best_val = row_vals.iloc[(row_vals - 0.5).abs().argmin()]
                else:                             best_val = row_vals.min()
                
                for col in data_cols:
                    if row_vals[col] == best_val:
                        styler.set_properties(subset=pd.IndexSlice[idx, col], **{'background-color': '#d4edda', 'color': 'green', 'font-weight': 'bold'})
            return styler

        st.table(style_comparison(comp_df.style.format("{:.5f}", subset=cols)))

    # TAB 2: ENCRYPTION
    with tab_demo:
        st.header("Real-World Deployment")
        subtab_text, subtab_image = st.tabs(["🔤 Text Tool", "🖼️ Image Laboratory"])

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

        with subtab_image:
            st.markdown("### Image Cryptography & Entropy Analysis")
            i_mode = st.radio("Image Operation", ["Encrypt Image", "Decrypt Image"], horizontal=True)
            
            if i_mode == "Encrypt Image":
                img_file = st.file_uploader("Upload Original", type=["png", "jpg", "jpeg"])
                k_img = st.text_input("Key", "MYSECRETKEY", type="password", key="kimg_enc")
                
                if img_file and k_img:
                    img_orig = Image.open(img_file).convert("RGB")
                    if st.button("Encrypt & Analyze"):
                        arr = np.array(img_orig)
                        shape = arr.shape
                        enc_bytes = encrypt_bytes(arr.flatten(), k_img, current_sbox)
                        img_enc = Image.fromarray(np.array(enc_bytes, dtype=np.uint8).reshape(shape), "RGB")
                        
                        ent_orig = calculate_entropy(img_orig)
                        ent_enc = calculate_entropy(img_enc)
                        
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.image(img_orig, caption=f"Original (Entropy: {ent_orig:.4f})", use_container_width=True)
                            st.pyplot(plot_rgb_histogram(img_orig))
                        with c_r:
                            st.image(img_enc, caption=f"Encrypted (Entropy: {ent_enc:.4f})", use_container_width=True)
                            st.pyplot(plot_rgb_histogram(img_enc))
                        
                        st.success(f"**Entropy Analysis:** Original `{ent_orig:.4f}` → Encrypted `{ent_enc:.4f}`. (Closer to 8.0 is better)")
                        
                        buf = io.BytesIO()
                        img_enc.save(buf, format="PNG")
                        st.download_button("Download Encrypted PNG", buf.getvalue(), "encrypted.png", "image/png")

            else: 
                enc_file = st.file_uploader("Upload Encrypted PNG", type=["png"])
                k_img_d = st.text_input("Key", "MYSECRETKEY", type="password", key="kimg_dec")
                if enc_file and k_img_d:
                    img_enc = Image.open(enc_file).convert("RGB")
                    if st.button("Decrypt & Analyze"):
                        arr = np.array(img_enc)
                        shape = arr.shape
                        dec_bytes = decrypt_bytes(arr.flatten(), k_img_d, current_inv)
                        img_dec = Image.fromarray(np.array(dec_bytes, dtype=np.uint8).reshape(shape), "RGB")
                        
                        c_l, c_r = st.columns(2)
                        with c_l:
                            st.image(img_enc, caption="Encrypted Input", use_container_width=True)
                            st.pyplot(plot_rgb_histogram(img_enc))
                        with c_r:
                            st.image(img_dec, caption="Decrypted Result", use_container_width=True)
                            st.pyplot(plot_rgb_histogram(img_dec))
                        
                        buf = io.BytesIO()
                        img_dec.save(buf, format="PNG")
                        st.download_button("Download Decrypted PNG", buf.getvalue(), "decrypted.png", "image/png")

def render_about_page():
    st.title("ℹ️ About the Project")
    
    # --- ADDED PHOTO SECTION ---
    # Attempts to load a local file named "project_photo.png" or displays a placeholder tip
    if os.path.exists("project_photo.jpg"):
        st.image("project_photo.png", caption="Project Team / Research Group", use_container_width=True)
    else:
        st.info("📷 **Tip:** To display a Team/University photo here, rename your image file to `project_photo.png` and upload it to the same folder as this script.")

    st.header("Project Title")
    st.markdown("""
    **AES S-box modification uses affine matrices exploration for increased S-box strength**
    """)
    
    st.header("Research Summary")
    st.info("""
    This research focuses on enhancing the cryptographic strength of the Advanced Encryption Standard (AES) 
    by replacing its standard S-box with a new one derived from an optimized Affine Matrix.
    
    **Methodology:**
    1. [cite_start]**Exploration:** Analyzed $2^{64}$ possible affine matrices[cite: 132, 1942].
    2. [cite_start]**Construction:** Generated candidate S-boxes using the irreducible polynomial $x^8 + x^4 + x^3 + x + 1$[cite: 39, 1191].
    3. [cite_start]**Validation:** Screened candidates for Bijectivity and Balance[cite: 433, 1957].
    4. [cite_start]**Testing:** Selected the best candidate ($S-box_{44}$) based on SAC, BIC, and Nonlinearity metrics [cite: 1376-1385].
    
    **Key Finding:** The proposed **$S-box_{44}$** achieves a Strict Avalanche Criterion (SAC) of **0.50073**, 
    [cite_start]which is closer to the ideal 0.5 than the standard AES S-box (0.50488)[cite: 1322, 1644].
    """)
    
    st.header("Our Team")
    # [cite_start]Data from [cite: 1843]
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Ara Bela Zulfa Laila**")
        st.caption("2304130050")
    
    with col2:
        st.markdown("**Khumaerotu Zahra**")
        st.caption("2304130072")
        
    with col3:
        st.markdown("**Desty Eka Syawfitri**")
        st.caption("2304130073")
    
    with col4:
        st.markdown("**Nabilla Marsha Amanda Putri**")
        st.caption("2304130081")
        
    st.divider()
    st.markdown("""
    **Institution:** Department of Informatics Engineering  
    Faculty of Mathematics and Natural Sciences  
    **Universitas Negeri Semarang (UNNES)** 2025
    """)
    
    st.header("References")
    st.markdown("""
    1. Alamsyah et al. (2025). *AES S-box modification uses affine matrices exploration for increased S-box strength*. Nonlinear Dynamics.
    """)

# ==========================================
# 6. MAIN APP LOGIC
# ==========================================

st.set_page_config(page_title="AES S-Box Project", layout="wide", page_icon="🔐")

# --- NAVIGATION ---
# Position at bottom of sidebar
st.sidebar.markdown("---")
page_selection = st.sidebar.radio("Navigate", ["🛠️ Main Tool", "ℹ️ About Project"])

if page_selection == "🛠️ Main Tool":
    render_main_tool()
else:
    render_about_page()
