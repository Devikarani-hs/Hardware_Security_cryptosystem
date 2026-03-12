import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless environment execution
import matplotlib.pyplot as plt
import re
import os
import gc
import time

# ==========================================
# 1. CONFIGURATION & PATHS
# ==========================================
CLK_PERIOD_PS = 10000  # 10ns Clock Period
CYCLES_PER_OP = 1050   # N=1024 operation cycles

# Unmasked / Baseline Design Paths
FILES_UNMASKED = [
     (0,  "/behav/xsim/trace_N1024_hd0.vcd"),
    (2,  "/behav/xsim/trace_N1024_hd2.vcd"),
    (4,  "behav/xsim/trace_N1024_hd4.vcd"),
    (8,  "/behav/xsim/trace_N1024_hd8.vcd"),
    (16, "/behav/xsim/trace_N1024_hd16.vcd"),
    (32, "/behav/xsim/trace_N1024_hd32.vcd")
]

# Masked / Secure Design Paths
FILES_MASKED = [
    (0,  "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd0_sweep.vcd"),
    (2,  "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd2_sweep.vcd"),
    (4,  "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd4_sweep.vcd"),
    (8,  "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd8_sweep.vcd"),
    (16, "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd16_sweep.vcd"),
    (32, "/Proposed with Masking/secure_vcd files/trace_sec_N1024_hd32_sweep.vcd")
]

# ==========================================
# 2. PARSING ENGINE
# ==========================================
def parse_vcd_to_profile(vcd_file):
    if not os.path.exists(vcd_file):
        print(f"  ⚠️ Warning: File not found: {os.path.basename(vcd_file)}")
        return None

    counts = {}
    curr_time = 0
    re_time = re.compile(r'^#(\d+)')
    re_val = re.compile(r'^([01xXzZ]|^[bB][01xXzZ]+)')

    with open(vcd_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            match = re_time.match(line)
            if match:
                curr_time = int(match.group(1))
            elif not line.startswith('$') and re_val.match(line):
                cyc = curr_time // CLK_PERIOD_PS
                counts[cyc] = counts.get(cyc, 0) + 1

    if not counts: return None
    
    max_c = max(counts.keys())
    trace = np.array([counts.get(i, 0) for i in range(max_c + 1)])
    
    num_ops = len(trace) // CYCLES_PER_OP
    if num_ops == 0: return None
    
    return trace[:num_ops*CYCLES_PER_OP].reshape(num_ops, CYCLES_PER_OP)

def calculate_peak_t_test(ref_matrix, target_matrix):
    max_t = 0
    n0, n1 = ref_matrix.shape[0], target_matrix.shape[0]
    
    for c in range(CYCLES_PER_OP):
        col0 = ref_matrix[:, c]
        col1 = target_matrix[:, c]
        
        mu0, var0 = np.mean(col0), np.var(col0)
        mu1, var1 = np.mean(col1), np.var(col1)
        
        denom = np.sqrt((var0/n0) + (var1/n1) + 1e-12)
        t_val = np.abs((mu0 - mu1) / denom)
        max_t = max(max_t, t_val)
        
    return max_t

def process_dataset(file_list, label):
    print(f"\n--- Extracting {label} ---")
    hd_values = []
    t_scores = []
    
    ref_hd, ref_file = file_list[0]
    ref_matrix = parse_vcd_to_profile(ref_file)
    
    if ref_matrix is None:
        print(f"❌ Failed to load Reference HD=0 for {label}")
        return [], []

    for hd, file_path in file_list:
        print(f"Processing HD={hd}...")
        tgt_matrix = parse_vcd_to_profile(file_path)
        if tgt_matrix is not None:
            score = calculate_peak_t_test(ref_matrix, tgt_matrix)
            hd_values.append(hd)
            t_scores.append(score)
            
    # Free memory
    del ref_matrix
    gc.collect()
    
    return hd_values, t_scores

# ==========================================
# 3. DATA EXTRACTION
# ==========================================
start_time = time.time()

hd_unmasked, t_unmasked = process_dataset(FILES_UNMASKED, "Unmasked (Baseline)")
hd_masked, t_masked = process_dataset(FILES_MASKED, "Masked (Secure)")

# ==========================================
# 4. PAPER-READY PLOTTING
# ==========================================
print("\nGenerating Unified High-Quality Plot...")

# Use a wider, shorter aspect ratio perfect for 2-column papers
fig, ax = plt.subplots(figsize=(7, 4.5))
plt.style.use('seaborn-v0_8-whitegrid')

# 1. Plot Unmasked Design (Red / Vulnerable)
if hd_unmasked:
    ax.plot(hd_unmasked, t_unmasked, marker='s', markersize=7, 
            linestyle='-', color='#e74c3c', linewidth=2.5, label='Proposed (Unmasked)')
    # Add Trendline
    zu = np.polyfit(hd_unmasked, t_unmasked, 1)
    pu = np.poly1d(zu)
    ax.plot(hd_unmasked, pu(hd_unmasked), color='#c0392b', linestyle='--', alpha=0.6)

# 2. Plot Masked Design (Green / Secure)
if hd_masked:
    ax.plot(hd_masked, t_masked, marker='o', markersize=7, 
            linestyle='-', color='#27ae60', linewidth=2.5, label='Proposed (Masked)')
    # Add Trendline
    zm = np.polyfit(hd_masked, t_masked, 1)
    pm = np.poly1d(zm)
    ax.plot(hd_masked, pm(hd_masked), color='#2ecc71', linestyle='--', alpha=0.6)

# 3. TVLA Threshold & Secure Zone Shading
ax.axhline(y=4.5, color='darkred', linestyle=':', linewidth=2, label='TVLA Threshold (+4.5)')
ax.axhspan(0, 4.5, color='#2ecc71', alpha=0.1, label='Secure Zone')

# 4. Formatting (No Title to save space)
ax.set_xlabel("Hamming Distance (HD)", fontsize=12, fontweight='bold')
ax.set_ylabel("Peak Absolute t-value (|t|)", fontsize=12, fontweight='bold')
ax.tick_params(axis='both', labelsize=11)
ax.grid(True, linestyle='--', alpha=0.6)

# Intelligent limits
all_hds = hd_unmasked + hd_masked
all_ts = t_unmasked + t_masked
ax.set_xlim(-1, max(all_hds) + 2 if all_hds else 35)
ax.set_ylim(0, max(all_ts) + 5 if all_ts else 40)

# 5. Compact Legend Placement
# Placed inside the plot in an empty space (usually upper left for this specific data curve)
ax.legend(loc='upper left', frameon=True, shadow=True, fontsize=10)

# Save the plot with tight bounding box to eliminate wasted white margins
out_file = "TVLA_Combined_HD_Analysis.png"
plt.savefig(out_file, dpi=300, facecolor='white', bbox_inches='tight')

print(f"✅ Success! Completed in {time.time() - start_time:.1f}s. Saved to: {out_file}")
