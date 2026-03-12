import numpy as np
import matplotlib
# Forces matplotlib to not look for a GUI/Display window
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import re
import os

# =========================================================
# CONFIGURATION
# =========================================================
FILE_KEY0 = "/VCD files/Proposed/trace_N1024_key0_full.vcd"
FILE_KEY1 = "/VCD files/Proposed/trace_N1024_key1_full.vcd"

CLK_PERIOD = 10000     # Your clock is 10ns (10,000ps)
CYCLES_PER_OP = 22     # Latency of one encryption operation

# Path where the resulting image will be saved
SAVE_PATH = "KL_Divergence_Analysis_Proposed_without_masking.png"

def get_real_kl_profile(vcd0, vcd1):
    # 1. PARSE VCDs
    def parse(filename):
        if not os.path.exists(filename): 
            print(f"❌ ERROR: File not found: {filename}")
            return None
        print(f"Parsing {filename}...")
        counts = {}
        curr_time = 0
        re_time = re.compile(r'^#(\d+)')
        re_change = re.compile(r'^([01xXzZ]|^[bB][01xXzZ]+)')

        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                match = re_time.match(line)
                if match:
                    curr_time = int(match.group(1))
                elif not line.startswith('$') and re_change.match(line):
                    cyc = curr_time // CLK_PERIOD
                    counts[cyc] = counts.get(cyc, 0) + 1

        if not counts: return np.array([])
        max_c = max(counts.keys())
        return np.array([counts.get(i, 0) for i in range(max_c + 1)])

    t0 = parse(vcd0)
    t1 = parse(vcd1)

    if t0 is None or t1 is None or len(t0) == 0 or len(t1) == 0: 
        return np.zeros(CYCLES_PER_OP)

    # 2. FOLD TRACES (Reshape into [Num_Ops, Cycles_Per_Op])
    min_len = min(len(t0), len(t1))
    num_ops = min_len // CYCLES_PER_OP
    
    if num_ops == 0: 
        print("⚠️ Warning: Trace file too short for specified CYCLES_PER_OP.")
        return np.zeros(CYCLES_PER_OP)

    limit = num_ops * CYCLES_PER_OP
    mat0 = t0[:limit].reshape(num_ops, CYCLES_PER_OP)
    mat1 = t1[:limit].reshape(num_ops, CYCLES_PER_OP)

    # 3. CALCULATE KL
    kl_profile = []
    for c in range(CYCLES_PER_OP):
        col0 = mat0[:, c]
        col1 = mat1[:, c]

        mu0, std0 = np.mean(col0), np.std(col0)
        mu1, std1 = np.mean(col1), np.std(col1)

        # Add epsilon to avoid divide-by-zero
        epsilon = 1e-9
        if std0 < epsilon: std0 = epsilon
        if std1 < epsilon: std1 = epsilon

        term1 = np.log(std1/std0)
        term2 = (std0**2 + (mu0 - mu1)**2) / (2 * std1**2)
        kl = term1 + term2 - 0.5
        kl_profile.append(max(0, kl))

    return kl_profile

# =========================================================
# EXECUTE ANALYSIS
# =========================================================
real_kl_data = get_real_kl_profile(FILE_KEY0, FILE_KEY1)

# =========================================================
# PLOT AND SAVE
# =========================================================
print("Generating Plot...")
plt.figure(figsize=(10, 6))

# Check if seaborn style is available, otherwise use default
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')

plt.plot(real_kl_data, 'r-o', linewidth=2, label='Measured Leakage (Your Design)')
plt.axhline(y=0.5, color='k', linestyle='--', label='Vulnerability Threshold')
plt.title('Side Channel Analysis: KL Divergence per Cycle(Proposed without masking)', fontsize=14)
plt.xlabel('Clock Cycle within Operation')
plt.ylabel('KL Divergence Score')
plt.legend()
plt.grid(True)

# Save the figure to the current folder
plt.savefig(SAVE_PATH, dpi=300)
print(f"✅ Success! Plot saved as: {os.path.abspath(SAVE_PATH)}")

# Final Status Printout
max_kl = max(real_kl_data)
print("\n" + "="*30)
print(f"Max KL Divergence: {max_kl:.4f}")
if max_kl > 0.5:
    print("STATUS: 🔴 VULNERABLE")
else:
    print("STATUS: 🟢 SECURE")
print("="*30)
