import re
import os
import numpy as np
import matplotlib
# Use 'Agg' backend to save files without a pop-up window
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# =========================================================
# CONFIGURATION
# =========================================================
FILE_KEY0 = "/VCD files/Proposed/trace_N1024_key0_full.vcd"
FILE_KEY1 = "/VCD files/Proposed/trace_N1024_key1_full.vcd"

# Output filename for the graph
SAVE_PATH = "/Proposed_SCA_Plot_N1024.png"

# Match the VCD timescale (1ps) -> Clock is 10ns = 10,000ps
CLK_PERIOD = 10000

# =========================================================
# 1. PARSER
# =========================================================
def parse_vcd_manual(vcd_file):
    print(f"Parsing {vcd_file}...")
    if not os.path.exists(vcd_file):
        print(f"❌ ERROR: File not found: {vcd_file}")
        return {}

    activity = {}
    current_time = 0

    # Regex for VCD parsing
    re_time = re.compile(r'^#(\d+)')
    re_change = re.compile(r'^([01xXzZ]|^[bB][01xXzZ]+)')

    with open(vcd_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue

            # Match Time (#10000)
            match_time = re_time.match(line)
            if match_time:
                current_time = int(match_time.group(1))
                continue

            # Match Value Change
            if not line.startswith('$') and re_change.match(line):
                # Calculate Cycle: integer division by 10,000
                cycle = current_time // CLK_PERIOD
                activity[cycle] = activity.get(cycle, 0) + 1

    return activity

# =========================================================
# 2. ANALYSIS
# =========================================================
db0 = parse_vcd_manual(FILE_KEY0)
db1 = parse_vcd_manual(FILE_KEY1)

if db0 and db1:
    # Align traces
    max_cycle = min(max(db0.keys()), max(db1.keys()))
    trace0 = np.array([db0.get(i, 0) for i in range(max_cycle)])
    trace1 = np.array([db1.get(i, 0) for i in range(max_cycle)])

    # Stats
    mu0, std0 = np.mean(trace0), np.std(trace0)
    mu1, std1 = np.mean(trace1), np.std(trace1)

    print(f"\nStats [Key 0]: Mean={mu0:.2f}")
    print(f"Stats [Key 1]: Mean={mu1:.2f}")

    # Plot
    print(f"Generating plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(trace0[:500], label="Key 0 (Fixed)", alpha=0.7)
    plt.plot(trace1[:500], label="Key 1 (Random)", alpha=0.7)
    plt.title(f"Switching Activity (N=1024, Proposed Design)")
    plt.xlabel("Clock Cycle")
    plt.ylabel("Toggle Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # SAVE instead of SHOW
    plt.savefig(SAVE_PATH)
    print(f"✅ Graph saved to: {SAVE_PATH}")

    # KL Divergence Calculation
    epsilon = 1e-9
    kl = np.log((std1+epsilon)/(std0+epsilon)) + (std0**2 + (mu0-mu1)**2)/(2*(std1**2)+epsilon) - 0.5
    kl = max(0, kl)

    print("\n" + "="*40)
    print(f"FINAL SCA STATUS (N=1024)")
    print("="*40)
    print(f"KL Divergence Score: {kl:.5f}")
    
    if kl > 0.5:
        print(">>> STATUS: 🔴 VULNERABLE (Significant Leakage Detected)")
    else:
        print(">>> STATUS: 🟢 SECURE (Masking effectively hidden key signature)")
    print("="*40)

else:
    print("❌ Analysis failed. Check if VCD files contain valid data.")
