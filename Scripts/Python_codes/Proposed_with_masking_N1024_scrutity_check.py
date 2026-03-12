import os
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# =========================================================================
# 1. CONFIGURATION & FILE PATHS
# =========================================================================
FILE_KEY0 = "/xsim/trace_N1024_key0_secure.vcd"
FILE_KEY1 = "/xsim/trace_N1024_key1_secure.vcd"

NUM_TRACES = 1000
THRESHOLD = 4.5

# =========================================================================
# 2. MEMORY-EFFICIENT VCD PARSER
# =========================================================================
def extract_trace_toggles(filepath, num_traces):
    print(f"Parsing {os.path.basename(filepath)}...")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ File not found: {filepath}")

    # Pass 1: Find the total simulation time (min and max timestamps)
    min_time = float('inf')
    max_time = 0
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                try:
                    t = int(line[1:].strip())
                    if t < min_time: min_time = t
                    if t > max_time: max_time = t
                except ValueError:
                    pass

    print(f"  -> Detected time range: {min_time} to {max_time} ns/ps")
    
    # Calculate the time window for each trace
    trace_window = (max_time - min_time) / num_traces
    if trace_window <= 0:
        trace_window = 1 
        
    trace_toggles = np.zeros(num_traces)
    
    # Pass 2: Count signal toggles and bin them into their respective traces
    current_time = min_time
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('$'): 
                continue # Skip VCD header and definitions
                
            if line.startswith('#'):
                current_time = int(line[1:])
            else:
                # Any other line is a signal state change (a toggle)
                idx = int((current_time - min_time) / trace_window)
                if idx >= num_traces: 
                    idx = num_traces - 1
                trace_toggles[idx] += 1
                
    print(f"  -> Extracted {num_traces} traces successfully.\n")
    return trace_toggles

# =========================================================================
# 3. MAIN ANALYSIS (WELCH'S T-TEST)
# =========================================================================
def main():
    print("========================================")
    print("STARTING TVLA (WELCH'S T-TEST) ANALYSIS")
    print("========================================")
    
    # 1. Parse Data
    try:
        traces_key0 = extract_trace_toggles(FILE_KEY0, NUM_TRACES)
        traces_key1 = extract_trace_toggles(FILE_KEY1, NUM_TRACES)
    except Exception as e:
        print(e)
        return

    # 2. Perform Welch's t-test
    # equal_var=False enforces Welch's t-test (assumes unequal variances between sets)
    t_stat, p_val = stats.ttest_ind(traces_key0, traces_key1, equal_var=False)
    
    max_abs_t_val = np.abs(t_stat)

    # 3. Print Statistics
    mean_k0, var_k0 = np.mean(traces_key0), np.var(traces_key0)
    mean_k1, var_k1 = np.mean(traces_key1), np.var(traces_key1)
    
    print("--- STATISTICAL PROFILE ---")
    print(f"Key 0 (All 0s): Mean = {mean_k0:.2f} toggles, Variance = {var_k0:.2f}")
    print(f"Key 1 (All 1s): Mean = {mean_k1:.2f} toggles, Variance = {var_k1:.2f}")
    print(f"Calculated |t-value|: {max_abs_t_val:.4f}")

    # 4. Security Status Evaluation
    print("\n========================================")
    print("FINAL SCA STATUS (N=1024 -PROPOSEDDESIGN WITH MASKING)")
    print("========================================")
    if max_abs_t_val > THRESHOLD:
        print(f">>> STATUS: 🔴 VULNERABLE (Leakage Detected, |t| > {THRESHOLD})")
        print("    The masking logic is failing to completely hide the key activity.")
    else:
        print(f">>> STATUS: 🟢 SECURE (No Significant Leakage, |t| <= {THRESHOLD})")
        print("    The distributions are statistically indistinguishable!")

    # =========================================================================
    # 4. VISUALIZATION (PLOT GENERATION)
    # =========================================================================
    plt.figure(figsize=(8, 6))
    
    # Plotting the t-value as a bar comparison against the threshold limits
    plt.bar(['TVLA Result'], [max_abs_t_val], color='blue' if max_abs_t_val <= THRESHOLD else 'red', width=0.3)
    
    # Draw Threshold Lines
    plt.axhline(y=THRESHOLD, color='red', linestyle='--', linewidth=2, label=f'Vulnerability Threshold (+{THRESHOLD})')
    
    # Formatting
    plt.ylabel('Absolute t-value (|t|)', fontsize=12)
    plt.title('Welch\'s t-test Leakage Assessment\nLTE_SMA_PSC_SECURE (N=1024)', fontsize=14, fontweight='bold')
    plt.ylim(0, max(THRESHOLD + 2, max_abs_t_val + 2))
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save Graph in the same folder as the VCD files
    save_dir = os.path.dirname(FILE_KEY0)
    plot_filename = os.path.join(save_dir, "TVLA_SCA_Plot_N1024_Secure.png")
    plt.savefig(plot_filename, dpi=300)
    print(f"\n✅ Graph saved to: {plot_filename}")

if __name__ == "__main__":
    main()
