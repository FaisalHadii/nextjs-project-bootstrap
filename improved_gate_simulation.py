import subprocess
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import uproot
import pandas as pd

# --- Configuration ---
GATE_EXECUTABLE = "Gate"  # Command to run GATE (ensure it's in your PATH)
BASE_MACRO_FILE = "test.mac" # Your finalized GATE macro template with FIXED detector position
OUTPUT_DIR = "./output/"    # Directory for simulation outputs
RESULTS_FILE = os.path.join(OUTPUT_DIR, "attenuation_results_150keV_fixedDet.csv") # To store intermediate results

# List of phantom thicknesses to test (in cm)
thicknesses_cm = [0.0001, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# Number of primary particles per simulation
N_PRIMARIES = 1000000 # Value from the last provided test.mac

# Energy filtering settings (for 100 Kev primary beam)
TARGET_ENERGY_MEV = 0.150
ENERGY_WINDOW_MEV = 0.001 # +/- 1 keV window around the target energy (0.169 - 0.171 MeV)
MIN_ENERGY_MEV = TARGET_ENERGY_MEV - ENERGY_WINDOW_MEV
MAX_ENERGY_MEV = TARGET_ENERGY_MEV + ENERGY_WINDOW_MEV

# --- Helper Functions ---

def modify_macro(base_macro, temp_macro, thickness_cm, output_root_file, output_stat_file):
    if not os.path.exists(base_macro):
        print(f"Error: Base macro file '{base_macro}' not found.")
        return False
    try:
        with open(base_macro, 'r') as infile, open(temp_macro, 'w') as outfile:
            for line in infile:
                stripped_line = line.strip()
                if stripped_line.startswith("/gate/phantom/geometry/setZLength"):
                    outfile.write(f"/gate/phantom/geometry/setZLength  {thickness_cm:.4f} cm\n")
                elif stripped_line.startswith("/gate/actor/phsp_actor/save"):
                    outfile.write(f"/gate/actor/phsp_actor/save {output_root_file}\n")
                elif stripped_line.startswith("/gate/actor/stat/save"):
                    outfile.write(f"/gate/actor/stat/save {output_stat_file}\n")
                elif stripped_line.startswith("/gate/phsp_volume/placement/setTranslation"):
                    outfile.write(line) # Write the original line from template
                elif stripped_line.startswith("/gate/application/setTotalNumberOfPrimaries"):
                    outfile.write(f"/gate/application/setTotalNumberOfPrimaries {N_PRIMARIES}\n")
                else:
                    outfile.write(line)
        return True
    except Exception as e:
        print(f"Error modifying macro '{temp_macro}': {e}")
        return False

def run_gate(temp_macro):
    print(f"Running GATE with {temp_macro}...")
    log_file_name = temp_macro + ".log"
    try:
        command = [GATE_EXECUTABLE, temp_macro]
        with open(log_file_name, 'w') as logfile:
            result = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, text=True, check=False)

        if result.returncode != 0:
            print(f"GATE exited with error code {result.returncode}. Check log file: {log_file_name}")
            return False
        else:
            print(f"GATE run completed successfully. Log file: {log_file_name}")
            return True
    except FileNotFoundError:
        print(f"Error: GATE executable '{GATE_EXECUTABLE}' not found. Is it in your PATH?")
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running GATE: {e}")
        return False

def get_total_and_filtered_counts(root_file, min_E, max_E):
    print(f"Reading {root_file}...")
    total_count = None
    filtered_count = None
    if not os.path.exists(root_file):
        print(f"Error: Output ROOT file '{root_file}' not found.")
        return total_count, filtered_count

    try:
        time.sleep(0.5) # Brief pause to ensure file is fully written
        with uproot.open(root_file) as file:
            tree_name = "PhaseSpace" # Default TTree name for PhaseSpaceActor
            if tree_name in file:
                tree = file[tree_name]
                total_count = tree.num_entries # Get total entries first
                print(f"  Total particles recorded: {total_count}")

                if "Ekine" in tree.keys() and total_count > 0:
                    print(f"  Filtering for energy {min_E:.3f}-{max_E:.3f} MeV...")
                    energies = tree["Ekine"].array(library="np")
                    mask = (energies >= min_E) & (energies <= max_E)
                    filtered_count = np.sum(mask)
                    print(f"  Found {filtered_count} particles within energy window.")
                elif "Ekine" not in tree.keys():
                    print(f"  Error: 'Ekine' branch not found in TTree '{tree_name}'. Cannot filter by energy.")
                    filtered_count = None
                else:
                    filtered_count = 0
                    print(f"  Found {filtered_count} particles within energy window.")

            else:
                print(f"Error: '{tree_name}' TTree not found in {root_file}")
                if os.path.getsize(root_file) == 0:
                    print("  ROOT file size is 0 bytes. GATE run might have crashed before writing.")

    except Exception as e:
        print(f"Error reading/filtering ROOT file {root_file}: {e}")
        if os.path.exists(root_file) and os.path.getsize(root_file) == 0:
            print("  ROOT file size is 0 bytes. GATE run might have crashed before writing.")

    return total_count, filtered_count

# --- Main Script ---

# Ensure output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    print(f"Created output directory: {OUTPUT_DIR}")

# Store results: list of tuples (thickness, total_count, filtered_count)
results = []

# --- Run Simulations ---
print(f"Starting GATE simulations for {len(thicknesses_cm)} thicknesses...")
for i, thickness in enumerate(thicknesses_cm):
    print("-" * 40)
    print(f"Processing thickness: {thickness:.4f} cm")
    temp_macro_file = f"temp_macro_{i}.mac"
    output_root_filename = os.path.join(OUTPUT_DIR, f"phantom_transmission_{thickness:.4f}cm.root")
    output_stat_filename = os.path.join(OUTPUT_DIR, f"stats_{thickness:.4f}cm.txt")

    if not modify_macro(BASE_MACRO_FILE, temp_macro_file, thickness, output_root_filename, output_stat_filename):
        print("Stopping script due to macro modification error.")
        exit()

    if not run_gate(temp_macro_file):
        print(f"GATE run failed for thickness {thickness:.4f} cm. Skipping. Check log file.")
        if os.path.exists(temp_macro_file):
            os.remove(temp_macro_file)
        continue

    total_count, filtered_count = get_total_and_filtered_counts(output_root_filename, MIN_ENERGY_MEV, MAX_ENERGY_MEV)

    if total_count is not None and filtered_count is not None:
        results.append((thickness, total_count, filtered_count))
    else:
        print(f"Could not get counts for thickness {thickness:.4f} cm. Check ROOT file and GATE log.")

    if os.path.exists(temp_macro_file):
        os.remove(temp_macro_file)

print("-" * 40)
print("All GATE simulations finished.")

# --- Data Analysis and Plotting ---
if not results:
    print("No results collected. Cannot plot or fit.")
    exit()

# Convert results to numpy array and pandas DataFrame
results_array = np.array(results)
thickness_data = results_array[:, 0]
total_counts_data = results_array[:, 1] # All particles entering detector
filtered_counts_data = results_array[:, 2] # Particles in 150Kev window

results_df = pd.DataFrame(results, columns=['Thickness_cm', 'Total_Counts', 'Filtered_Counts_150Kev'])
results_df.to_csv(RESULTS_FILE, index=False)
print(f"\nResults saved to {RESULTS_FILE}")
print(results_df) # Print the results table

# --- Calculate u values ---
I0 = 100000  # Given initial intensity
u_values = []
ln_I_over_I0 = []  # List to store ln(I/I0)

for thickness, filtered_count in zip(thickness_data, filtered_counts_data):
    if filtered_count > 0:  # Avoid division by zero or log of zero
        u = -1 / thickness * np.log(filtered_count / I0)
        u_values.append((thickness, u))
        ln_I_over_I0.append(np.log(filtered_count / I0))  # Calculate ln(I/I0)
    else:
        u_values.append((thickness, None))  # No valid u value if filtered_count is zero
        ln_I_over_I0.append(None)  # No valid ln(I/I0)

# Convert u_values to DataFrame for easier handling
u_df = pd.DataFrame(u_values, columns=['Thickness_cm', 'u_value'])
print(u_df)

# Optionally save u values to a CSV file
u_df.to_csv(os.path.join(OUTPUT_DIR, "u_values.csv"), index=False)
print(f"u values saved to {os.path.join(OUTPUT_DIR, 'u_values.csv')}")

# --- Plotting ln(I/I0) vs Thickness ---
plt.style.use('classic')  # Use a widely available classic style to avoid errors
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(thickness_data, ln_I_over_I0, marker='o', linestyle='-', color='#1f77b4', label='ln(I/I0) values', markersize=8, markerfacecolor='white', markeredgewidth=1.5)

# Fit line for u value
valid_indices = np.array([i for i, val in enumerate(ln_I_over_I0) if val is not None])
if len(valid_indices) > 1:
    thickness_fit = thickness_data[valid_indices]
    ln_I_over_I0_fit = np.array(ln_I_over_I0)[valid_indices]
    
    # Fit a linear model
    fit_params = np.polyfit(thickness_fit, ln_I_over_I0_fit, 1)
    fit_line = np.polyval(fit_params, thickness_data)

    # Calculate u from the slope
    u_fit = -fit_params[0]
    ln_I0 = fit_params[1]  # Intercept

    # Calculate R-squared
    residuals = ln_I_over_I0_fit - np.polyval(fit_params, thickness_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((ln_I_over_I0_fit - np.mean(ln_I_over_I0_fit))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # Plot the fitted line
    ax.plot(thickness_data, fit_line, color='#d62728', linestyle='--', linewidth=2, label=f'Fit: ln(I/I0) = {-u_fit:.4f} * Thickness + {ln_I0:.4f}')

    # Add equation and R2 to the plot
    equation_text = f'ln(I/I0) = {-u_fit:.4f} * Thickness + {ln_I0:.4f}\n$R^2$ = {r_squared:.4f}'
    # Remove any previous text boxes that do not contain R2 by only adding this one
    for txt in ax.texts:
        txt.remove()
    ax.text(0.05, 0.8, equation_text, fontsize=14, fontweight='bold', transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.5', fc='whitesmoke', ec='gray', alpha=0.8))

ax.set_title('Natural Logarithm of Intensity Ratio (ln(I/I0)) vs. Thickness', fontsize=16, fontweight='bold')
ax.set_xlabel('Thickness (cm)', fontsize=14)
ax.set_ylabel('ln(I/I0)', fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=12)
ax.grid(True, linestyle='--', linewidth=0.7, alpha=0.7)
ax.axhline(0, color='black', linewidth=0.8, linestyle='--')  # Add a horizontal line at y=0
ax.set_xlim(-1, 10)  # Expand x-axis from 1 cm before 0 to 10 cm
ax.set_ylim(min(ln_I_over_I0) - 0.5, max(ln_I_over_I0) + 0.5)  # Adjust y-axis limits based on data with padding
ax.legend(fontsize=12)
ax.set_facecolor('#eaeaf2')  # Different subtle background for plot area

plt.tight_layout()
plt.show()

# Ensure plot is shown even if script is run in non-interactive environment
plt.pause(0.001)
