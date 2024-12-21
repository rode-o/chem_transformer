import os
from subprocess import run

# Define paths for the scripts
data_gen_script = "data_gen.py"
csv_to_h5 = "csv_to_h5.py"
data_vis_script = "data_vis.py"

# Check if scripts exist
for script in [data_gen_script, data_vis_script]:
    if not os.path.exists(script):
        raise FileNotFoundError(f"{script} not found. Please ensure it's in the current directory.")

# Step 1: Generate synthetic data
print("Step 1: Generating synthetic data...")
run(["python", data_gen_script], check=True)

# Step 3: Convert csv to h5
print("Step 3: Convert csv to h5...")
run(["python", csv_to_h5], check=True)

# Step 3: Visualizing the data
print("Step 3: Visualizing the data...")
run(["python", data_vis_script], check=True)

print("Pipeline Complete")
