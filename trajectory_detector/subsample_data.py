import os
import shutil
import random
from pathlib import Path

def subsample(src_dir, dest_dir, sample_size):
    os.makedirs(dest_dir, exist_ok=True)
    all_files = os.listdir(src_dir)
    json_files = [f for f in all_files if f.endswith('.json')]
    
    # Shuffle and select
    random.seed(42)
    selected = random.sample(json_files, min(len(json_files), sample_size))
    
    print(f"Sampling {len(selected)} files from {src_dir} to {dest_dir}...")
    for f in selected:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dest_dir, f))

# Set up test env
base = "/Users/williammuji/Codes/AntiCheat/T-Detector/train_data"
test_base = "/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_sampled"

subsample(os.path.join(base, "move"), os.path.join(test_base, "move"), 3000)
subsample(os.path.join(base, "mouse"), os.path.join(test_base, "mouse"), 2000)

# Create a master dummy label mapping
print("Generating test label csv...")
with open(os.path.join(test_base, "label.csv"), "w") as f:
    f.write("id,label\n")
    for fname in os.listdir(os.path.join(test_base, "move")) + os.listdir(os.path.join(test_base, "mouse")):
        if fname.endswith(".json"):
            clean_name = fname.replace(".json", "")
            # Randomly assign 0 or 1 for POC visualization purposes to see if code works
            lbl = random.choice([0, 0, 0, 0, 1]) 
            f.write(f"{clean_name},{lbl}\n")

print("Done! Use train_data_sampled for testing the pipeline.")
