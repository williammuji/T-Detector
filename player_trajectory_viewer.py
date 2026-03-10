import json
import matplotlib.pyplot as plt
import os
import argparse
import re

def plot_player_trajectory(mapid, player_id, train_data_dir, output_root="analysis_results"):
    print(f"Finding data for Player: {player_id} on Map: {mapid}")
    
    # Create player-specific output directory
    out_dir = os.path.join(output_root, f"player_{player_id}")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Search for files in train_data/move and train_data/mouse
    # Filename format: user_idx-accid_mapid_time_session.json
    # We need to support both accid and userid (which is the full filename usually)
    
    move_dir = os.path.join(train_data_dir, "move")
    mouse_dir = os.path.join(train_data_dir, "mouse")
    
    if not os.path.exists(move_dir) or not os.path.exists(mouse_dir):
        print(f"Error: {train_data_dir} does not contain move/ or mouse/ subdirectories.")
        return

    # List all files and filter
    move_files = os.listdir(move_dir)
    
    matches = []
    pid_clean = player_id.lower().strip()
    
    for f in move_files:
        if not f.endswith(".json"): continue
        
        # Check if mapid matches
        parts = f.replace(".json", "").split("_")
        if len(parts) < 2 or parts[1] != str(mapid):
            continue
            
        # Fuzzy matching: check if clean ID is in the filename (case-insensitive)
        if pid_clean in f.lower():
            matches.append(f)
        else:
            # Check accid specifically (the part after the dash in the first part)
            first_part = parts[0]
            acc_id = first_part.split("-")[-1] if "-" in first_part else first_part
            if pid_clean == acc_id.lower():
                matches.append(f)
            
    if not matches:
        print(f"No match found for Player ID: {player_id} on Map: {mapid}")
        return

    print(f"Found {len(matches)} matching sessions. Generating plots...")
    
    os.makedirs(out_dir, exist_ok=True)
    
    for match in matches:
        session_id = match.replace(".json", "")
        move_file = os.path.join(move_dir, match)
        mouse_file = os.path.join(mouse_dir, match)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Movement path
        with open(move_file, "r") as f:
            move_data = json.load(f)
        if move_data:
            mx = [d['x'] for d in move_data]
            my = [d['y'] for d in move_data]
            ax1.plot(mx, my, '-o', markersize=2, alpha=0.6, color='blue', label='Move')
            ax1.set_title("Movement Path")
            ax1.grid(True, alpha=0.3)
        
        # Mouse path
        if os.path.exists(mouse_file):
            with open(mouse_file, "r") as f:
                mouse_data = json.load(f)
            if mouse_data:
                msx = [d['x'] for d in mouse_data]
                msy = [d['y'] for d in mouse_data]
                ax2.plot(msx, msy, '-o', markersize=1, alpha=0.4, color='green', label='Mouse')
                ax2.set_title("Mouse Trajectory")
                ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No Mouse Data", ha='center')

        plt.suptitle(f"Map: {mapid} | Session: {session_id}", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        out_path = os.path.join(out_dir, f"map{mapid}_{session_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapid", type=str, required=True, help="Map ID to filter by")
    parser.add_argument("--id", type=str, required=True, help="User ID or Acc ID to filter by")
    parser.add_argument("--train_data", type=str, default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data")
    parser.add_argument("--output", type=str, default="analysis_results")
    args = parser.parse_args()
    
    plot_player_trajectory(args.mapid, args.id, args.train_data, args.output)
