import json
import matplotlib.pyplot as plt
import os
import argparse
import re

def get_cluster_map(html_path):
    try:
        with open(html_path, "r", encoding='utf-8') as f:
            content = f.read()
        # 更加健壮的正则匹配，捕获 ID 和 Cluster
        matches = re.findall(r'\"id\":\s*\"([^\"]+)\".*?\"cluster\":\s*(-?\d+)', content, re.DOTALL)
        return {m[0]: m[1] for m in matches}
    except Exception as e:
        print(f"Error reading HTML: {e}")
        return {}

def plot_trajectory(user_id, base_dir, out_dir, cluster_id="Unknown"):
    move_file = os.path.join(base_dir, "move", f"{user_id}.json")
    mouse_file = os.path.join(base_dir, "mouse", f"{user_id}.json")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    found_any = False
    
    # 绘制人物移动
    if os.path.exists(move_file):
        with open(move_file, "r") as f:
            data = json.load(f)
        if data:
            x = [d['x'] for d in data]
            y = [d['y'] for d in data]
            ax1.plot(x, y, '-o', markersize=3, alpha=0.7, color='#1f77b4', label='Path')
            ax1.set_title(f"Movement Path (Cluster: {cluster_id})")
            ax1.grid(True, linestyle='--', alpha=0.5)
            found_any = True
        else:
            ax1.text(0.5, 0.5, "Move Data Empty", ha='center')
    else:
        # 尝试检查目录是否存在其他命名的文件
        ax1.text(0.5, 0.5, f"Missing:\n{os.path.basename(move_file)}", ha='center', color='red')

    # 绘制鼠标移动
    if os.path.exists(mouse_file):
        with open(mouse_file, "r") as f:
            data = json.load(f)
        if data:
            x = [d['x'] for d in data]
            y = [d['y'] for d in data]
            ax2.plot(x, y, '-o', markersize=2, alpha=0.6, color='#2ca02c', label='Mouse')
            ax2.set_title(f"Mouse Trajectory (Cluster: {cluster_id})")
            ax2.grid(True, linestyle='--', alpha=0.5)
            found_any = True
        else:
            ax2.text(0.5, 0.5, "Mouse Data Empty", ha='center')
    else:
        ax2.text(0.5, 0.5, f"Missing:\n{os.path.basename(mouse_file)}", ha='center', color='red')

    plt.suptitle(f"User ID: {user_id}\nTarget Cluster: {cluster_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    os.makedirs(out_dir, exist_ok=True)
    clean_id = user_id.replace(" ", "_").replace("/", "_")
    out_path = os.path.join(out_dir, f"cluster{cluster_id}_{clean_id}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    
    if found_any:
        print(f"Successfully plotted: {out_path}")
    else:
        print(f"Warning: No data found for {user_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", nargs="+")
    parser.add_argument("--html", default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_radar_vision.html")
    parser.add_argument("--base_dir", default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data")
    parser.add_argument("--out_dir", default="/Users/williammuji/Codes/AntiCheat/T-Detector/plots_train_data")
    args = parser.parse_args()
    
    cluster_map = get_cluster_map(args.html)
    
    ids_to_plot = args.ids if args.ids else []
    
    # 如果没有指定 ID，则从 HTML 提取 Cluster 4, 9, 11 的样本
    if not ids_to_plot:
        target_clusters = ['4', '9', '11']
        found = {c: [] for c in target_clusters}
        for uid, cid in cluster_map.items():
            if cid in target_clusters and len(found[cid]) < 2:
                found[cid].append(uid)
                ids_to_plot.append(uid)
        # 加入 436001100 作为对比（人类基准）
        for uid in cluster_map:
            if '436001100' in uid:
                ids_to_plot.append(uid)

    for uid in ids_to_plot:
        cid = cluster_map.get(uid, "Unknown")
        plot_trajectory(uid, args.base_dir, args.out_dir, cid)
