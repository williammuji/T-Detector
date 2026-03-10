import numpy as np
import json
import os
import argparse

def get_dim_category(dim):
    if 0 <= dim <= 400: return "Movement Dynamics"
    if 401 <= dim <= 800: return "Mouse Precision"
    if 801 <= dim <= 1200: return "Path Complexity"
    return "Action Sequence"

def explain_player_behavior(features_path, meta_path, user_id, focus_dims=None):
    """
    提供全面的玩家行为特征画像。
    """
    if focus_dims is None:
        focus_dims = [1174, 1044, 1132, 360]
        
    print(f"Loading data from {features_path}...")
    features = np.load(features_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # 找到所有匹配的玩家索引
    matches = []
    pid_clean = str(user_id).lower().strip()
    for i, m in enumerate(meta):
        if pid_clean in str(m['user_id']).lower():
            matches.append((i, m['user_id']))
            
    if not matches:
        print(f"Error: User ID containing '{user_id}' not found in metadata.")
        return

    print(f"\n找到 {len(matches)} 个匹配会话。正在生成综合画像...")

    # 获取维度定义
    dim_logs = {
        1174: "平滑度 (Traj Smoothness)",
        1044: "加速度峰值 (Accel Peak)",
        1132: "微操作特征 (Micro-operation)",
        360: "高频震颤 (HF Jitter)"
    }

    # 计算整体基准
    pop_means = np.mean(features, axis=0)
    pop_stds = np.std(features, axis=0) + 1e-6

    for i, (p_idx, full_id) in enumerate(matches):
        p_features = features[p_idx]
        z_scores = (p_features - pop_means) / pop_stds
        
        print(f"\n{'='*60}")
        print(f" 会话 {i+1} 分析: {full_id}")
        print(f"{'='*60}")
        
        # 1. 核心业务维度分析
        print("\n[关键业务指标 (Key Business Metrics)]")
        print(f"{'维度':<8} | {'数值':<10} | {'Z-Score':<8} | {'状态评估':<12} | {'含义'}")
        print("-" * 80)
        for dim in focus_dims:
            val = p_features[dim]
            z = z_scores[dim]
            status = "正常 (Normal)"
            if abs(z) > 3: status = "高度异常 (Extreme)"
            elif abs(z) > 2: status = "显著偏离 (Suspicious)"
            
            dim_name = dim_logs.get(dim, f"Dim-{dim}")
            print(f"{dim:<8} | {val:<10.4f} | {z:<8.2f} | {status:<12} | {dim_name}")

        # 2. 自动捕获最显著的异常维度
        print("\n[异常维度自动捕获 (Top Anomalies)]")
        # 排除已展示的 focus_dims
        candidate_dims = [d for d in range(len(z_scores)) if d not in focus_dims]
        top_indices = sorted(candidate_dims, key=lambda d: abs(z_scores[d]), reverse=True)[:5]
        
        for idx in top_indices:
            z = z_scores[idx]
            trait = "偏高 (High)" if z > 0 else "偏低 (Low)"
            category = get_dim_category(idx)
            print(f" - Dim-{idx:<5}: Z-Score {z:>6.2f} -> {trait} {category}")

        # 3. 综合评估建议
        print("\n[综合评估 (Aggregated Insight)]")
        extreme_dims = np.sum(np.abs(z_scores) > 3)
        suspicious_dims = np.sum(np.abs(z_scores) > 2)
        
        if extreme_dims > 5 or z_scores[1174] < -2.5:
            print("结论: 高风险。该玩家表现出极度非人特征，建议立即对照 Radar 聚类并执行追踪。")
        elif suspicious_dims > 10:
            print("结论: 中风险。虽无极端单项，但整体行为模式与常人偏离点过多，可能为定制化脚本或资深外挂。")
        else:
            print("结论: 风险较低。未发现系统性作弊特征，可能为一般性行为波动。")

    print("\n" + "="*60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, required=True)
    parser.add_argument("--features", type=str, default="train_full_features.npy")
    parser.add_argument("--meta", type=str, default="train_full_meta.json")
    # 增加 dimension 参数兼容性，虽不再是唯一核心，但可作为重点关注维度传入
    parser.add_argument("--dimension", type=int, default=1174)
    
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Note: Ignoring unrecognized arguments: {unknown}")
    
    # 兼容传入的单个维度，将其加入关注列表
    focus = [1174, 1044, 1132, 360]
    if args.dimension not in focus:
        focus.append(args.dimension)
        
    explain_player_behavior(args.features, args.meta, args.user_id, focus_dims=focus)
