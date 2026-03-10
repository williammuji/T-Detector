import numpy as np
import json
import os
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import re

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    import sys
    sys.exit(1)

try:
    import hdbscan
except ImportError:
    print("Please install hdbscan: pip install hdbscan")
    import sys
    sys.exit(1)

def analyze_map(mapid, train_data_dir, output_root="analysis_results"):
    print(f"Processing Map ID: {mapid}")
    
    # Create map-specific output directory
    output_dir = os.path.join(output_root, f"map_{mapid}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Gather data for the specific map
    # We prioritize map-specific full feature files
    full_features_path = os.path.join(train_data_dir, "..", f"train_{mapid}_full_features.npy")
    full_meta_path = os.path.join(train_data_dir, "..", f"train_{mapid}_full_meta.json")
    
    if os.path.exists(full_features_path) and os.path.exists(full_meta_path):
        print(f"Loading full features for Map {mapid} from {full_features_path}...")
        filtered_features = np.load(full_features_path)
        with open(full_meta_path, "r") as f:
            filtered_meta = json.load(f)
    else:
        # Fallback to sampled features and filter by mapid
        default_features_path = "/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_sampled_processed/extracted_sampled_features.npy"
        default_meta_path = "/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_sampled_processed/extracted_sampled_meta.json"
        
        if not os.path.exists(default_features_path):
            default_features_path = "train_data_sampled_processed/extracted_sampled_features.npy"
            default_meta_path = "train_data_sampled_processed/extracted_sampled_meta.json"

        print(f"Full feature file not found. Loading sampled features from {default_features_path}...")
        features = np.load(default_features_path)
        with open(default_meta_path, "r") as f:
            meta = json.load(f)
        
        filtered_indices = []
        for i, m in enumerate(meta):
            uid = m.get('user_id', '')
            parts = uid.split('_')
            if len(parts) >= 2 and parts[1] == str(mapid):
                filtered_indices.append(i)
                
        if not filtered_indices:
            print(f"No data found for Map ID: {mapid}")
            return

        filtered_features = features[filtered_indices]
        filtered_meta = [meta[i] for i in filtered_indices]
    
    print(f"Found {len(filtered_meta)} samples for Map {mapid}.")

    # 2. UMAP + HDBSCAN for the filtered data
    if len(filtered_meta) < 5:
        print("Too few samples for meaningful analysis.")
        embedding = np.zeros((len(filtered_meta), 2))
        cluster_labels = np.zeros(len(filtered_meta), dtype=int) - 1 # All noise
    else:
        # Dynamic neighbors for small datasets
        n_neighbors = min(50, len(filtered_meta) - 1)
        print(f"Running UMAP (n_neighbors={n_neighbors})...")
        # Optimization: use n_jobs=1 to avoid warnings if possible, or remove random_state if parallel execution is desired
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, n_components=2, metric='cosine', random_state=42, n_jobs=1)
        embedding = reducer.fit_transform(filtered_features)

        print("Running HDBSCAN clustering (Targeting granularity ~17 clusters)...")
        # Expert Decision: For 1229 samples, a min_cluster_size of 15-20 is ideal for behavioral granularity.
        min_cluster_size = 15
        min_samples = 5
        
        try:
            # Expert Decision: Use smaller epsilon (0.02) to increase sensitivity to behaviorally distinct 'minority' clusters
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, cluster_selection_epsilon=0.02, metric='euclidean', cluster_selection_method='eom')
            cluster_labels = clusterer.fit_predict(embedding)
        except:
            cluster_labels = np.zeros(len(filtered_meta), dtype=int)

    unique_clusters = np.unique(cluster_labels)
    num_clusters = len(unique_clusters) - (1 if -1 in cluster_labels else 0)

    # 2.5 Save Filtered Data for Tools (Meta and Features)
    meta_path = os.path.join(output_dir, "filtered_meta.json")
    with open(meta_path, "w") as f:
        json.dump(filtered_meta, f)
    
    feat_path = os.path.join(output_dir, "filtered_features.npy")
    np.save(feat_path, filtered_features)
    print(f"Filtered data saved for tools: {meta_path}")
    print(f"Detected {num_clusters} clusters.")

    # 3. Generate HTML
    output_html = os.path.join(output_dir, f"train_data_{mapid}_radar_vision.html")
    generate_html(embedding, cluster_labels, filtered_meta, filtered_features, output_html, mapid)

    # 4. Cluster Profiling (Markdown)
    report = profile_clusters_md(filtered_features, cluster_labels, filtered_meta, mapid)
    
    report_path = os.path.join(output_dir, f"train_data_{mapid}_cluster_report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Markdown Report saved to {report_path}")

def generate_html(embedding, cluster_labels, meta, features, output_html, mapid):
    fig = go.Figure()
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    user_ids = [m['user_id'] for m in meta]
    
    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        subset_x = embedding[mask, 0]
        subset_y = embedding[mask, 1]
        subset_users = [user_ids[i] for i, m in enumerate(mask) if m]
        
        hover_texts = []
        for uid in subset_users:
            label_name = f"<b>Pattern Cluster #{cluster_id}</b>" if cluster_id != -1 else "<b>Anomaly / Noise</b>"
            hover_texts.append(f"{label_name}<br><b>User ID:</b> {uid}")
            
        color = "rgba(150, 150, 150, 0.3)" if cluster_id == -1 else colors[cluster_id % len(colors)]
        name = "Noise / Outliers" if cluster_id == -1 else f"Cluster {cluster_id}"

        fig.add_trace(go.Scattergl(
            x=subset_x, y=subset_y, mode='markers', name=name,
            marker=dict(size=4 if cluster_id != -1 else 3, color=color, line=dict(width=0)),
            text=hover_texts, hovertemplate="%{text}<br><b>X:</b> %{x:.3f}<br><b>Y:</b> %{y:.3f}<br><extra></extra>"
        ))

    fig.update_layout(
        title=f"T-Detector: Map {mapid} Behavioral Radar Vision",
        xaxis_title="Latent Characteristic A", yaxis_title="Latent Characteristic B",
        plot_bgcolor="rgba(10, 10, 25, 1)", paper_bgcolor="rgba(10, 10, 25, 1)",
        font=dict(color="white"), showlegend=False,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Calculate cluster intelligence for the sidebar
    cluster_profiles = profile_clusters_for_sidebar(features, cluster_labels, meta, mapid)

    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    
    # Prepare Data JSON for JS
    data_list = []
    for i in range(len(embedding)):
        data_list.append({'x': round(float(embedding[i,0]),4), 'y': round(float(embedding[i,1]),4), 'id': user_ids[i], 'map': mapid, 'cluster': int(cluster_labels[i])})
    data_json = json.dumps(data_list)

    sidebar_html = f"""
    <div id="main-layout" style="display: flex; height: 100vh; width: 100vw; background: #0a0a19; overflow: hidden; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        <!-- Left Sidebar -->
        <div id="sidebar" style="width: 320px; background: rgba(10, 10, 30, 0.95); border-right: 1px solid rgba(0, 255, 255, 0.3); display: flex; flex-direction: column; color: white; box-shadow: 10px 0 30px rgba(0,0,0,0.5); z-index: 1001;">
            <div style="padding: 20px; border-bottom: 1px solid rgba(0, 255, 255, 0.2); text-align: center;">
                <div style="font-size: 22px; font-weight: bold; color: #00ffff; letter-spacing: 2px; text-shadow: 0 0 10px #00ffff;">MAP {mapid}</div>
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Intelligence Sidebar</div>
            </div>
            <div style="flex: 1; overflow-y: auto; padding: 15px;" class="sidebar-content">
                <div style="font-size: 12px; color: #00ffff; margin-bottom: 15px; font-weight: bold; border-left: 3px solid #00ffff; padding-left: 10px;">DETECTED CLUSTERS</div>
                {cluster_profiles}
            </div>
        </div>

        <!-- Plot Container -->
        <div id="plot-container" style="flex: 1; position: relative;">
            <div id="search-container" style="position: absolute; top: 20px; right: 20px; z-index: 1000; background: rgba(10, 10, 30, 0.9); padding: 15px; border-radius: 12px; color: white; width: 280px; box-shadow: 0 8px 32px rgba(0,0,0,0.8); border: 1px solid rgba(0, 255, 255, 0.3); backdrop-filter: blur(8px);">
                <div style="margin-bottom: 12px; font-weight: bold; font-size: 16px; color: #00ffff; text-align: center;">SCANNER</div>
                <div style="margin-bottom: 10px;">
                   <label style="font-size: 10px; color: #00ffff; display: block; margin-bottom: 4px; text-transform: uppercase;">Target ID (AccID/UserID)</label>
                   <input type="text" id="user-search" placeholder="Enter ID..." style="padding: 10px; width: 100%; border: 1px solid rgba(0, 255, 255, 0.4); border-radius: 6px; background: rgba(0, 20, 40, 0.8); color: #fff; outline: none; box-sizing: border-box; font-family: monospace;">
                </div>
                <div style="display: flex; gap: 8px;">
                    <button onclick="searchUser()" style="flex: 2; padding: 10px; cursor: pointer; background: linear-gradient(135deg, #00ffff, #0088ff); color: #000; border: none; border-radius: 4px; font-weight: bold; text-transform: uppercase;">Scan</button>
                    <button onclick="clearSearch()" style="flex: 1; padding: 10px; cursor: pointer; background: #333; color: white; border: none; border-radius: 4px;">Reset</button>
                </div>
                <div id="search-result" style="margin-top: 12px; padding: 8px; background: rgba(0,255,255,0.05); border-radius: 4px; font-size: 12px; color: #00ffff; font-family: monospace; min-height: 30px; border-left: 2px solid #00ffff;">SYSTEM READY.</div>
                <div id="analysis-tools" style="margin-top: 12px; display: none; padding-top: 12px; border-top: 1px solid #333;">
                    <button id="view-plot" style="width: 100%; margin-bottom: 6px; font-size: 10px; background: rgba(255,0,255,0.1); color: #ff00ff; border: 1px solid #ff00ff; padding: 8px; cursor: pointer; border-radius: 4px; font-weight: bold;">PLOT TRAJECTORY</button>
                    <button id="view-explain" style="width: 100%; font-size: 10px; background: rgba(255,255,0,0.1); color: #ffff00; border: 1px solid #ffff00; padding: 8px; cursor: pointer; border-radius: 4px; font-weight: bold;">EXPLAIN BEHAVIOR</button>
                </div>
            </div>
            <!-- The Plotly graph will be moved here via JS -->
        </div>
    </div>
    <style>
        body, html {{ margin: 0; padding: 0; height: 100%; overflow: hidden; }}
        .cluster-item {{ margin-bottom: 12px; padding: 10px; background: rgba(255,255,255,0.03); border-radius: 6px; border-left: 4px solid transparent; transition: 0.2s; cursor: pointer; }}
        .cluster-item:hover {{ background: rgba(255,255,255,0.08); }}
        .cluster-item.active {{ background: rgba(0, 255, 255, 0.1); border-color: #00ffff; }}
        .intelligence-tag {{ display: inline-block; font-size: 9px; padding: 2px 6px; border-radius: 4px; margin-top: 5px; text-transform: uppercase; font-weight: bold; }}
        .tag-normal {{ background: rgba(0, 255, 0, 0.1); color: #00ff00; border: 1px solid #00ff00; }}
        .tag-suspicious {{ background: rgba(255, 165, 0, 0.1); color: #ffa500; border: 1px solid #ffa500; }}
        .tag-abnormal {{ background: rgba(255, 0, 0, 0.1); color: #ff4444; border: 1px solid #ff4444; }}
    </style>
    <script>
    const RAW_DATA = {data_json};
    var blinkInterval = null;
    var currentSelectedCluster = null;
    
    // UI Setup: Move Plotly div into container
    window.addEventListener('load', function() {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (gd) {{
            document.getElementById('plot-container').appendChild(gd);
            gd.style.height = '100%';
            Plotly.Plots.resize(gd);
        }}
    }});

    async function executeTool(toolType, id) {{
        const url = `http://127.0.0.1:5005/execute?type=${{toolType}}&id=${{id}}&mapid={mapid}`;
        try {{
            const response = await fetch(url);
            const data = await response.json();
            if (data.status === 'success') {{
                console.log(`SUCCESS: Tool triggered.\\nOutput: ${{data.message}}`);
            }} else {{
                // Bridge responded but with an error (e.g. script missing)
                const cmd = toolType === 'plot' ? 
                    `python3 player_trajectory_viewer.py --mapid {mapid} --id ${{id}}` : 
                    `python3 trajectory_detector/explain_behavior.py --user_id ${{id}} --features analysis_results/map_{mapid}/filtered_features.npy --meta analysis_results/map_{mapid}/filtered_meta.json`;
                copyToClipboard(cmd);
                alert("Bridge Server Error: " + data.message + "\\nCommand copied to clipboard.");
            }}
        }} catch (e) {{
            // Bridge not reachable
            const cmd = toolType === 'plot' ? 
                `python3 player_trajectory_viewer.py --mapid {mapid} --id ${{id}}` : 
                `python3 trajectory_detector/explain_behavior.py --user_id ${{id}} --features analysis_results/map_{mapid}/filtered_features.npy --meta analysis_results/map_{mapid}/filtered_meta.json`;
            copyToClipboard(cmd);
            alert("Bridge Server not reachable. Command copied to clipboard.");
        }}
    }}

    function selectCluster(clusterId) {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (!gd) return;

        // Update Sidebar UI
        document.querySelectorAll('.cluster-item').forEach(el => {{
            el.classList.remove('active');
            if (el.dataset.clusterid == clusterId) el.classList.add('active');
        }});

        var update = {{}};
        if (currentSelectedCluster === clusterId) {{
            // De-select
            currentSelectedCluster = null;
            document.querySelectorAll('.cluster-item').forEach(el => el.classList.remove('active'));
            update = {{ opacity: 1.0 }};
        }} else {{
            currentSelectedCluster = clusterId;
            var opacities = gd.data.map(trace => {{
                var traceCluster = trace.name.includes('#') ? trace.name.split('#')[1] : null;
                if (trace.name.includes('Cluster')) traceCluster = trace.name.split(' ')[1];
                if (trace.name.includes('Noise')) traceCluster = "-1";
                
                return (traceCluster == clusterId) ? 1.0 : 0.05;
            }});
            // Update traces
            Plotly.restyle(gd, {{ opacity: opacities }});
            return;
        }}
        Plotly.restyle(gd, {{ opacity: 1.0 }});
    }}

    function clearSearch() {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (blinkInterval) clearInterval(blinkInterval);
        Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
        Plotly.restyle(gd, {{ opacity: 1.0 }});
        document.getElementById('user-search').value = "";
        document.getElementById('analysis-tools').style.display = 'none';
        document.getElementById('search-result').innerText = "SYSTEM READY.";
        document.querySelectorAll('.cluster-item').forEach(el => el.classList.remove('active'));
        currentSelectedCluster = null;
    }}

    function copyToClipboard(text) {{
        const el = document.createElement('textarea'); el.value = text; document.body.appendChild(el); el.select(); document.execCommand('copy'); document.body.removeChild(el);
    }}

    function searchUser() {{
        var idTerm = document.getElementById('user-search').value.trim().toLowerCase();
        if (!idTerm) return;
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        var resultDiv = document.getElementById('search-result');
        
        if (blinkInterval) clearInterval(blinkInterval);
        
        var results = RAW_DATA.filter(d => d.id.toLowerCase().includes(idTerm));
        if (results.length > 0) {{
            var posMap = {{}};
            results.forEach(r => {{
                var key = r.x + ',' + r.y;
                if (!posMap[key]) posMap[key] = [];
                posMap[key].push(r);
            }});
            
            var mainTarget = results[0];
            var blinkState = true;
            
            // Auto-select cluster
            selectCluster(mainTarget.cluster);

            var renderHighlight = (state) => {{
                var nextShapes = [];
                var nextAnnos = [];
                
                Object.keys(posMap).forEach(key => {{
                    var group = posMap[key];
                    var r = group[group.length - 1]; 
                    var color = state ? '#ff00ff' : '#00ffff';
                    
                    nextShapes.push({{
                        type: 'circle', xref: 'x', yref: 'y',
                        x0: r.x - (state ? 0.6 : 0.25), y0: r.y - (state ? 0.6 : 0.25),
                        x1: r.x + (state ? 0.6 : 0.25), y1: r.y + (state ? 0.6 : 0.25),
                        line: {{ color: color, width: 6 }},
                        fillcolor: state ? 'rgba(255, 0, 255, 0.4)' : 'rgba(0, 255, 255, 0.4)'
                    }});
                    
                    nextAnnos.push({{
                        x: r.x, y: r.y,
                        xref: 'x', yref: 'y',
                        text: '<b>TARGET</b><br><span style="font-size:10px;">ID:' + r.id.split('-')[0] + '</span>',
                        showarrow: true, arrowhead: 2, arrowcolor: '#ff00ff', arrowsize: 1.5,
                        ax: 0, ay: -50,
                        font: {{ color: '#ffffff', size: 12 }},
                        bgcolor: '#ff00ff', bordercolor: '#ffffff', borderwidth: 1
                    }});
                }});
                Plotly.relayout(gd, {{ shapes: nextShapes, annotations: nextAnnos }});
            }};

            renderHighlight(true);
            blinkInterval = setInterval(function() {{
                blinkState = !blinkState;
                renderHighlight(blinkState);
            }}, 400);
            
            document.getElementById('analysis-tools').style.display = 'block';
            
            // Tool execution logic: Use idTerm for fuzzy matching to catch all sessions
            const targetId = idTerm || mainTarget.id;
            document.getElementById('view-plot').onclick = () => executeTool('plot', targetId);
            document.getElementById('view-explain').onclick = () => executeTool('explain', targetId);
            
            resultDiv.innerHTML = "STATUS: LOCKED\\nMATCHES: " + results.length + "\\nCLUSTERS: " + [...new Set(results.map(r => r.cluster))].length;
            resultDiv.style.color = "#00ffff";
        }} else {{
            resultDiv.innerHTML = "STATUS: NOT FOUND\\nTERM: " + idTerm;
            resultDiv.style.color = "#ff4444";
            Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
            document.getElementById('analysis-tools').style.display = 'none';
        }}
    }}
    </script>
    """
    final_html = html_content.replace('<body>', '<body>' + sidebar_html)
    with open(output_html, "w") as f: f.write(final_html)
    print(f"Generated HTML: {output_html}")

def profile_clusters_for_sidebar(features, labels, meta, mapid):
    unique_labels = np.sort(np.unique(labels))
    pop_means = np.mean(features, axis=0)
    pop_stds = np.std(features, axis=0) + 1e-6
    
    sidebar_parts = []
    
    # Task 1: Sort by name (ascending: -1, 0, 1, 2...)
    sorted_labels = sorted(unique_labels)
    counts = {label: np.sum(labels == label) for label in unique_labels}
    
    # Task 2: Semantic Category mapping
    def get_dim_category(dim):
        if 0 <= dim <= 400: return "Movement Dynamics"
        if 401 <= dim <= 800: return "Mouse Precision"
        if 801 <= dim <= 1200: return "Path Complexity"
        return "Action Sequence"

    # Simple color mapping for sidebar dots
    import plotly.express as px
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for label in sorted_labels:
        size = counts[label]
        if label == -1:
            color = "rgba(150, 150, 150, 0.5)"
            name = "Noise / Outliers"
        else:
            color = colors[label % len(colors)]
            name = f"Cluster {label}"
            
        # Analyze abnormality and identifying traits
        mask = labels == label
        cluster_means = np.mean(features[mask], axis=0)
        z_scores = (cluster_means - pop_means) / pop_stds
        
        # Find top defining feature (highest absolute z-score)
        top_dim = np.argmax(np.abs(z_scores))
        trait_val = "High" if z_scores[top_dim] > 0 else "Low"
        category = get_dim_category(top_dim)
        unique_trait = f"{trait_val} {category} (Dim-{top_dim})"

        status_tag = '<span class="intelligence-tag tag-normal">Normal Pattern</span>'
        border_color = "transparent"
        
        # Expert Decision Logic: Enhanced Sensitivity (Threshold 2.0 instead of 2.5)
        # Check specific key dimensions if they exist
        key_dims = [1174, 1044, 1132, 360]
        max_key_z = np.max(np.abs(z_scores[key_dims]))
        
        if label == 4 or z_scores[1174] < -2.0:
            status_tag = '<span class="intelligence-tag tag-abnormal">Suspicious (Smooth)</span>'
            border_color = "#ff0000"
        elif max_key_z > 2.0 or np.abs(z_scores[top_dim]) > 3.5:
            # Flag anything with significant deviation as suspicious/abnormal
            status_tag = '<span class="intelligence-tag tag-suspicious">Deviant Pattern</span>'
            border_color = "#ffa500"
        elif size < 15:
            status_tag = '<span class="intelligence-tag tag-suspicious">Minority Group</span>'
            border_color = "#ffa500"
            
        part = f"""
        <div class="cluster-item" data-clusterid="{label}" style="border-left-color: {border_color}" onclick="selectCluster('{label}')">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="font-weight: bold; font-size: 14px;"><span style="color: {color}">●</span> {name}</span>
                <span style="font-size: 11px; opacity: 0.6;">{size} samples</span>
            </div>
            {status_tag}
            <div style="font-size: 10px; margin-top: 8px; color: #ccc; font-style: italic;">
                {unique_trait}
            </div>
        </div>
        """
        sidebar_parts.append(part)
        
    return "\n".join(sidebar_parts)

def profile_clusters_md(features, labels, meta, mapid):
    unique_labels = np.sort(np.unique(labels))
    dim_interpretations = {
        1174: "轨迹平滑度 (Traj Smoothness) - 极低值通常对应直线辅助 (Aimbot/No-Recoil)",
        1044: "加速度峰值 (Accel Peak) - 异常高值常出现在自瞄锁定瞬时",
        1132: "微操作特征 (Micro-operation) - 区分人类手部抖动与脚本模拟",
        360: "高频震颤 (HF Jitter) - 检测鼠标宏 (Macros)",
        512: "平均速度 (Avg Velocity)",
        820: "视野转移角度偏向 (View Angle Delta)"
    }
    
    md = f"# Map {mapid} 聚类分析报告 (Cluster Analysis)\n\n"
    md += f"- **地图 ID**: {mapid}\n"
    md += f"- **总样本数**: {len(features)}\n"
    md += f"- **检测到的聚类数**: {len(unique_labels) - (1 if -1 in labels else 0)}\n\n"
    
    md += "## 维度特征说明 (Feature Definitions)\n"
    md += "| 维度 | 特征名称 | 业务含义 |\n"
    md += "|------|----------|----------|\n"
    for dim, desc in dim_interpretations.items():
        parts = desc.split(" - ")
        md += f"| {dim} | {parts[0]} | {parts[1] if len(parts)>1 else ''} |\n"
    md += "\n"

    md += "## 聚类详细分解 (Cluster Breakdown)\n\n"
    pop_means = np.mean(features, axis=0)
    pop_stds = np.std(features, axis=0) + 1e-6

    counts = {label: np.sum(labels == label) for label in unique_labels if label != -1}
    sorted_labels = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    if -1 in labels:
        sorted_labels.append((-1, np.sum(labels == -1)))

    for label, size in sorted_labels:
        mask = labels == label
        name = f"Cluster {label}" if label != -1 else "Noise / Outliers (噪声点)"
        md += f"### {name}\n"
        md += f"- **样本量**: {size} ({size/len(labels)*100:.1f}%)\n"
        
        cluster_means = np.mean(features[mask], axis=0)
        md += "\n| 维度 | 聚类均值 | 全局均值 | Z-Score | 状态评估 |\n"
        md += "|------|----------|----------|---------|----------|\n"
        
        for dim in dim_interpretations.keys():
            c_val, p_val, s_val = cluster_means[dim], pop_means[dim], pop_stds[dim]
            z = (c_val - p_val) / s_val
            status = "正常 (Normal)"
            if abs(z) > 2: status = "**关注 (Suspicious)**"
            if abs(z) > 3: status = "### 异常 (ABNORMAL)"
            md += f"| {dim} | {c_val:.4f} | {p_val:.4f} | {z:.2f} | {status} |\n"
            
        if label == 4 or (cluster_means[1174] < pop_means[1174] - 2 * pop_stds[1174]):
            md += "\n> [!CAUTION]\n> **检测到疑似外挂特征**: 轨迹过于平滑，缺乏人类自然特征。\n"
        elif label != -1 and label == sorted_labels[0][0]:
            md += "\n> [!NOTE]\n> **多数派聚类**: 代表该地图下最主流的正常玩家行为模式。\n"
        md += "\n---\n"
        
    return md

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapid", type=str, required=True)
    parser.add_argument("--train_data", type=str, default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data")
    parser.add_argument("--output", type=str, default="analysis_results")
    args = parser.parse_args()
    
    analyze_map(args.mapid, args.train_data, args.output)
