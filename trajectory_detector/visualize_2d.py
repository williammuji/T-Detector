import numpy as np
import json
import os
import argparse

try:
    import umap
except ImportError:
    print("Please install umap-learn: pip install umap-learn")
    import sys
    sys.exit(1)

try:
    import plotly.graph_objects as go
    import plotly.express as px
except ImportError:
    print("Please install plotly: pip install plotly")
    import sys
    sys.exit(1)

def visualize_features(features_path, meta_path, output_html="radar_vision.html"):
    print(f"Loading high-dimensional features from {features_path}...")
    features = np.load(features_path)

    print(f"Loading metadata from {meta_path}...")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} dimensions.")

    # Run UMAP
    print("Running UMAP dimensionality reduction (this may take a few minutes for large datasets)...")
    reducer = umap.UMAP(
        n_neighbors=50,
        min_dist=0.1,
        n_components=2,
        metric='cosine',
        random_state=42
    )

    embedding = reducer.fit_transform(features)
    print("UMAP reduction complete.")

    # Run HDBSCAN Clustering
    print("Running HDBSCAN anomaly clustering...")
    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=15,
            min_samples=5,
            cluster_selection_epsilon=0.5,
            metric='euclidean'
        )
        cluster_labels = clusterer.fit_predict(embedding)
        print(f"Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} clusters.")
    except Exception as e:
        print(f"HDBSCAN clustering failed: {e}. Defaulting to single cluster.")
        cluster_labels = np.zeros(embedding.shape[0], dtype=int)

    # Extract metadata properties for visualization
    # User ID format is often like "accid-userid_mapid_time" or similar based on preprocess.py
    user_ids = []
    map_ids = []
    
    for m in meta:
        uid = str(m.get('user_id', 'Unknown'))
        user_ids.append(uid)
        
        # Try to extract map_id from user_id if it's there
        # parts = file.strip(".json").split("_") -> user_idx, map_idx, begin, end, session_id
        # Actually in meta it's just the 'user_id' which is likely the session id
        parts = uid.split("_")
        if len(parts) >= 2:
            map_ids.append(parts[1])
        else:
            map_ids.append("0")

    # Some logic to handle large datasets effectively in Plotly
    # For WebGL rendering performance, plotly express scatter is best

    print("Generating interactive 2D Radar Plotly HTML...")

    fig = go.Figure()

    unique_clusters = list(set(cluster_labels))
    unique_clusters.sort()

    # import plotly.express as px # Already imported at the top
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24
    
    for cluster_id in unique_clusters:
        mask = cluster_labels == cluster_id
        subset_x = embedding[mask, 0]
        subset_y = embedding[mask, 1]
        
        # Original user IDs for this cluster
        subset_users = [user_ids[i] for i, m in enumerate(mask) if m]
        
        # Enhanced text array for hovering
        hover_texts = []
        for i in range(len(subset_users)):
            if cluster_id == -1:
                label_name = "<b>Anomaly / Noise</b>"
            else:
                label_name = f"<b>Pattern Cluster #{cluster_id}</b>"
            hover_text = f"{label_name}<br><b>User ID:</b> {subset_users[i]}"
            hover_texts.append(hover_text)
            
        if cluster_id == -1:
            name = "Noise / Outliers"
            color = "rgba(150, 150, 150, 0.3)" # Gray for noise
            marker_size = 3
        else:
            name = f"Cluster {cluster_id}"
            color = colors[cluster_id % len(colors)]
            marker_size = 4

        fig.add_trace(go.Scattergl(
            x=subset_x,
            y=subset_y,
            mode='markers',
            name=name,
            marker=dict(
                size=marker_size,
                color=color,
                line=dict(width=0)
            ),
            text=hover_texts,
            hovertemplate=
            "%{text}<br>" +
            "<b>X:</b> %{x:.3f}<br>" +
            "<b>Y:</b> %{y:.3f}<br>" +
            "<extra></extra>"
        ))

    fig.update_layout(
        title="T-Detector AntiCheat: Behavioral Latent Pattern Map",
        xaxis_title="Latent Characteristic A",
        yaxis_title="Latent Characteristic B",
        plot_bgcolor="rgba(10, 10, 25, 1)", # Dark Mode Background
        paper_bgcolor="rgba(10, 10, 25, 1)",
        font=dict(color="white"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.7)",
            font=dict(size=10)
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Prepare Data JSON for direct injection into JS (bypassing Plotly scraping issues)
    data_list = []
    for i in range(len(embedding)):
        data_list.append({
            'x': round(float(embedding[i, 0]), 4),
            'y': round(float(embedding[i, 1]), 4),
            'id': user_ids[i],
            'map': map_ids[i],
            'cluster': int(cluster_labels[i])
        })
    data_json = json.dumps(data_list)

    # Add a search bar via custom HTML template
    import plotly.io as pio

    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    
    # Prepare Data JSON for direct injection into JS
    # This block is a duplicate from the original code, keeping it as is for faithful reproduction
    data_list = []
    for i in range(len(embedding)):
        data_list.append({
            'x': round(float(embedding[i, 0]), 4),
            'y': round(float(embedding[i, 1]), 4),
            'id': user_ids[i],
            'cluster': int(cluster_labels[i]) # Added cluster info here too
        })
    data_json = json.dumps(data_list)

    # Add a search bar via custom HTML template
    # This import is a duplicate from the original code, keeping it as is for faithful reproduction
    import plotly.io as pio

    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    
    # Inject search bar and high-visibility highlighting logic
    search_html = f"""
    <div id="search-container" style="position: absolute; top: 10px; right: 10px; z-index: 1000; background: rgba(10, 10, 30, 0.95); padding: 15px; border-radius: 12px; color: white; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; box-shadow: 0 8px 32px rgba(0,0,0,0.8); border: 1px solid rgba(0, 255, 255, 0.3); width: 320px; backdrop-filter: blur(10px);">
        <div style="margin-bottom: 12px; font-weight: bold; font-size: 18px; color: #00ffff; letter-spacing: 2px; text-align: center; text-shadow: 0 0 10px #00ffff;">T-DETECTOR RADAR</div>
        <div style="font-size: 11px; color: #aaa; margin-bottom: 10px; text-align: center;">Combat Pattern Analysis System</div>
        
        <div style="margin-bottom: 10px;">
           <label style="font-size: 11px; color: #00ffff; display: block; margin-bottom: 4px;">TARGET ID (AccID/UserID)</label>
           <input type="text" id="user-search" placeholder="Enter ID..." style="padding: 10px; width: 100%; border: 1px solid rgba(0, 255, 255, 0.5); border-radius: 6px; background: rgba(0, 20, 40, 0.8); color: #fff; outline: none; box-sizing: border-box; font-family: monospace; font-size: 13px;">
        </div>
        
        <div style="margin-bottom: 15px;">
           <label style="font-size: 11px; color: #00ffff; display: block; margin-bottom: 4px;">MAP ID</label>
           <input type="text" id="map-search" placeholder="Enter MapID..." style="padding: 10px; width: 100%; border: 1px solid rgba(0, 255, 255, 0.5); border-radius: 6px; background: rgba(0, 20, 40, 0.8); color: #fff; outline: none; box-sizing: border-box; font-family: monospace; font-size: 13px;">
        </div>

        <div style="display: flex; gap: 8px;">
            <button onclick="searchUser()" style="flex: 2; padding: 12px; cursor: pointer; background: linear-gradient(135deg, #00ffff, #0088ff); color: #000; border: none; border-radius: 6px; font-weight: bold; text-transform: uppercase; transition: 0.3s; box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);">Scan Area</button>
            <button onclick="clearSearch()" style="flex: 1; padding: 12px; cursor: pointer; background: rgba(60, 60, 80, 0.8); color: white; border: none; border-radius: 6px; transition: 0.3s;">Reset</button>
        </div>
        
        <div id="search-result" style="margin-top: 15px; padding: 10px; background: rgba(0,0,0,0.3); border-radius: 6px; font-size: 12px; color: #00ffff; font-family: monospace; min-height: 40px; border-left: 3px solid #00ffff;">SYSTEM READY.</div>

        <div id="analysis-tools" style="margin-top: 15px; display: none; padding-top: 15px; border-top: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 11px; color: #ff00ff, margin-bottom: 8px;">INTELLIGENCE TOOLS:</div>
            <div id="copy-plot" style="margin-bottom: 8px; font-size: 10px; background: rgba(255,0,255,0.1); padding: 5px; border: 1px dashed #ff00ff; cursor: pointer; border-radius: 4px;" title="Click to copy plot command">
               PLOT TRAJECTORY
            </div>
            <div id="copy-explain" style="font-size: 10px; background: rgba(255,255,0,0.1); padding: 5px; border: 1px dashed #ffff00; cursor: pointer; border-radius: 4px;" title="Click to copy explain command">
               EXPLAIN BEHAVIOR
            </div>
        </div>
    </div>
    <script>
    const RAW_DATA = {data_json};
    var blinkState = false;
    var blinkInterval = null;

    function clearSearch() {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (blinkInterval) {{ clearInterval(blinkInterval); blinkInterval = null; }}
        Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
        document.getElementById('search-result').innerText = "SYSTEM RESET.";
        document.getElementById('user-search').value = "";
        document.getElementById('map-search').value = "";
        document.getElementById('analysis-tools').style.display = 'none';
    }}

    function copyToClipboard(text) {{
        const el = document.createElement('textarea');
        el.value = text;
        document.body.appendChild(el);
        el.select();
        document.execCommand('copy');
        document.body.removeChild(el);
        alert("Command copied to clipboard!");
    }}

    function searchUser() {{
        var idTerm = document.getElementById('user-search').value.trim().toLowerCase();
        var mapTerm = document.getElementById('map-search').value.trim().toLowerCase();
        
        if (!idTerm && !mapTerm) return;
        
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (blinkInterval) {{ clearInterval(blinkInterval); blinkInterval = null; }}

        var results = RAW_DATA.filter(d => {{
            var idMatch = !idTerm || d.id.toLowerCase().includes(idTerm);
            var mapMatch = !mapTerm || d.map.toLowerCase().includes(mapTerm);
            return idMatch && mapMatch;
        }});
        
        var resultDiv = document.getElementById('search-result');
        var toolsDiv = document.getElementById('analysis-tools');

        if (results.length > 0) {{
            // De-duplicate positions
            var posMap = {{}};
            results.forEach(r => {{
                var key = r.x + "," + r.y;
                if (!posMap[key]) posMap[key] = [];
                posMap[key].push(r);
            }});

            var shapes = [];
            var annotations = [];
            Object.keys(posMap).forEach(key => {{
                var group = posMap[key];
                var r = group[0];
                var count = group.length;
                
                shapes.push({{
                    type: 'circle',
                    xref: 'x', yref: 'y',
                    x0: r.x - 0.25, y0: r.y - 0.25,
                    x1: r.x + 0.25, y1: r.y + 0.25,
                    line: {{ color: '#ff00ff', width: 4 }},
                    fillcolor: 'rgba(255, 0, 255, 0.2)'
                }});

                var label = (count > 1) ? `<b>TARGET ×${{count}}</b>` : `<b>TARGET</b>`;
                
                annotations.push({{
                    x: r.x, y: r.y,
                    xref: 'x', yref: 'y',
                    text: label + '<br><span style="font-size:10px;">ID:' + r.id.split('-')[0] + '</span>',
                    showarrow: true,
                    arrowhead: 2,
                    arrowcolor: '#ff00ff',
                    arrowsize: 2,
                    ax: 0, ay: -60,
                    font: {{ color: '#ffffff', size: 14 }},
                    bgcolor: '#ff00ff',
                    bordercolor: '#ffffff',
                    borderwidth: 2,
                    opacity: 1,
                    hovertext: group.map(g => g.id).join('\\n')
                }});
            }});

            Plotly.relayout(gd, {{ shapes: shapes, annotations: annotations }});

            blinkInterval = setInterval(function() {{
                blinkState = !blinkState;
                var color = blinkState ? '#ff00ff' : '#00ffff';
                var nextShapes = [];
                Object.keys(posMap).forEach(key => {{
                    var r = posMap[key][0];
                    nextShapes.push({{
                        type: 'circle',
                        xref: 'x', yref: 'y',
                        x0: r.x - (blinkState ? 0.6 : 0.25), 
                        y0: r.y - (blinkState ? 0.6 : 0.25),
                        x1: r.x + (blinkState ? 0.6 : 0.25), 
                        y1: r.y + (blinkState ? 0.6 : 0.25),
                        line: {{ color: color, width: 6 }},
                        fillcolor: blinkState ? 'rgba(0, 255, 255, 0.4)' : 'rgba(255, 0, 255, 0.4)'
                    }});
                }});
                Plotly.relayout(gd, {{ shapes: nextShapes }});
            }}, 400);
            
            resultDiv.innerHTML = "STATUS: LOCKED<br>MATCHES: " + results.length + "<br>CLUSTERS: " + Object.keys(posMap).length;
            resultDiv.style.color = "#00ffff";
            
            // Show analysis tools for the first match
            if (results.length > 0) {{
                const firstId = results[0].id;
                toolsDiv.style.display = 'block';
                document.getElementById('copy-plot').onclick = () => copyToClipboard(`python plot_trajectory_script.py --ids ${{firstId}}`);
                document.getElementById('copy-explain').onclick = () => copyToClipboard(`python explain_behavior.py --user_id ${{firstId}}`);
            }}

        }} else {{
            resultDiv.innerHTML = "STATUS: NOT FOUND<br>TERM: " + idTerm + " " + mapTerm;
            resultDiv.style.color = "#ff4444";
            toolsDiv.style.display = 'none';
            Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
        }}
    }}
    
    [document.getElementById('user-search'), document.getElementById('map-search')].forEach(el => {{
        el.addEventListener("keyup", function(event) {{
            if (event.keyCode === 13) {{
                event.preventDefault();
                searchUser();
            }}
        }});
    }});
    </script>
    </script>
    """
    
    final_html = html_content.replace('</body>', search_html + '</body>')
    
    with open(output_html, "w") as f:
        f.write(final_html)
    print(f"Visualization complete! Open '{output_html}' in your browser.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_processed/extracted_features.npy")
    parser.add_argument("--meta", type=str, default="/Users/williammuji/Codes/AntiCheat/T-Detector/train_data_processed/extracted_meta.json")
    parser.add_argument("--output", type=str, default="/Users/williammuji/Codes/AntiCheat/T-Detector/radar_vision.html")

    args = parser.parse_args()

    visualize_features(args.features, args.meta, args.output)
