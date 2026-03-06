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

    # Extract metadata properties for visualization
    user_ids = [str(m.get('user_id', 'Unknown')) for m in meta]
    labels = [str(m.get('label', '0')) for m in meta]

    # Some logic to handle large datasets effectively in Plotly
    # For WebGL rendering performance, plotly express scatter is best

    print("Generating interactive 2D Radar Plotly HTML...")

    fig = go.Figure()

    # Add traces based on labels (if available to show black/white ground truths - though in your case it might be mostly unknown)
    # All points are now treated as a single dataset for visualization
    subset_x = embedding[:, 0]
    subset_y = embedding[:, 1]
    subset_users = user_ids

    # Since data is currently unlabeled, we treat everything as general behavioral data
    name = "Behavioral Data"
    color = "rgba(100, 150, 250, 0.4)" # Sleek blue

    fig.add_trace(go.Scattergl(
        x=subset_x,
        y=subset_y,
        mode='markers',
        name=name,
        marker=dict(
            size=4,
            color=color,
            line=dict(width=0)
        ),
        text=subset_users,
        hovertemplate=
        "<b>User ID:</b> %{text}<br>" +
        "<b>X:</b> %{x:.3f}<br>" +
        "<b>Y:</b> %{y:.3f}<br>" +
        "<extra></extra>"
    ))

    fig.update_layout(
        title="T-Detector AntiCheat: 2D Behavior Radar",
        xaxis_title="Behvioral Characteristic A",
        yaxis_title="Behvioral Characteristic B",
        plot_bgcolor="rgba(10, 10, 25, 1)", # Dark Mode Background
        paper_bgcolor="rgba(10, 10, 25, 1)",
        font=dict(color="white"),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    # Prepare Data JSON for direct injection into JS (bypassing Plotly scraping issues)
    data_list = []
    for i in range(len(embedding)):
        data_list.append({
            'x': float(embedding[i, 0]),
            'y': float(embedding[i, 1]),
            'id': user_ids[i]
        })
    data_json = json.dumps(data_list)

    # Add a search bar via custom HTML template
    import plotly.io as pio

    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    
    # Prepare Data JSON for direct injection into JS
    data_list = []
    for i in range(len(embedding)):
        data_list.append({
            'x': round(float(embedding[i, 0]), 4),
            'y': round(float(embedding[i, 1]), 4),
            'id': user_ids[i]
        })
    data_json = json.dumps(data_list)

    # Add a search bar via custom HTML template
    import plotly.io as pio

    html_content = pio.to_html(fig, full_html=True, include_plotlyjs='cdn')
    
    # Inject search bar and high-visibility highlighting logic
    search_html = f"""
    <div id="search-container" style="position: absolute; top: 50px; right: 10px; z-index: 1000; background: rgba(0,0,0,0.95); padding: 15px; border-radius: 10px; color: white; font-family: sans-serif; box-shadow: 0 0 25px rgba(0,0,0,1); border: 2px solid #00ffff; width: 280px;">
        <div style="margin-bottom: 12px; font-weight: bold; font-size: 16px; color: #00ffff; letter-spacing: 1px;">COMBAT RADAR v1.2</div>
        <div style="font-size: 11px; color: #888; margin-bottom: 8px;">* Sampled View (Max 100k samples for WebGL)</div>
        <input type="text" id="user-search" placeholder="Search AccID or UserID..." style="padding: 12px; width: 100%; border: 1px solid #00ffff; border-radius: 4px; background: #000; color: #fff; outline: none; box-sizing: border-box; font-family: monospace; font-size: 14px;">
        <div style="margin-top: 12px; display: flex; gap: 8px;">
            <button onclick="searchUser()" style="flex: 1; padding: 10px; cursor: pointer; background: #00ffff; color: #000; border: none; border-radius: 4px; font-weight: bold; text-transform: uppercase;">Scan</button>
            <button onclick="clearSearch()" style="padding: 10px; cursor: pointer; background: #444; color: white; border: none; border-radius: 4px;">Reset</button>
        </div>
        <div id="search-result" style="margin-top: 12px; font-size: 13px; color: #00ffff; font-family: monospace; white-space: pre-wrap;">RADAR STANDBY...</div>
    </div>
    <script>
    const RAW_DATA = {data_json};
    var blinkState = false;
    var blinkInterval = null;

    function clearSearch() {{
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (blinkInterval) {{ clearInterval(blinkInterval); blinkInterval = null; }}
        Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
        document.getElementById('search-result').innerText = "RADAR RESET.";
        document.getElementById('user-search').value = "";
    }}

    function searchUser() {{
        var searchTerm = document.getElementById('user-search').value.trim().toLowerCase();
        if (!searchTerm) return;
        
        var gd = document.getElementsByClassName('plotly-graph-div')[0];
        if (blinkInterval) {{ clearInterval(blinkInterval); blinkInterval = null; }}

        var results = RAW_DATA.filter(d => d.id.toLowerCase().includes(searchTerm));
        
        var resultDiv = document.getElementById('search-result');
        if (results.length > 0) {{
            // De-duplicate positions to handle overlapping points (clustered segments)
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

                // Label shows ID and count if multiple segments overlap
                var label = (count > 1) ? `<b>TARGET ×${{count}}</b>` : `<b>TARGET</b>`;
                var detail = group.map(g => g.id.split('_').slice(-1)[0]).join('\\n');
                
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
            
            resultDiv.innerHTML = "STATUS: LOCKED\\nMATCHES: " + results.length + "\\nCLUSTERS: " + Object.keys(posMap).length + "\\n(Check tooltips for full IDs)";
            resultDiv.style.color = "#00ffff";
        }} else {{
            resultDiv.innerHTML = "STATUS: NOT FOUND\\nTERM: " + searchTerm;
            resultDiv.style.color = "#ff4444";
            Plotly.relayout(gd, {{ shapes: [], annotations: [] }});
        }}
    }}
    
    document.getElementById('user-search').addEventListener("keyup", function(event) {{
        if (event.keyCode === 13) {{
            event.preventDefault();
            searchUser();
        }}
    }});
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
