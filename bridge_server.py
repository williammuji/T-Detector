import http.server
import socketserver
import urllib.parse
import json
import subprocess
import os
import sys

PORT = 5005

class BridgeHandler(http.server.BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'X-Requested-With')
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-type', 'application/json')
        self.end_headers()

        parsed_path = urllib.parse.urlparse(self.path)
        if parsed_path.path == '/execute':
            params = urllib.parse.parse_qs(parsed_path.query)
            tool_type = params.get('type', [''])[0]
            player_id = params.get('id', [''])[0]
            map_id = params.get('mapid', [''])[0]

            if not tool_type or not player_id or not map_id:
                self.wfile.write(json.dumps({'status': 'error', 'message': 'Missing parameters'}).encode())
                return

            try:
                if tool_type == 'plot':
                    cmd = [sys.executable, "player_trajectory_viewer.py", "--mapid", map_id, "--id", player_id]
                    print(f"Executing: {' '.join(cmd)}")
                    subprocess.run(cmd, check=True)
                    
                    # Logic to open the result folder
                    # Result folder naming: analysis_results/player_{id}
                    # We might have fuzzy id, so we should look for the actual folder
                    output_dir = os.path.join("analysis_results", f"player_{player_id}")
                    if not os.path.exists(output_dir):
                        # Try to find a folder that contains the player_id
                        results_base = "analysis_results"
                        folders = [f for f in os.listdir(results_base) if f.startswith("player_") and player_id in f]
                        if folders:
                            output_dir = os.path.join(results_base, folders[0])
                    
                    if os.path.exists(output_dir):
                        if sys.platform == 'darwin':
                            subprocess.run(['open', output_dir])
                        elif sys.platform == 'win32':
                            os.startfile(output_dir)
                        else:
                            subprocess.run(['xdg-open', output_dir])
                    
                    self.wfile.write(json.dumps({'status': 'success', 'message': f'Plot generated and opened: {output_dir}'}).encode())

                elif tool_type == 'explain':
                    # Find the correct full feature and meta paths
                    features_path = os.path.abspath("train_data_sampled_processed/extracted_sampled_features.npy")
                    meta_path = os.path.abspath("train_data_sampled_processed/extracted_sampled_meta.json")
                    
                    # Check if map-specific filtered data exists (preferred)
                    map_feat = f"analysis_results/map_{map_id}/filtered_features.npy"
                    map_meta = f"analysis_results/map_{map_id}/filtered_meta.json"
                    
                    if os.path.exists(map_feat) and os.path.exists(map_meta):
                        features_path = os.path.abspath(map_feat)
                        meta_path = os.path.abspath(map_meta)
                    else:
                        # Fallback to map-specific full features if available
                        full_feat = f"train_{map_id}_full_features.npy"
                        if os.path.exists(full_feat):
                            features_path = os.path.abspath(full_feat)
                            # Check for meta in analysis results
                            if os.path.exists(map_meta):
                                meta_path = os.path.abspath(map_meta)

                    cmd = [sys.executable, "trajectory_detector/explain_behavior.py", "--user_id", player_id, "--features", features_path, "--meta", meta_path]
                    print(f"Executing: {' '.join(cmd)}")
                    
                    # Ensure logs folder exists
                    os.makedirs("analysis_results/logs", exist_ok=True)
                    log_file = os.path.abspath(f"analysis_results/logs/explain_{player_id}.md")
                    
                    with open(log_file, "w") as f:
                        subprocess.run(cmd, stdout=f, check=True)
                    
                    if sys.platform == 'darwin':
                        subprocess.run(['open', log_file])
                    elif sys.platform == 'win32':
                        os.startfile(log_file)
                    else:
                        subprocess.run(['xdg-open', log_file])

                    self.wfile.write(json.dumps({'status': 'success', 'message': f'Analysis generated and opened: {log_file}'}).encode())
                
                else:
                    self.wfile.write(json.dumps({'status': 'error', 'message': 'Invalid tool type'}).encode())

            except Exception as e:
                self.wfile.write(json.dumps({'status': 'error', 'message': str(e)}).encode())
        else:
            self.wfile.write(json.dumps({'status': 'error', 'message': 'Unknown path'}).encode())

def kill_existing_process(port):
    """Kills any process currently listening on the specified port."""
    try:
        # Find PIDs using the port (works on macOS/Linux)
        result = subprocess.run(["lsof", "-ti", f":{port}"], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                print(f"Cleaning up existing process {pid} on port {port}...")
                subprocess.run(["kill", "-9", pid], check=False)
    except Exception as e:
        # Silently fail if lsof is not available or no process found
        pass

if __name__ == "__main__":
    kill_existing_process(PORT)
    with socketserver.TCPServer(("", PORT), BridgeHandler) as httpd:
        print(f"Bridge Server started at http://127.0.0.1:{PORT}")
        print("Keep this terminal open and refresh your Radar HTML to enable direct execution.")
        httpd.serve_forever()
