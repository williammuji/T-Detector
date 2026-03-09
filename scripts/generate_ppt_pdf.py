import os
import subprocess
import tempfile
import re
import sys

def run_command(cmd, cwd=None):
    try:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {cmd}")
        print(f"Details: {e}")
        sys.exit(1)

def main():
    # Use the local path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    t_detector_root = os.path.dirname(script_dir)
    source_md = os.path.join(t_detector_root, "docs", "presentation_outline.md")

    if not os.path.exists(source_md):
        print(f"Source file not found: {source_md}")
        sys.exit(1)

    with open(source_md, 'r', encoding='utf-8') as f:
        content = f.read()

    # Strip frontmatter from source if exists
    if content.startswith('---'):
        parts = re.split(r'^---$', content, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            content = parts[2].strip()

    # Create temporary directory for processing
    tmp_dir = tempfile.mkdtemp(prefix="tdetector_pres_")
    tmp_md_path = os.path.join(tmp_dir, "build.md")
    print(f"Working in temporary directory: {tmp_dir}")

    # Process mermaid diagrams into SVGs
    mermaid_blocks = re.findall(r'```mermaid\n(.*?)\n```', content, re.DOTALL)

    for i, m_content in enumerate(mermaid_blocks):
        mmd_file = os.path.join(tmp_dir, f"chart_{i}.mmd")
        svg_file = os.path.join(tmp_dir, f"chart_{i}.svg")

        with open(mmd_file, 'w', encoding='utf-8') as f:
            f.write(m_content)

        # Run mmdc
        print(f"Generating SVG for chart {i}...")
        run_command(f"mmdc -i {mmd_file} -o {svg_file} -b transparent")

        # Replace mermaid block with image link
        svg_rel_path = f"chart_{i}.svg"
        content = content.replace(f"```mermaid\n{m_content}\n```", f"![chart]({svg_rel_path})")

    # Split into slides using Marp standard separator
    slides_raw = re.split(r'^---$', content, flags=re.MULTILINE)
    slides = [s.strip() for s in slides_raw if s.strip()]

    processed_slides = []
    for idx, slide in enumerate(slides):
        # Determine if slide is only a chart
        lines = [l.strip() for l in slide.split('\n') if l.strip()]

        has_h1 = any(l.startswith('# ') for l in lines)
        has_img = sum(1 for l in lines if l.startswith('![chart]')) == 1

        # If it's short and has a chart, assume it's a chart slide
        is_chart_only = has_h1 and has_img and len(lines) <= 5

        if idx == 0:
             processed_slides.append("<!-- _class: title-slide -->\n" + slide)
        elif is_chart_only:
             processed_slides.append("<!-- _class: centered-chart -->\n" + slide)
        else:
             processed_slides.append(slide)

    # Rejoin with horizontal rules for Marp
    final_markdown = "\n\n---\n\n".join(processed_slides)

    # Add Marp frontmatter and custom CSS
    frontmatter = """---
marp: true
theme: default
paginate: true
style: |
  section {
    justify-content: flex-start;
    align-items: flex-start;
    padding-top: 100px;
    padding-left: 50px;
    padding-right: 50px;
    background-color: #0d1117;
    color: #c9d1d9;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  }

  h1 {
    position: absolute;
    top: 30px;
    left: 50px;
    margin: 0;
    width: calc(100% - 100px);
    text-align: left;
    color: #58a6ff;
    border-bottom: 2px solid #30363d;
    padding-bottom: 10px;
    font-size: 1.6em;
  }

  h2 {
    color: #79c0ff;
  }

  strong {
    color: #f0883e;
  }

  code {
    background-color: #161b22;
    color: #ff7b72;
    border-radius: 4px;
    padding: 0.2em 0.4em;
  }

  pre {
    background-color: #010409;
    border: 1px solid #30363d;
    border-radius: 6px;
    padding: 15px;
    width: 95%;
  }

  table {
    width: 100%;
    border-collapse: collapse;
  }

  th {
    background-color: #161b22;
    color: #58a6ff;
  }

  td, th {
    border: 1px solid #30363d;
    padding: 8px;
  }

  /* Specific class for first page */
  section.title-slide {
    justify-content: center;
    align-items: center;
    padding-top: 0;
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
  }

  section.title-slide h1 {
    position: static;
    text-align: center;
    width: auto;
    border-bottom: none;
    font-size: 3em;
    color: #58a6ff;
    text-shadow: 0 0 20px rgba(88, 166, 255, 0.3);
  }

  section.title-slide h2 {
    text-align: center;
    color: #8b949e;
    margin-top: 20px;
  }

  /* Specific class for charts */
  section.centered-chart {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  section.centered-chart h1 {
    position: absolute; /* Keep title at top */
  }

  section.centered-chart p {
    text-align: center;
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    height: 70%;
    margin-top: 60px;
  }

  section.centered-chart img {
    max-height: 70vh;
    max-width: 85vw;
    object-fit: contain;
    border: 1px solid #30363d;
    border-radius: 8px;
    box-shadow: 0 0 30px rgba(0, 0, 0, 0.5);
  }
---

"""

    final_content = frontmatter + final_markdown

    with open(tmp_md_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    docs_dir = os.path.join(t_detector_root, "docs")
    os.makedirs(docs_dir, exist_ok=True)

    print("Generating PDF...")
    pdf_out = os.path.join(docs_dir, "presentation_outline.pdf")
    run_command(f"marp '{tmp_md_path}' --pdf -o '{pdf_out}' --allow-local-files", cwd=tmp_dir)

    print("Generating PPTX...")
    pptx_out = os.path.join(docs_dir, "presentation_outline.pptx")
    run_command(f"marp '{tmp_md_path}' --pptx -o '{pptx_out}' --allow-local-files", cwd=tmp_dir)

    print("Done! Generated files:")
    print(f"- {pdf_out}")
    print(f"- {pptx_out}")

if __name__ == "__main__":
    main()
