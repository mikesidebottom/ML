import os
import json
import re
import shutil
import nbformat
from pathlib import Path
import base64

def convert_notebooks_to_md(source_dir, target_dir, repo_owner='CLDiego', repo_name='uom_fse_dl_workshop'):
    """
    Convert Jupyter notebooks to Markdown files for Jekyll website with Colab links.
    
    Args:
        source_dir: Directory containing the Jupyter notebooks
        target_dir: Directory to save the converted markdown files
        repo_owner: GitHub username for the repository
        repo_name: GitHub repository name
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Create directory for images
    img_dir = os.path.join(target_dir, '..', 'assets', 'images')
    os.makedirs(img_dir, exist_ok=True)
    
    # Find all .ipynb files in the source directory
    notebook_files = list(Path(source_dir).glob("*.ipynb"))
    
    notebook_map = {
        "SE01_CA_Intro_to_pytorch.ipynb": "session1.md",
        "SE02_CA_Artificial_neural_networks.ipynb": "session2.md",
        "SE03_CA_Training_neural_networks.ipynb": "session3.md",
        "SE04_CA_Convolutional_Neural_Networks.ipynb": "session4.md", 
        "SE05_CA_Transfer_Learning.ipynb": "session5.md"
    }
    
    title_map = {
        "SE01_CA_Intro_to_pytorch.ipynb": "SESSION 1: INTRODUCTION TO PYTORCH",
        "SE02_CA_Artificial_neural_networks.ipynb": "SESSION 2: ARTIFICIAL NEURAL NETWORKS",
        "SE03_CA_Training_neural_networks.ipynb": "SESSION 3: MODEL TRAINING & OPTIMIZATION",
        "SE04_CA_Convolutional_Neural_Networks.ipynb": "SESSION 4: CONVOLUTIONAL NEURAL NETWORKS", 
        "SE05_CA_Transfer_Learning.ipynb": "SESSION 5: TRANSFER LEARNING & U-NET"
    }
    
    for nb_path in notebook_files:
        notebook_filename = os.path.basename(nb_path)
        if notebook_filename not in notebook_map:
            continue
            
        md_filename = notebook_map[notebook_filename]
        notebook_name = os.path.splitext(md_filename)[0]
        
        # Read the notebook
        with open(nb_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Create markdown file path
        md_path = os.path.join(target_dir, md_filename)
        
        # Generate front matter for Jekyll
        session_num = re.search(r'session(\d+)', notebook_name).group(1)
        front_matter = f"""---
layout: notebook
title: "{title_map.get(notebook_filename, 'Workshop Notebook')}"
notebook_file: {notebook_filename}
permalink: /notebooks/session{session_num}/
---

"""
        
        # Process notebook cells
        md_content = []
        image_counter = 1
        
        for i, cell in enumerate(notebook.cells):
            if i == 0 and cell.cell_type == "markdown" and "![" in cell.source:
                # Skip the banner image - we'll include it in the layout
                continue
                
            if cell.cell_type == 'markdown':
                # Process markdown content: fix image links
                content = cell.source
                
                # Fix image paths to use GitHub raw URLs
                content = re.sub(
                    r'!\[(.*?)\]\((https://raw.githubusercontent.com/.*?)\)',
                    r'![\1](\2)',
                    content
                )
                
                # Fix GitHub-style image references
                content = re.sub(
                    r'<img src="https://github.com/CLDiego/uom_fse_dl_workshop/raw/main/(.*?)"(.*?)>',
                    r'<img src="https://raw.githubusercontent.com/CLDiego/uom_fse_dl_workshop/main/\1"\2>',
                    content
                )
                
                md_content.append(content)
                
            elif cell.cell_type == 'code':
                # Format code cells with syntax highlighting
                if cell.source.strip():  # Only add non-empty code cells
                    md_content.append(f"```python\n{cell.source.strip()}\n```")
                
                # Add cell outputs if any
                if cell.outputs:
                    output_content = []
                    for output in cell.outputs:
                        if output.output_type == 'stream':
                            output_content.append(f"```\n{output.text.strip()}\n```")
                            
                        elif output.output_type == 'execute_result':
                            if 'text/plain' in output.data:
                                text = output.data['text/plain']
                                output_content.append(f"```\n{text.strip()}\n```")
                                
                            # Handle HTML output
                            if 'text/html' in output.data:
                                html = output.data['text/html']
                                if isinstance(html, list):
                                    html = ''.join(html)
                                output_content.append(f"\n{html}\n")
                                
                        elif output.output_type == 'display_data':
                            # Handle images in output
                            if 'image/png' in output.data:
                                img_data = output.data['image/png']
                                img_filename = f"output_{notebook_name}_{image_counter}.png"
                                image_counter += 1
                                
                                # Save the image to assets folder
                                img_path = os.path.join(img_dir, img_filename)
                                with open(img_path, 'wb') as img_file:
                                    img_file.write(base64.b64decode(img_data))
                                
                                # Add image reference to markdown
                                output_content.append(f"\n![Output]({{ site.baseurl }}/assets/images/{img_filename})\n")
                    
                    if output_content:
                        md_content.append("<div class='cell-output'>\n" + "\n".join(output_content) + "\n</div>")
        
        # Write the markdown file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(front_matter)
            f.write('\n\n'.join(md_content))
        
        print(f"Created {md_path}")

if __name__ == "__main__":
    # Adjust paths to your repository structure
    notebooks_dir = "notebooks"
    target_dir = "_notebooks"
    
    convert_notebooks_to_md(notebooks_dir, target_dir)