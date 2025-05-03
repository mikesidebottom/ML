import os
import json
import re
import shutil
from pathlib import Path
import nbformat

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
    
    # Find all .ipynb files in the source directory
    notebook_files = list(Path(source_dir).glob("*.ipynb"))
    
    for nb_path in notebook_files:
        print(f"Converting {nb_path}...")
        notebook_filename = os.path.basename(nb_path)
        notebook_name = os.path.splitext(notebook_filename)[0]
        
        # Read the notebook
        with open(nb_path, 'r', encoding='utf-8') as f:
            notebook = nbformat.read(f, as_version=4)
        
        # Create markdown file name
        md_filename = notebook_name + ".md"
        md_path = os.path.join(target_dir, md_filename)
        
        # Generate front matter for Jekyll
        front_matter = f"""---
layout: notebook
title: "{generate_title(notebook_name)}"
permalink: /notebooks/{notebook_name}/
notebook_file: {notebook_filename}
---

"""
        
        # Process notebook cells
        md_content = []
        for cell in notebook.cells:
            if cell.cell_type == 'markdown':
                # Add markdown content directly
                md_content.append(cell.source)
            elif cell.cell_type == 'code':
                # Format code cells with syntax highlighting
                code_output = f"```python\n{cell.source}\n```"
                md_content.append(code_output)
                
                # Add cell outputs if any
                if cell.outputs:
                    md_content.append("<div class='cell-output'>")
                    for output in cell.outputs:
                        if output.output_type == 'stream':
                            md_content.append(f"```\n{output.text}\n```")
                        elif output.output_type == 'execute_result':
                            if 'text/plain' in output.data:
                                md_content.append(f"```\n{output.data['text/plain']}\n```")
                        elif output.output_type == 'display_data':
                            if 'image/png' in output.data:
                                # For images, we need to save them separately
                                img_dir = os.path.join(target_dir, 'images', notebook_name)
                                os.makedirs(img_dir, exist_ok=True)
                                img_filename = f"output_{len(md_content)}.png"
                                img_path = os.path.join(img_dir, img_filename)
                                
                                # Note: In a real script, you'd decode the base64 image data and save it
                                # Here we're just adding a placeholder for the image
                                md_content.append(f"![Output]({{{{ site.baseurl }}}}/assets/images/{notebook_name}/{img_filename})")
                    md_content.append("</div>")
        
        # Write the markdown file
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(front_matter)
            f.write('\n\n'.join(md_content))
        
        print(f"Created {md_path}")

def generate_title(notebook_name):
    """Generate a nice title from notebook filename."""
    # Remove any numbers and underscores from the beginning
    clean_name = re.sub(r'^[0-9_]+', '', notebook_name)
    # Replace underscores with spaces
    clean_name = clean_name.replace('_', ' ')
    # Capitalize words
    title = ' '.join(word.capitalize() for word in clean_name.split())
    
    # Map known session names to proper titles
    session_map = {
        "session1": "SESSION 1: INTRODUCTION TO PYTORCH",
        "session2": "SESSION 2: ARTIFICIAL NEURAL NETWORKS",
        "session3": "SESSION 3: MODEL TRAINING & OPTIMIZATION",
        "session4": "SESSION 4: CONVOLUTIONAL NEURAL NETWORKS",
        "session5": "SESSION 5: TRANSFER LEARNING & U-NET"
    }
    
    return session_map.get(notebook_name, title)

def copy_notebook_assets(source_dir, target_dir):
    """Copy notebook assets (images, data files) to the website assets directory."""
    # This is a simplified version - you might need to expand this
    # based on what assets your notebooks are using
    assets_source = os.path.join(source_dir, 'assets')
    assets_target = os.path.join(target_dir, 'assets')
    
    if os.path.exists(assets_source):
        print(f"Copying assets from {assets_source} to {assets_target}")
        if not os.path.exists(assets_target):
            os.makedirs(assets_target)
        
        # Copy all files recursively
        for item in os.listdir(assets_source):
            s = os.path.join(assets_source, item)
            d = os.path.join(assets_target, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

if __name__ == "__main__":
    # Paths should be adjusted to your repository structure
    notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'notebooks')
    website_notebooks_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '_notebooks')
    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
    
    convert_notebooks_to_md(notebooks_dir, website_notebooks_dir)
    copy_notebook_assets(notebooks_dir, assets_dir)