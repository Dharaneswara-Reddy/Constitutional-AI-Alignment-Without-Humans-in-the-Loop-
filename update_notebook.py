import json
import sys

file_path = 'LOCAL_QUICKSTART.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'def install_package(package):' in line:
                source[i] = line.replace('def install_package(package):', 'def install_package(package_str):')
            if 'pip + ["install", package],' in line:
                source[i] = line.replace('pip + ["install", package],', 'pip + ["install"] + package_str.split(),')
            if 'FAILED TO INSTALL: {package}' in line:
                source[i] = line.replace('{package}', '{package_str}')
            if 'SUCCESS: {package}' in line:
                source[i] = line.replace('{package}', '{package_str}')
            if '"torch>=2.3.0",\n' in line:
                source[i] = line.replace('"torch>=2.3.0",\n', '"torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121",\n')
            if '"torch",\n' in line:
                source[i] = line.replace('"torch",\n', '"torch --index-url https://download.pytorch.org/whl/cpu",\n')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    
print('Updated LOCAL_QUICKSTART.ipynb successfully')
