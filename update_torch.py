import json
file_path = 'LOCAL_QUICKSTART.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if '"torch>=2.3.0 --index-url' in line:
                source[i] = line.replace('torch>=2.3.0', 'torch>=2.3.0,<2.6.0')
            if '"torch --index-url' in line:
                source[i] = line.replace('torch --index-url', 'torch<2.6.0 --index-url')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
