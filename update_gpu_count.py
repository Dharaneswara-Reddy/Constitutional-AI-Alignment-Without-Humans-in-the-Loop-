import json

file_path = 'LOCAL_QUICKSTART.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for cell in notebook.get('cells', []):
    if cell.get('cell_type') == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'print(f"\\n  ✅ NVIDIA GPU: {torch.cuda.get_device_name(0)}")' in line:
                source[i] = line.replace('print(f"\\n  ✅ NVIDIA GPU: {torch.cuda.get_device_name(0)}")', 
                    'count = torch.cuda.device_count()\n        print(f"\\n  ✅ NVIDIA GPU(s): {count}x {torch.cuda.get_device_name(0)}")')
            if 'print(f"     VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")' in line:
                source[i] = line.replace('print(f"     VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")',
                    'print(f"     VRAM (GPU 0): {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")')

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
