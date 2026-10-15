import json

with open('notebooks/colab_inference.ipynb', 'r') as f:
    nb = json.load(f)

# Find Cell 4 and add gate5_pass variable
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if any('def score_batch' in line for line in source):
            for i, line in enumerate(source):
                if 'GATE 5: PASSED' in line:
                    source.insert(i, "    gate5_pass = True\n")
                    break

with open('notebooks/colab_inference.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
