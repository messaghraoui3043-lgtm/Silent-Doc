import json

nb = json.load(open(r'c:\Users\UserDev\Desktop\silent-doctor\silent-doctor\models\ml6505-report.ipynb', 'r', encoding='utf-8'))
cells = nb['cells']
print(f"Total cells: {len(cells)}")
print(f"Kernel: {nb.get('metadata', {}).get('kernelspec', {})}")
print("="*80)

for i, c in enumerate(cells):
    ctype = c['cell_type']
    source = ''.join(c['source'])
    print(f"\n--- Cell {i} [{ctype}] ---")
    print(source[:500])
    if len(source) > 500:
        print(f"... [TRUNCATED, total {len(source)} chars]")
