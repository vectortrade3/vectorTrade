import json
import os

def add_inf_replace(file_name):
    if not os.path.exists(file_name):
        print(f"File {file_name} not found.")
        return
        
    with open(file_name, 'r', encoding='utf-8') as f:
        d = json.load(f)
    
    modified = False
    for cell in d.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            final_source = []
            for i, line in enumerate(source):
                final_source.append(line)
                if 'pd.read_csv' in line:
                    if i+1 < len(source) and 'np.inf' not in source[i+1]:
                        final_source.append("df.replace([np.inf, -np.inf], np.nan, inplace=True)\n")
                        modified = True
            cell['source'] = final_source
            
    if modified:
        with open(file_name, 'w', encoding='utf-8') as f:
            json.dump(d, f, indent=1)
        print(f"Fixed {file_name}")
    else:
        print(f"{file_name} already fixed or no changes needed.")

add_inf_replace('5_years_5_inference.ipynb')
add_inf_replace('5_years_5.2_global_inference.ipynb')
