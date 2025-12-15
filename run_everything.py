import os
import pandas as pd
import ssgetpy
import benchmark_runner as bm
import scipy.sparse as sp
DOWNLOAD_DIR = 'matrices_collection'
REGISTRY_FILE = 'matrix_registry.csv'

def ensure_data_exists():
    if os.path.exists(REGISTRY_FILE) and os.path.exists(DOWNLOAD_DIR):
        if len(os.listdir(DOWNLOAD_DIR)) > 0:
            print(f"Data found in '{DOWNLOAD_DIR}' and registry exists. Skipping download.")
            return

    print("Data missing. Searching SuiteSparse collection...")
    
    
    results = ssgetpy.search(rowbounds=(1000, 50000), isspd=True, limit=10, kind='Structural Problem')
    
    if len(results) == 0:
        print("Warning: No matrices found matching criteria!")
        return

    print(f"Found {len(results)} matrices. Downloading to '{DOWNLOAD_DIR}'...")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    results.download(destpath=DOWNLOAD_DIR, extract=True)
    
    print("Generating registry...")
    data = []
    for matrix in results:
        mtx_path = os.path.join(DOWNLOAD_DIR, matrix.name, f"{matrix.name}.mtx")
        status = "Ready" if os.path.exists(mtx_path) else "Missing"
        
        data.append({
            'name': matrix.name,
            'group': matrix.group,
            'rows': matrix.rows,
            'cols': matrix.cols,
            'nnz': matrix.nnz,
            'path': mtx_path,
            'status': status
        })
    
    pd.DataFrame(data).to_csv(REGISTRY_FILE, index=False)
    print(f"Registry saved to {REGISTRY_FILE}")

def main():
    print("--- PROJECT START ---")
    
    ensure_data_exists()

    bm.init_results_file()
    
    bm.run_baselines()
    
    bm.run_ilu_sensitivity()
    
    bm.run_ric2s_sensitivity()
    
    bm.run_exact_cholesky_ric()

    print("\n------------------------------------------------")
    print(f"Benchmarks Complete. Data saved in '{bm.RESULTS_FILE}'")
    print("Run 'python plot_results.py' to see the graphs.")

if __name__ == "__main__":
    main()