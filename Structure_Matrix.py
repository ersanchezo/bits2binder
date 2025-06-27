import os
import subprocess
import pandas as pd
from multiprocessing import Pool, cpu_count
from pathlib import Path
import re
from itertools import combinations_with_replacement

# CONFIGURATION
USALIGN_EXEC = "USalign"  # Ensure it's in PATH or set full path
PDB_DIR = "pdbs/"         # Directory with input PDBs
N_CPU = max(1, cpu_count() - 1)
OUTPUT_CSV = "similarity_matrix_avg.csv"

# Extract TM-score normalized by average length from US-align output
def parse_tm_avg(output):
    match = re.search(r"TM-score=\s*([0-9.]+).*normalized by average length: L=\d+", output)
    if match:
        return float(match.group(1))
    return None

# Run US-align between two PDB files using -a normalization
def run_usalign_avg(pair):
    pdb1, pdb2 = pair
    cmd = [USALIGN_EXEC, str(pdb1), str(pdb2), "-a"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        score = parse_tm_avg(result.stdout + result.stderr)
        return (pdb1.name, pdb2.name, score)
    except subprocess.CalledProcessError:
        return (pdb1.name, pdb2.name, None)

def main():
    pdb_files = sorted(Path(PDB_DIR).glob("*.pdb"))
    if len(pdb_files) < 2:
        print("Need at least two PDB files.")
        return

    pairs = list(combinations_with_replacement(pdb_files, 2))
    print(f"Running {len(pairs)} pairwise alignments with -a normalization using {N_CPU} CPUs...")

    with Pool(N_CPU) as pool:
        results = pool.map(run_usalign_avg, pairs)

    pdb_names = [p.name for p in pdb_files]
    matrix = pd.DataFrame(index=pdb_names, columns=pdb_names, dtype=float)

    for pdb1, pdb2, score in results:
        matrix.loc[pdb1, pdb2] = score
        matrix.loc[pdb2, pdb1] = score  # ensure symmetry

    matrix.to_csv(OUTPUT_CSV)
    print(f"TM-score matrix (normalized by average length) saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
