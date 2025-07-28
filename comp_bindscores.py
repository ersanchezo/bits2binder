#!/usr/bin/env python3
"""
Code partially adapted from https://github.com/dunbracklab/IPSAE
comp_bindscores.py

Simplified script to compute and output maximum DockQ, pDockQ2, iPsAE, LIS, and iPAE scores
for each chain pair in an AlphaFold3/Chai1 mmCIF model.

Usage:
    python simplified_ipsae.py <path_to_pae_json> <path_to_mmcif> [<pae_cutoff>]

Outputs tab-separated lines:
    Chain1  Chain2  DockQ  pDockQ2  iPsAE  LIS  iPAE

Requires: numpy
"""
import sys
import json
import math
import numpy as np

# Functions for PTM and d0

def ptm_func(x, d0):
    return 1.0 / (1 + (x / d0) ** 2)

vector_ptm = np.vectorize(ptm_func)

def calc_d0(L, pair_type):
    L = float(L)
    if L < 27:
        L = 27.0
    min_val = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    return max(min_val, d0)

# Parse mmCIF to extract atom_site fields and coordinates

def parse_cif_atoms(cif_path):
    field_idx = {}
    atoms = []
    with open(cif_path) as f:
        for line in f:
            if line.startswith('_atom_site.'):
                _, field = line.strip().split('.')
                field_idx[field] = len(field_idx)
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.split()
                if not field_idx:
                    continue
                atom = {
                    'atom_name': parts[field_idx['label_atom_id']],
                    'chain_id': parts[field_idx['label_asym_id']],
                    'residue': parts[field_idx['label_comp_id']],
                    'resnum': parts[field_idx['label_seq_id']],
                    'x': float(parts[field_idx['Cartn_x']]),
                    'y': float(parts[field_idx['Cartn_y']]),
                    'z': float(parts[field_idx['Cartn_z']])
                }
                atoms.append(atom)
    return atoms

# Classify chain types based on residue names

def classify_chains(chain_ids, residues):
    nuc_set = {"DA","DC","DT","DG","A","C","U","G"}
    types = {}
    for ch in set(chain_ids):
        mask = [cid == ch and res in nuc_set for cid, res in zip(chain_ids, residues)]
        types[ch] = 'nucleic_acid' if any(mask) else 'protein'
    return types

# Main computation
def main():
    if len(sys.argv) not in (3,4):
        print(__doc__)
        sys.exit(1)
    json_path, cif_path = sys.argv[1], sys.argv[2]
    pae_cutoff = float(sys.argv[3]) if len(sys.argv) == 4 else 10.0

    # Load JSON PAE and plDDT
    with open(json_path) as jf:
        data = json.load(jf)
    if 'atom_plddts' not in data or 'pae' not in data:
        raise ValueError("JSON must contain 'atom_plddts' and 'pae'")
    atom_plddts = np.array(data['atom_plddts'])
    pae_full = np.array(data['pae'])

    # Parse CIF atoms
    atoms = parse_cif_atoms(cif_path)

    # Build token mask, coordinate arrays, and chain/residue lists
    ca_idx, cb_idx = [], []
    token_mask = []
    chain_ids = []
    residues = []
    for i, atom in enumerate(atoms):
        a = atom['atom_name']
        chain_ids.append(atom['chain_id'])
        residues.append(atom['residue'])
        if a == 'CA':
            ca_idx.append(i)
            token_mask.append(True)
        elif a == 'CB':
            cb_idx.append(i)
            token_mask.append(True)
        else:
            token_mask.append(False)
    token_mask = np.array(token_mask)

    # Subset plDDT and PAE
    plddt = atom_plddts[ca_idx]
    cb_plddt = atom_plddts[cb_idx]
    pae = pae_full[np.ix_(token_mask, token_mask)]

    # Coordinates for CB atoms
    coords = np.array([[atoms[i]['x'], atoms[i]['y'], atoms[i]['z']] for i in cb_idx])
    chains = np.array([atoms[i]['chain_id'] for i in cb_idx])
    unique = list(sorted(set(chains)))

    # Distance matrix
    dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)

    # Classify chains
    chain_types = classify_chains(chains, residues)

    # Results container
    results = []

    # Parameters
    pDockQ_cutoff = 8.0

    # Loop over chain pairs
    for i, ch1 in enumerate(unique):
        for ch2 in unique[i+1:]:
            # Indices for each chain
            idx1 = np.where(chains == ch1)[0]
            idx2 = np.where(chains == ch2)[0]

            # --- DockQ ---
            residues_set = set()
            npairs = 0
            for ii in idx1:
                valid = idx2[dist[ii, idx2] <= pDockQ_cutoff]
                if valid.size > 0:
                    residues_set.update(valid.tolist())
                    npairs += valid.size
            if npairs > 0:
                mean_pl = cb_plddt[list(residues_set)].mean()
                x = mean_pl * math.log10(npairs)
                dockq = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
            else:
                dockq = 0.0

            # --- pDockQ2 ---
            sum_ptm = 0.0
            for ii in idx1:
                valid = idx2[dist[ii, idx2] <= pDockQ_cutoff]
                if valid.size > 0:
                    pae_vals = pae[ii, valid]
                    sum_ptm += vector_ptm(pae_vals, 10.0).sum()
            p2 = (1.31 / (1 + math.exp(-0.075 * ((cb_plddt[list(residues_set)].mean() * (sum_ptm/npairs)) - 84.733))) + 0.005) if npairs > 0 else 0.0

            # --- iPsAE ---
            ptm_list = []
            for ii in idx1:
                valid = np.where((chains == ch2) & (pae[ii] < pae_cutoff))[0]
                if valid.size > 0:
                    d0 = calc_d0(valid.size, 'protein')
                    ptm_list.append(vector_ptm(pae[ii, valid], d0).mean())
            ipsae_max = max(ptm_list) if ptm_list else 0.0

            # --- LIS ---
            sel = pae[np.ix_(idx1, idx2)]
            valid_pae = sel[sel <= 12]
            lis = valid_pae.size > 0 and ((12 - valid_pae)/12).mean() or 0.0

            # --- iPAE ---
            if idx1.size and idx2.size:
                mean12 = pae[np.ix_(idx1, idx2)].mean()
                mean21 = pae[np.ix_(idx2, idx1)].mean()
                ipae = (mean12 + mean21) / 2.0
            else:
                ipae = 0.0

            results.append((ch1, ch2, dockq, p2, ipsae_max, lis, ipae))

    # Print header and values
    print("Chain1\tChain2\tDockQ\tpDockQ2\tiPsAE\tLIS\tiPAE")
    for r in results:
        print(f"{r[0]}\t{r[1]}\t{r[2]:.4f}\t{r[3]:.4f}\t{r[4]:.6f}\t{r[5]:.4f}\t{r[6]:.4f}")

if __name__ == '__main__':
    main()
