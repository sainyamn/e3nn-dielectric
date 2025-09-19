import os
import ase.io
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def analyze_dataset(filepath):
    # Load the dataset
    print(f"Loading dataset from {filepath}...")
    structures = ase.io.read(filepath, index=":")
    
    # Basic information
    print(f"\n=== Dataset Overview ===")
    print(f"Number of structures: {len(structures)}")
    
    # Check available properties
    if len(structures) > 0:
        print("\nAvailable properties:")
        for key, value in structures[0].arrays.items():
            print(f"- {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Element analysis
    element_counts = defaultdict(int)
    for atoms in structures:
        for symbol in atoms.get_chemical_symbols():
            element_counts[symbol] += 1
    
    print("\nElement counts:")
    for element, count in sorted(element_counts.items()):
        print(f"{element}: {count}")
    
    # Structure size analysis
    num_atoms = [len(atoms) for atoms in structures]
    print(f"\nStructure size statistics:")
    print(f"  Min atoms: {min(num_atoms)}")
    print(f"  Max atoms: {max(num_atoms)}")
    print(f"  Mean atoms: {np.mean(num_atoms):.2f} ± {np.std(num_atoms):.2f}")
    
    # Property analysis
    if 'becs' in structures[0].arrays:
        all_becs = np.concatenate([atoms.arrays['becs'] for atoms in structures])
        print("\nBECS statistics:")
        print(f"  Shape: {all_becs.shape}")
        print(f"  Mean: {np.mean(all_becs):.4f} ± {np.std(all_becs):.4f}")
        print(f"  Min: {np.min(all_becs):.4f}")
        print(f"  Max: {np.max(all_becs):.4f}")
    
    if 'epsilon' in structures[0].arrays:
        all_epsilon = np.concatenate([atoms.arrays['epsilon'] for atoms in structures])
        print("\nEpsilon statistics:")
        print(f"  Shape: {all_epsilon.shape}")
        print(f"  Mean: {np.mean(all_epsilon):.4f} ± {np.std(all_epsilon):.4f}")
        print(f"  Min: {np.min(all_epsilon):.4f}")
        print(f"  Max: {np.max(all_epsilon):.4f}")

if __name__ == "__main__":
    filepath = os.path.join("data", "phonon_becs_epsilon.xyz")
    analyze_dataset(filepath)
