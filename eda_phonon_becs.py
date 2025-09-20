import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
from ase.visualize.plot import plot_atoms
from collections import defaultdict
from tqdm import tqdm
import pandas as pd

def load_xyz(file_path):
    """Load an XYZ file using ASE."""
    return read(file_path, index=':')

def analyze_frames(atoms_list):
    """Analyze all frames and gather statistics."""
    print(f"Analyzing {len(atoms_list)} frames...")
    
    # Initialize data structures
    all_elements = []
    all_becs = defaultdict(list)
    all_epsilons = []
    structure_sizes = []
    
    for atoms in tqdm(atoms_list, desc="Processing frames"):
        # Get elements
        elements = [atom.symbol for atom in atoms]
        all_elements.append(elements)
        structure_sizes.append(len(elements))
        
        # Get BECs if available
        if 'becs' in atoms.info:
            becs = np.array(atoms.info['becs']).reshape(-1, 3, 3)
            for i, element in enumerate(elements):
                all_becs[element].append(becs[i])
                
        # Get epsilon if available
        if 'epsilon' in atoms.info:
            epsilon = np.array(atoms.info['epsilon'])
            all_epsilons.append(epsilon)
    
    return {
        'elements': all_elements,
        'becs': dict(all_becs),
        'epsilons': np.array(all_epsilons),
        'structure_sizes': structure_sizes
    }

def print_statistics(data):
    """Print comprehensive statistics about the data."""
    print("\n=== Structure Statistics ===")
    unique_sizes = set(data['structure_sizes'])
    print(f"Number of different structure sizes: {len(unique_sizes)}")
    for size in sorted(unique_sizes):
        count = data['structure_sizes'].count(size)
        print(f"  - {size} atoms: {count} frames ({count/len(data['structure_sizes']):.1%})")
    
    # Element statistics
    element_counts = defaultdict(int)
    for elements in data['elements']:
        for element in set(elements):
            element_counts[element] += 1
    
    print("\n=== Element Distribution ===")
    for element, count in sorted(element_counts.items()):
        print(f"  {element}: {count} occurrences")
    
    # BEC statistics
    if data['becs']:
        print("\n=== BEC Statistics ===")
        for element, becs in data['becs'].items():
            becs = np.array(becs)
            print(f"\nElement: {element}")
            print(f"  Number of BEC tensors: {len(becs)}")
            print("  Mean BEC tensor:")
            print(f"  {np.mean(becs, axis=0).round(4)}")
            print("  Standard deviation:")
            print(f"  {np.std(becs, axis=0).round(4)}")
    
    # Epsilon statistics
    if len(data['epsilons']) > 0:
        print("\n=== Dielectric Tensor Statistics ===")
        print("Mean epsilon tensor:")
        print(np.mean(data['epsilons'], axis=0).round(4))
        print("\nStandard deviation:")
        print(np.std(data['epsilons'], axis=0).round(4))

def plot_bec_distribution(becs_data):
    """Plot distribution of BEC tensor components."""
    plt.figure(figsize=(15, 10))
    for element, becs in becs_data.items():
        becs = np.array(becs)
        for i in range(3):
            for j in range(3):
                plt.hist(becs[:, i, j].flatten(), 
                        alpha=0.5, 
                        label=f'{element} ({i+1},{j+1})',
                        bins=30)
    plt.title('Distribution of BEC Tensor Components')
    plt.xlabel('BEC Value')
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_epsilon_evolution(epsilons):
    """Plot evolution of epsilon tensor components over frames."""
    plt.figure(figsize=(12, 6))
    for i in range(3):
        for j in range(3):
            plt.plot(epsilons[:, i, j], label=f'ε{i+1}{j+1}')
    plt.title('Evolution of Dielectric Tensor Components')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def main(xyz_file):
    # Load the XYZ file
    print(f"Loading {xyz_file}...")
    structures = load_xyz(xyz_file)
    
    # Analyze all frames
    data = analyze_frames(structures)
    
    # Print statistics
    print_statistics(data)
    
    # Plot BEC distribution if available
    if data['becs']:
        plot_bec_distribution(data['becs'])
    
    # Plot epsilon evolution if available
    if len(data['epsilons']) > 0:
        plot_epsilon_evolution(data['epsilons'])

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python eda_phonon_becs.py <path_to_xyz_file>")
        sys.exit(1)
    
    main(sys.argv[1])