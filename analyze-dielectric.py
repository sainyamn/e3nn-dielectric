import os
import ase.io
import numpy as np
import matplotlib
# Use a non-interactive backend to avoid display issues
matplotlib.use('Agg')  # This will save plots to files instead of displaying them
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import sys

def extract_eps_from_comment(comment):
    """Extract epsilon tensor from comment line and return its mean diagonal value."""
    import re
    # Find the eps field in the comment
    eps_match = re.search(r'eps="([^"]+)"', comment)
    if not eps_match:
        return None
    
    # Convert string to numpy array and reshape to 3x3
    eps_values = np.array(eps_match.group(1).split(), dtype=float)
    if len(eps_values) != 9:
        return None
    
    # Calculate mean of diagonal elements (indices 0, 4, 8 for 0-based indexing)
    return (eps_values[0] + eps_values[4] + eps_values[8]) / 3

def analyze_dielectric_properties(filepath):
    # Load the dataset
    print("Loading dataset...")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        structures = []
        i = 0
        while i < len(lines):
            if i + 1 >= len(lines):
                break
                
            # Read number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except ValueError:
                i += 1
                continue
                
            # Read comment line with epsilon
            comment = lines[i+1].strip()
            i += 2
            
            # Read atom data
            atom_lines = []
            for _ in range(num_atoms):
                if i >= len(lines):
                    break
                atom_lines.append(lines[i].strip())
                i += 1
                
            # Parse the structure
            try:
                atoms = ase.Atoms('H' * num_atoms)  # Dummy atoms, we just need the object
                atoms.info['comment'] = comment
                
                # Extract positions and other properties
                positions = []
                becs = []
                for line in atom_lines:
                    parts = line.split()
                    symbol = parts[0]
                    pos = list(map(float, parts[1:4]))
                    bec = list(map(float, parts[4:13]))  # 9 BECs components
                    positions.append(pos)
                    becs.append(bec)
                
                atoms.positions = np.array(positions)
                atoms.arrays['becs'] = np.array(becs)
                
                # Extract epsilon from comment
                eps = extract_eps_from_comment(comment)
                if eps is not None:
                    atoms.info['epsilon'] = eps
                
                structures.append(atoms)
                
            except Exception as e:
                print(f"Error parsing structure: {e}")
                continue
        
        print(f"Successfully parsed {len(structures)} structures")
        
    except Exception as e:
        print(f"Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Initialize storage
    epsilons = []
    composition_eps = defaultdict(list)
    
    # Collect dielectric data
    for i, atoms in enumerate(structures):
        comp = atoms.get_chemical_formula(mode='hill')
        if 'epsilon' in atoms.info:
            eps = atoms.info['epsilon']
            epsilons.append(eps)
            composition_eps[comp].append(eps)
    
    print(f"Found epsilon data for {len(epsilons)} out of {len(structures)} structures")
    
    print(f"Found epsilon data for {len(epsilons)} out of {len(structures)} structures")
    
    # Basic statistics
    if epsilons:
        eps_array = np.array(epsilons)
        print("\n=== Dielectric Constant Statistics ===")
        print(f"Number of structures with ε data: {len(epsilons)}")
        print(f"Min ε: {np.min(eps_array):.2f}")
        print(f"Max ε: {np.max(eps_array):.2f}")
        print(f"Mean ε: {np.mean(eps_array):.2f} ± {np.std(eps_array):.2f}")
        print(f"Median ε: {np.median(eps_array):.2f}")
        
        # Print percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        print("\nPercentiles:")
        for p in percentiles:
            print(f"  {p}%: {np.percentile(eps_array, p):.2f}")
        
        # Plot distribution
        plt.figure(figsize=(14, 6))
        
        # Histogram
        plt.subplot(1, 2, 1)
        sns.histplot(epsilons, bins=50, kde=True)
        plt.axvline(np.mean(eps_array), color='r', linestyle='--', label=f'Mean: {np.mean(eps_array):.2f}')
        plt.axvline(np.median(eps_array), color='g', linestyle='-', label=f'Median: {np.median(eps_array):.2f}')
        plt.title("Distribution of Dielectric Constants")
        plt.xlabel("Dielectric Constant (ε)")
        plt.ylabel("Count")
        plt.legend()
        
        # Box plot
        plt.subplot(1, 2, 2)
        sns.boxplot(y=epsilons)
        plt.title("Box Plot of Dielectric Constants")
        plt.ylabel("Dielectric Constant (ε)")
        
        output_dir = "analysis_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the plots
        output_file = os.path.join(output_dir, "dielectric_distribution.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
        plt.close()
        
        # Top 10 compositions by occurrence with their dielectric properties
        if composition_eps:
            print("\n=== Dielectric Properties by Composition ===")
            print(f"{'Composition':<15} {'Count':<8} {'Mean ε':<10} {'Min ε':<10} {'Max ε':<10} {'Std ε':<10}")
            print("-" * 60)
            
            # Sort by count
            sorted_comps = sorted(composition_eps.items(), 
                                key=lambda x: len(x[1]), 
                                reverse=True)[:10]  # Top 10
            
            for comp, eps_list in sorted_comps:
                eps_arr = np.array(eps_list)
                print(f"{comp:<15} {len(eps_list):<8} {np.mean(eps_arr):<10.2f} "
                      f"{np.min(eps_arr):<10.2f} {np.max(eps_arr):<10.2f} {np.std(eps_arr):<10.2f}")
            
            # Plot top compositions
            plt.figure(figsize=(14, 6))
            data = []
            for comp, eps_list in sorted_comps:
                for eps in eps_list:
                    data.append({'Composition': comp, 'ε': eps})
            
            import pandas as pd
            df = pd.DataFrame(data)
            
            plt.subplot(1, 2, 1)
            sns.boxplot(x='Composition', y='ε', data=df)
            plt.xticks(rotation=90)
            plt.title("Dielectric Constants by Composition")
            
            plt.subplot(1, 2, 2)
            sns.violinplot(x='Composition', y='ε', data=df, inner="quartile")
            plt.xticks(rotation=90)
            plt.title("Distribution by Composition")
            
            output_file = os.path.join(output_dir, "dielectric_by_composition.png")
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved composition plot to {output_file}")
            plt.close()

if __name__ == "__main__":
    filepath = os.path.join("data", "phonon_becs_epsilon.xyz")
    print(f"Analyzing file: {os.path.abspath(filepath)}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {os.path.abspath(filepath)}")
        sys.exit(1)
    analyze_dielectric_properties(filepath)
    print("Analysis complete. Check the 'analysis_results' directory for output files.")