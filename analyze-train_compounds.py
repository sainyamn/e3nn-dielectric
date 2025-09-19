import os
import ase.io
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from pymatgen.core.composition import Composition
import seaborn as sns

def analyze_oxides(filepath):
    # Load the dataset
    print("Loading dataset...")
    structures = ase.io.read(filepath, index=":")
    
    # Initialize storage
    oxide_compositions = []
    epsilons = []
    composition_count = defaultdict(int)
    composition_eps = defaultdict(list)
    
    # Analyze each structure
    for atoms in structures:
        # Get composition
        comp = atoms.get_chemical_formula(mode='hill')
        composition_count[comp] += 1
        
        # Get epsilon if available
        if 'epsilon' in atoms.arrays:
            eps = np.mean(atoms.arrays['epsilon'])
            epsilons.append(eps)
            composition_eps[comp].append(eps)
    
    # Print most common compositions
    print("\nMost common compositions:")
    sorted_comps = sorted(composition_count.items(), key=lambda x: x[1], reverse=True)
    for comp, count in sorted_comps[:20]:  # Show top 20
        print(f"{comp}: {count} structures")
    
    # Plot composition distribution
    plt.figure(figsize=(12, 6))
    top_20 = dict(sorted_comps[:20])
    plt.bar(top_20.keys(), top_20.values())
    plt.xticks(rotation=90)
    plt.title("Top 20 Most Common Compositions")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
    
    # Plot epsilon distribution
    if epsilons:
        plt.figure(figsize=(12, 6))
        sns.histplot(epsilons, bins=50)
        plt.title("Distribution of Average Dielectric Constants (ε)")
        plt.xlabel("Dielectric Constant (ε)")
        plt.ylabel("Count")
        plt.show()
        
        # Plot epsilon vs composition for common compositions
        common_comps = [comp for comp, _ in sorted_comps[:10]]  # Top 10 most common
        data = []
        for comp in common_comps:
            if comp in composition_eps:
                for eps in composition_eps[comp]:
                    data.append({'Composition': comp, 'ε': eps})
        
        if data:
            import pandas as pd
            df = pd.DataFrame(data)
            plt.figure(figsize=(14, 6))
            sns.boxplot(x='Composition', y='ε', data=df)
            plt.xticks(rotation=90)
            plt.title("Dielectric Constant Distribution by Composition")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    filepath = os.path.join("data", "phonon_becs_epsilon.xyz")
    analyze_oxides(filepath)