import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import os

def extract_eps_info(filepath):
    epsilons = []
    compositions = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        # Read number of atoms
        try:
            num_atoms = int(lines[i].strip())
        except:
            i += 1
            continue
        
        # Read comment line with epsilon
        if i + 1 >= len(lines):
            break
            
        comment = lines[i+1].strip()
        
        # Extract composition
        elements = []
        for j in range(num_atoms):
            if i + 2 + j >= len(lines):
                break
            element = lines[i + 2 + j].split()[0]
            elements.append(element)
        
        # Count elements for composition
        comp = {}
        for el in elements:
            comp[el] = comp.get(el, 0) + 1
        comp_str = ''.join(f"{k}{v}" for k, v in sorted(comp.items()))
        
        # Extract epsilon tensor
        eps_match = re.search(r'eps="([^"]+)"', comment)
        if eps_match:
            eps_values = list(map(float, eps_match.group(1).split()))
            if len(eps_values) == 9:
                # Calculate mean of diagonal elements (eps_xx, eps_yy, eps_zz)
                eps_mean = (eps_values[0] + eps_values[4] + eps_values[8]) / 3
                epsilons.append(eps_mean)
                compositions.append(comp_str)
        
        i += 2 + num_atoms  # Move to next structure
    
    return np.array(epsilons), compositions

def analyze_epsilons(epsilons, compositions):
    if len(epsilons) == 0:
        print("No epsilon data found!")
        return
    
    # Basic statistics
    print(f"\n=== Dielectric Constant Analysis ===")
    print(f"Number of structures with ε data: {len(epsilons)}")
    print(f"Min ε: {np.min(epsilons):.2f}")
    print(f"Max ε: {np.max(epsilons):.2f}")
    print(f"Mean ε: {np.mean(epsilons):.2f} ± {np.std(epsilons):.2f}")
    print(f"Median ε: {np.median(epsilons):.2f}")
    
    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(epsilons, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(epsilons), color='r', linestyle='--', label=f'Mean: {np.mean(epsilons):.2f}')
    plt.axvline(np.median(epsilons), color='g', linestyle='-', label=f'Median: {np.median(epsilons):.2f}')
    plt.title("Distribution of Dielectric Constants (ε)")
    plt.xlabel("Dielectric Constant (ε)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    os.makedirs("analysis_results", exist_ok=True)
    plt.savefig("analysis_results/dielectric_distribution.png", dpi=300, bbox_inches='tight')
    print("\nSaved dielectric distribution plot to 'analysis_results/dielectric_distribution.png'")
    
    # Show top 10 compositions by occurrence
    print("\nTop 10 most common compositions:")
    comp_counts = defaultdict(int)
    comp_eps = defaultdict(list)
    
    for comp, eps in zip(compositions, epsilons):
        comp_counts[comp] += 1
        comp_eps[comp].append(eps)
    
    sorted_comps = sorted(comp_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    print(f"{'Composition':<15} {'Count':<8} {'Mean ε':<10} {'Min ε':<10} {'Max ε':<10}")
    print("-" * 50)
    for comp, count in sorted_comps:
        eps_data = comp_eps[comp]
        print(f"{comp:<15} {count:<8} {np.mean(eps_data):<10.2f} {min(eps_data):<10.2f} {max(eps_data):<10.2f}")

if __name__ == "__main__":
    filepath = os.path.join("data", "phonon_becs_epsilon.xyz")
    print(f"Analyzing file: {os.path.abspath(filepath)}")
    
    if not os.path.exists(filepath):
        print(f"Error: File not found at {os.path.abspath(filepath)}")
    else:
        epsilons, compositions = extract_eps_info(filepath)
        analyze_epsilons(epsilons, compositions)
        print("\nAnalysis complete!")
