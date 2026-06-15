import os
import pandas as pd
import numpy as np

# Directory containing the CSV files
directory = "solid/elasticity_tensors"

# Percentage threshold for similarity check
percentage_threshold = 0.01  

# List to store averaged C matrices and corresponding filenames
matrices = []
filenames = []

# Step 1: Load CSV files and extract averaged C matrix when bbr = 0
for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        
        # Load the CSV file
        df = pd.read_csv(filepath)
        
        # Filter rows where bbr = 0
        if 'bbr' in df.columns:
            df = df[df['bbr'] == 0]
        
        # Ensure there are rows with bbr = 0
        if len(df) == 0:
            continue

        # Extract the C matrix (3rd to 4th columns of each row)
        C = df.iloc[:, 2:4].values.astype(float)

        # Average the C matrix across all rows with bbr = 0
        C_avg = np.mean(C, axis=0, keepdims=True)  # Shape will be (1, 2)

        matrices.append(C_avg)
        filenames.append(filename)

# Step 2: Group matrices that are element-wise similar within the percentage threshold
def matrices_are_similar(C1, C2, threshold):
    """Check if all corresponding elements are within the given percentage threshold."""
    diff = np.abs(C1 - C2)
    tolerance = threshold * np.abs(C1)
    
    mask = (C1 != 0)
    
    # Check similarity for nonzero elements
    similar = np.all(diff[mask] <= tolerance[mask])
    
    # For zero values, check absolute difference
    zero_match = np.all(diff[~mask] <= threshold)
    
    return similar and zero_match

# Step 3: Grouping similar matrices
similar_groups = []
visited = set()

for i in range(len(matrices)):
    if i in visited:
        continue
    group = [filenames[i]]
    visited.add(i)
    
    for j in range(i + 1, len(matrices)):
        if matrices_are_similar(matrices[i], matrices[j], percentage_threshold):
            group.append(filenames[j])
            visited.add(j)
    
    # Sort the group alphabetically
    group.sort()
    
    similar_groups.append(group)

# Step 4: Save results to a text file
output_file = "similar_groups_output.txt"
with open(output_file, "w") as f:
    f.write(f"--- Similar Averaged C Matrix Groups (All values within ±{percentage_threshold*100:.2f}%) ---\n\n")

    for idx, group in enumerate(similar_groups):
        f.write(f"\nGroup {idx + 1} (Total {len(group)} files):\n")
        print(f"\nGroup {idx + 1} (Total {len(group)} files):")
        
        for fname in group:
            f.write(f"  - {fname}\n")
            print(f"  - {fname}")
        
        f.write("\nAveraged C Matrices in this group:\n")
        for fname in group:
            idx = filenames.index(fname)
            f.write(f"\n{fname}:\n{matrices[idx]}\n")

# Print confirmation
print(f"Results saved to {output_file}")
