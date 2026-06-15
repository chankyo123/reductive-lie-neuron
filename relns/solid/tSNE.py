import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Directory containing CSV files (relative to solid/)
data_dir = "solid/elastic"

# List all CSV files in solid/elastic/
files = sorted(list(set(f for f in os.listdir(data_dir) if f.endswith(".csv"))))
print(f"Found {len(files)} unique CSV files: {files[:5]}...")

# Functions to extract category and motion from filename
def get_category(filename):
    base_name = filename.replace(".csv", "")
    parts = base_name.split("_")
    return parts[0]  # Take only the first part as category (e.g., "kubc")

def get_motion(filename):
    base_name = filename.replace(".csv", "")
    parts = base_name.split("_")
    try:
        number_idx = next(i for i, part in enumerate(parts) if part.isdigit())
        return parts[number_idx - 1] if number_idx > 1 else "unknown"
    except StopIteration:
        return "unknown"

# Load and preprocess data
data = []
category_labels = []
motion_labels = []
expected_cols = [
    "t", "bbr", "C_1111", "C_1112", "C_1122", "C_1222", "C_2222", "C_1212",
    "eps_1_xx", "eps_1_xy", "eps_1_yx", "eps_1_yy", "sig_1_xx", "sig_1_xy", "sig_1_yx", "sig_1_yy",
    "eps_2_xx", "eps_2_xy", "eps_2_yx", "eps_2_yy", "sig_2_xx", "sig_2_xy", "sig_2_yx", "sig_2_yy",
    "eps_3_xx", "eps_3_xy", "eps_3_yx", "eps_3_yy", "sig_3_xx", "sig_3_xy", "sig_3_yx", "sig_3_yy"
]

successful_files = []
failed_files = []

for file in files:
    filepath = os.path.join(data_dir, file)
    try:
        df = pd.read_csv(filepath, names=expected_cols, header=0)
        if df.shape[1] != len(expected_cols):
            raise ValueError(f"Expected {len(expected_cols)} columns, but found {df.shape[1]}")
        
        state_1_cols = ["eps_1_xx", "eps_1_xy", "eps_1_yy", "sig_1_xx", "sig_1_xy", "sig_1_yy"]
        state_2_cols = ["eps_2_xx", "eps_2_xy", "eps_2_yy", "sig_2_xx", "sig_2_xy", "sig_2_yy"]
        state_3_cols = ["eps_3_xx", "eps_3_xy", "eps_3_yy", "sig_3_xx", "sig_3_xy", "sig_3_yy"]
        
        features_1 = df[state_1_cols].values
        features_2 = df[state_2_cols].values
        features_3 = df[state_3_cols].values
        
        n_rows = features_1.shape[0]
        if not (features_2.shape[0] == n_rows and features_3.shape[0] == n_rows):
            raise ValueError(f"Inconsistent row counts: {features_1.shape[0]}, {features_2.shape[0]}, {features_3.shape[0]}")
        
        features = np.hstack([features_1, features_2, features_3])
        print(f"{file} - Features shape: {features.shape}")
        
        data.append(features)
        
        category = get_category(file)
        motion = get_motion(file)
        category_labels.extend([category] * n_rows)
        motion_labels.extend([motion] * n_rows)
        
        print(f"{file} - Added {n_rows} labels - Category: {category}, Motion: {motion}")
        successful_files.append(file)
    except Exception as e:
        failed_files.append((file, str(e)))
        print(f"Failed to load {file}: {e}")
        with open(filepath, 'r') as f:
            lines = f.readlines()
            print(f"First few lines of {file}:")
            for i, line in enumerate(lines[:5]):
                print(f"Line {i}: {line.strip()} (length: {len(line.split(','))})")

if not data:
    print("No data was loaded. Failed files:")
    for file, error in failed_files:
        print(f"{file}: {error}")
    raise ValueError("No data was loaded from the CSV files.")

# Combine all data
X = np.vstack(data)
print(f"X shape: {X.shape}")
print(f"Category labels length: {len(category_labels)}")
print(f"Motion labels length: {len(motion_labels)}")

# Verify lengths match
if X.shape[0] != len(category_labels) or X.shape[0] != len(motion_labels):
    raise ValueError(f"Length mismatch: X has {X.shape[0]} rows, category_labels has {len(category_labels)}, motion_labels has {len(motion_labels)}")

# Standardize the features
X = StandardScaler().fit_transform(X)

# Apply t-SNE
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(X)

# 1. t-SNE by Category (fix motion to 'tension')
fixed_motion = "tension"
mask_motion = np.array(motion_labels) == fixed_motion
X_tsne_category = X_tsne[mask_motion]
category_labels_filtered = np.array(category_labels)[mask_motion]
print(f"t-SNE by Category - Fixed Motion: {fixed_motion}, Data points: {len(X_tsne_category)}")

if len(X_tsne_category) > 0:
    tsne_df_category = pd.DataFrame({
        "x": X_tsne_category[:, 0],
        "y": X_tsne_category[:, 1],
        "category": category_labels_filtered
    })

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x="x", y="y", hue="category", data=tsne_df_category, palette="deep", s=50, alpha=0.6)
    plt.title(f"t-SNE Visualization by Category (Fixed Motion: {fixed_motion})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")
    plt.tight_layout()
    plt.savefig("tsne_by_category.png", bbox_inches="tight")
    plt.show()
else:
    print(f"No data points for fixed motion '{fixed_motion}'. Skipping plot.")

# 2. t-SNE by Motion (fix category to 'kubc')
fixed_category = "kubc"
mask_category = np.array(category_labels) == fixed_category
X_tsne_motion = X_tsne[mask_category]
motion_labels_filtered = np.array(motion_labels)[mask_category]
print(f"t-SNE by Motion - Fixed Category: {fixed_category}, Data points: {len(X_tsne_motion)}")

if len(X_tsne_motion) > 0:
    tsne_df_motion = pd.DataFrame({
        "x": X_tsne_motion[:, 0],
        "y": X_tsne_motion[:, 1],
        "motion": motion_labels_filtered
    })

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x="x", y="y", hue="motion", data=tsne_df_motion, palette="deep", s=50, alpha=0.6)
    plt.title(f"t-SNE Visualization by Motion (Fixed Category: {fixed_category})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Motion")
    plt.tight_layout()
    plt.savefig("tsne_by_motion.png", bbox_inches="tight")
    plt.show()
else:
    print(f"No data points for fixed category '{fixed_category}'. Skipping plot.")

# 3. t-SNE by Category (fix motion to 'shear')
fixed_motion_2 = "shear"
mask_motion_2 = np.array(motion_labels) == fixed_motion_2
X_tsne_category_2 = X_tsne[mask_motion_2]
category_labels_filtered_2 = np.array(category_labels)[mask_motion_2]
print(f"t-SNE by Category - Fixed Motion: {fixed_motion_2}, Data points: {len(X_tsne_category_2)}")

if len(X_tsne_category_2) > 0:
    tsne_df_category_2 = pd.DataFrame({
        "x": X_tsne_category_2[:, 0],
        "y": X_tsne_category_2[:, 1],
        "category": category_labels_filtered_2
    })

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x="x", y="y", hue="category", data=tsne_df_category_2, palette="deep", s=50, alpha=0.6)
    plt.title(f"t-SNE Visualization by Category (Fixed Motion: {fixed_motion_2})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Category")
    plt.tight_layout()
    plt.savefig("tsne_by_category_shear.png", bbox_inches="tight")
    plt.show()
else:
    print(f"No data points for fixed motion '{fixed_motion_2}'. Skipping plot.")

# 4. t-SNE by Motion (fix category to 'pbc')
fixed_category_2 = "pbc"
mask_category_2 = np.array(category_labels) == fixed_category_2
X_tsne_motion_2 = X_tsne[mask_category_2]
motion_labels_filtered_2 = np.array(motion_labels)[mask_category_2]
print(f"t-SNE by Motion - Fixed Category: {fixed_category_2}, Data points: {len(X_tsne_motion_2)}")

if len(X_tsne_motion_2) > 0:
    tsne_df_motion_2 = pd.DataFrame({
        "x": X_tsne_motion_2[:, 0],
        "y": X_tsne_motion_2[:, 1],
        "motion": motion_labels_filtered_2
    })

    plt.figure(figsize=(14, 10))
    sns.scatterplot(x="x", y="y", hue="motion", data=tsne_df_motion_2, palette="deep", s=50, alpha=0.6)
    plt.title(f"t-SNE Visualization by Motion (Fixed Category: {fixed_category_2})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Motion")
    plt.tight_layout()
    plt.savefig("tsne_by_motion_pbc.png", bbox_inches="tight")
    plt.show()
else:
    print(f"No data points for fixed category '{fixed_category_2}'. Skipping plot.")

# Print summary
print("\nSummary:")
print(f"Total datapoints: {len(X)}")
print(f"Unique categories: {len(np.unique(category_labels))}")
print(f"Categories: {np.unique(category_labels)}")
print(f"Unique motions: {len(np.unique(motion_labels))}")
print(f"Motions: {np.unique(motion_labels)}")
print(f"Successful files: {len(successful_files)} / {len(files)}")
print(f"Failed files: {len(failed_files)}")
if failed_files:
    print("Failed files details:")
    for file, error in failed_files:
        print(f"{file}: {error}")