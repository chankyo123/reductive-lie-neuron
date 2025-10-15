import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data (adjust paths)
imu_df = pd.read_csv('V1_01_easy/mav0/imu0/data.csv')
gt_df = pd.read_csv('V1_01_easy/mav0/state_groundtruth_estimate0/data.csv')

# Extract features and velocity
X = imu_df[['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]', 
            'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].values
velocities = gt_df[['v_RS_R_x [m s^-1]', 'v_RS_R_y [m s^-1]', 'v_RS_R_z [m s^-1]']].values
y = np.sqrt(np.sum(velocities**2, axis=1))

# Synchronize and subset
n_samples = min(len(X), len(y))
X, y = X[:n_samples], y[:n_samples]

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# #### Downsampling
# from sklearn.decomposition import PCA
# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(X_scaled)
# X_tsne = tsne.fit_transform(X_pca)

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Velocity Magnitude (m/s)')
plt.title('t-SNE on EuRoC V1_01_easy')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.show()