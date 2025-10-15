import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

# Parameters
NUM_POINTS = 1000  # Total points
RADIUS = 1.0       # Original radius
HEIGHT = 2.0       # Original height
NUM_SUBPARTS = 4   # Arbitrary number of subparts (e.g., 3, 5, 10, 14)

# Set random seed for reproducibility
torch.manual_seed(42)

def generate_cylinder_point_cloud(num_points, radius, height):
    """Generate a cylindrical point cloud."""
    theta = torch.rand(num_points) * 2 * torch.pi
    z = torch.rand(num_points) * height - height / 2
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)
    points = torch.stack([x, y, z], dim=-1)  # (N, 3)
    return points

def split_point_cloud(points, num_subparts, z_axis=2):
    """Split points into num_subparts along z-axis."""
    z_values = points[:, z_axis]
    z_min, z_max = z_values.min(), z_values.max()
    z_step = (z_max - z_min) / num_subparts
    subparts = []
    for i in range(num_subparts):
        z_lower = z_min + i * z_step
        z_upper = z_min + (i + 1) * z_step if i < num_subparts - 1 else z_max + 1e-6
        mask = (z_values >= z_lower) & (z_values < z_upper)
        subpart = points[mask]
        if subpart.shape[0] > 0:
            subparts.append(subpart)
    return subparts

def generate_varied_sl4_transformations(num_subparts):
    """Generate varied SL(4) transformations for num_subparts."""
    transformations = []
    for i in range(num_subparts):
        s_x = 0.5 + (i / (num_subparts - 1)) if num_subparts > 1 else 1.0  # 0.5 to 1.5
        s_y = 1.25 - 0.5 * (i / (num_subparts - 1)) if num_subparts > 1 else 1.0  # 1.25 to 0.75
        s_h = 1 / (s_x * s_y)
        shear_xy = 0.2 * torch.sin(torch.tensor(i * 1.0))
        shear_yz = 0.1 * torch.cos(torch.tensor(i * 2.0))
        perturbation = 0.05 * (i % 2 - 0.5)
        
        A = torch.eye(4, dtype=torch.float32)
        A[0, 0] = s_x
        A[1, 1] = s_y
        A[2, 2] = s_h
        A[0, 1] = shear_xy
        A[1, 2] = shear_yz
        A[3, 0] = perturbation
        det = torch.det(A)
        A[3, 3] = 1 / det if det != 0 else 1.0
        transformations.append(A)
    return transformations

def sl4_deform_subpart(points, A):
    """Apply SL(4) deformation to a subpart."""
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)  # (N, 4)
    deformed_homo = points_homo @ A.T  # (N, 4)
    deformed = deformed_homo[:, :3] / deformed_homo[:, 3:4].clamp(min=1e-6)  # (N, 3)
    return deformed

def compute_bounding_volume(points):
    """Approximate volume via bounding box (simplified, sans π)."""
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    return (x_range * y_range * z_range).item()

def visualize_point_clouds(original, deformed_subparts):
    """Visualize original and deformed point clouds with colors per subpart."""
    fig = plt.figure(figsize=(12, 5))
    
    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1, c='b', label='Original')
    ax1.set_title("Original Cylinder")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-HEIGHT, HEIGHT)
    ax1.legend()
    
    # Deformed with colors
    ax2 = fig.add_subplot(122, projection='3d')
    colormap = cm.get_cmap('tab10', len(deformed_subparts))  # Distinct colors
    for i, subpart in enumerate(deformed_subparts):
        color = colormap(i)
        ax2.scatter(subpart[:, 0], subpart[:, 1], subpart[:, 2], s=1, c=[color], label=f'Subpart {i+1}')
    ax2.set_title(f"Deformed Sponge-like Cylinder ({NUM_SUBPARTS} Subparts)")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-HEIGHT * 2, HEIGHT * 2)
    ax2.legend()
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate original cylinder
    original_points = generate_cylinder_point_cloud(NUM_POINTS, RADIUS, HEIGHT)
    
    # Test different subpart counts
    # subpart_lists = [3, 5, 10, 14]
    subpart_lists = [5]
    for NUM_SUBPARTS in subpart_lists:
        print(f"\nProcessing {NUM_SUBPARTS} subparts:")
        
        # Split into subparts
        subparts = split_point_cloud(original_points, NUM_SUBPARTS)
        print(f"Number of subparts generated: {len(subparts)}")
        
        # Generate varied SL(4) transformations
        transformations = generate_varied_sl4_transformations(NUM_SUBPARTS)
        
        # Apply deformations and keep subparts separate
        deformed_subparts = []
        for i, subpart in enumerate(subparts):
            deformed = sl4_deform_subpart(subpart, transformations[i])
            deformed_subparts.append(deformed)
        
        # Combine for volume check
        deformed_points = torch.cat(deformed_subparts, dim=0)
        
        # Print shapes and sample points
        print("Original shape:", original_points.shape)
        print("Deformed shape:", deformed_points.shape)
        print("Sample original points:\n", original_points[:5])
        print("Sample deformed points:\n", deformed_points[:5])
        
        # Verify volume
        orig_volume = compute_bounding_volume(original_points)
        def_volume = compute_bounding_volume(deformed_points)
        print("Original bounding volume:", orig_volume)
        print("Deformed bounding volume (approx):", def_volume)
        
        # Visualize with colors
        original_np = original_points.numpy()
        deformed_subparts_np = [subpart.numpy() for subpart in deformed_subparts]
        visualize_point_clouds(original_np, deformed_subparts_np)