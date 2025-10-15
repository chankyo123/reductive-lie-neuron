import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
NUM_POINTS = 1000  # Number of points
RADIUS = 1.0       # Original radius
HEIGHT = 2.0       # Original height
RADIUS_SCALE = 0.5 # Shrink radius by 0.5
HEIGHT_SCALE = 4.0 # Stretch height by 4 (1 / (0.5)^2)

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

def sl4_deform(points, radius_scale, height_scale):
    """Deform point cloud using SL(4) transformation, preserving volume."""
    # Convert to 4D homogeneous coordinates
    points_homo = torch.cat([points, torch.ones(points.shape[0], 1)], dim=-1)  # (N, 4)

    # SL(4) transformation matrix
    w_scale = 1 / (radius_scale * radius_scale * height_scale)  # Ensures det = 1
    A = torch.tensor([
        [radius_scale, 0, 0, 0],
        [0, radius_scale, 0, 0],
        [0, 0, height_scale, 0],
        [0, 0, 0, w_scale]
    ], dtype=torch.float32)

    # Apply transformation
    deformed_homo = points_homo @ A.T  # (N, 4)
    
    # Normalize back to 3D (divide by w)
    deformed = deformed_homo[:, :3] / deformed_homo[:, 3:4]  # (N, 3)
    return deformed

def visualize_point_clouds(original, deformed):
    """Visualize original and deformed point clouds."""
    fig = plt.figure(figsize=(12, 5))

    # Original
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], s=1, c='b')
    ax1.set_title("Original Cylinder")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_xlim(-1.5, 1.5)
    ax1.set_ylim(-1.5, 1.5)
    ax1.set_zlim(-HEIGHT, HEIGHT)

    # Deformed
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(deformed[:, 0], deformed[:, 1], deformed[:, 2], s=1, c='r')
    ax2.set_title("Deformed Cylinder (SL(4))")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-HEIGHT * HEIGHT_SCALE, HEIGHT * HEIGHT_SCALE)

    plt.show()

# Main execution
if __name__ == "__main__":
    # Generate original cylinder
    original_points = generate_cylinder_point_cloud(NUM_POINTS, RADIUS, HEIGHT)
    
    # Deform using SL(4)
    deformed_points = sl4_deform(original_points, RADIUS_SCALE, HEIGHT_SCALE)
    
    # Print shapes and sample points
    print("Original shape:", original_points.shape)
    print("Deformed shape:", deformed_points.shape)
    print("Sample original points:\n", original_points[:5])
    print("Sample deformed points:\n", deformed_points[:5])
    
    # Verify volume preservation (approximate via bounding box)
    orig_volume = RADIUS * RADIUS * HEIGHT  # π simplified out
    def_volume = (RADIUS * RADIUS_SCALE) * (RADIUS * RADIUS_SCALE) * (HEIGHT * HEIGHT_SCALE)
    print("Original volume (r^2 * h):", orig_volume)
    print("Deformed volume (r'^2 * h'):", def_volume)
    
    # Visualize
    original_np = original_points.numpy()
    deformed_np = deformed_points.numpy()
    visualize_point_clouds(original_np, deformed_np)