import numpy as np
from scipy.spatial.transform import Rotation

# Define tetrahedron vertices (centered at origin)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

# Define faces (each face is indices of 3 vertices)
faces = np.array([
    [0, 1, 2],  # v0-v1-v2
    [0, 1, 3],  # v0-v1-v3
    [0, 2, 3],  # v0-v2-v3
    [1, 2, 3]   # v1-v2-v3
])

# Identify adjacent face pairs
adjacent_pairs = []
for i in range(4):
    for j in range(i + 1, 4):
        common_vertices = set(faces[i]) & set(faces[j])
        if len(common_vertices) == 2:
            adjacent_pairs.append((i, j))

def order_points(face1_idx, face2_idx):
    """Order points according to specifications"""
    f1 = faces[face1_idx]
    f2 = faces[face2_idx]
    
    # Find common edge vertices
    common = list(set(f1) & set(f2))
    not_common_f1 = list(set(f1) - set(common))[0]
    not_common_f2 = list(set(f2) - set(common))[0]
    
    # Order v1: non-common point first, then common points counterclockwise
    v1 = vertices[f1]
    center = np.mean(vertices, axis=0)
    normal = np.cross(v1[1] - v1[0], v1[2] - v1[0])
    if np.dot(normal, v1[0] - center) < 0:  # Ensure normal points outward
        normal = -normal
    
    # Check and fix counterclockwise order
    ordered_common = common
    if np.dot(np.cross(vertices[ordered_common[0]] - vertices[not_common_f1], 
                       vertices[ordered_common[1]] - vertices[not_common_f1]), normal) < 0:
        ordered_common.reverse()
    
    v1_points = vertices[[not_common_f1] + ordered_common]
    
    # v2: [second from v1, non-common, remaining]
    v2_ordered = [ordered_common[0], not_common_f2, ordered_common[1]]
    v2_points = vertices[v2_ordered]
    
    return v1_points, v2_points

def compute_affine_transform(source, target):
    """Compute affine transformation from source to target points"""
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))
    
    transform_matrix, _, _, _ = np.linalg.lstsq(source_h, target_h, rcond=None)
    return transform_matrix.T.flatten()

def affine_transformation(scale_range=(0.5, 2.0), shear_std=0.1, translation_range=(-1, 1)):
    """Generate a random affine GL(4) matrix."""
    n = 3

    R = Rotation.random().as_matrix()
    scale = np.random.uniform(*scale_range, size=(n,))
    S = np.diag(scale)
    shear = np.eye(n) + np.random.normal(0, shear_std, size=(n, n))
    affine_3x3 = R @ S @ shear
    translation = np.random.uniform(*translation_range, size=(3,))
    
    affine = np.eye(4)
    affine[:3, :3] = affine_3x3
    affine[:3, 3] = translation
    return affine

def to_homogeneous(points):
    """Convert 3D points to homogeneous coordinates."""
    ones = np.ones((points.shape[0], 1))
    return np.hstack((points, ones))

def check_equivariance(vertices, transformations):
    """Check if applying affine transformation commutes with the computed transformations"""
    affine = affine_transformation()

    # Compute new affine transformations from transformed vertices
    new_transformations = []
    for face1_idx, face2_idx in adjacent_pairs:
        v1, v2 = order_points(face1_idx, face2_idx)
        v1_h = to_homogeneous(v1)
        v2_h = to_homogeneous(v2)
        
        # Apply affine to v1 and v2
        v1_affine = (v1_h @ affine.T)[:, :3]
        v2_affine = (v2_h @ affine.T)[:, :3]

        # Compute the transformation from affine transformed faces
        t12_affine = compute_affine_transform(v1_affine, v2_affine)
        new_transformations.append(t12_affine)
        
        # v2 to v1 transformation
        t21_affine = compute_affine_transform(v2_affine, v1_affine)
        new_transformations.append(t21_affine)

    consistent = True
    tol = 1e-5
    for i, (original, transformed) in enumerate(zip(transformations, new_transformations)):
        equivariant_output = affine @ original.reshape(4,4) @ np.linalg.inv(affine)

        if not np.allclose(equivariant_output, transformed.reshape(4,4), atol=tol):
            print(f"Equivariance failed for transformation {i}")
            consistent = False
    
    if consistent:
        print("Equivariance holds for all transformations.")
    else:
        print("Equivariance failed for some transformations.")

# Calculate all transformations
transformations = []
for face1_idx, face2_idx in adjacent_pairs:
    v1, v2 = order_points(face1_idx, face2_idx)
    
    # v1 to v2 transformation
    t12 = compute_affine_transform(v1, v2)
    transformations.append(t12)
    
    # v2 to v1 transformation
    t21 = compute_affine_transform(v2, v1)
    transformations.append(t21)

transformations = np.array(transformations)

# Verify we have 12 transformations
assert len(transformations) == 12, f"Expected 12 transformations, got {len(transformations)}"
assert transformations.shape[1] == 16, f"Expected 16D vectors, got {transformations.shape[1]}D"

# Run equivariance check
check_equivariance(vertices, transformations)