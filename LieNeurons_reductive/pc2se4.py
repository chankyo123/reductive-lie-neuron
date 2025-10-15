import numpy as np
from scipy.spatial.transform import Rotation

# Define tetrahedron vertices (centered at origin)
vertices = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]) 

# Define faces (each row is indices of 3 vertices)
faces = np.array([
    [0, 1, 2],  # f0: v0-v1-v2
    [0, 1, 3],  # f1: v0-v1-v3
    [0, 2, 3],  # f2: v0-v2-v3
    [1, 2, 3]   # f3: v1-v2-v3
])

# Find adjacent face pairs (sharing an edge)
adjacent_pairs = []
for i in range(4):
    for j in range(i+1, 4):
        common_vertices = set(faces[i]) & set(faces[j])
        if len(common_vertices) == 2:  # Adjacent if they share exactly 2 vertices
            adjacent_pairs.append((i, j))

def order_points(face1_idx, face2_idx):
    """Order points according to specifications"""
    f1 = faces[face1_idx]
    f2 = faces[face2_idx]
    # Find common edge vertices
    common = set(f1) & set(f2)
    not_common_f1 = list(set(f1) - common)[0]
    not_common_f2 = list(set(f2) - common)[0]
    common_list = list(common)
    
    # Order v1: non-common point first, then common points clockwise
    v1 = vertices[f1]
    center = np.mean(vertices, axis=0)
    normal = np.cross(v1[1] - v1[0], v1[2] - v1[0])
    if np.dot(normal, v1[0] - center) < 0:  # Ensure normal points outward
        normal = -normal
    
    # v1: [non-common, common1, common2] in clockwise order
    v1_ordered = [not_common_f1]
    # common_list.reverse()
    v1_ordered.extend(common_list)
    v1_points = vertices[v1_ordered]
    
    # Check and fix clockwise order
    edge1 = v1_points[1] - v1_points[0]
    edge2 = v1_points[2] - v1_points[0]
    if np.dot(np.cross(edge1, edge2), normal) > 0:
        v1_points[[1, 2]] = v1_points[[2, 1]]
    
    # v2: [second from v1, non-common, remaining]
    v2_ordered = [v1_ordered[1], not_common_f2, v1_ordered[2]]
    v2_points = vertices[v2_ordered]
    
    return v1_points, v2_points

def compute_se4_transform(source, target):
    """Compute SE(4) transformation from source to target points"""
    # Center points
    src_center = np.mean(source, axis=0)
    tgt_center = np.mean(target, axis=0)
    
    # Translate to origin
    src_centered = source - src_center
    tgt_centered = target - tgt_center
    
    # Find rotation using Kabsch algorithm
    H = src_centered.T @ tgt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (not reflection)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # Translation
    t = tgt_center - R @ src_center
    
    # Construct SE(4) matrix (4x4)
    se4 = np.eye(4)
    se4[:3, :3] = R
    se4[:3, 3] = t
    
    # Convert to 16D vector
    return se4.flatten()

def gl4_transformation(scale_range=(0.5, 2.0), shear_std=0.1, translation_range=(-1, 1)):
    """Generate a random GL(4) transformation matrix."""
    n = 3  # For the 3D part

    # Random rotation
    R = Rotation.random().as_matrix()

    # Random scaling
    scale = np.random.uniform(*scale_range, size=(n,))
    S = np.diag(scale)

    # Random shear
    shear = np.random.normal(0, shear_std, size=(n, n))
    np.fill_diagonal(shear, 1)  # Ensure diagonal is 1 to preserve the main scaling

    # Combine rotation, scaling, and shear
    gl3 = R @ S @ shear

    # Random translation
    translation = np.random.uniform(*translation_range, size=(3,))

    # Form the GL(4) matrix
    gl4 = np.eye(4)
    gl4[:3, :3] = gl3
    gl4[:3, 3] = translation

    return gl4

def affine_transformation(scale_range=(0.5, 2.0), shear_std=0.1, translation_range=(-1, 1)):
    """Generate a random affine transformation matrix (GL(4))."""
    n = 3  # 3D transformation

    # Random rotation
    R = Rotation.random().as_matrix()

    # Random scaling
    scale = np.random.uniform(*scale_range, size=(n,))
    S = np.diag(scale)

    # Random shear
    shear = np.random.normal(0, shear_std, size=(n, n))
    np.fill_diagonal(shear, 1)  # Keep diagonal as 1 to avoid degenerate scaling

    # Combine rotation, scaling, and shear
    affine_3x3 = R @ S @ shear
    affine_3x3 = np.array([[4,0,0],
                          [0,2,0],
                          [0,0,1]])  # Ensure scaling is uniform for simplicity
    affine_3x3 = np.eye(3)
    # Random translation vector
    # translation = np.random.uniform(*translation_range, size=(3,))
    translation = np.array([10,0,0])

    # Form the affine GL(4) matrix
    affine = np.eye(4)
    affine[:3, :3] = affine_3x3
    affine[:3, 3] = translation
    return affine

def se4_transformation(translation_range=(-1, 1)):
    """Generate a random SE(4) transformation matrix."""
    n = 3  # For the 3D part

    # Random rotation
    R = Rotation.random().as_matrix()

    # Translation
    translation = np.random.uniform(*translation_range, size=(3,))

    # Form the SE(4) matrix
    se4 = np.eye(4)
    se4[:3, :3] = R
    se4[:3, 3] = translation

    return se4


def to_homogeneous(points):
    """Convert 3D points to homogeneous coordinates."""
    ones = np.ones((points.shape[0], 1))
    return np.hstack((points, ones))

def check_equivariance(vertices, transformations):
    """Check if applying GL(n) transformation commutes with the affine transformation"""
    # gln = gl4_transformation()
    # gln = affine_transformation()
    gln = se4_transformation()

    # Compute new affine transformations from transformed vertices
    new_transformations = []
    for face1_idx, face2_idx in adjacent_pairs:
        v1, v2 = order_points(face1_idx, face2_idx)
        v1_h = to_homogeneous(v1)  # 3x4 matrix
        v2_h = to_homogeneous(v2)
        
        # Apply GL(n) to v1 and v2
        v1_gln = v1_h @ gln.T
        v2_gln = v2_h @ gln.T
        v1_gln = v1_gln[:, :3]
        v2_gln = v2_gln[:, :3]

        # Compute SE(4) transformation from GL(n) transformed faces
        t12_gln = compute_se4_transform(v1_gln, v2_gln)
        new_transformations.append(t12_gln)
        
        # v2 to v1 transformation
        t21_gln = compute_se4_transform(v2_gln, v1_gln)
        new_transformations.append(t21_gln)

    # Check equivariance condition
    consistent = True
    tol = 1e-5
    for i, (original, transformed) in enumerate(zip(transformations, new_transformations)):
        # Apply GL(n) to the output transformation
        equivariant_output = gln @ original.reshape(4,4) @ np.linalg.inv(gln)
        
        # print(equivariant_output)
        # print(transformed.reshape(4,4))
        # Check if the two transformations are close
        if not np.allclose(equivariant_output, transformed.reshape(4,4), atol=tol):
            # print(f"Equivariance failed for transformation {i}")
            consistent = False
    
    if consistent:
        print("Equivariance holds for all transformations.")
    else:
        print("Equivariance failed for some transformations.")
        
    
# Calculate all transformations
transformations = []
for face1_idx, face2_idx in adjacent_pairs:
    # Get ordered points
    v1, v2 = order_points(face1_idx, face2_idx)
    
    # v1 to v2 transformation
    t12 = compute_se4_transform(v1, v2)
    transformations.append(t12)
    
    # v2 to v1 transformation
    t21 = compute_se4_transform(v2, v1)
    transformations.append(t21)
# Convert to numpy array
transformations = np.array(transformations)

# Print results
print(f"Number of transformations: {len(transformations)}")
print(f"Shape of each transformation: {transformations[0].shape}")
# print("\nFirst transformation:")
# print(transformations[0].reshape(4, 4))

# Verify we have 12 transformations
assert len(transformations) == 12, f"Expected 12 transformations, got {len(transformations)}"
assert transformations.shape[1] == 16, f"Expected 16D vectors, got {transformations.shape[1]}D"

# Run equivariance check
check_equivariance(vertices, transformations)