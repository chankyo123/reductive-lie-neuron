import numpy as np
from scipy.spatial.transform import Rotation

def generate_ground_truth_affine():
    """Generate a random ground truth affine transformation."""
    R = Rotation.random().as_matrix()  # Random rotation
    translation = np.random.uniform(-5, 5, size=(3,))  # Random translation
    scale = np.random.uniform(0.5, 2.0, size=(3,))
    S = np.diag(scale)  # Random scaling

    # Ground truth affine matrix
    affine_gt = np.eye(4)
    affine_gt[:3, :3] = R @ S
    affine_gt[:3, 3] = translation
    return affine_gt

def apply_affine(points, affine):
    """Apply an affine transformation to a set of 3D points."""
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Transform to homogeneous coordinates
    transformed_points_h = points_h @ affine.T  # Apply affine transformation
    return transformed_points_h[:, :3]  # Back to 3D coordinates

def solve_affine(source, target):
    """Solve the affine transformation between source and target points."""
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))
    affine_matrix, _, _, _ = np.linalg.lstsq(source_h, target_h, rcond=None)
    return affine_matrix.T

def test_affine_transformation():
    # Define four non-coplanar points on the first plane
    plane1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Generate a random ground truth affine transformation
    gt_affine = generate_ground_truth_affine()
    print("Ground Truth Affine Transformation:")
    print(gt_affine)

    # Transform the points on the first plane using the ground truth transformation
    plane2 = apply_affine(plane1, gt_affine)

    # Solve for the affine transformation that maps plane1 to plane2
    computed_affine = solve_affine(plane1, plane2)
    print("\nComputed Affine Transformation:")
    print(computed_affine)

    # Check if the computed affine transformation is close to the ground truth
    if np.allclose(computed_affine, gt_affine, atol=1e-6):
        print("\nThe computed affine transformation is correct.")
    else:
        print("\nThe computed affine transformation is incorrect.")

# test_affine_transformation()

def generate_random_affine(scale_range=(0.5, 2.0), translation_range=(-1, 1)):
    """Generate a random affine transformation matrix."""
    R = Rotation.random().as_matrix()  # Random rotation
    scale = np.random.uniform(*scale_range, size=(3,))
    S = np.diag(scale)  # Random scaling
    translation = np.random.uniform(*translation_range, size=(3,))  # Random translation

    affine = np.eye(4)
    affine[:3, :3] = R @ S
    affine[:3, 3] = translation
    return affine

def apply_affine(points, affine):
    """Apply an affine transformation to a set of 3D points."""
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))  # Transform to homogeneous coordinates
    transformed_points_h = points_h @ affine.T  # Apply affine transformation
    return transformed_points_h[:, :3]  # Back to 3D coordinates

def solve_affine(source, target):
    """Solve the affine transformation between source and target points."""
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))
    affine_matrix, _, _, _ = np.linalg.lstsq(source_h, target_h, rcond=None)
    return affine_matrix.T

def test_affine_conjugation():
    # Define four non-coplanar points on the first plane
    plane1 = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    # Generate a random ground truth affine transformation
    gt_affine = generate_random_affine()
    print("Ground Truth Affine Transformation:")
    print(gt_affine)

    # Transform the points on the first plane using the ground truth transformation
    plane2 = apply_affine(plane1, gt_affine)

    # Generate another random affine transformation
    additional_affine = generate_random_affine()
    print("\nAdditional Affine Transformation:")
    print(additional_affine)

    # Transform the second plane using this additional affine transformation
    plane3 = apply_affine(plane1, additional_affine)
    plane4 = apply_affine(plane2, additional_affine)

    # Solve for the affine transformation that maps plane1 to plane3
    final_affine = solve_affine(plane3, plane4)
    print("\nFinal Affine Transformation:")
    print(final_affine)

    # Compute the conjugate of the ground truth by the additional affine
    conjugate_affine = additional_affine @ gt_affine @ np.linalg.inv(additional_affine)
    print("\nConjugate Affine Transformation:")
    print(conjugate_affine)

    # Check if the final affine transformation is the same as the conjugate transformation
    if np.allclose(final_affine, conjugate_affine, atol=1e-6):
        print(final_affine, conjugate_affine)
        print("\nThe final affine transformation matches the conjugate transformation.")
    else:
        print("\nThe final affine transformation does not match the conjugate transformation.")

# test_affine_conjugation()



def generate_random_plane():
    """Generate 4 non-coplanar random points to define a 3D plane."""
    while True:
        points = np.random.uniform(-5, 5, (4, 3))  # 4 points in 3D space
        # Ensure points are non-coplanar by checking the volume of the tetrahedron they form
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        v3 = points[3] - points[0]
        volume = np.abs(np.dot(v1, np.cross(v2, v3)))
        if volume > 1e-6:
            return points

def generate_random_affine(scale_range=(0.5, 2.0), translation_range=(-1, 1)):
    """Generate a random affine transformation matrix."""
    R = Rotation.random().as_matrix()  # Random rotation
    scale = np.random.uniform(*scale_range, size=(3,))
    S = np.diag(scale)  # Random scaling
    translation = np.random.uniform(*translation_range, size=(3,))  # Random translation

    affine = np.eye(4)
    affine[:3, :3] = R @ S
    affine[:3, 3] = translation
    return affine

def apply_affine(points, affine):
    """Apply an affine transformation to a set of 3D points."""
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points_h = points_h @ affine.T
    return transformed_points_h[:, :3]

def solve_affine(source, target):
    """Solve the affine transformation between source and target points."""
    source_h = np.hstack((source, np.ones((source.shape[0], 1))))
    target_h = np.hstack((target, np.ones((target.shape[0], 1))))
    affine_matrix, _, _, _ = np.linalg.lstsq(source_h, target_h, rcond=None)
    return affine_matrix.T

def test_conjugate_affine_transformation():
    # Generate two random non-coplanar planes
    plane1 = generate_random_plane()
    plane2 = generate_random_plane()

    # Compute the initial affine transformation between plane1 and plane2
    initial_affine = solve_affine(plane1, plane2)
    print("Initial Affine Transformation:")
    print(initial_affine)

    # Generate a random affine transformation
    random_affine = generate_random_affine()
    print("\nRandom Affine Transformation:")
    print(random_affine)

    # Apply the random affine transformation to both planes
    transformed_plane1 = apply_affine(plane1, random_affine)
    transformed_plane2 = apply_affine(plane2, random_affine)

    # Compute the affine transformation between the transformed planes
    final_affine = solve_affine(transformed_plane1, transformed_plane2)
    print("\nAffine Transformation between Transformed Planes:")
    print(final_affine)

    # Compute the conjugate of the initial transformation
    conjugate_affine = random_affine @ initial_affine @ np.linalg.inv(random_affine)
    print("\nConjugate of Initial Affine Transformation:")
    print(conjugate_affine)

    # Compare the final affine transformation with the conjugate transformation
    if np.allclose(final_affine, conjugate_affine, atol=1e-6):
        print("\nThe affine transformation between the transformed planes matches the conjugate transformation.")
    else:
        print("\nThe affine transformation between the transformed planes does not match the conjugate transformation.")

test_conjugate_affine_transformation()