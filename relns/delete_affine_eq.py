import numpy as np

# Function to apply an affine transformation T(x) = Ax + b in 3D
def apply_affine(points, A, b):
    return (A @ points.T).T + b

# Function to compute affine transformation between two 3D point sets
def compute_affine(P1, P2):
    P1_h = np.hstack((P1, np.ones((P1.shape[0], 1))))
    P2_h = np.hstack((P2, np.ones((P2.shape[0], 1))))
    T, _, _, _ = np.linalg.lstsq(P1_h, P2_h, rcond=None)
    
    print(T)
    return T[:3, :3], T[:3, 3]  # A (3x3) and b (3D vector)

# Define initial 3D points for P1 (square plane in xy-plane at z=0)
P1 = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0]
], dtype=float)

# Define T1: P2 = T1(P1)
A1 = np.array([
    [2, 0, 0],  # Scale x by 2
    [0, 1, 0],
    [0, 0, 1]
])
b1 = np.array([1, 1, 1])
P2 = apply_affine(P1, A1, b1)

# Define T2
A2 = np.array([
    [1, 0, 0.5],  # Shear x along z
    [0, 1, 0],
    [0, 0, 1]
])
b2 = np.array([0, 2, 1])

# Apply T2 to both P1 and P2
P1_prime = apply_affine(P1, A2, b2)
P2_prime = apply_affine(P2, A2, b2)

# Compute T3: P2_prime = T3(P1_prime)
A3, b3 = compute_affine(P1_prime, P2_prime)

# Compute T2^-1
A2_inv = np.linalg.inv(A2)
b2_inv = -A2_inv @ b2

# Compute conjugate: T2 ∘ T1 ∘ T2^-1
A_conjugate = A2 @ A1 @ A2_inv
b_conjugate = A2 @ (A1 @ b2_inv + b1) + b2

# # Print the requested transformations
# print("Affine transformation between transformed planes (T3):")
# print("A3:")
# print(A3)
# print("b3:", b3)
# print("\nConjugate of original affine transformation (T2 ∘ T1 ∘ T2^-1):")
# print("A_conjugate:")
# print(A_conjugate)
# print("b_conjugate:", b_conjugate)

# Check if they are the same within tolerance
tolerance = 1e-10
are_same = (np.allclose(A3, A_conjugate, atol=tolerance) and 
            np.allclose(b3, b_conjugate, atol=tolerance))
print(f"\nAre they the same within tolerance {tolerance}? {are_same}")