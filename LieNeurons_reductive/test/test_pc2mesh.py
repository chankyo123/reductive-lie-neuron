import open3d as o3d
import numpy as np
from itertools import combinations

# Angle of rotation in radians
theta = np.pi / 4  # 45 degrees

# Source points in 3D
S = np.array([[1, 1, 1],
              [2, 1, 1],
              [1, 2, 1]])

# Rotation matrix about the z-axis
R_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]])

# Compute target points
T = S.dot(R_z.T)

# Solve for affine transformation using least squares
A, _, _, _ = np.linalg.lstsq(S, T, rcond=None)

# Compare the computed A with the rotation matrix R_z
print("Computed Transformation Matrix A:")
print(A)

print("\nGround Truth Rotation Matrix R_z:")
print(R_z)

# Calculate the difference
difference = np.abs(A - R_z)
print("\nDifference between computed A and ground truth R_z:")
print(difference)

# Check if the difference is within a small threshold
threshold = 1e-6
is_same = np.all(difference < threshold)
print("\nIs the obtained transformation the same as the ground truth (within tolerance)?", is_same)
assert False

def create_subdivided_icosahedron(radius=1.0, subdivision_level=2):
    icosahedron = o3d.geometry.TriangleMesh.create_icosahedron(radius=radius)
    icosahedron.compute_vertex_normals()
    mesh = icosahedron.subdivide_loop(number_of_iterations=subdivision_level)
    mesh.compute_vertex_normals()
    return mesh

def compute_affine_transformation(v1, v2):
    v1_h = np.hstack([v1, np.ones((3, 1))])
    v2_h = np.hstack([v2, np.ones((3, 1))])
    M, _, _, _ = np.linalg.lstsq(v1_h, v2_h, rcond=None)
    return M.T

def compute_all_affine_transformations(mesh):
    vertex_array = np.asarray(mesh.vertices)
    triangle_array = np.asarray(mesh.triangles)

    affine_transformations = {}
    affine_transformations_stack = []

    for i, j in combinations(range(len(triangle_array)), 2):
        tri_a = triangle_array[i]
        tri_b = triangle_array[j]

        shared_vertices = set(tri_a) & set(tri_b)
        if len(shared_vertices) >= 2:
            vertices_a = vertex_array[tri_a]
            vertices_b = vertex_array[tri_b]
            
            M = compute_affine_transformation(vertices_a, vertices_b)
            affine_transformations[(i, j)] = M
            affine_transformations_stack.append(M)
    return affine_transformations, np.array(affine_transformations_stack)

def order_triangle_vertices(vertices, triangle):
    """
    Order vertices of a triangle in counterclockwise order.
    """
    # Fetch the vertex positions
    v1, v2, v3 = vertices[triangle]
    
    # Utilize a simple criteria based on cross product and sum to ensure order. 
    # Create a vector from v1 to v2 and from v1 to v3
    vec1 = v2 - v1
    vec2 = v3 - v1

    # Cross product should be consistent using normal direction and ensure v1 is the start
    cross_product = np.cross(vec1, vec2)
    
    if cross_product[2] < 0:
        return [triangle[0], triangle[2], triangle[1]]  # Swap the order to follow a given convention
    else:
        return [triangle[0], triangle[1], triangle[2]]
    
def compute_ordered_affine_transformations(mesh):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    affine_transformations = {}

    for i in range(len(triangles)):
        for j in range(i + 1, len(triangles)):
            tri_a = order_triangle_vertices(vertices, triangles[i])
            tri_b = order_triangle_vertices(vertices, triangles[j])

            if len(set(tri_a) & set(tri_b)) == 2:  # Check adjacency by shared edge
                # Extract the correctly ordered vertices
                ordered_vertices_a = vertices[tri_a]
                ordered_vertices_b = vertices[tri_b]
                
                # Compute the transformation
                M = compute_affine_transformation(ordered_vertices_a, ordered_vertices_b)
                affine_transformations[(i, j)] = M
                
    return affine_transformations

def apply_gln_to_vertices(mesh, gln_matrix):
    vertices = np.asarray(mesh.vertices)
    ones = np.ones((vertices.shape[0], 1))
    homogeneous_vertices = np.hstack((vertices, ones))
    transformed_vertices = homogeneous_vertices @ gln_matrix.T
    mesh_transformed = o3d.geometry.TriangleMesh()
    mesh_transformed.vertices = o3d.utility.Vector3dVector(transformed_vertices[:, :3])
    mesh_transformed.triangles = mesh.triangles
    mesh_transformed.compute_vertex_normals()
    return mesh_transformed

def apply_transformed_affine(mesh, transformation_dict):
    transformed_vertices = np.asarray(mesh.vertices).copy()

    for (i, j), M in transformation_dict.items():
        tri_a = mesh.triangles[i]
        # Reorder vertices to get counterclockwise orientation
        ordered_tri_a = order_triangle_vertices(transformed_vertices, tri_a)
        
        # Apply transformation to the triangle vertices
        transformed_vertices[ordered_tri_a] = (
            np.dot(np.hstack((transformed_vertices[ordered_tri_a], np.ones((3, 1)))), M.T)
        )[:, :3]

    transformed_mesh = o3d.geometry.TriangleMesh()
    transformed_mesh.vertices = o3d.utility.Vector3dVector(transformed_vertices)
    transformed_mesh.triangles = mesh.triangles
    transformed_mesh.compute_vertex_normals()
    return transformed_mesh

def remesh_uniformly(mesh, target_number_of_faces=5000):
    """
    Remesh the given mesh to have approximately target_number_of_faces.
    This function adjusts the mesh to ensure more uniform face sizes.
    """
    mesh_simplified = mesh.simplify_quadric_decimation(target_number_of_faces)
    mesh_simplified.compute_vertex_normals()
    return mesh_simplified

def smooth_mesh(mesh, number_of_iterations=10):
    """
    Apply Laplacian smoothing to the mesh to make it smoother.
    """
    mesh_smooth = mesh.filter_smooth_laplacian(number_of_iterations=number_of_iterations)
    mesh_smooth.compute_vertex_normals()  # Recalculate normals for better lighting view
    return mesh_smooth

# GL(4) transformation matrix
gln_matrix = np.array([
    [10, 10, 0.0, 0.0],
    [0.0, 2, 0.1, 0.0],
    [0.1, 0.0, 3, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

# gln_matrix = np.eye(4)

# Create the original mesh
sphere_mesh = create_subdivided_icosahedron(radius=1.0, subdivision_level=2)
# Directly apply the GL(4) matrix to vertices
sphere_mesh_gln_applied = apply_gln_to_vertices(sphere_mesh, gln_matrix)

# Compute the original affine transformations
affine_transforms_original, affine_transforms_original_stack = compute_all_affine_transformations(sphere_mesh)
affine_transforms_ordered = compute_ordered_affine_transformations(sphere_mesh)
# Apply the GL(n) matrix to each affine transformation
transformed_affine_due_to_gln = {
    key: gln_matrix @ M @ np.linalg.inv(gln_matrix) for key, M in affine_transforms_original.items()
}

# Apply the transformed affine transformations to create a new mesh
mesh_transformed_affine = apply_transformed_affine(sphere_mesh, transformed_affine_due_to_gln)
mesh_transformed_affine_remeshed = remesh_uniformly(mesh_transformed_affine, target_number_of_faces=3000)
mesh_transformed_affine_smoothed = smooth_mesh(mesh_transformed_affine_remeshed, number_of_iterations=10)

# Visualize both remeshed objects
sphere_mesh_gln_applied.paint_uniform_color([1, 0.706, 0])  # Original casting paint
mesh_transformed_affine.paint_uniform_color([0, 1, 0])  # Smoothed painted green

o3d.visualization.draw_geometries([sphere_mesh_gln_applied], window_name='Mesh with Direct GL(n) Transformation', width=800, height=600)
o3d.visualization.draw_geometries([mesh_transformed_affine], window_name='Smoothed Mesh with Regularized Faces', width=800, height=600)