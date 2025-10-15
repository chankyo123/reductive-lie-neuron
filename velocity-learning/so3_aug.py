import os
import numpy as np
from scipy.spatial.transform import Rotation
import shutil
import pandas as pd

def rand_rotation_matrix(theta, roll, pitch):
    """
    Creates an SO(3) rotation matrix from yaw (theta), roll, and pitch angles.
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    
    # ZYX rotation sequence
    Mz = np.array([[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]])
    My = np.array([[c_p, 0, s_p], [0, 1, 0], [-s_p, 0, c_p]])
    Mx = np.array([[1, 0, 0], [0, c_r, -s_r], [0, s_r, c_r]])
    
    rotation_matrix = Mz @ My @ Mx
    return rotation_matrix

def apply_rotation_to_vectors(vectors, rotation_matrix):
    """
    Applies rotation to a batch of 3D vectors.
    vectors: (N, 3) array
    rotation_matrix: (3, 3) array
    Returns: (N, 3) array of rotated vectors.
    """
    # Note: (R @ v.T).T is equivalent to v @ R.T
    return np.dot(vectors, rotation_matrix.T)

def apply_rotation_to_covariances(covariances_flat, rotation_matrix):
    """
    Applies rotation to a batch of flattened 3x3 covariance matrices.
    covariances_flat: (N, 9) array
    rotation_matrix: (3, 3) array
    Returns: (N, 9) array of rotated flattened covariances.
    """
    # Reshape from (N, 9) to (N, 3, 3)
    cov_matrices = covariances_flat.reshape(-1, 3, 3)
    
    # Apply the transformation C' = R @ C @ R.T using einsum for efficiency
    # 'ik,nkl,jl->nij' corresponds to R[i,k] * C[n,k,l] * R.T[l,j]
    rotated_cov_matrices = np.einsum('ik,nkl,lj->nij', rotation_matrix, cov_matrices, rotation_matrix.T)
    
    # Reshape back to (N, 9)
    return rotated_cov_matrices.reshape(-1, 9)

# --- Configuration ---
data_directory = "./local_data/results_cov3"
save_directory = "./local_data/results_cov3_so3_2"
# ---

# Use the test_list.txt for processing
list_path = os.path.join(data_directory, "test_list.txt")

# Load subdirectory names
if not os.path.exists(list_path):
    print(f"Error: Could not find {list_path}")
else:
    with open(list_path, "r") as file:
        subdirectories = [line.strip() for line in file]

    rpy_values = []
    print(f"Found {len(subdirectories)} sequences to process...")

    for subdirectory in subdirectories:
        subdirectory_path = os.path.join(data_directory, subdirectory)
        npy_file_path = os.path.join(subdirectory_path, "combined_flight_data.npy")

        if os.path.exists(npy_file_path):
            original_data = np.load(npy_file_path)

            # Generate a random SO(3) rotation
            rad_range_roll_pitch = 60 * (np.pi / 180) # +/- 60 degrees for roll and pitch
            random_roll = np.random.uniform(-rad_range_roll_pitch, rad_range_roll_pitch)
            random_pitch = np.random.uniform(-rad_range_roll_pitch, rad_range_roll_pitch)
            random_theta = np.random.uniform(0, 2 * np.pi) # Full range for yaw
            
            rpy_values.append([random_roll, random_pitch, random_theta])
            rotation_matrix = rand_rotation_matrix(random_theta, random_roll, random_pitch)
            
            # --- Apply Rotations to Corresponding Fields ---
            
            # Extract columns
            ts_us = original_data[:, 0:1]
            gyr_world = original_data[:, 1:4]
            acc_world = original_data[:, 4:7]
            q_World_Device = original_data[:, 7:11] # qxyzw
            pos_World_Device = original_data[:, 11:14]
            vel_World = original_data[:, 14:17]
            cov_World_flat = original_data[:, 17:26]
            vel_World_gt = original_data[:, 26:29]

            # Rotate vectors
            rotated_gyr = apply_rotation_to_vectors(gyr_world, rotation_matrix)
            rotated_acc = apply_rotation_to_vectors(acc_world, rotation_matrix)
            rotated_pos = apply_rotation_to_vectors(pos_World_Device, rotation_matrix)
            rotated_vel = apply_rotation_to_vectors(vel_World, rotation_matrix)
            rotated_vel_gt = apply_rotation_to_vectors(vel_World_gt, rotation_matrix)
            
            # Rotate covariance
            rotated_cov_flat = apply_rotation_to_covariances(cov_World_flat, rotation_matrix)

            # Rotate quaternion
            # R_new = R_rot @ R_old
            r_old = Rotation.from_quat(q_World_Device)
            r_rot = Rotation.from_matrix(rotation_matrix)
            r_new = r_rot * r_old
            rotated_q = r_new.as_quat()

            # Concatenate the final rotated data array
            rotated_data = np.concatenate([
                ts_us,
                rotated_gyr,
                rotated_acc,
                rotated_q,
                rotated_pos,
                rotated_vel,
                rotated_cov_flat,
                rotated_vel_gt
            ], axis=1)
            
            # --- Save the Rotated Data ---
            save_path = os.path.join(save_directory, subdirectory)
            os.makedirs(save_path, exist_ok=True)
            np.save(os.path.join(save_path, "combined_flight_data.npy"), rotated_data)
            print(f"Processed and saved: {os.path.join(save_path, 'combined_flight_data.npy')}")

    # Save the rotation values for reference
    np.save(os.path.join(save_directory, "rpy_values.npy"), rpy_values)
    print(f"Saved rotation values to: {os.path.join(save_directory, 'rpy_values.npy')}")
    
    # --- Copy Ancillary Files ---
    print("Copying ancillary files (json, txt, etc.)...")
    specific_files = ['all_ids.txt', 'spline_metrics.csv', 'test_list.txt', 'train_list.txt', 'val_list.txt']
    for file in specific_files:
        file_path = os.path.join(data_directory, file)
        if os.path.exists(file_path):
            shutil.copy(file_path, save_directory)

    for subdir in subdirectories:
        subdir_path = os.path.join(data_directory, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".json"):
                    dest_dir = os.path.join(save_directory, subdir)
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(os.path.join(subdir_path, file), dest_dir)
    print("Done.")