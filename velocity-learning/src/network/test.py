"""
This file includes the main libraries in the network testing module
"""

import json
import os
from os import path as osp

import matplotlib.pyplot as plt
import torch
#from dataloader.dataset_fb import FbSequenceDataset
from dataloader.tlio_data import TlioData
from dataloader.memmapped_sequences_dataset import MemMappedSequencesDataset
from network.losses import get_loss
from network.model_factory import get_model
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation
from torch.utils.data import DataLoader
from utils.dotdict import dotdict
from utils.utils import to_device
from utils.logging import logging
from utils.math_utils import *
import time


def compute_rpe(rpe_ns, ps, ps_gt, yaw, yaw_gt):
    ns = ps_gt.shape[0]
    assert ns - rpe_ns > 100
    assert ps.shape == ps_gt.shape
    assert yaw.shape == yaw_gt.shape

    rpes = []
    relative_yaw_errors = []
    for i in range(0, ns - rpe_ns, 100):
        chunk = ps[i : i + rpe_ns, :]
        chunk_gt = ps_gt[i : i + rpe_ns, :]
        chunk_yaw = yaw[i : i + rpe_ns, :]
        chunk_yaw_gt = yaw_gt[i : i + rpe_ns, :]
        initial_error_yaw = wrap_rpy(chunk_yaw[0, :] - chunk_yaw_gt[0, :])
        final_error_p_relative = Rotation.from_euler(
            "z", initial_error_yaw, degrees=True
        ).as_matrix().dot((chunk[[-1], :] - chunk[[0], :]).T)[0, :, :].T - (
            chunk_gt[[-1], :] - chunk_gt[[0], :]
        )
        final_error_yaw = wrap_rpy(chunk_yaw[[-1], :] - chunk_yaw_gt[[-1], :])
        rpes.append(final_error_p_relative)
        relative_yaw_errors.append(wrap_rpy(final_error_yaw - initial_error_yaw))
    rpes = np.concatenate(rpes, axis=0)
    relative_yaw_errors = np.concatenate(relative_yaw_errors, axis=0)

    plt.figure("relative yaw error")
    plt.plot(relative_yaw_errors)
    plt.figure("rpes list")
    plt.plot(rpes)
    # compute statistics over z separately
    rpe_rmse = np.sqrt(np.mean(np.sum(rpes ** 2, axis=1)))
    rpe_rmse_z = np.sqrt(np.mean(rpes[:, 2] ** 2))
    relative_yaw_rmse = np.sqrt(np.mean(relative_yaw_errors ** 2))
    return rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes


def pose_integrate(args, dataset, preds, use_pred_vel, body_frame):
    """
    Concatenate predicted velocity to reconstruct sequence trajectory
    """      
    
    #ind = np.array([i[1] for i in dataset.index_map], dtype=np.int)
    #delta_int = int(
    #    args.window_time * args.imu_freq / 2.0
    #)  # velocity as the middle of the segment
    if not (args.window_time * args.imu_freq / 2.0).is_integer():
        logging.info("Trajectory integration point is not centered.")
    #ind_intg = ind + delta_int  # the indices of doing integral

    ts = dataset.get_ts_last_imu_us() * 1e-6
    if body_frame:
        # r_gt, pos_gt, vel_body_gt = dataset.get_gt_traj_center_window_times(body_frame)
        r_gt, pos_gt, vel_body_gt, r_gt_last = dataset.get_gt_traj_center_window_times(body_frame)
    else:
        r_gt, pos_gt = dataset.get_gt_traj_end_window_times(body_frame)
        vel_body_gt= np.zeros(pos_gt.shape)
    eul_gt = r_gt.as_euler("xyz", degrees=True)
    
    if body_frame:
        # rotation_matrices = r_gt.as_matrix() 
        rotation_matrices = r_gt_last.as_matrix() 
        if use_pred_vel:
            pred_vels_bd = preds   #body frame velocity
            pred_vels = np.einsum('ijk,ik->ij', rotation_matrices, pred_vels_bd)
        else:
            dp_t = args.window_time
            pred_vels_bd = preds / dp_t
            pred_vels = np.einsum('ijk,ik->ij', rotation_matrices, pred_vels_bd)
            # pred_vels = pred_vels_bd
            
        # pred_vels = np.einsum('ijk,ik->ij', rotation_matrices, pred_vels_bd)
        vel_world_gt= np.einsum('ijk,ik->ij', rotation_matrices, vel_body_gt)
        
        vel_world_gt= vel_body_gt
    else:
        if use_pred_vel:
            pred_vels = preds
        else:
            dp_t = args.window_time
            pred_vels = preds / dp_t
            
    #dts = np.mean(ts[ind_intg[1:]] - ts[ind_intg[:-1]])
    dts = np.mean(ts[1:] - ts[:-1])
    #pos_intg = np.zeros([pred_vels.shape[0] + 1, args.output_dim])
    pos_intg = np.zeros([pred_vels.shape[0], args.output_dim])
    #pos_intg[0] = pos_gt[0]
    if use_pred_vel:
        if "sim_imu_longerseq/"==args.root_dir:
            displacement_intg = np.cumsum(pred_vels[:, :] * dts * 1e6, axis=0)
        else:
            displacement_intg = np.cumsum(pred_vels[:, :] * dts, axis=0)
            
            
        
            
        # displacement_intg = np.cumsum(pred_vels[:, :] * dts, axis=0)
        
        # R_z = Rotation.from_euler("xyz", [0, 0, -30], degrees=True).as_matrix()
        # print(r_gt.as_matrix()[0,:,:])
        # displacement_intg = np.einsum('jk,ik->ij', R_z, displacement_intg)
    else:
        displacement_intg = np.cumsum(pred_vels[:, :] * dts, axis=0)
        # print(pred_vels.shape)
        # print(ts.shape)
        # print(pos_gt.shape)
        # assert False
            
    pos_intg =  displacement_intg+ pos_gt[0]

        # pos_gt = pos_intg_gt + pos_gt[0]
    #ts_intg = np.append(ts[ind_intg], ts[ind_intg[-1]] + dts)
    ts_intg = np.append(ts[0], ts[-1] + dts)

    #ts_in_range = ts[ind_intg[0] : ind_intg[-1]]  # s
    pos_pred = pos_intg #interp1d(ts_intg, pos_intg, axis=0)(ts_in_range)
    #ori_pred = dataset.orientations[0][ind_intg[0] : ind_intg[-1], :]
    eul_pred = eul_gt #Rotation.from_quat(ori_pred).as_euler("xyz", degrees=True)
    
    #print("SHAPES", ts.shape, pos_pred.shape, pos_gt.shape, eul_pred.shape, eul_gt.shape)

    traj_attr_dict = {
        "ts": ts, #ts_in_range,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "eul_pred": eul_pred,
        "eul_gt": eul_gt,
        "vel_body_gt": vel_body_gt,
    }

    return traj_attr_dict


def compute_metrics_and_plotting(args, net_attr_dict, traj_attr_dict, body_frame):
    """
    Obtain trajectory and compute metrics.
    """

    """ ------------ Trajectory metrics ----------- """
    ts = traj_attr_dict["ts"]
    pos_pred = traj_attr_dict["pos_pred"]
    pos_gt = traj_attr_dict["pos_gt"]
    eul_pred = traj_attr_dict["eul_pred"]
    eul_gt = traj_attr_dict["eul_gt"]
    vel_body_gt = traj_attr_dict["vel_body_gt"]

    # ################################################################## #
    # ############### START: CODE TO SLICE DATA FROM 0-200s ############ #
    # ################################################################## #
    end_time_s = 200.0
    
    # Find the first index where the timestamp is >= end_time_s
    end_index = np.searchsorted(ts, end_time_s)
    
    # Check if any data exists before 200s
    if end_index == 0:
        logging.warning(f"No data available before {end_time_s}s. Skipping metrics for this sequence.")
        # Return empty dictionaries to avoid crashing
        return {}, {}

    # Slice all arrays from the beginning up to the end index
    ts = ts[:end_index]
    pos_pred = pos_pred[:end_index]
    pos_gt = pos_gt[:end_index]
    eul_pred = eul_pred[:end_index]
    eul_gt = eul_gt[:end_index]
    vel_body_gt = vel_body_gt[:end_index]

    # Also slice the raw network output arrays to keep all metrics consistent
    net_attr_dict["preds"] = net_attr_dict["preds"][:end_index]
    net_attr_dict["targets"] = net_attr_dict["targets"][:end_index]
    net_attr_dict["preds_vel"] = net_attr_dict["preds_vel"][:end_index]
    net_attr_dict["targets_vel"] = net_attr_dict["targets_vel"][:end_index]
    if "losses" in net_attr_dict:
        net_attr_dict["losses"] = net_attr_dict["losses"][:end_index]

    logging.info(f"--- Evaluation sliced to end at t={ts[-1]:.2f}s ---")
    # ################################################################## #
    # ########################### END OF SLICING CODE #################### #
    # ################################################################## #
    
    preds = np.array(net_attr_dict["preds"])
    targets = np.array(net_attr_dict["targets"])
    preds_vel = np.array(net_attr_dict["preds_vel"])
    targets_vel =  np.array(net_attr_dict["targets_vel"])
    
    # get RMSE
    rmse = np.sqrt(np.mean(np.linalg.norm(pos_pred - pos_gt, axis=1) ** 2))
    if body_frame:
        rmse_vel = np.sqrt(np.mean(np.linalg.norm(preds_vel - targets_vel, axis=1) ** 2))
        rmse_vel = rmse_vel.astype(np.float64)
    else:
        rmse_vel = np.sqrt(np.mean(np.linalg.norm(preds - targets, axis=1) ** 2))
        rmse_vel = rmse_vel.astype(np.float64)
        
    # get ATE
    diff_pos = pos_pred - pos_gt
    ate = np.mean(np.linalg.norm(diff_pos, axis=1))
    # get RMHE (yaw)
    diff_eul = wrap_rpy(eul_pred - eul_gt)
    rmhe = np.sqrt(np.mean(diff_eul[:, 2] ** 2))
    # get position drift
    traj_lens = np.sum(np.linalg.norm(pos_gt[1:] - pos_gt[:-1], axis=1))
    drift_pos = np.linalg.norm(pos_pred[-1, :] - pos_gt[-1, :])
    drift_ratio = drift_pos / traj_lens
    # get yaw drift
    duration = ts[-1] - ts[0]
    drift_ang = np.linalg.norm(
        diff_eul[-1, 2] - diff_eul[0, 2]
    )  # beginning not aligned
    drift_ang_ratio = drift_ang / duration
    # get RPE on position and yaw
    ns_rpe = int(args.rpe_window * args.imu_freq)
    rpe_rmse, rpe_rmse_z, relative_yaw_rmse, rpes = compute_rpe(
        ns_rpe, pos_pred, pos_gt, eul_pred[:, [2]], eul_gt[:, [2]]
    )
    
    # compute ATE relative to trajectory length (avoid division by zero)
    traj_length_m = float(traj_lens) if traj_lens > 0 else float("nan")
    ate_rel = float(ate / traj_length_m) if traj_lens > 0 else float("nan")
    ate_pct = float(100.0 * ate / traj_length_m) if traj_lens > 0 else float("nan")

    metrics = {
        "ronin": {
            "rmse": rmse,
            "ate": ate,
            "ate_rel": ate_rel,          # unitless fraction (ATE / trajectory_length)
            "ate_pct": ate_pct,          # percentage (100 * ATE / trajectory_length)
            "traj_length_m": traj_length_m,  # total ground-truth path length (meters, after slicing)
            "rmhe": rmhe,
            "drift_pos (m/m)": drift_ratio,
            "drift_yaw (deg/s)": drift_ang_ratio,
            "rpe": rpe_rmse,
            "rpe_z": rpe_rmse_z,
            "rpe_yaw": relative_yaw_rmse,
            "rmse_vel": rmse_vel,
        }
    }

    """ ------------ Network loss metrics ----------- """
    mse_loss = np.mean(
        (net_attr_dict["targets"] - net_attr_dict["preds"]) ** 2, axis=0
    )  # 3x1
    likelihood_loss = np.mean(net_attr_dict["losses"], axis=0)  # 3x1
    avg_mse_loss = np.mean(mse_loss)
    avg_likelihood_loss = np.mean(likelihood_loss)
    metrics["ronin"]["mse_loss_x"] = float(mse_loss[0])
    metrics["ronin"]["mse_loss_y"] = float(mse_loss[1])
    metrics["ronin"]["mse_loss_z"] = float(mse_loss[2])
    metrics["ronin"]["mse_loss_avg"] = float(avg_mse_loss)
    metrics["ronin"]["likelihood_loss_x"] = float(likelihood_loss[0])
    metrics["ronin"]["likelihood_loss_y"] = float(likelihood_loss[1])
    metrics["ronin"]["likelihood_loss_z"] = float(likelihood_loss[2])
    # metrics["ronin"]["likelihood_loss_x"] = float(likelihood_loss)
    # metrics["ronin"]["likelihood_loss_y"] = float(likelihood_loss)
    # metrics["ronin"]["likelihood_loss_z"] = float(likelihood_loss)
    metrics["ronin"]["likelihood_loss_avg"] = float(avg_likelihood_loss)

    """ ------------ Data for plotting ----------- """
    total_pred = net_attr_dict["preds"].shape[0]
    pred_ts = (1.0 / args.sample_freq) * np.arange(total_pred)
    plot_dict = {
        "ts": ts,
        "pos_pred": pos_pred,
        "pos_gt": pos_gt,
        "vel_body_gt": vel_body_gt,
        "pred_ts": pred_ts,
        "preds": net_attr_dict["preds"],
        "targets": net_attr_dict["targets"],
        "rmse": rmse,
        "rpe_rmse": rpe_rmse,
        "rpes": rpes,
        "preds_vel": net_attr_dict["preds_vel"],
        "targets_vel": net_attr_dict["targets_vel"],
    }

    return metrics, plot_dict


def plot_3d_2var(x, y1, y2, xlb, ylbs, lgs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1[:, i], label=lgs[0])
        plt.plot(x, y2[:, i], label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var(x, y, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        if x is not None:
            plt.plot(x, y[:, i])
        else:
            plt.plot(y[:, i])
        plt.ylabel(ylbs[i])
        plt.grid(True)
    if xlb is not None:
        plt.xlabel(xlb)
    return fig


def plot_3d_2var_with_sigma(
    x, y1, y2, sig, xlb, ylbs, lgs, num=None, dpi=None, figsize=None
):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    y1_plus_sig = y1 + 3 * sig
    y1_minus_sig = y1 - 3 * sig
    for i in range(3):
        plt.subplot(3, 1, i + 1)
        plt.plot(x, y1_plus_sig[:, i], "-g", linewidth=0.2)
        plt.plot(x, y1_minus_sig[:, i], "-g", linewidth=0.2)
        plt.fill_between(
            x, y1_plus_sig[:, i], y1_minus_sig[:, i], facecolor="green", alpha=0.5
        )
        plt.plot(x, y1[:, i], "-b", linewidth=0.5, label=lgs[0])
        plt.plot(x, y2[:, i], "-r", linewidth=0.5, label=lgs[1])
        plt.ylabel(ylbs[i])
        plt.legend()
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def plot_3d_1var_with_sigma(x, y, sig, xlb, ylbs, num=None, dpi=None, figsize=None):
    fig = plt.figure(num=num, dpi=dpi, figsize=figsize)
    plus_sig = 3 * sig
    minus_sig = -3 * sig
    for i in range(3):
        # plt.subplot(3, 1, i + 1)
        # plt.plot(x, plus_sig[:, i], "-g", linewidth=0.2)
        # plt.plot(x, minus_sig[:, i], "-g", linewidth=0.2)
        # plt.fill_between(
        #     x, plus_sig[:, i], minus_sig[:, i], facecolor="green", alpha=0.5
        # )
        plt.plot(x, y[:, i], "-b", linewidth=0.5)
        plt.ylabel(ylbs[i])
        plt.grid(True)
    plt.xlabel(xlb)
    return fig


def make_plots(args, plot_dict, outdir, use_pred_vel=False):
    # This function is simplified as `use_pred_vel` is no longer relevant.
    ts = plot_dict["ts"]
    pos_pred = plot_dict["pos_pred"]
    pos_gt = plot_dict["pos_gt"]
    pred_ts = plot_dict["pred_ts"]
    preds = plot_dict["preds"] # This is now absolute position
    targets = plot_dict["targets"] # This is now absolute position GT
    rmse = plot_dict["rmse"]
    rpe_rmse = plot_dict["rpe_rmse"]
    rpes = plot_dict["rpes"]

    dpi = 90
    figsize = (16, 9)

    # Main trajectory plot
    fig1 = plt.figure(num="prediction vs gt", dpi=dpi, figsize=figsize)
    plt.subplot2grid((3, 2), (0, 0), rowspan=2)
    plt.plot(pos_pred[:, 0], pos_pred[:, 1], 'r-', label='Estimated Trajectory')
    plt.plot(pos_gt[:, 0], pos_gt[:, 1], 'b--', label='Ground Truth')
    plt.axis("equal")
    plt.legend()
    plt.title("2D Trajectory Comparison")

    plt.subplot2grid((3, 2), (2, 0))
    plt.plot(ts, np.linalg.norm(pos_pred - pos_gt, axis=1))
    plt.ylabel("Absolute Position Error [m]")
    plt.xlabel("Time [s]")
    plt.legend(["ATE:{:.3f}".format(rmse)])
    plt.grid(True)
    
    # Position components over time
    for i, axis in enumerate(['X', 'Y', 'Z']):
        plt.subplot2grid((3, 2), (i, 1))
        plt.plot(ts, pos_pred[:, i], 'r-', label='Predicted')
        plt.plot(ts, pos_gt[:, i], 'b--', label='Ground Truth')
        plt.legend()
        plt.title(f"Position ({axis}-axis)")
        plt.ylabel("Position [m]")
        plt.grid(True)
    
    plt.tight_layout()
    fig1.savefig(osp.join(outdir, "trajectory_comparison.png"))

    plt.close("all")


def torch_to_numpy(torch_arr):
    return torch_arr.cpu().detach().numpy()


def get_inference(network, data_loader, device, epoch, body_frame_3regress, body_frame, transforms=[]):
    """
    Obtain attributes from a data loader given a network state
    Outputs all targets, predicts, predicted covariance params, and losses in numpy arrays
    Enumerates the whole data loader
    """
    targets_all, preds_all, preds_cov_all, losses_all = [], [], [], []
    targets_vel_all, preds_vel_all = [], []
    network.eval()

    for bid, sample in enumerate(data_loader):
        sample = to_device(sample, device)
        
        for transform in transforms:
            sample = transform(sample)
            
            
        # feat = sample["feats"]["velocity"] # 'velocity' 키로 피처에 접근
        feat = sample["feats"]["vel_and_cov"] # 'velocity' 키로 피처에 접근
        # feat = sample["feats"]["vel_and_acc"] # 'velocity' 키로 피처에 접근
        # feat = sample["feats"]["vel_acc_cov"] # 'velocity' 키로 피처에 접근
        
        # #if random SO(3) rotate in test stage
        # rotation_matrix = np.array([[0.1097, 0.1448, 0.9834],[0.8754, -0.4827, -0.0266],[0.4708, 0.8637, -0.1797]])
        # rotation_matrix = torch.from_numpy(rotation_matrix).to('cuda').to(torch.float32)
        # accelerometer_data = feat[:, :3,:].to(torch.float32)
        # accelerometer_data = accelerometer_data.permute(1,0,2).reshape(3,-1)
        # gyroscope_data = feat[:, 3:, :].to(torch.float32)
        # gyroscope_data = gyroscope_data.permute(1,0,2).reshape(3,-1)

        # rotated_accelerometer_data = torch.matmul(rotation_matrix, accelerometer_data)
        # rotated_accelerometer_data = rotated_accelerometer_data.reshape(rotated_accelerometer_data.size(0), feat.size(0), feat.size(2))
        # rotated_accelerometer_data = rotated_accelerometer_data.permute(1,0,2)
        # rotated_gyroscope_data = torch.matmul(rotation_matrix, gyroscope_data)
        # rotated_gyroscope_data = rotated_gyroscope_data.reshape(rotated_gyroscope_data.size(0), feat.size(0), feat.size(2))
        # rotated_gyroscope_data = rotated_gyroscope_data.permute(1,0,2)
        
        if body_frame_3regress : 
            pred, pred_cov, pred_vel = network(feat)
            if not body_frame :
                targ_vel = sample["vel_World"][:,-1,:]
                # print("here")
            else : 
                targ_vel = sample["vel_Body"][:,-1,:]
        else:
            pred = network(feat)
            # pred = pred * 10
            pred_vel = pred.clone()
            if body_frame : 
                
                # targ_vel =torch.zeros_like(pred_vel)
                targ_vel = sample["vel_Body"][:,-1,:]
                targ = sample["vel_Body"][:,-1,:]
                # targ_vel = sample["targ_dt_Body"][:,-1,:]
                # targ = sample["targ_dt_Body"][:,-1,:]
                
                # #using gt
                # targ = sample["targ_dt_World"][:,-1,:]
                targ = sample["targ_pos_World"][:,-1,:]
            else:
                targ = sample["targ_pos_World"][:,-1,:]  # 수정된 코드
                targ_vel= targ.clone()
                
        # Only grab the last prediction in this case
        if len(pred.shape) == 3:
            pred = pred[:,:,-1]
            pred_cov = pred_cov[:,:,-1]
        
        assert len(pred.shape) == 2
        if body_frame_3regress: 
            loss = get_loss(pred_vel, pred_cov, targ_vel, epoch, body_frame_3regress)
        else:
            loss = get_loss(pred, targ, epoch, False)

        targets_all.append(torch_to_numpy(targ))
        preds_all.append(torch_to_numpy(pred))
        losses_all.append(torch_to_numpy(loss))
        
        targets_vel_all.append(torch_to_numpy(targ_vel))
        preds_vel_all.append(torch_to_numpy(pred_vel))
    
    targets_all = np.concatenate(targets_all, axis=0)
    preds_all = np.concatenate(preds_all, axis=0)
    # print("preds_cov_all : ", len(preds_cov_all))
    losses_all = np.concatenate(losses_all, axis=0)
    targets_vel_all = np.concatenate(targets_vel_all, axis=0)
    preds_vel_all = np.concatenate(preds_vel_all, axis=0)
    # print("preds_cov_all, np: ", preds_cov_all.shape)
    # print("preds_vel_all, np: ", preds_vel_all.shape)
    # assert False
    attr_dict = {
        "targets": targets_all,
        "preds": preds_all,
        "losses": losses_all,
        "targets_vel": targets_vel_all,
        "preds_vel": preds_vel_all,
    }
    return attr_dict


def get_datalist(list_path):
    with open(list_path) as f:
        data_list = [s.strip() for s in f.readlines() if len(s.strip()) > 0]
    return data_list


def arg_conversion(args):
    """ Conversions from time arguments to data size """

    if not (args.past_time * args.imu_freq).is_integer():
        raise ValueError(
            "past_time cannot be represented by integer number of IMU data."
        )
    if not (args.window_time * args.imu_freq).is_integer():
        raise ValueError(
            "window_time cannot be represented by integer number of IMU data."
        )
    if not (args.future_time * args.imu_freq).is_integer():
        raise ValueError(
            "future_time cannot be represented by integer number of IMU data."
        )
    if not (args.imu_freq / args.sample_freq).is_integer():
        raise ValueError("sample_freq must be divisible by imu_freq.")

    data_window_config = dotdict()
    data_window_config.past_data_size = int(args.past_time * args.imu_freq)
    data_window_config.window_size = int(args.window_time * args.imu_freq)
    data_window_config.future_data_size = int(args.future_time * args.imu_freq)
    data_window_config.step_size = int(args.imu_freq / args.sample_freq)
    data_window_config.data_style = "resampled"
    data_window_config.input_sensors = ["imu0"]
    data_window_config.decimator = 10
    data_window_config.express_in_t0_yaw_normalized_frame = False

    net_config = {
        "in_dim": (
            data_window_config["past_data_size"]
            + data_window_config["window_size"]
            + data_window_config["future_data_size"]
        )
        // 32
        + 1
    }

    # Display
    np.set_printoptions(formatter={"all": "{:.6f}".format})
    logging.info(f"Training/testing with {args.imu_freq} Hz IMU data")
    logging.info(
        "Size: "
        + str(data_window_config["past_data_size"])
        + "+"
        + str(data_window_config["window_size"])
        + "+"
        + str(data_window_config["future_data_size"])
        + ", "
        + "Time: "
        + str(args.past_time)
        + "+"
        + str(args.window_time)
        + "+"
        + str(args.future_time)
    )
    logging.info("Perturb on bias: %s" % args.do_bias_shift)
    logging.info("Perturb on gravity: %s" % args.perturb_gravity)
    logging.info("Sample frequency: %s" % args.sample_freq)
    return data_window_config, net_config


def net_test(args):
    """
    Main function for network testing
    Generate trajectories, plots, and metrics.json file
    """

    try:
        if args.root_dir is None:
            raise ValueError("root_dir must be specified.")
        if args.out_dir is not None:
            if not osp.isdir(args.out_dir):
                os.makedirs(args.out_dir)
            logging.info(f"Testing output writes to {args.out_dir}")
        else:
            raise ValueError("out_dir must be specified.")
        data_window_config, net_config = arg_conversion(args)
    except ValueError as e:
        logging.error(e)
        return

    test_list_path = osp.join(args.root_dir, "test_list.txt")
    test_list = get_datalist(test_list_path)

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    checkpoint = torch.load(args.model_path, map_location=device)
    network = get_model(args.arch, net_config, args.input_dim, args.output_dim).to(
        device
    )
    network.load_state_dict(checkpoint["model_state_dict"])
    network.eval()
    logging.info(f"Model {args.model_path} loaded to device {device}.")

    # initialize containers
    all_metrics = {}
    consumed_times = []
    consumed_gpu = []

    for data in test_list:
        logging.info(f"Processing {data}...")
        try:
            #seq_dataset = FbSequenceDataset(
            #    args.root_dir, [data], args, data_window_config, mode="test"
            #)

            seq_dataset = MemMappedSequencesDataset(
                args.root_dir,
                "test",
                data_window_config,
                sequence_subset=[data],
                store_in_ram=True,
            )

            seq_loader = DataLoader(seq_dataset, batch_size=1024, shuffle=False)
        except OSError as e:
            print(e)
            continue

        # use_pred_vel = True
        # body_frame_3regress = True
        # body_frame = True
        
        # use_pred_vel = False
        # body_frame_3regress = False
        # body_frame = False
        
        if "3res" in args.model_path:
        # if "res_3res" in args.model_path:
            print("we regress 2 value!!")
            body_frame_3regress = False
        elif "/res" in args.out_dir or "resnet" in args.out_dir or "/eq_" in args.out_dir or "/vn_" in args.out_dir or "/ln_" in args.out_dir:
            body_frame_3regress = False
        else:
            body_frame_3regress = True
        # body_frame_3regress = False
        body_frame = eval(args.body_frame)
        if "disp" not in args.out_dir:
            use_pred_vel = True
        else:
            use_pred_vel = False
            
        # Obtain trajectory
        start_t = time.time()
        
        test_transforms = []
        # ##UNCOMMENT IF WANT TO ADD BIAS IN TEST DATASET
        # test_transforms = seq_dataset.get_test_transforms_bodyframe() 
        # # assert False
        # ##UNCOMMENT IF WANT TO ADD BIAS IN TEST DATASET
        
        net_attr_dict = get_inference(network, seq_loader, device, epoch=50, body_frame_3regress = body_frame_3regress, body_frame = body_frame, transforms = test_transforms)
        end_t = time.time()
        mem_used_max_GB = torch.cuda.max_memory_allocated() / (1024*1024*1024)
        torch.cuda.reset_peak_memory_stats()
        mem_str = f'GPU Mem: {mem_used_max_GB:.3f}GB'
        logging.info(mem_str)
        consumed_gpu.append(mem_used_max_GB)    
        
        logging.info(f"inference time usage: {end_t - start_t:.3f}s")
        consumed_times.append(end_t - start_t)    
        time_mem_log = osp.join(args.out_dir, 'time_mem_log.txt')
        with open(time_mem_log, 'a') as log_file:
            log_file.write(f"Inference time : {end_t - start_t:.4f} seconds, {mem_used_max_GB:.3f}GB\n")
        
        if use_pred_vel:
            traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds_vel"], use_pred_vel, body_frame)
        else:
            traj_attr_dict = pose_integrate(args, seq_dataset, net_attr_dict["preds"], use_pred_vel, body_frame)
            
        outdir = osp.join(args.out_dir, data)
        if osp.exists(outdir) is False:
            os.mkdir(outdir)
        outfile = osp.join(outdir, "trajectory.txt")
        trajectory_data = np.concatenate(
            [
                traj_attr_dict["ts"].reshape(-1, 1),
                traj_attr_dict["pos_pred"],
                traj_attr_dict["pos_gt"],
            ],
            axis=1,
        )
        np.savetxt(outfile, trajectory_data, delimiter=",")

        # obtain metrics
        metrics, plot_dict = compute_metrics_and_plotting(
            args, net_attr_dict, traj_attr_dict, body_frame
        )
        
        logging.info(metrics)
        all_metrics[data] = metrics

        outfile_net = osp.join(outdir, "net_outputs.txt")
        if use_pred_vel:
            net_outputs_data = np.concatenate(
                [
                    plot_dict["pred_ts"].reshape(-1, 1),
                    plot_dict["preds_vel"],
                    plot_dict["targets_vel"],
                ],
                axis=1,
            )
        else:
            net_outputs_data = np.concatenate(
                [
                    plot_dict["pred_ts"].reshape(-1, 1),
                    plot_dict["preds"],
                    plot_dict["targets"],
                ],
                axis=1,
            )
        np.savetxt(outfile_net, net_outputs_data, delimiter=",")

        if args.save_plot:
            make_plots(args, plot_dict, outdir, use_pred_vel)

        try:
            with open(args.out_dir + "/metrics.json", "w") as f:
                json.dump(all_metrics, f, indent=1)
        except ValueError as e:
            raise e
        except OSError as e:
            print(e)
            continue
        except Exception as e:
            raise e

    mean_epoch_time = np.mean(consumed_times)
    mean_epoch_gpu = np.mean(consumed_gpu)
    
    with open(time_mem_log, 'a') as log_file:
        log_file.write(f"Mean Inference time: {mean_epoch_time:.4f} seconds, {mean_epoch_gpu:.3f}GB\n")
    logging.info(f"Mean inference time usage: {mean_epoch_time:.3f}s")
    mem_str = f'GPU Mem: {mean_epoch_gpu:.3f}GB'
    logging.info(mem_str)
    
    return
