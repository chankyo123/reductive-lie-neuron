import os
import json
import numpy as np
import matplotlib.pyplot as plt
from os import path as osp
from mpl_toolkits.mplot3d import Axes3D

TOP_N_HIGHLIGHT = 2            # number of best models to highlight (besides GT)
FAILED_ALPHA = 0.25            # transparency for poor models
FAILED_LW = 0.9                # linewidth for poor models
HIGHLIGHT_LW = 2.8             # linewidth for highlighted models
GT_LW = 2.5                    # linewidth for ground truth
MARKER_SIZE = 6
ZOOM_FRACTION = 0.10           # zoom on last 10% of trajectory
PALETTE = plt.get_cmap("tab10")  # color palette for models

GOOD_MODELS = [
    "ln_2regress",
    "ln_2regress_w_cov_correct_lr1e5",
]
# If you prefer automatic detection, set GOOD_MODELS = None and the function will pick top-N by RMSE.
AUTO_PICK_TOP_N = None  # e.g. 2 or None to use GOOD_MODELS list

# visual params
GT_LW = 2.6
GOOD_LW = 2.2
BAD_LW = 1.0
BAD_STYLE = (0, (5, 3))   # dashed
PALETTE = plt.get_cmap("tab10")  # distinct colors for good models

def plot_sequence_clean(sequence_name, traj_entries, out_dir, show=False):
    """
    ICLR-ready, serif-font version of plot_sequence_clean.
    Saves high-quality PNG + SVG and returns (png_path, svg_path).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from os import path as osp

    # ===== Publication-ready rc settings (serif / Times) =====
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        # "font.size": 11,
        # "axes.titlesize": 11,
        # "axes.labelsize": 11,
        # "legend.fontsize": 9,
        # "xtick.labelsize": 10,
        # "ytick.labelsize": 10,
        "font.size": 14,          # ↑ from 11
        "axes.titlesize": 16,     # ↑ from 11
        "axes.labelsize": 15,     # ↑ from 11
        "legend.fontsize": 13,    # ↑ from 9
        "xtick.labelsize": 13,    # ↑ from 10
        "ytick.labelsize": 13,    # ↑ from 10
        "axes.titleweight": "semibold",
        # vector font embedding good defaults
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    # Visual parameters tuned for paper
    FIGSIZE = (6, 6)
    DPI = 300
    GT_COLOR = "k"
    GT_LW = 1.7
    PRED_LW = 1.1
    # GT_LW = 3.0
    # PRED_LW = 1.8
    MARKER_START = "o"
    MARKER_END = "*"
    MARKER_SIZE = 36
    GRID_STYLE = {"linestyle": ":", "linewidth": 0.6, "alpha": 0.9}
    ZOOM_FRAC = 0.12  # for inset bbox padding

    # optional globals (fallbacks)
    GOOD_MODELS = globals().get("GOOD_MODELS", ["ln_2regress", "ln_2regress_w_cov_correct_lr1e5"])
    PALETTE = globals().get("PALETTE", plt.get_cmap("tab10"))

    os.makedirs(out_dir, exist_ok=True)

    # Ensure rmse exists
    for e in traj_entries:
        if e.get("rmse") is None:
            pred = e["traj"]["pos_pred"]
            gt = e["traj"]["pos_gt"]
            e["rmse"] = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))

    if len(traj_entries) == 0:
        raise ValueError("No trajectories provided for plotting.")

    # --- choose two models for inset (prefer GOOD_MODELS else top-2 by RMSE) ---
    models_present = [e["model"] for e in traj_entries]
    good_candidates = [m for m in GOOD_MODELS if m in models_present]
    if len(good_candidates) >= 2:
        inset_models = good_candidates[:2]
    else:
        sorted_by_rmse = sorted(traj_entries, key=lambda x: x["rmse"])
        inset_models = [e["model"] for e in sorted_by_rmse[:2]]

    entry_map = {e["model"]: e for e in traj_entries}

    # Use local rc context so changes are temporary
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(**GRID_STYLE)

        # Plot GT first (bold black)
        pos_gt = traj_entries[0]["traj"]["pos_gt"]
        ts = traj_entries[0]["traj"]["ts"]
        mask = ts <= 200.0
        pos_gt = pos_gt[mask]
        ts = ts[mask]
        
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], color=GT_COLOR, linewidth=GT_LW, label="GT", zorder=40)
        ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker=MARKER_START, s=MARKER_SIZE, color=GT_COLOR, zorder=41)
        ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 1.2), color=GT_COLOR, zorder=41)
        label_map = {
            "ln_2regress_w_cov_correct_lr1e5": "ReLN (ours) — vel + cov",
            "ln_2regress": "ReLN (ours) — vel",
            "vn_2regress_w_cov": "VN — vel + cov",
            "resnet_w_cov": "ResNet — vel + cov",
            "resnet": "ResNet — vel",
        }
        # Plot all predictions (default color cycle), straight solid lines
        for model_key in label_map.keys():
            if model_key not in entry_map:
                continue  # skip if this model is not present
            entry = entry_map[model_key]
            pos_pred = entry["traj"]["pos_pred"]
            ts = entry["traj"]["ts"]
            mask = ts <= 200.0
            pos_pred = pos_pred[mask]
            ts = ts[mask]
            
            rmse = entry.get("rmse")
            label = f"{label_map[model_key]} "

            # plot line and start/end markers
            line = ax.plot(pos_pred[:, 0], pos_pred[:, 1], linewidth=PRED_LW, label=label, zorder=30)[0]
            lc = line.get_color()
            ax.scatter(pos_pred[0, 0], pos_pred[0, 1], marker=MARKER_START, s=int(MARKER_SIZE * 0.45), color=lc, zorder=31)
            ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 0.7), color=lc, zorder=31)
            

        # Labels & title (serif font now)
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_position([0.12, 0.12, 0.6, 0.6])
        # ax.set_title(f"{sequence_name}: Estimated vs Ground Truth (XY)")

        # Legend to right (compact, frameless)
        handles, labels = ax.get_legend_handles_labels()
        if len(handles) > 0:
            ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

        # Nice scale bar (rounded)
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            scalelen = (xlim[1] - xlim[0]) * 0.12
            def nice_round(x):
                if x <= 0:
                    return 0.0
                exp = np.floor(np.log10(x))
                base = x / (10 ** exp)
                if base < 1.5:
                    nice = 1.0
                elif base < 3.5:
                    nice = 2.0
                elif base < 7.5:
                    nice = 5.0
                else:
                    nice = 10.0
                return nice * (10 ** exp)
            scalelen_nice = nice_round(scalelen)
            sb_x = xlim[0] + 0.06 * (xlim[1] - xlim[0])
            sb_y = ylim[0] + 0.06 * (ylim[1] - ylim[0])
            # ax.hlines(sb_y, sb_x, sb_x + scalelen_nice, colors="k", linewidth=2, zorder=200)
            # ax.text(sb_x + scalelen_nice / 2.0, sb_y - 0.035 * (ylim[1] - ylim[0]),
            #         f"{scalelen_nice:.2f} m", ha="center", va="top", fontsize=9)
        except Exception:
            pass

        # ----- bottom-right inset: show full GT + two selected models (start/end indicated) -----
        try:
            points = [pos_gt]
            for m in inset_models:
                if m in entry_map:
                    points.append(entry_map[m]["traj"]["pos_gt"])
                    points.append(entry_map[m]["traj"]["pos_pred"])
            all_pts = np.vstack(points)
            x_min, y_min = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
            pad_x = max(1e-6, 0.12 * (x_max - x_min + 1e-6))
            pad_y = max(1e-6, 0.12 * (y_max - y_min + 1e-6))

            # position inset slightly left to make room for legend (adjustable)
            x_range = (x_max + pad_x) - (x_min - pad_x)
            y_range = (y_max + pad_y) - (y_min - pad_y)

            # Define a desired height for your inset
            inset_height = 0.42 # In figure coordinates

            # Calculate the width needed to maintain the data's aspect ratio
            # This is the key step
            data_aspect_ratio = x_range / y_range
            inset_width = inset_height * data_aspect_ratio

            # Now, create the axes with the correct dimensions from the start
            inset_ax = fig.add_axes([0.52, 0.19, inset_width, inset_height])

            # Set your limits and aspect ratio as before
            inset_ax.set_xlim(x_min - pad_x, x_max + pad_x)
            inset_ax.set_ylim(y_min - pad_y, y_max + pad_y)

            # This will now ensure the axes box perfectly fits the data with an equal aspect
            inset_ax.set_aspect("equal", "box") 
            
            # GT full
            inset_ax.plot(pos_gt[:, 0], pos_gt[:, 1], color=GT_COLOR, linewidth=GT_LW, label="GT (full)")
            inset_ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker=MARKER_START, s=int(MARKER_SIZE * 0.6), color=GT_COLOR)
            inset_ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 0.9), color=GT_COLOR)

            # two selected models full curves, using same main colors
            main_colors = {}
            main_colors = {k: PALETTE.colors[i % len(PALETTE.colors)] 
               for i, k in enumerate(label_map.keys())}

            # Plot inset trajectories
            for model_key in label_map.keys():
                if model_key not in entry_map or model_key not in inset_models:
                    continue
                ent = entry_map[model_key]
                pred = ent["traj"]["pos_pred"]
                ts = ent["traj"]["ts"]
                mask = ts <= 200.0
                pred = pred[mask]

                color = main_colors[model_key]

                inset_ax.plot(
                    pred[:, 0], pred[:, 1], linewidth=PRED_LW,
                    color=color, label=f"{label_map[model_key]} (full)"
                )
                inset_ax.scatter(pred[0, 0], pred[0, 1], marker=MARKER_START,
                                s=int(MARKER_SIZE * 0.6), color=color)
                inset_ax.scatter(pred[-1, 0], pred[-1, 1], marker=MARKER_END,
                                s=int(MARKER_SIZE * 0.9), color=color)

            # inset_ax.set_xticks([])
            # inset_ax.set_yticks([])
            # compute nice ticks for inset
            x_ticks = np.linspace(np.floor(x_min), np.ceil(x_max), 5)  # 5 nice steps
            y_ticks = np.linspace(np.floor(y_min), np.ceil(y_max), 5)

            inset_ax.set_xticks(x_ticks)
            inset_ax.set_yticks(y_ticks)
            inset_ax.set_xticklabels([f"{t:.0f}" for t in x_ticks], fontsize=8)
            inset_ax.set_yticklabels([f"{t:.0f}" for t in y_ticks], fontsize=8)
            # inset_ax.set_xticklabels([])  # hide numbers
            # inset_ax.set_yticklabels([])  # hide numbers
            inset_ax.set_title("Inset: GT + Top-2 Predictions", fontsize=9)
            inset_ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            # inset_ax.grid(
            #     b=ax.xaxis._gridOnMajor,  # match whether main grid is on
            #     linestyle=ax.xaxis.get_gridlines()[0].get_linestyle(),
            #     linewidth=ax.xaxis.get_gridlines()[0].get_linewidth(),
            #     alpha=ax.xaxis.get_gridlines()[0].get_alpha()
            # )
            # inset_ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)


            # mark maximum deviation among inset models (red 'x')
            max_err_val = -1.0
            max_pt = None
            for m in inset_models:
                if m not in entry_map:
                    continue
                pred = entry_map[m]["traj"]["pos_pred"]
                gt_local = entry_map[m]["traj"]["pos_gt"]
                errs = np.linalg.norm(pred - gt_local, axis=1)
                idx = int(np.argmax(errs))
                if errs[idx] > max_err_val:
                    max_err_val = float(errs[idx])
                    max_pt = (pred[idx, 0], pred[idx, 1])
            # if max_pt is not None:
            #     inset_ax.scatter(max_pt[0], max_pt[1], marker="x", s=56, color="red", zorder=250)
        except Exception:
            pass

        # finalize & save high-quality outputs
        fig.tight_layout(rect=[0, 0, 0.78, 1.0])
        clean_dir = osp.join(out_dir, "clean2")
        os.makedirs(clean_dir, exist_ok=True)
        png_path = osp.join(clean_dir, f"{sequence_name}_compare_xy_clean.png")
        svg_path = osp.join(clean_dir, f"{sequence_name}_compare_xy_clean.svg")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        fig.savefig(svg_path, dpi=DPI, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return png_path, svg_path

def plot_sequence_clean2(sequence_name, traj_entries, out_dir, show=False):
    """
    ICLR-ready, serif-font version of plot_sequence_clean.
    Saves high-quality PNG + SVG and returns (png_path, svg_path).
    MODIFIED: Enforces consistent, 'nice' tick intervals on both main and inset axes.
    """
    # ===== Publication-ready rc settings (serif / Times) =====
    rc = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.titleweight": "semibold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }

    # Visual parameters tuned for paper
    FIGSIZE = (6, 6)
    DPI = 300
    GT_COLOR = "k"
    GT_LW = 1.7
    PRED_LW = 1.1
    MARKER_START = "o"
    MARKER_END = "*"
    MARKER_SIZE = 36
    GRID_STYLE = {"linestyle": ":", "linewidth": 0.6, "alpha": 0.9}
    MAIN_TICK_INTERVAL = 50.0
    INSET_TICK_INTERVAL = 30.0
    INSET_PADDING_FRACTION = 0.02

    # optional globals (fallbacks)
    GOOD_MODELS = globals().get("GOOD_MODELS", ["ln_2regress", "ln_2regress_w_cov_correct_lr1e5"])
    PALETTE = globals().get("PALETTE", plt.get_cmap("tab10"))

    os.makedirs(out_dir, exist_ok=True)

    # Ensure rmse exists
    for e in traj_entries:
        if e.get("rmse") is None:
            pred = e["traj"]["pos_pred"]
            gt = e["traj"]["pos_gt"]
            e["rmse"] = float(np.sqrt(np.mean(np.sum((pred - gt) ** 2, axis=1))))

    if len(traj_entries) == 0:
        raise ValueError("No trajectories provided for plotting.")

    # --- choose two models for inset ---
    models_present = [e["model"] for e in traj_entries]
    good_candidates = [m for m in GOOD_MODELS if m in models_present]
    if len(good_candidates) >= 2:
        inset_models = good_candidates[:2]
    else:
        sorted_by_rmse = sorted(traj_entries, key=lambda x: x["rmse"])
        inset_models = [e["model"] for e in sorted_by_rmse[:2]]

    entry_map = {e["model"]: e for e in traj_entries}

    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.set_facecolor("white")
        ax.set_axisbelow(True)
        ax.grid(**GRID_STYLE)

        # Plot GT
        pos_gt = traj_entries[0]["traj"]["pos_gt"]
        ts = traj_entries[0]["traj"]["ts"]
        mask = ts <= 200.0
        pos_gt = pos_gt[mask]
        
        ax.plot(pos_gt[:, 0], pos_gt[:, 1], color=GT_COLOR, linewidth=GT_LW, label="GT", zorder=40)
        ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker=MARKER_START, s=MARKER_SIZE, color=GT_COLOR, zorder=41)
        ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 1.2), color=GT_COLOR, zorder=41)
        
        label_map = {
            "ln_2regress_w_cov_correct_lr1e5": "ReLN (ours) — velocity + covariance",
            "ln_2regress": "ReLN (ours) — velocity",
            "vn_2regress_w_cov": "VN — velocity + covariance",
            "resnet_w_cov": "ResNet — velocity + covariance",
            "resnet": "ResNet — velocity",
        }
        
        # Plot all predictions
        for model_key in label_map.keys():
            if model_key not in entry_map:
                continue
            entry = entry_map[model_key]
            pos_pred = entry["traj"]["pos_pred"]
            ts = entry["traj"]["ts"]
            mask = ts <= 200.0
            pos_pred = pos_pred[mask]
            
            line = ax.plot(pos_pred[:, 0], pos_pred[:, 1], linewidth=PRED_LW, label=label_map[model_key], zorder=30)[0]
            lc = line.get_color()
            ax.scatter(pos_pred[0, 0], pos_pred[0, 1], marker=MARKER_START, s=int(MARKER_SIZE * 0.45), color=lc, zorder=31)
            ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 0.7), color=lc, zorder=31)
            
        ax.set_aspect("equal", "box")
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")

        # Set main axis ticks to be 'nice' multiples
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        start_xtick = np.floor(xmin / MAIN_TICK_INTERVAL) * MAIN_TICK_INTERVAL
        end_xtick = np.ceil(xmax / MAIN_TICK_INTERVAL) * MAIN_TICK_INTERVAL
        ax.set_xticks(np.arange(start_xtick, end_xtick + 1, MAIN_TICK_INTERVAL))
        start_ytick = np.floor(ymin / MAIN_TICK_INTERVAL) * MAIN_TICK_INTERVAL
        end_ytick = np.ceil(ymax / MAIN_TICK_INTERVAL) * MAIN_TICK_INTERVAL
        ax.set_yticks(np.arange(start_ytick, end_ytick + 1, MAIN_TICK_INTERVAL))

        # ----- Inset Plot -----
        try:
            points_for_bbox = [pos_gt]
            for m in inset_models:
                if m in entry_map:
                    points_for_bbox.append(entry_map[m]["traj"]["pos_pred"])
            
            all_pts = np.vstack(points_for_bbox)
            x_min, y_min = np.min(all_pts[:, 0]), np.min(all_pts[:, 1])
            x_max, y_max = np.max(all_pts[:, 0]), np.max(all_pts[:, 1])
            
            # --- NEW TIGHTER INSET LOGIC ---
            x_range_raw = x_max - x_min + 1e-6
            y_range_raw = y_max - y_min + 1e-6

            pad_x = INSET_PADDING_FRACTION * x_range_raw
            pad_y = INSET_PADDING_FRACTION * y_range_raw

            final_x_min = x_min - pad_x
            final_x_max = x_max + pad_x
            final_y_min = y_min - pad_y
            final_y_max = y_max + pad_y

            data_aspect_ratio = (final_x_max - final_x_min) / (final_y_max - final_y_min)

            inset_base_size = 0.55
            if data_aspect_ratio > 1:
                inset_width = inset_base_size
                inset_height = inset_base_size / data_aspect_ratio
            else:
                inset_height = inset_base_size
                inset_width = inset_base_size * data_aspect_ratio

            inset_ax = fig.add_axes([0.81, 0.12, inset_width, inset_height])
            inset_ax.set_xlim(final_x_min, final_x_max)
            inset_ax.set_ylim(final_y_min, final_y_max)
            inset_ax.set_aspect("equal", "box") 
            
            # Plot trajectories in inset
            inset_ax.plot(pos_gt[:, 0], pos_gt[:, 1], color=GT_COLOR, linewidth=GT_LW)
            inset_ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker=MARKER_START, s=int(MARKER_SIZE * 0.6), color=GT_COLOR)
            inset_ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 0.9), color=GT_COLOR)

            main_colors = {k: PALETTE.colors[i % len(PALETTE.colors)] for i, k in enumerate(label_map.keys())}

            for model_key in inset_models:
                if model_key not in entry_map:
                    continue
                pred = entry_map[model_key]["traj"]["pos_pred"]
                ts = entry_map[model_key]["traj"]["ts"]
                mask = ts <= 200.0
                pred = pred[mask]
                color = main_colors[model_key]
                inset_ax.plot(pred[:, 0], pred[:, 1], linewidth=PRED_LW, color=color)
                inset_ax.scatter(pred[0, 0], pred[0, 1], marker=MARKER_START, s=int(MARKER_SIZE * 0.6), color=color)
                inset_ax.scatter(pred[-1, 0], pred[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE * 0.9), color=color)
            
            # Set inset axis ticks to be 'nice' multiples inside the tight view
            start_inset_xtick = np.ceil(final_x_min / INSET_TICK_INTERVAL) * INSET_TICK_INTERVAL
            end_inset_xtick = np.floor(final_x_max / INSET_TICK_INTERVAL) * INSET_TICK_INTERVAL
            if end_inset_xtick >= start_inset_xtick:
                inset_ax.set_xticks(np.arange(start_inset_xtick, end_inset_xtick + 1e-6, INSET_TICK_INTERVAL))

            start_inset_ytick = np.ceil(final_y_min / INSET_TICK_INTERVAL) * INSET_TICK_INTERVAL
            end_inset_ytick = np.floor(final_y_max / INSET_TICK_INTERVAL) * INSET_TICK_INTERVAL
            if end_inset_ytick >= start_inset_ytick:
                inset_ax.set_yticks(np.arange(start_inset_ytick, end_inset_ytick + 1e-6, INSET_TICK_INTERVAL))
            
            inset_ax.tick_params(axis='x', labelsize=10)
            inset_ax.tick_params(axis='y', labelsize=10)
            inset_ax.set_title("Inset: GT + Top-2 Predictions", fontsize=11)
            inset_ax.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
            
            mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="0.6", lw=1.2, zorder=50)

        except Exception as e:
            print(f"Could not generate inset plot for {sequence_name}: {e}")

        fig.tight_layout()
        clean_dir = osp.join(out_dir, "clean2")
        os.makedirs(clean_dir, exist_ok=True)
        png_path = osp.join(clean_dir, f"{sequence_name}_compare_xy_clean.png")
        svg_path = osp.join(clean_dir, f"{sequence_name}_compare_xy_clean.svg")
        fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
        fig.savefig(svg_path, dpi=DPI, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    return png_path, svg_path







def plot_sequence_comparison_highlight(sequence_name, traj_entries, out_dir, show=False):
    """
    traj_entries: list of dicts {"model": str, "traj_path": str, "traj": dict, "rmse": float}
    Saves: <out_dir>/<sequence_name>_compare_xy_highlight.png
    """
    os.makedirs(out_dir, exist_ok=True)

    # compute/ensure RMSE and sort by RMSE (ascending -> best first)
    for e in traj_entries:
        if e.get("rmse") is None:
            pos_pred = e["traj"]["pos_pred"]
            pos_gt = e["traj"]["pos_gt"]
            e["rmse"] = float(np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1))))
    traj_entries = sorted(traj_entries, key=lambda x: x["rmse"])

    # assign colors, highlight best TOP_N_HIGHLIGHT
    n_models = len(traj_entries)
    colors = [PALETTE(i % 10) for i in range(n_models)]

    fig, ax = plt.subplots(figsize=(9, 9))
    # first plot the failing/weak models (drawn beneath)
    for idx in range(n_models - 1, -1, -1):  # plot worst -> best so best drawn last
        e = traj_entries[idx]
        model = e["model"]
        pos_pred = e["traj"]["pos_pred"]
        pos_gt = e["traj"]["pos_gt"]
        rmse = e["rmse"]

        # determine style
        rank = idx  # since sorted asc, idx=0 best, idx=n_models-1 worst; but we loop reversed
        # compute rank_from_best
        rank_from_best = n_models - 1 - idx  # 0 => worst, n_models-1 => best (used for loop reversed)
        # better approach: compute index in sorted list:
        sorted_index = traj_entries.index(e)
        if sorted_index < TOP_N_HIGHLIGHT:
            lw = HIGHLIGHT_LW
            alpha = 1.0
            linestyle = "-"
            zord = 20 - sorted_index  # ensure best on top
            color = colors[sorted_index]
        else:
            lw = FAILED_LW
            alpha = FAILED_ALPHA
            linestyle = (0, (3, 5))  # dashed-dot pattern
            zord = 5
            color = colors[sorted_index]

        # plot predicted trajectory
        ax.plot(
            pos_pred[:, 0],
            pos_pred[:, 1],
            label=f"{model} (RMSE={rmse:.3f})",
            linewidth=lw,
            alpha=alpha,
            linestyle=linestyle,
            zorder=zord,
            color=color,
        )
        # start & end markers
        ax.scatter(pos_pred[0, 0], pos_pred[0, 1], marker="o", s=MARKER_SIZE**2 * 0.6, zorder=zord + 1, alpha=alpha, color=color)
        ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], marker="*", s=MARKER_SIZE**2, zorder=zord + 1, alpha=alpha, color=color)

    # plot GT on top (black)
    # pick pos_gt from first entry (they should be identical)
    pos_gt = traj_entries[0]["traj"]["pos_gt"]
    ax.plot(pos_gt[:, 0], pos_gt[:, 1], color="k", linewidth=GT_LW, linestyle="-", label="GT", zorder=999)
    ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker="o", s=MARKER_SIZE**2, color="k", zorder=1000)
    ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker="*", s=MARKER_SIZE**2 * 1.2, color="k", zorder=1000)

    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Sequence {sequence_name}: XY Trajectory Comparison (best highlighted)")
    ax.grid(True)

    # legend: place outside to avoid overlapping; smaller font
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize="small", frameon=True)

    # ANNOTATION: annotate best model RMSE on plot
    best = traj_entries[0]
    txt = f"Best model: {best['model']}\nRMSE = {best['rmse']:.3f} m"
    ax.annotate(txt, xy=(0.99, 0.01), xycoords="axes fraction", ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8), fontsize=9)

    # INSET ZOOM of last ZOOM_FRACTION of trajectory (show last 10% where drift often occurs)
    try:
        # compute index start
        N = pos_gt.shape[0]
        start_idx = max(0, int(N * (1.0 - ZOOM_FRACTION)))
        x_min = np.min(pos_gt[start_idx:, 0])
        x_max = np.max(pos_gt[start_idx:, 0])
        y_min = np.min(pos_gt[start_idx:, 1])
        y_max = np.max(pos_gt[start_idx:, 1])
        pad_x = (x_max - x_min) * 0.2 + 1e-6
        pad_y = (y_max - y_min) * 0.2 + 1e-6
        inset_ax = fig.add_axes([0.60, 0.60, 0.32, 0.32])  # x, y, width, height in figure coords
        inset_ax.set_xlim(x_min - pad_x, x_max + pad_x)
        inset_ax.set_ylim(y_min - pad_y, y_max + pad_y)
        # re-plot everything in inset but with simpler styles
        for i, e in enumerate(traj_entries):
            pos_pred = e["traj"]["pos_pred"]
            model = e["model"]
            rmse = e["rmse"]
            if i < TOP_N_HIGHLIGHT:
                lw = HIGHLIGHT_LW
                alpha = 1.0
                color = colors[i]
                linestyle = "-"
            else:
                lw = FAILED_LW
                alpha = FAILED_ALPHA
                color = colors[i]
                linestyle = (0, (3, 5))
            inset_ax.plot(pos_pred[:, 0], pos_pred[:, 1], linewidth=lw, alpha=alpha, linestyle=linestyle, color=color)
        inset_ax.plot(pos_gt[:, 0], pos_gt[:, 1], color="k", linewidth=1.8)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.set_title("Zoom (last 10%)", fontsize=9)
        inset_ax.grid(True, linestyle=":", linewidth=0.4)
    except Exception:
        pass  # if anything fails, ignore inset

    fig.tight_layout(rect=[0, 0, 0.78, 1.0])  # leave space for legend on right
    outpath = osp.join(out_dir, f"{sequence_name}_compare_xy_highlight.png")
    fig.savefig(outpath, dpi=300)
    if show:
        plt.show()
    plt.close(fig)
    return outpath

def find_model_dirs(root_models_dir, whitelist=None):
    subs = []
    for name in sorted(os.listdir(root_models_dir)):
        p = osp.join(root_models_dir, name)
        if osp.isdir(p):
            if whitelist is None or name in whitelist:
                subs.append(p)
    return subs

def load_metrics_if_exists(model_dir):
    metrics_path = osp.join(model_dir, "metrics.json")
    if osp.isfile(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def gather_trajectories_for_model(model_dir):
    seq_map = {}
    for entry in sorted(os.listdir(model_dir)):
        seq_dir = osp.join(model_dir, entry)
        if osp.isdir(seq_dir):
            traj_file = osp.join(seq_dir, "trajectory.txt")
            if osp.isfile(traj_file):
                seq_map[entry] = traj_file
    return seq_map

def read_traj_file(traj_path):
    try:
        data = np.loadtxt(traj_path, delimiter=",")
    except Exception:
        data = np.loadtxt(traj_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] >= 7:
        ts = data[:, 0]
        pos_pred = data[:, 1:4]
        pos_gt = data[:, 4:7]
    elif data.shape[1] == 6:
        ts = np.arange(data.shape[0])
        pos_pred = data[:, 0:3]
        pos_gt = data[:, 3:6]
    else:
        raise ValueError(f"Trajectory file {traj_path} has unexpected shape {data.shape}")
    return {"ts": ts, "pos_pred": pos_pred, "pos_gt": pos_gt}

def get_rmse_from_metrics(metrics_json, seq_name):
    if not metrics_json:
        return None
    try:
        v = metrics_json.get(seq_name)
        if v is None:
            return None
        if "ronin" in v and "rmse" in v["ronin"]:
            return float(v["ronin"]["rmse"])
        for k in v:
            if isinstance(v[k], dict) and "rmse" in v[k]:
                return float(v[k]["rmse"])
    except Exception:
        pass
    return None

def plot_sequence_comparison(sequence_name, traj_entries, out_dir, show=False):
    """
    Publication-quality XY comparison while preserving original colorization.
    Saves both PNG and SVG (300 DPI) under out_dir with filenames:
      <sequence_name>_compare_xy.png
      <sequence_name>_compare_xy.svg
    Returns tuple (png_path, svg_path).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from os import path as osp

    # Visual config (keeps default color cycle for model lines)
    FIGSIZE = (8, 8)
    DPI = 300
    GT_COLOR = "k"
    GT_LW = 2.2
    PRED_LW = 1.4
    MARKER_START = "o"
    MARKER_END = "*"
    MARKER_SIZE = 36
    LEGEND_FONT = 9
    TITLE_FONT = 12
    AXIS_FONT = 10
    GRID_STYLE = {"linestyle": ":", "linewidth": 0.6, "alpha": 0.8}

    os.makedirs(out_dir, exist_ok=True)

    # --- ensure rmse present and compute if missing ---
    for e in traj_entries:
        if e.get("rmse") is None:
            pos_pred = e["traj"]["pos_pred"]
            pos_gt = e["traj"]["pos_gt"]
            e["rmse"] = float(np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1))))

    # --- create figure and axis ---
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_facecolor("white")
    ax.set_axisbelow(True)  # grid below lines

    # Plot ground truth first/seeded so line color black remains consistent
    gt_plotted = False
    for entry in traj_entries:
        traj = entry["traj"]
        pos_gt = traj["pos_gt"]
        if not gt_plotted:
            ax.plot(pos_gt[:, 0], pos_gt[:, 1],
                    color=GT_COLOR, linestyle="-", linewidth=GT_LW, label="GT")
            # start/end marker for GT
            ax.scatter(pos_gt[0, 0], pos_gt[0, 1], marker=MARKER_START, s=MARKER_SIZE, color=GT_COLOR, zorder=20)
            ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE*1.2), color=GT_COLOR, zorder=20)
            gt_plotted = True
            break

    # Plot model predictions in the order given (this preserves original matplotlib color cycle)
    for entry in traj_entries:
        traj = entry["traj"]
        model = entry["model"]
        pos_pred = traj["pos_pred"]
        pos_gt = traj["pos_gt"]
        rmse = entry.get("rmse")
        if rmse is None:
            rmse = np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1)))
        label = f"{model} (RMSE={rmse:.3f})"

        # use matplotlib default color cycle automatically by not specifying color
        ax.plot(pos_pred[:, 0], pos_pred[:, 1], linewidth=PRED_LW, label=label, zorder=15)

        # small start/end markers for model (use same color as line)
        # get last plotted line's color:
        lc = ax.lines[-1].get_color()
        ax.scatter(pos_pred[0, 0], pos_pred[0, 1], marker=MARKER_START, s=int(MARKER_SIZE*0.45), color=lc, zorder=16)
        ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], marker=MARKER_END, s=int(MARKER_SIZE*0.7), color=lc, zorder=16)

    # axis labels, title, grid
    ax.set_aspect("equal", "box")
    ax.set_xlabel("X [m]", fontsize=AXIS_FONT)
    ax.set_ylabel("Y [m]", fontsize=AXIS_FONT)
    ax.set_title(f"Sequence {sequence_name}: Estimated vs Ground Truth (XY)", fontsize=TITLE_FONT, fontweight="semibold")
    ax.tick_params(axis="both", which="major", labelsize=AXIS_FONT)
    ax.grid(**GRID_STYLE)

    # neat legend to the right, compact, no frame for a cleaner look
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=LEGEND_FONT, frameon=False)

    # add a scale bar (nice rounded value) in lower-left of axes
    try:
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        scalelen = (xlim[1] - xlim[0]) * 0.12  # 12% of width
        # round to 1,2,5 * 10^k
        def nice_round(x):
            if x <= 0:
                return 0.0
            exp = np.floor(np.log10(x))
            base = x / (10 ** exp)
            if base < 1.5:
                nice = 1.0
            elif base < 3.5:
                nice = 2.0
            elif base < 7.5:
                nice = 5.0
            else:
                nice = 10.0
            return nice * (10 ** exp)
        scalelen_nice = nice_round(scalelen)
        sb_x = xlim[0] + 0.06 * (xlim[1] - xlim[0])
        sb_y = ylim[0] + 0.06 * (ylim[1] - ylim[0])
        ax.hlines(sb_y, sb_x, sb_x + scalelen_nice, colors="k", linewidth=2)
        ax.text(sb_x + scalelen_nice / 2.0, sb_y - 0.035 * (ylim[1] - ylim[0]),
                f"{scalelen_nice:.2f} m", ha="center", va="top", fontsize=9)
    except Exception:
        pass

    # finalize layout & save high-quality PNG + SVG for vector edits
    fig.tight_layout(rect=[0, 0, 0.78, 1.0])  # leave room for legend
    png_path = osp.join(out_dir, f"{sequence_name}_compare_xy.png")
    svg_path = osp.join(out_dir, f"{sequence_name}_compare_xy.svg")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return png_path, svg_path

def plot_sequence_comparison_3d(sequence_name, traj_entries, out_dir, show=False):
    """
    Publication-quality 3D trajectory comparison (X, Y, Z).
    Saves both PNG and SVG (300 DPI) under out_dir with filenames:
      <sequence_name>_compare_xyz.png
      <sequence_name>_compare_xyz.svg
    Returns tuple (png_path, svg_path).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    from os import path as osp

    # Visual config
    FIGSIZE = (8, 8)
    DPI = 300
    GT_COLOR = "k"
    GT_LW = 2.2
    PRED_LW = 1.4
    MARKER_START = "o"
    MARKER_END = "*"
    MARKER_SIZE = 36
    LEGEND_FONT = 9
    TITLE_FONT = 12
    AXIS_FONT = 10

    os.makedirs(out_dir, exist_ok=True)

    # --- ensure rmse present and compute if missing ---
    for e in traj_entries:
        if e.get("rmse") is None:
            pos_pred = e["traj"]["pos_pred"]
            pos_gt = e["traj"]["pos_gt"]
            e["rmse"] = float(np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1))))

    # --- create figure and 3D axis ---
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Plot ground truth once
    gt_plotted = False
    for entry in traj_entries:
        traj = entry["traj"]
        pos_gt = traj["pos_gt"]
        if not gt_plotted:
            ax.plot(pos_gt[:, 0], pos_gt[:, 1], pos_gt[:, 2],
                    color=GT_COLOR, linestyle="-", linewidth=GT_LW, label="GT")
            # start/end marker
            ax.scatter(pos_gt[0, 0], pos_gt[0, 1], pos_gt[0, 2],
                       marker=MARKER_START, s=MARKER_SIZE, color=GT_COLOR, zorder=20)
            ax.scatter(pos_gt[-1, 0], pos_gt[-1, 1], pos_gt[-1, 2],
                       marker=MARKER_END, s=int(MARKER_SIZE*1.2), color=GT_COLOR, zorder=20)
            gt_plotted = True
            break

    # Plot predictions
    for entry in traj_entries:
        traj = entry["traj"]
        model = entry["model"]
        pos_pred = traj["pos_pred"]
        pos_gt = traj["pos_gt"]
        rmse = entry.get("rmse")
        if rmse is None:
            rmse = np.sqrt(np.mean(np.sum((pos_pred - pos_gt) ** 2, axis=1)))
        label = f"{model} (RMSE={rmse:.3f})"

        # Default color cycle
        line = ax.plot(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                       linewidth=PRED_LW, label=label, zorder=15)[0]

        # Start/end markers with same line color
        lc = line.get_color()
        ax.scatter(pos_pred[0, 0], pos_pred[0, 1], pos_pred[0, 2],
                   marker=MARKER_START, s=int(MARKER_SIZE*0.45), color=lc, zorder=16)
        ax.scatter(pos_pred[-1, 0], pos_pred[-1, 1], pos_pred[-1, 2],
                   marker=MARKER_END, s=int(MARKER_SIZE*0.7), color=lc, zorder=16)

    # axis labels, title
    ax.set_xlabel("X [m]", fontsize=AXIS_FONT)
    ax.set_ylabel("Y [m]", fontsize=AXIS_FONT)
    ax.set_zlabel("Z [m]", fontsize=AXIS_FONT)
    ax.set_title(f"Sequence {sequence_name}: Estimated vs Ground Truth (3D)",
                 fontsize=TITLE_FONT, fontweight="semibold")

    # legend to the right, no frame
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 0:
        ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                  fontsize=LEGEND_FONT, frameon=False)

    # fix equal aspect ratio in 3D (approximate)
    try:
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        ranges = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
        centers = np.array([np.mean(xlim), np.mean(ylim), np.mean(zlim)])
        radius = 0.5 * max(ranges)
        ax.set_xlim(centers[0] - radius, centers[0] + radius)
        ax.set_ylim(centers[1] - radius, centers[1] + radius)
        ax.set_zlim(centers[2] - radius, centers[2] + radius)
    except Exception:
        pass

    # view angle (optional fixed)
    ax.view_init(elev=25, azim=-60)

    # save
    fig.tight_layout(rect=[0, 0, 0.80, 1.0])  # leave room for legend
    png_path = osp.join(out_dir, f"{sequence_name}_compare_xyz.png")
    svg_path = osp.join(out_dir, f"{sequence_name}_compare_xyz.svg")
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(svg_path, dpi=DPI, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return png_path, svg_path


def main(root_models_dir, out_plots_dir, models_whitelist=None, show=False):
    model_dirs = find_model_dirs(root_models_dir, whitelist=models_whitelist)
    print("Found model dirs:", model_dirs)
    all_models = []
    for md in model_dirs:
        model_name = osp.basename(md)
        seq_map = gather_trajectories_for_model(md)
        if not seq_map:
            print(f"[warn] no sequences with trajectory.txt found in {md}; skipping")
            continue
        metrics = load_metrics_if_exists(md)
        all_models.append({"name": model_name, "path": md, "seq_map": seq_map, "metrics": metrics})

    seq_set = set()
    for m in all_models:
        seq_set.update(m["seq_map"].keys())
    seq_list = sorted(seq_set)
    print(f"Found sequences: {seq_list}")

    for seq in seq_list:
        traj_entries = []
        for m in all_models:
            if seq in m["seq_map"]:
                traj_path = m["seq_map"][seq]
                try:
                    traj = read_traj_file(traj_path)
                except Exception as e:
                    print(f"[error] reading {traj_path}: {e}")
                    continue
                rmse = get_rmse_from_metrics(m["metrics"], seq)
                traj_entries.append({"model": m["name"], "traj_path": traj_path, "traj": traj, "rmse": rmse})
        if not traj_entries:
            continue
        # saved = plot_sequence_comparison(seq, traj_entries, out_plots_dir, show=show)
        # saved = plot_sequence_comparison_3d(seq, traj_entries, out_plots_dir, show=show)
        # saved = plot_sequence_comparison_highlight(seq, traj_entries, out_plots_dir, show=show)
        saved = plot_sequence_clean2(seq, traj_entries, out_plots_dir, show=show)
        # saved = plot_sequence_clean(seq, traj_entries, out_plots_dir, show=show)
        print(f"Saved {saved} (models: {[e['model'] for e in traj_entries]})")

if __name__ == "__main__":
    ROOT_MODELS_DIR = "batch_test_outputs_cov3"
    OUT_DIR = "batch_test_outputs_cov3/sequence_comparisons"  # <-- PNGs saved directly here
    # MODELS_TO_USE = None  # e.g. ["resnet", "resnet_w_cov", "vn_2regress_w_cov"]; None => use all subfolders
    MODELS_TO_USE = ["resnet", "resnet_w_cov", "vn_2regress_w_cov", "ln_2regress", "ln_2regress_w_cov_correct_lr1e5"]  # e.g. ["resnet", "resnet_w_cov", "vn_2regress_w_cov"]; None => use all subfolders
    SHOW_FIGURES = False

    main(ROOT_MODELS_DIR, OUT_DIR, models_whitelist=MODELS_TO_USE, show=SHOW_FIGURES)
