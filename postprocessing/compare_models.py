import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d import Axes3D

# --- Configuration ---
GT_DEPTH_DIR = r"c:\Users\Trinh Nguyen\Desktop\simcol3dcolon\Depth"
GT_POSE_FILE = r"c:\Users\Trinh Nguyen\Desktop\simcol3dcolon\ConvertedPoses_S1.txt"

ENDO_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\EndoDAC"
AF_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\AF-SfMLearner"
DA3_LARGE_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\DepthAnythingV3\da3-large"
DA3_MONO_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\DepthAnythingV3\da3-mono-large"

OUTPUT_DIR = r"c:\Users\Trinh Nguyen\Desktop\Comparison_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Outputs will be saved to: {OUTPUT_DIR}")

# --- Helper Functions ---

def load_depth_gt(path):
    # GT is 16-bit PNG. Max depth 20cm.
    # depth_cm = raw_val / (256*255) * 20
    # depth_mm = raw_val / 65280.0 * 200.0
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None: return None
    img_float = img.astype(np.float32)
    depth_mm = img_float * (200.0 / 65280.0)
    return depth_mm

def load_depth_pred(path):
    # NPY files (EndoDAC / AF-SfM)
    try:
        data = np.load(path)
        return data.squeeze()
    except:
        return None

def load_depth_da3_npy(path):
    # DA3 Large: XXXXX_depth.npy
    # DA3 Mono: XXXXX.npy
    try:
        data = np.load(path)
        return data.squeeze()
    except:
        return None

def compute_depth_metrics(gt, pred):
    # Median scaling
    mask = gt > 0
    if mask.sum() == 0: return None
    
    gt_masked = gt[mask]
    pred_masked = pred[mask]
    
    scale = np.median(gt_masked) / np.median(pred_masked)
    pred_scaled = pred * scale
    pred_masked_scaled = pred_scaled[mask]

    # Metrics
    # To avoid log(0) or div by zero, clamp pred to a minimal value (e.g., 0.001mm)
    # Also clamp to MAX_DEPTH (200mm) to handle outliers
    pred_masked_clean = np.maximum(pred_masked_scaled, 1e-3)
    pred_masked_clean = np.minimum(pred_masked_clean, 200.0)

    # Metrics
    thresh = np.maximum((gt_masked / pred_masked_clean), (pred_masked_clean / gt_masked))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt_masked - pred_masked_clean) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt_masked) - np.log(pred_masked_clean)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt_masked - pred_masked_clean) / gt_masked)
    sq_rel = np.mean(((gt_masked - pred_masked_clean) ** 2) / gt_masked)

    return {
        "abs_rel": abs_rel, "sq_rel": sq_rel, "rmse": rmse, "rmse_log": rmse_log, 
        "a1": a1, "a2": a2, "a3": a3
    }

def parse_gt_poses(path):
    # Comma-separated, 16 values per line (4x4 matrix flattened)
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            if len(vals) == 16:
                mat = np.array([float(x) for x in vals]).reshape(4, 4)
                # Inspecting the data: last row is tx ty tz 1, which implies column-major storage
                # if we loaded it as row-major. So we need to transpose it to get standard 
                # [R t; 0 1] format.
                mat = mat.T
                poses.append(mat)
    return np.array(poses)

def parse_pred_trajectory(path):
    # Space-separated, 12 values per line (3x4 matrix flattened)
    poses = []
    with open(path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 12:
                # Reshape to 3x4
                mat34 = np.array([float(x) for x in vals]).reshape(3, 4)
                # Append row [0,0,0,1] to make 4x4
                mat44 = np.eye(4)
                mat44[:3, :] = mat34
                poses.append(mat44)
    return np.array(poses)

def umeyama_alignment(model_points, data_points):
    """
    Computes Sim3 alignment (s, R, t) to minimize ||s*R*model + t - data||^2
    model_points: 3xN matrix
    data_points: 3xN matrix
    Returns: s, R, t
    """
    m = model_points.mean(axis=1).reshape(3, 1)
    d = data_points.mean(axis=1).reshape(3, 1)

    model_centered = model_points - m
    data_centered = data_points - d

    n = model_points.shape[1]
    
    # Covariance
    S = (model_centered @ data_centered.T) / n
    U, D, Vh = np.linalg.svd(S)
    V = Vh.T
    
    C = np.eye(3)
    if np.linalg.det(V @ U.T) < 0:
        C[2, 2] = -1

    R = V @ C @ U.T
    var_model = np.var(model_centered, axis=1).sum()
    s = np.trace(np.diag(D) @ C) / var_model
    t = d - s * R @ m
    
    return s, R, t

def apply_transform(points, s, R, t):
    # points: 3xN
    return s * R @ points + t

def export_trajectory(path, timestamps, poses):
    # Export in TUM format: timestamp tx ty tz qx qy qz qw
    # For now, we just export tx ty tz for simple plotting if needed external
    with open(path, 'w') as f:
        for i in range(len(poses)):
            t = poses[i][:3, 3]
            f.write(f"{timestamps[i]} {t[0]} {t[1]} {t[2]}\n")

# --- Main Evaluation Logic ---

def evaluate_datasets():
    print("\n--- Evaluating Depth ---")
    
    # Identify GT files (PNGs)
    gt_files = sorted(glob(os.path.join(GT_DEPTH_DIR, "*.png")))
    if not gt_files:
        print(f"[Error] No GT depth files found in {GT_DEPTH_DIR}!")
        return
    
    print(f"Found {len(gt_files)} GT depth files.")
    print(f"Sample GT: {gt_files[0]}")

    # Metrics accumulators
    endo_metrics_list = []
    af_metrics_list = []
    da3_large_metrics_list = []
    da3_mono_metrics_list = []
    
    sample_vis_idx = len(gt_files) // 2 

    for i, gt_path in enumerate(gt_files):
        basename = os.path.basename(gt_path)
        try:
            frame_id = int(basename.split('_')[1].split('.')[0])
        except:
            frame_id = int(os.path.splitext(basename)[0])
        
        pred_filename_npy = f"{frame_id:010d}.npy"
        
        # DA3 Large: XXXXX_depth.npy
        da3_large_filename = f"{frame_id:05d}_depth.npy"
        # DA3 Mono: XXXXX.npy
        da3_mono_filename = f"{frame_id:05d}.npy"
        
        # Construct paths
        endo_path = os.path.join(ENDO_DIR, "depth_corrected", pred_filename_npy)
        af_path = os.path.join(AF_DIR, "depth_corrected", pred_filename_npy)
        da3_large_path = os.path.join(DA3_LARGE_DIR, da3_large_filename)
        da3_mono_path = os.path.join(DA3_MONO_DIR, da3_mono_filename)
        
        if i < 3: # Debug first few frame paths
            print(f"Frame {i}: GT={basename} -> ID={frame_id}")
            if not os.path.exists(endo_path): print(f"  [MISSING] Endo: {endo_path}")
            if not os.path.exists(af_path):   print(f"  [MISSING] AF:   {af_path}")
            if not os.path.exists(da3_large_path): print(f"  [MISSING] DA3 Large: {da3_large_path}")
            if not os.path.exists(da3_mono_path):  print(f"  [MISSING] DA3 Mono:  {da3_mono_path}")
        
        # Load GT
        gt_depth = load_depth_gt(gt_path)
        if gt_depth is None: continue
        H, W = gt_depth.shape
        
        # --- EndoDAC ---
        if os.path.exists(endo_path):
            endo_pred = load_depth_pred(endo_path)
            if endo_pred is not None:
                endo_pred_resized = cv2.resize(endo_pred, (W, H), interpolation=cv2.INTER_LINEAR)
                m = compute_depth_metrics(gt_depth, endo_pred_resized)
                if m: endo_metrics_list.append(m)
                if i == sample_vis_idx: endo_vis = endo_pred_resized

        # --- AF-SfM ---
        if os.path.exists(af_path):
            af_pred = load_depth_pred(af_path)
            if af_pred is not None:
                af_pred_resized = cv2.resize(af_pred, (W, H), interpolation=cv2.INTER_LINEAR)
                m = compute_depth_metrics(gt_depth, af_pred_resized)
                if m: af_metrics_list.append(m)
                if i == sample_vis_idx: af_vis = af_pred_resized

        # --- DA3 Large ---
        if os.path.exists(da3_large_path):
            da3_l_pred = load_depth_da3_npy(da3_large_path)
            if da3_l_pred is not None:
                if da3_l_pred.shape != (H, W):
                    da3_l_pred = cv2.resize(da3_l_pred, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # Apply Median Scaling
                mask_gt = gt_depth > 0
                mask_pred = da3_l_pred > 0
                mask = mask_gt & mask_pred
                if np.sum(mask) > 0:
                    scale = np.median(gt_depth[mask]) / np.median(da3_l_pred[mask])
                    da3_l_pred_scaled = da3_l_pred * scale
                    
                    m = compute_depth_metrics(gt_depth, da3_l_pred_scaled)
                    if m: da3_large_metrics_list.append(m)
                    if i == sample_vis_idx: da3_l_vis = da3_l_pred_scaled

        # --- DA3 Mono ---
        if os.path.exists(da3_mono_path):
            da3_m_pred = load_depth_da3_npy(da3_mono_path)
            if da3_m_pred is not None:
                if da3_m_pred.shape != (H, W):
                    da3_m_pred = cv2.resize(da3_m_pred, (W, H), interpolation=cv2.INTER_LINEAR)
                
                # Apply Median Scaling
                mask_gt = gt_depth > 0
                mask_pred = da3_m_pred > 0
                mask = mask_gt & mask_pred
                if np.sum(mask) > 0:
                    scale = np.median(gt_depth[mask]) / np.median(da3_m_pred[mask])
                    da3_m_pred_scaled = da3_m_pred * scale
                    
                    m = compute_depth_metrics(gt_depth, da3_m_pred_scaled)
                    if m: da3_mono_metrics_list.append(m)
                    if i == sample_vis_idx: da3_m_vis = da3_m_pred_scaled

        if i == sample_vis_idx:
            gt_vis = gt_depth

    # Average metrics
    summary_path = os.path.join(OUTPUT_DIR, "comparison_summary_new.txt")
    with open(summary_path, "w") as f:
        f.write("--- Depth Comparison Results ---\n")

    def average_m(mlist, name):
        if not mlist:
            msg = f"No valid data for {name}\n"
            print(msg)
            with open(summary_path, "a") as f: f.write(msg)
            return
        
        keys = mlist[0].keys()
        
        # Prepare table string
        lines = []
        lines.append(f"\nResults for {name}:")
        lines.append(f"{'Metric':<10} {'Value':<10}")
        lines.append("-" * 20)
        for k in keys:
            val = np.mean([x[k] for x in mlist])
            lines.append(f"{k:<10} {val:.4f}")
        
        output_str = "\n".join(lines) + "\n"
        print(output_str)
        with open(summary_path, "a") as f:
            f.write(output_str)

    average_m(endo_metrics_list, "EndoDAC")
    average_m(af_metrics_list, "AF-SfMLearner")
    average_m(da3_large_metrics_list, "DA3-Large")
    average_m(da3_mono_metrics_list, "DA3-Mono")
    
    print(f"Summary table saved to: {summary_path}")

    # Visualization Depth
    if 'gt_vis' in locals():
        plt.figure(figsize=(25, 5))
        
        plt.subplot(1, 5, 1)
        plt.title("Ground Truth")
        plt.imshow(gt_vis, cmap='plasma')
        plt.axis('off')
        
        if 'endo_vis' in locals():
            plt.subplot(1, 5, 2)
            plt.title("EndoDAC")
            plt.imshow(endo_vis, cmap='plasma')
            plt.axis('off')

        if 'af_vis' in locals():
            plt.subplot(1, 5, 3)
            plt.title("AF-SfM")
            plt.imshow(af_vis, cmap='plasma')
            plt.axis('off')

        if 'da3_l_vis' in locals():
            plt.subplot(1, 5, 4)
            plt.title("DA3-Large")
            plt.imshow(da3_l_vis, cmap='plasma')
            plt.axis('off')
            
        if 'da3_m_vis' in locals():
            plt.subplot(1, 5, 5)
            plt.title("DA3-Mono")
            plt.imshow(da3_m_vis, cmap='plasma')
            plt.axis('off')


            
        plt.tight_layout()
        save_img_path = os.path.join(OUTPUT_DIR, "depth_comparison.png")
        plt.savefig(save_img_path)
        print(f"\nSaved depth_comparison.png to {save_img_path}")
        # Interactive show
        print("Opening interactive depth comparison window...")
        plt.show()

    # --- Pose Evaluation ---
    print("\n--- Evaluating Pose ---")
    if not os.path.exists(GT_POSE_FILE):
        print(f"GT Pose file not found at {GT_POSE_FILE}")
        return

    gt_poses_all = parse_gt_poses(GT_POSE_FILE)
    print(f"Loaded {len(gt_poses_all)} GT poses.")

    # Get trajectories (just translation components)
    gt_traj = gt_poses_all[:, :3, 3].T # 3xN
    print(f"GT Trajectory Sample (First 3 pts):\n{gt_traj[:, :3]}")

    def eval_pose_trajectory(name, traj_file_path, gt_traj_full):
        if not os.path.exists(traj_file_path):
            print(f"Trajectory file not found for {name}")
            return None
        
        pred_poses = parse_pred_trajectory(traj_file_path)
        if len(pred_poses) == 0:
            print(f"No poses found in {traj_file_path}")
            return None
            
        pred_traj = pred_poses[:, :3, 3].T # 3xN
        print(f"{name} Traj Sample (First 3 pts):\n{pred_traj[:, :3]}")
        
        # Truncate to match length (usually predicted is same length, but just in case)
        n = min(gt_traj_full.shape[1], pred_traj.shape[1])
        gt_t = gt_traj_full[:, :n]
        pred_t = pred_traj[:, :n]
        
        # Umeyama alignment
        s, R, t = umeyama_alignment(pred_t, gt_t)
        print(f"{name} Alignment: Scale={s:.4f}")
        
        aligned_pred = apply_transform(pred_t, s, R, t)
        
        # ATE
        error = gt_t - aligned_pred
        rmse = np.sqrt((error ** 2).mean())
        print(f"{name} ATE (RMSE): {rmse:.4f}")
        
        # Export aligned
        save_path = os.path.join(OUTPUT_DIR, f"aligned_trajectory_{name.lower().split('-')[0]}.txt")
        # Dummy timestamps 0..N
        timestamps = list(range(n))
        with open(save_path, 'w') as f:
            for i in range(n):
                pt = aligned_pred[:, i]
                f.write(f"{i} {pt[0]} {pt[1]} {pt[2]}\n")
        print(f"Saved aligned trajectory to {os.path.basename(save_path)}")

        return aligned_pred

    endo_traj_path = os.path.join(ENDO_DIR, "trajectory_S1.txt")
    af_traj_path = os.path.join(AF_DIR, "trajectory_S1.txt")

    aligned_endo = eval_pose_trajectory("EndoDAC", endo_traj_path, gt_traj)
    aligned_af = eval_pose_trajectory("AF-SfMLearner", af_traj_path, gt_traj)

    # Plot Trajectories
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Downsample for plotting if too dense
    step = 1
    
    ax.plot(gt_traj[0, ::step], gt_traj[1, ::step], gt_traj[2, ::step], 'k-', label='GT', linewidth=2)
    
    if aligned_endo is not None:
        ax.plot(aligned_endo[0, ::step], aligned_endo[1, ::step], aligned_endo[2, ::step], 'b--', label='EndoDAC (Aligned)')
        
    if aligned_af is not None:
        ax.plot(aligned_af[0, ::step], aligned_af[1, ::step], aligned_af[2, ::step], 'r--', label='AF-SfM (Aligned)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title("Aligned 3D Trajectories (Sim3)")
    
    plt.savefig(os.path.join(OUTPUT_DIR, "pose_comparison.png"))
    print("Saved pose_comparison.png")

if __name__ == "__main__":
    evaluate_datasets()
