import os
import sys
import copy
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from glob import glob

# EVO imports
try:
    from evo.core import trajectory, metrics, sync
    from evo.core.trajectory import PoseTrajectory3D
    from evo.tools import plot
    import evo.main_ape as ape
    import evo.main_rpe as rpe
except ImportError:
    print("[ERROR] Please install evo: pip install evo")
    sys.exit(1)

# Open3D for interactive 3D viz
try:
    import open3d as o3d
except ImportError:
    print("[ERROR] Please install open3d: pip install open3d")
    sys.exit(1)

# --- Configuration ---
GT_POSE_FILE = r"c:\Users\Trinh Nguyen\Desktop\simcol3dcolon\ConvertedPoses_S1.txt"
ENDO_TRAJ = r"c:\Users\Trinh Nguyen\Desktop\Result\EndoDAC\trajectory_S1.txt"
AF_TRAJ = r"c:\Users\Trinh Nguyen\Desktop\Result\AF-SfMLearner\trajectory_S1.txt"
DA3_POSE_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\DepthAnythingV3\da3-large"
COLMAP_BIN = r"c:\Users\Trinh Nguyen\Desktop\Result\Colmap\FrameBuffer_RGB_PINHOLE_20260126_172439_results\sfm\images.bin"
OUTPUT_DIR = r"c:\Users\Trinh Nguyen\Desktop\Comparison_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Parsers ---

def parse_gt_poses(path):
    """Parse GT poses: 16 comma-separated values per line (4x4 matrix, column-major)"""
    poses = []
    positions = []
    quaternions = []
    
    if not os.path.exists(path):
        print(f"[Error] GT file not found: {path}")
        return None, None, None
    
    with open(path, 'r') as f:
        for line in f:
            vals = line.strip().split(',')
            if len(vals) == 16:
                mat = np.array([float(x) for x in vals]).reshape(4, 4).T
                poses.append(mat)
                positions.append(mat[:3, 3])
                # Extract quaternion (wxyz format for evo)
                rot = R.from_matrix(mat[:3, :3])
                quat = rot.as_quat()  # xyzw
                quaternions.append([quat[3], quat[0], quat[1], quat[2]])  # wxyz
    
    return np.array(poses), np.array(positions), np.array(quaternions)

def parse_pred_trajectory(path):
    """Parse predicted trajectory: 12 space-separated values per line (3x4 matrix)"""
    poses = []
    positions = []
    quaternions = []
    
    if not os.path.exists(path):
        print(f"[Error] Pred file not found: {path}")
        return None, None, None
    
    with open(path, 'r') as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) == 12:
                mat34 = np.array([float(x) for x in vals]).reshape(3, 4)
                mat44 = np.eye(4)
                mat44[:3, :] = mat34
                poses.append(mat44)
                positions.append(mat44[:3, 3])
                rot = R.from_matrix(mat44[:3, :3])
                quat = rot.as_quat()
                quaternions.append([quat[3], quat[0], quat[1], quat[2]])
    
    return np.array(poses), np.array(positions), np.array(quaternions)

def read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    fmt = format_char_sequence
    if not fmt.startswith("<") and not fmt.startswith("=") and not fmt.startswith(">"):
        fmt = "<" + fmt
    return struct.unpack(fmt, data)

def parse_colmap_images(bin_path):
    """Parse COLMAP images.bin file"""
    if not os.path.exists(bin_path):
        print(f"[Error] COLMAP file not found: {bin_path}")
        return None
    
    results = {}  # frame_id -> (position, quaternion_wxyz)
    
    print(f"Reading COLMAP bin: {bin_path}")
    with open(bin_path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        print(f"  COLMAP has {num_reg_images} registered images.")
        
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(binary_image_properties[1:5])  # qw, qx, qy, qz
            tvec = np.array(binary_image_properties[5:8])
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24)
            
            # Convert world-to-camera to camera-to-world
            r_obj = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])  # scipy: xyzw
            rot_mat = r_obj.as_matrix()
            pos_wc = -rot_mat.T @ tvec
            
            # Get quaternion in wxyz format
            quat_c2w = R.from_matrix(rot_mat.T).as_quat()  # xyzw
            quat_wxyz = [quat_c2w[3], quat_c2w[0], quat_c2w[1], quat_c2w[2]]
            
            # Extract frame ID
            try:
                parts = image_name.replace('.', '_').split('_')
                for p in reversed(parts):
                    if p.isdigit():
                        results[int(p)] = (pos_wc, quat_wxyz)
                        break
            except:
                pass
    
    return results

def parse_da3_poses(directory):
    """
    Parse DA3 poses from individual text files: XXXXX_pose.txt
    Each file contains 3x4 matrix (space separated)
    """
    if not os.path.exists(directory):
        print(f"[Error] DA3 dir not found: {directory}")
        return None, None, None

    pose_files = sorted(glob(os.path.join(directory, "*_pose.txt")))
    if not pose_files:
        print(f"[Error] No pose files found in {directory}")
        return None, None, None
        
    poses = []
    positions = []
    quaternions = []
    
    print(f"    Found {len(pose_files)} DA3 pose files.")
    
    # Initialize global pose as Identity
    current_pose = np.eye(4)
    
    poses.append(current_pose)
    positions.append(current_pose[:3, 3])
    rot = R.from_matrix(current_pose[:3, :3])
    quat = rot.as_quat()
    quaternions.append([quat[3], quat[0], quat[1], quat[2]])

    for i, fpath in enumerate(pose_files):
        with open(fpath, 'r') as f:
            content = f.read().strip()
            vals = content.split()
            if len(vals) == 12:
                mat34 = np.array([float(x) for x in vals]).reshape(3, 4)
                
                rel_pose = np.eye(4)
                rel_pose[:3, :] = mat34
                
                # Accumulate: T_global_new = T_global_old @ T_relative
                # Reverting to this logic as verified by process_da3_pose.py
                current_pose = current_pose @ rel_pose
                
                poses.append(current_pose)
                positions.append(current_pose[:3, 3])
                rot = R.from_matrix(current_pose[:3, :3])
                quat = rot.as_quat() 
                quaternions.append([quat[3], quat[0], quat[1], quat[2]])

    return np.array(poses), np.array(positions), np.array(quaternions)
                
    return np.array(poses), np.array(positions), np.array(quaternions)

# --- EVO Trajectory Creation ---

def create_evo_trajectory(positions, quaternions, timestamps=None):
    """Create evo PoseTrajectory3D from positions and quaternions"""
    if timestamps is None:
        timestamps = np.arange(len(positions))
    
    return PoseTrajectory3D(
        positions_xyz=positions,
        orientations_quat_wxyz=quaternions,
        timestamps=timestamps
    )

# --- Open3D Interactive Visualization ---

def create_trajectory_lineset(positions, color):
    """Create Open3D LineSet from trajectory positions"""
    n = len(positions)
    if n < 2:
        return None
    
    points = o3d.utility.Vector3dVector(positions)
    lines = [[i, i+1] for i in range(n-1)]
    colors = [color for _ in range(len(lines))]
    
    line_set = o3d.geometry.LineSet()
    line_set.points = points
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def visualize_trajectories_open3d(trajectories_dict):
    """
    Interactive 3D visualization using Open3D
    trajectories_dict: {name: (positions, color)}
    """
    geometries = []
    
    # Create coordinate frame at origin
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    geometries.append(coord_frame)
    
    for name, (positions, color) in trajectories_dict.items():
        lineset = create_trajectory_lineset(positions, color)
        if lineset is not None:
            geometries.append(lineset)
            
            # Add start point (sphere)
            start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
            start_sphere.translate(positions[0])
            start_sphere.paint_uniform_color(color)
            geometries.append(start_sphere)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Trajectory Comparison (EVO + Open3D)", width=1280, height=720)
    
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set view
    ctr = vis.get_view_control()
    ctr.set_zoom(0.5)
    
    print("\n" + "="*60)
    print("INTERACTIVE 3D VIEWER CONTROLS:")
    print("  - Left Mouse: Rotate")
    print("  - Middle Mouse / Scroll: Zoom")
    print("  - Right Mouse: Pan")
    print("  - Press 'Q' or close window to exit")
    print("="*60 + "\n")
    
    vis.run()
    vis.destroy_window()

# --- Main ---

def main():
    print("="*60)
    print("TRAJECTORY COMPARISON USING EVO LIBRARY")
    print("="*60)
    
    # 1. Load GT
    print("\n[1] Loading Ground Truth...")
    gt_poses, gt_positions, gt_quaternions = parse_gt_poses(GT_POSE_FILE)
    if gt_poses is None:
        return
    
    n_gt = len(gt_positions)
    timestamps = np.arange(n_gt).astype(float)
    gt_traj = create_evo_trajectory(gt_positions, gt_quaternions, timestamps)
    print(f"    GT: {n_gt} poses loaded")
    
    # 2. Load predictions
    print("\n[2] Loading Predictions...")
    results = {}  # name -> {"traj": aligned_traj, "ape": ape_result, "rpe": rpe_result}
    
    # EndoDAC
    endo_poses, endo_positions, endo_quaternions = parse_pred_trajectory(ENDO_TRAJ)
    if endo_positions is not None:
        n_endo = min(len(endo_positions), n_gt)
        endo_traj = create_evo_trajectory(
            endo_positions[:n_endo], 
            endo_quaternions[:n_endo], 
            timestamps[:n_endo]
        )
        results["EndoDAC"] = {"traj_orig": endo_traj}
        print(f"    EndoDAC: {n_endo} poses loaded")
    
    # AF-SfMLearner
    af_poses, af_positions, af_quaternions = parse_pred_trajectory(AF_TRAJ)
    if af_positions is not None:
        n_af = min(len(af_positions), n_gt)
        af_traj = create_evo_trajectory(
            af_positions[:n_af], 
            af_quaternions[:n_af], 
            timestamps[:n_af]
        )
        results["AF-SfM"] = {"traj_orig": af_traj}
        print(f"    AF-SfM: {n_af} poses loaded")
    
    # COLMAP
    colmap_dict = parse_colmap_images(COLMAP_BIN)
    if colmap_dict:
        sorted_ids = sorted([k for k in colmap_dict.keys() if k < n_gt])
        if len(sorted_ids) > 5:
            colmap_positions = np.array([colmap_dict[i][0] for i in sorted_ids])
            colmap_quaternions = np.array([colmap_dict[i][1] for i in sorted_ids])
            colmap_timestamps = np.array(sorted_ids).astype(float)
            colmap_traj = create_evo_trajectory(colmap_positions, colmap_quaternions, colmap_timestamps)
            results["COLMAP"] = {"traj_orig": colmap_traj, "indices": sorted_ids}
            results["COLMAP"] = {"traj_orig": colmap_traj, "indices": sorted_ids}
            print(f"    COLMAP: {len(sorted_ids)} poses loaded")

    # DA3-Large
    da3_poses, da3_positions, da3_quaternions = parse_da3_poses(DA3_POSE_DIR)
    if da3_positions is not None:
        n_da3 = min(len(da3_positions), n_gt)
        da3_traj = create_evo_trajectory(
            da3_positions[:n_da3],
            da3_quaternions[:n_da3],
            timestamps[:n_da3]
        )
        results["DA3-Large"] = {"traj_orig": da3_traj}
        print(f"    DA3-Large: {n_da3} poses loaded")
    
    # 3. Align and compute metrics
    print("\n[3] Aligning trajectories (Sim3) and computing metrics...")
    print("-"*60)
    
    summary_lines = ["TRAJECTORY EVALUATION RESULTS", "="*40, ""]
    
    for name, data in results.items():
        traj_orig = data["traj_orig"]
        
        # Sync with GT if needed (for COLMAP with sparse frames)
        if "indices" in data:
            gt_sub_positions = gt_positions[data["indices"]]
            gt_sub_quaternions = gt_quaternions[data["indices"]]
            gt_sub = create_evo_trajectory(gt_sub_positions, gt_sub_quaternions, np.array(data["indices"]).astype(float))
        else:
            n_common = min(len(traj_orig.timestamps), n_gt)
            gt_sub = create_evo_trajectory(gt_positions[:n_common], gt_quaternions[:n_common], timestamps[:n_common])
            traj_orig = create_evo_trajectory(
                np.asarray(traj_orig.positions_xyz)[:n_common],
                np.asarray(traj_orig.orientations_quat_wxyz)[:n_common],
                timestamps[:n_common]
            )
        
        # Align with Sim(3)
        traj_aligned = copy.deepcopy(traj_orig)
        traj_aligned.align(gt_sub, correct_scale=True, correct_only_scale=False)
        data["traj_aligned"] = traj_aligned
        
        # APE (Absolute Pose Error)
        ape_metric = metrics.APE(metrics.PoseRelation.translation_part)
        ape_metric.process_data((gt_sub, traj_aligned))
        ape_stats = ape_metric.get_all_statistics()
        data["ape_stats"] = ape_stats
        
        # RPE (Relative Pose Error)
        rpe_metric = metrics.RPE(metrics.PoseRelation.translation_part, delta=1, all_pairs=False)
        rpe_metric.process_data((gt_sub, traj_aligned))
        rpe_stats = rpe_metric.get_all_statistics()
        data["rpe_stats"] = rpe_stats
        
        # Print results
        print(f"\n{name}:")
        print(f"  APE (RMSE): {ape_stats['rmse']:.4f}")
        print(f"  APE (Mean): {ape_stats['mean']:.4f}")
        print(f"  APE (Std):  {ape_stats['std']:.4f}")
        print(f"  RPE (RMSE): {rpe_stats['rmse']:.4f}")
        
        summary_lines.append(f"{name}:")
        summary_lines.append(f"  APE RMSE: {ape_stats['rmse']:.4f}")
        summary_lines.append(f"  APE Mean: {ape_stats['mean']:.4f}")
        summary_lines.append(f"  APE Std:  {ape_stats['std']:.4f}")
        summary_lines.append(f"  RPE RMSE: {rpe_stats['rmse']:.4f}")
        summary_lines.append("")
    
    # Save summary
    summary_path = os.path.join(OUTPUT_DIR, "evo_comparison_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    print(f"\n[SAVED] Summary: {summary_path}")
    
    # 4. Visualization (Matplotlib with Axes & Table)
    print("\n[4] Generating Visualization (Matplotlib)...")
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot trajectories
    legend_labels = []
    
    # Plot GT
    gt_pos = gt_traj.positions_xyz
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], color='k', label='Ground Truth', linewidth=2, linestyle='--')
    
    # Prepare table data
    table_data = [] # [Model, ATE (RMSE), Color]
    cell_colors = []
    
    colors_matplotlib = {
        "EndoDAC": 'blue',
        "AF-SfM": 'red',
        "COLMAP": 'green',
        "DA3-Large": 'purple'
    }
    
    # Plot predictions
    for name, data in results.items():
        if "traj_aligned" in data:
            traj = data["traj_aligned"]
            pos = traj.positions_xyz
            color = colors_matplotlib.get(name, 'gray')
            
            # Get metrics
            ate_rmse = data.get("ape_stats", {}).get("rmse", 0.0)
            
            # Plot path with ATE in label
            label_text = f"{name} (ATE RMSE={ate_rmse:.4f} cm)"
            ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=color, label=label_text, linewidth=1.5)
            
    # Set labels and title
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'Trajectory Comparison', pad=30)
    ax.legend()
    
    # Set equal aspect ratio
    all_pos = np.vstack([t.positions_xyz for t in [gt_traj] + [data["traj_aligned"] for data in results.values() if "traj_aligned" in data]])
    min_bound = all_pos.min(axis=0)
    max_bound = all_pos.max(axis=0)
    center = np.mean(all_pos, axis=0)
    radius = 0.5 * np.max(max_bound - min_bound)
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])

    plt.tight_layout()
    viz_path = os.path.join(OUTPUT_DIR, "trajectory_comparison_plot.png")
    plt.savefig(viz_path, dpi=150)
    print(f"[SAVED] Plot: {viz_path}")
    plt.show() # Interactive matplotlib window
    
    print("\n[DONE] Comparison complete!")

if __name__ == "__main__":
    main()
