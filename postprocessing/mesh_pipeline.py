import os
import struct
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

# --- CONFIGURATION ---
# Paths
SFM_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\Colmap\FrameBuffer_RGB_PINHOLE_20260126_172439_results\sfm"
DA3_DEPTH_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\DepthAnythingV3\da3-large"
RGB_DIR = r"c:\Users\Trinh Nguyen\Desktop\simcol3dcolon\FrameBuffer_RGB"
OUTPUT_DIR = r"c:\Users\Trinh Nguyen\Desktop\Result\Mesh_Reconstruction"

# Constants
FX = 227.60416 # SimCol3D intrinsics
FY = 237.5
CX = 227.60416
CY = 237.5
DEPTH_SCALE_FACTOR = 1.0 # Initial relative scale.
SKIP_PIXEL = 4 # Downsample for dense cloud to save memory
POISSON_DEPTH = 9 # Depth for Poisson reconstruction (higher = more detail, more noise)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- COLMAP BINARY PARSERS ---

def read_next_bytes(fid, num_bytes, format_char_sequence):
    data = fid.read(num_bytes)
    return struct.unpack("<" + format_char_sequence, data)

def read_points3D_bin(path):
    print(f"Reading sparse points from {path}...")
    xyzs = []
    rgbs = []
    
    with open(path, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"  Found {num_points} sparse points.")
        
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            read_next_bytes(fid, track_length * 8, "ii" * track_length) # skip track info
            
            xyzs.append(xyz)
            rgbs.append(rgb)
            
    return np.array(xyzs), np.array(rgbs)

def read_images_bin(path):
    print(f"Reading images from {path}...")
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
                
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.read(num_points2D * 24) # skip 2d points
            
            rot_mat = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]]).as_matrix()
            center = -rot_mat.T @ tvec
            
            try:
                fid_str = ''.join(filter(str.isdigit, image_name))
                fid_int = int(fid_str)
            except:
                fid_int = image_id

            images[fid_int] = {
                'R': rot_mat,
                'center': center,
                'name': image_name
            }
    return images

# --- UTILS ---

def save_ply(path, points, colors=None, normals=None):
    print(f"Saving {len(points)} points to {path}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0 if colors.max() > 1.0 else colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.io.write_point_cloud(path, pcd)
    return pcd

# --- MAIN WORKFLOW ---

def main():
    # 1. Sparse Cloud
    points3D_path = os.path.join(SFM_DIR, "points3D.bin")
    sparse_out_path = os.path.join(OUTPUT_DIR, "sparse_colmap.ply")
    
    if os.path.exists(points3D_path):
        if not os.path.exists(sparse_out_path):
            xyzs, rgbs = read_points3D_bin(points3D_path)
            save_ply(sparse_out_path, xyzs, rgbs)
    else:
        print(f"[WARNING] points3D.bin not found at {points3D_path}, skipping sparse cloud.")

    # 2. Dense Cloud Generation
    dense_out_path = os.path.join(OUTPUT_DIR, "dense_fused_da3.ply")
    
    if os.path.exists(dense_out_path):
        print(f"Loading existing dense cloud from {dense_out_path}...")
        pcd = o3d.io.read_point_cloud(dense_out_path)
    else:
        images_bin_path = os.path.join(SFM_DIR, "images.bin")
        if not os.path.exists(images_bin_path):
            print("[ERROR] images.bin not found. Cannot proceed with dense fusion.")
            return

        images_data = read_images_bin(images_bin_path)
        sorted_ids = sorted(images_data.keys())
        
        all_points = []
        all_colors = []
        
        print("Generating dense point cloud from DA3 depth maps...")
        for i, fid in enumerate(sorted_ids):
            # Optional: Skip frames to speed up or reduce memory
            if i % 5 != 0: continue 

            info = images_data[fid]
            img_name = info['name']
            
            # Paths
            depth_name = f"{fid:05d}_depth.npy" # Assuming 00001_depth.npy config
            depth_path = os.path.join(DA3_DEPTH_DIR, depth_name)
            rgb_path = os.path.join(RGB_DIR, os.path.basename(img_name))
            
            if not os.path.exists(depth_path):
                continue
            if not os.path.exists(rgb_path):
                continue
                
            print(f"Processing frame {fid}...", end='\r')
            
            # Load Data
            depth_map = np.load(depth_path)
            rgb_img = cv2.imread(rgb_path)
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            
            H, W = depth_map.shape
            H_rgb, W_rgb, _ = rgb_img.shape
            
            # Resize RGB if mismatch
            if (H_rgb != H) or (W_rgb != W):
                rgb_img = cv2.resize(rgb_img, (W, H))
                
            # Downsample
            y_idx, x_idx = np.mgrid[0:H:SKIP_PIXEL, 0:W:SKIP_PIXEL]
            depth_sub = depth_map[0:H:SKIP_PIXEL, 0:W:SKIP_PIXEL] * DEPTH_SCALE_FACTOR
            rgb_sub = rgb_img[0:H:SKIP_PIXEL, 0:W:SKIP_PIXEL]
            
            # Filter valid depth
            valid_mask = (depth_sub > 0.1) & (depth_sub < 20.0) # Clip range
            
            z = depth_sub[valid_mask]
            x = (x_idx[valid_mask] - CX) * z / FX
            y = (y_idx[valid_mask] - CY) * z / FY
            
            pts_cam = np.vstack((x, y, z)).T
            colors_cam = rgb_sub[valid_mask]
            
            R_mat = info['R']
            center = info['center']
            
            pts_world = pts_cam @ R_mat + center
            
            all_points.append(pts_world)
            all_colors.append(colors_cam)
            
        print("\nFusing points...")
        if not all_points:
            print("[ERROR] No points generated.")
            return
            
        final_pts = np.vstack(all_points)
        final_cols = np.vstack(all_colors)
        
        pcd = save_ply(dense_out_path, final_pts, final_cols)
    
    # 3. Poisson Reconstruction
    print("Starting Poisson Reconstruction...")
    
    # Robust Check for Scale
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print(f"  Avg NN Distance: {avg_dist:.4f}")
    
    # Downsample
    voxel_size = avg_dist * 5 # Heuristic
    print(f"  Downsampling with voxel_size={voxel_size:.4f}...")
    pcd_down = pcd.voxel_down_sample(voxel_size)
    print(f"  Points remaining: {len(pcd_down.points)}")

    # Estimate Normals
    print("  Estimating normals...")
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    pcd_down.orient_normals_consistent_tangent_plane(10)
    
    print(f"  Running Poisson Mesh (depth={POISSON_DEPTH})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd_down, depth=POISSON_DEPTH)
    
    # Filter mesh
    print("  Filtering low density vertices...")
    vertices_to_remove = densities < np.quantile(densities, 0.05)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    mesh_out_path = os.path.join(OUTPUT_DIR, "poisson_mesh.ply")
    print(f"Saving mesh to {mesh_out_path}")
    o3d.io.write_triangle_mesh(mesh_out_path, mesh)
    print("Done.")

if __name__ == "__main__":
    main()
