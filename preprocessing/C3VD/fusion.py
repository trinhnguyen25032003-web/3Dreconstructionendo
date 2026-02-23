import open3d as o3d
import numpy as np
import cv2
import os
import glob

# --- CẤU HÌNH (Giữ nguyên như cũ) ---
BASE_DIR = r"C:\Users\Trinh Nguyen\Downloads\cecum_t2_b_rectified"
POSE_PATH = r"C:\Users\Trinh Nguyen\Desktop\C3VD prepare\C3VD_EndoGSLAM\C3VD\cecum_t2_b\pose.txt"

CAM = {"w": 1000, "h": 1000, "fx": 450, "fy": 450, "cx": 500, "cy": 500}

TSDF = {
    "voxel": 0.002,
    "sdf_trunc": 0.08,
    "depth_scale": 655350.0, 
    "depth_trunc": 0.10
}

def load_poses(path):
    poses = []
    if not os.path.exists(path): return []
    with open(path, 'r') as f:
        for line in f:
            try:
                mat = np.array([float(x) for x in line.strip().split(',')]).reshape(4, 4).T
                mat[:3, 3] /= 1000.0 
                poses.append(mat)
            except: pass
    return poses

def main():
    print(f"[INFO] Bắt đầu TSDF Fusion (Vá lỗ & Tô màu)...")
    
    # 1. Setup
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=TSDF["voxel"], 
        sdf_trunc=TSDF["sdf_trunc"],
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        CAM["w"], CAM["h"], CAM["fx"], CAM["fy"], CAM["cx"], CAM["cy"]
    )

    # 2. Load Data
    poses = load_poses(POSE_PATH)
    rgb_files = sorted(glob.glob(os.path.join(BASE_DIR, "color", "*.png")))
    depth_files = sorted(glob.glob(os.path.join(BASE_DIR, "depth", "*.tiff")))
    occ_files = sorted(glob.glob(os.path.join(BASE_DIR, "occlusion", "*.png")))
    
    n_frames = min(len(poses), len(rgb_files), len(depth_files))
    
    # 3. Integration
    for i in range(n_frames):
        print(f"Integrating: {i+1}/{n_frames}", end='\r')
        color = cv2.cvtColor(cv2.imread(rgb_files[i]), cv2.COLOR_BGR2RGB)
        depth = cv2.imread(depth_files[i], -1)
        
        if i < len(occ_files):
            mask = cv2.imread(occ_files[i], 0)
            if mask is not None: depth[mask == 255] = 0

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color), o3d.geometry.Image(depth),
            depth_scale=TSDF["depth_scale"], depth_trunc=TSDF["depth_trunc"],
            convert_rgb_to_intensity=False
        )
        try: volume.integrate(rgbd, intrinsic, np.linalg.inv(poses[i]))
        except: pass

    # 4. Trích xuất Mesh thô (Có màu nhưng bị thủng)
    print("\n[EXTRACT] Trích xuất mesh gốc...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    
    # --- BƯỚC MỚI: VÁ LỖ & TÔ MÀU ---
    print(f"[POISSON] Đang lấp lỗ hổng (Mesh gốc: {len(mesh.vertices)} đỉnh)...")

    # Tạo Point Cloud từ Mesh gốc để giữ lại MÀU
    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.normals = mesh.vertex_normals
    pcd.colors = mesh.vertex_colors # <--- Quan trọng: Lưu lại màu gốc

    # Chạy Poisson (Tạo ra mesh mới, trắng trơn)
    poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=9, scale=1.1, linear_fit=True
    )
    
    # Cắt bỏ phần thừa (bong bóng)
    vertices_to_remove = densities < np.quantile(densities, 0.04)
    poisson_mesh.remove_vertices_by_mask(vertices_to_remove)

    print("[COLOR] Đang tô màu lại cho mesh mới (Có thể mất vài giây)...")
    # Dùng KDTree để tìm điểm màu gần nhất từ dữ liệu gốc áp vào mesh mới
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    
    # Lấy mảng đỉnh của mesh mới
    new_vertices = np.asarray(poisson_mesh.vertices)
    original_colors = np.asarray(pcd.colors)
    recolored_colors = []

    # Duyệt qua từng đỉnh mới để tìm màu (Loop này nhanh với < 1 triệu điểm)
    for i in range(len(new_vertices)):
        # Tìm 1 điểm gần nhất trong pcd gốc
        [_, idx, _] = pcd_tree.search_knn_vector_3d(new_vertices[i], 1)
        # Lấy màu của điểm đó
        recolored_colors.append(original_colors[idx[0]])
    
    # Gán màu mới vào mesh
    poisson_mesh.vertex_colors = o3d.utility.Vector3dVector(recolored_colors)
    poisson_mesh.compute_vertex_normals() # Tính lại normal cho đẹp

    # Lưu kết quả
    out_name = "cecum_final_colored.ply"
    o3d.io.write_triangle_mesh(out_name, poisson_mesh)
    print(f"[DONE] Hoàn tất! File đã lưu: {out_name}")
    
    o3d.visualization.draw_geometries([poisson_mesh], 
                                      window_name="Cecum Final (Colored)",
                                      mesh_show_back_face=True)

if __name__ == "__main__":
    main()