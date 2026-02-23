import numpy as np
import cv2
import os
import glob
from scipy.interpolate import interp1d

# --- CẤU HÌNH INTRINSICS (Thông số chuẩn v2 bạn cung cấp) ---
INTRINSICS = {
    "width": 1350, "height": 1080,
    "cx": 677.739464094188,
    "cy": 543.057997844875,
    "a0": 767.733695862103,
    "a1": 0.0,
    "a2": -0.000592506426558248,
    "a3": -2.69440266600040e-07,
    "a4": -2.16380341010063e-10,
    "c": 0.9999,
    "d": 1.10e-4,
    "e": -1.83e-4
}

# Output Settings
NEW_SIZE = (1000, 1000)
NEW_CAM = {"fx": 450.0, "fy": 450.0, "cx": 500.0, "cy": 500.0}

def get_undistort_maps():
    """Tính toán Map X, Y cho Scaramuzza Model"""
    print("[1/3] Đang tính toán Lookup Table...")
    
    # 1. Tạo Lookup Table: Slope -> Rho
    # np.polyval tính theo thứ tự bậc cao xuống thấp: [a4, a3, a2, a1, a0]
    rhos = np.linspace(0, np.hypot(INTRINSICS['width'], INTRINSICS['height']), 5000)
    poly_coeffs = [INTRINSICS['a4'], INTRINSICS['a3'], INTRINSICS['a2'], INTRINSICS['a1'], INTRINSICS['a0']]
    z = np.polyval(poly_coeffs, rhos) 
    
    with np.errstate(divide='ignore'):
        slopes = z / rhos
    slopes[0] = 1e9 # Tại tâm
    
    # Hàm nội suy ngược: Slope -> Rho
    get_rho = interp1d(slopes[::-1], rhos[::-1], kind='linear', fill_value="extrapolate")

    print("[2/3] Đang tạo Pixel Map...")
    
    # 2. Tạo lưới pixel Pinhole
    u, v = np.meshgrid(np.arange(NEW_SIZE[0]), np.arange(NEW_SIZE[1]))
    X3d, Y3d = (u - NEW_CAM['cx']) / NEW_CAM['fx'], (v - NEW_CAM['cy']) / NEW_CAM['fy']
    R3d = np.hypot(X3d, Y3d)
    
    # 3. Mapping ngược
    rho_fish = get_rho(1.0 / (R3d + 1e-9))
    scale = np.divide(rho_fish, R3d, out=np.zeros_like(R3d), where=R3d!=0)
    
    # Áp dụng Affine (Stretch Matrix)
    uv_x, uv_y = X3d * scale, Y3d * scale
    map_x = INTRINSICS['c'] * uv_x + INTRINSICS['d'] * uv_y + INTRINSICS['cx']
    map_y = INTRINSICS['e'] * uv_x + 1.0 * uv_y + INTRINSICS['cy']
    
    return map_x.astype(np.float32), map_y.astype(np.float32)

def process_batch(src_root, dst_root, folder, ext, maps, is_raw=False):
    src_path = os.path.join(src_root, folder)
    if not os.path.exists(src_path): 
        print(f"[SKIP] Không tìm thấy thư mục: {src_path}")
        return

    dst_path = os.path.join(dst_root, folder)
    os.makedirs(dst_path, exist_ok=True)
    
    files = sorted(glob.glob(os.path.join(src_path, f"*{ext}")))
    print(f">> Xử lý {folder}: {len(files)} ảnh...")

    map_x, map_y = maps
    interp = cv2.INTER_NEAREST if is_raw else cv2.INTER_LINEAR
    
    count = 0
    for fpath in files:
        img = cv2.imread(fpath, -1)
        if img is None: continue
        
        # Remap
        rect = cv2.remap(img, map_x, map_y, interpolation=interp, borderMode=cv2.BORDER_CONSTANT)
        
        # Lưu file
        base_name = os.path.basename(fpath)
        cv2.imwrite(os.path.join(dst_path, base_name), rect)
        count += 1
        
        if count % 100 == 0:
            print(f"   Đã xong {count}/{len(files)}...", end='\r')
    print("")

def main():
    # --- ĐƯỜNG DẪN ---
    BASE_DIR = r"C:\Users\Trinh Nguyen\Downloads\cecum_t2_b"
    OUTPUT_DIR = r"C:\Users\Trinh Nguyen\Downloads\cecum_t2_b_rectified"
    
    maps = get_undistort_maps()
    
    # Danh sách task
    tasks = [
        ("color", ".png", False),
        ("depth", ".tiff", True),
        ("occlusion", ".png", True),
        ("normals", ".tiff", True),
        ("flow", ".tiff", True)
    ]
    
    # [FIX LỖI] Giải nén tuple rõ ràng trước khi truyền vào hàm
    for folder, ext, is_raw in tasks:
        process_batch(BASE_DIR, OUTPUT_DIR, folder, ext, maps, is_raw)
        
    print(f"\n[DONE] Hoàn tất! Output tại: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()