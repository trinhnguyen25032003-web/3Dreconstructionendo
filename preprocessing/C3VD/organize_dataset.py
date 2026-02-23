import os
import shutil

def organize_c3vd_dataset_sampled(source_path, output_path, step=4):
    """
    Hàm chia dataset C3VD thành các thư mục con và lọc theo bước nhảy (step).
    Mặc định step=1 (lấy các frame 0, 4, 8, 12...).
    """
    
    # 1. Cập nhật danh sách các loại file
    # Đã đổi lại thành .png đúng với dữ liệu gốc trên máy bạn
    categories = {
        "color": "_color.png",      # <--- GIỮ NGUYÊN LÀ .PNG
        "depth": "_depth.tiff",
        "normals": "_normals.tiff",
        "occlusion": "_occlusion.png",
        "flow": "_flow.tiff"
    }

    # Kiểm tra đường dẫn đầu vào
    if not os.path.exists(source_path):
        print(f"Lỗi: Đường dẫn đầu vào '{source_path}' không tồn tại.")
        return

    # Tạo thư mục đầu ra nếu chưa có
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"Đã tạo thư mục gốc đầu ra: {output_path}")

    # 2. Tạo các thư mục con
    for folder_name in categories.keys():
        folder_path = os.path.join(output_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

    print("-" * 30)
    print(f"Đang quét file tại: {source_path}")
    print(f"Chế độ lọc: Lấy mỗi {step} ảnh (frame % {step} == 0)")
    print("Bắt đầu phân loại...")

    processed_count = 0
    skipped_sampling_count = 0 
    skipped_unknown_count = 0  
    
    detail_counts = {k: 0 for k in categories.keys()}

    # 3. Duyệt và copy file
    files = os.listdir(source_path)
    
    for filename in files:
        src_file = os.path.join(source_path, filename)
        
        if not os.path.isfile(src_file):
            continue

        # --- LOGIC LỌC BƯỚC NHẢY (SAMPLING) ---
        try:
            # Tên file thường là: 0004_color.png -> Lấy phần "0004"
            prefix = filename.split('_')[0]
            frame_id = int(prefix)
            
            # Nếu frame_id không chia hết cho step (ví dụ 4), thì bỏ qua
            if frame_id % step != 0:
                skipped_sampling_count += 1
                continue 
        except ValueError:
            # File không có số ở đầu (ví dụ file rác), bỏ qua bước check step
            pass
        # --------------------------------------

        matched = False
        for folder_name, suffix in categories.items():
            if filename.endswith(suffix):
                dst_file = os.path.join(output_path, folder_name, filename)
                shutil.copy2(src_file, dst_file)
                
                processed_count += 1
                detail_counts[folder_name] += 1
                matched = True
                break
        
        # Chỉ tính là unknown nếu nó đã qua được vòng lọc step mà vẫn không khớp đuôi file nào
        if not matched and (frame_id % step == 0): 
            skipped_unknown_count += 1

    # 4. Báo cáo kết quả
    print("-" * 30)
    print("HOÀN THÀNH!")
    print(f"Tổng số file ĐƯỢC GIỮ LẠI: {processed_count}")
    print("-" * 15)
    print("Chi tiết số lượng từng loại:")
    for cate, count in detail_counts.items():
        print(f"  - {cate.ljust(10)}: {count} files")
    print("-" * 15)
    print(f"Số file bị lọc bỏ (do không thuộc bước nhảy {step}): {skipped_sampling_count}")
    print(f"Số file bị bỏ qua (file rác/khác): {skipped_unknown_count}")
    print(f"Thư mục kết quả: {output_path}")

# ==========================================
# CẤU HÌNH ĐƯỜNG DẪN
# ==========================================

INPUT_DATASET_PATH = r"C:\Users\Trinh Nguyen\Downloads\cecum_t2_b\cecum_t2_b" 
OUTPUT_DATASET_PATH = r"C:\Users\Trinh Nguyen\Downloads\cecum_t2_b"
SAMPLING_STEP = 1 

if __name__ == "__main__":
    organize_c3vd_dataset_sampled(INPUT_DATASET_PATH, OUTPUT_DATASET_PATH, SAMPLING_STEP)