import os
import shutil
from pathlib import Path
from datetime import datetime

# Đường dẫn đến folder gốc chứa dataset
SOURCE_ROOT = Path(r"c:\Users\Trinh Nguyen\Downloads\simcold3d\SyntheticColon_I\SyntheticColon_I")

# Đường dẫn đến folder đầu ra đã xử lý
OUTPUT_ROOT = Path(r"c:\Users\Trinh Nguyen\Downloads\simcold3d\SyntheticColon_I\Processed_SyntheticColon_I")

# HÀM XỬ LÝ

def organize_single_folder(source_folder: Path, output_folder: Path) -> dict:
    """
    Tách ảnh Depth và FrameBuffer từ một folder nguồn vào các subfolder riêng.
    
    Args:
        source_folder: Đường dẫn đến folder nguồn (vd: Frames_S2)
        output_folder: Đường dẫn đến folder đầu ra
    
    Returns:
        dict: Thống kê số lượng file đã xử lý
    """
    # Tạo các subfolder đầu ra
    depth_folder = output_folder / "Depth"
    framebuffer_folder = output_folder / "FrameBuffer"
    
    depth_folder.mkdir(parents=True, exist_ok=True)
    framebuffer_folder.mkdir(parents=True, exist_ok=True)
    
    # Đếm số file
    stats = {"depth": 0, "framebuffer": 0, "skipped": 0}
    
    # Lấy danh sách tất cả file PNG
    files = list(source_folder.glob("*.png"))
    
    for file_path in files:
        filename = file_path.name
        
        if filename.startswith("Depth_"):
            # Copy ảnh depth
            dest_path = depth_folder / filename
            shutil.copy2(file_path, dest_path)
            stats["depth"] += 1
            
        elif filename.startswith("FrameBuffer_"):
            # Copy ảnh framebuffer
            dest_path = framebuffer_folder / filename
            shutil.copy2(file_path, dest_path)
            stats["framebuffer"] += 1
            
        else:
            stats["skipped"] += 1
    
    return stats


def main():
    """Hàm chính để xử lý toàn bộ dataset."""
    
    print("=" * 70)
    print("SCRIPT TỔ CHỨC DATASET SYNTHETICCOLON_I")
    print("=" * 70)
    print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Folder nguồn: {SOURCE_ROOT}")
    print(f"Folder đầu ra: {OUTPUT_ROOT}")
    print()
    
    # Kiểm tra folder nguồn tồn tại
    if not SOURCE_ROOT.exists():
        print(f" LỖI: Folder nguồn không tồn tại: {SOURCE_ROOT}")
        return
    
    # Tạo folder đầu ra
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách tất cả subfolder Frames_*
    frame_folders = sorted([
        f for f in SOURCE_ROOT.iterdir() 
        if f.is_dir() and f.name.startswith("Frames_")
    ])
    
    if not frame_folders:
        print(" LỖI: Không tìm thấy folder Frames_* nào!")
        return
    
    print(f" Tìm thấy {len(frame_folders)} folder cần xử lý:")
    for folder in frame_folders:
        print(f"   - {folder.name}")
    print()
    
    # Thống kê tổng hợp
    total_stats = {"depth": 0, "framebuffer": 0, "skipped": 0}
    
    # Xử lý từng folder
    for idx, source_folder in enumerate(frame_folders, 1):
        folder_name = source_folder.name
        output_folder = OUTPUT_ROOT / folder_name
        
        print(f"[{idx}/{len(frame_folders)}] Đang xử lý: {folder_name}...", end=" ")
        
        stats = organize_single_folder(source_folder, output_folder)
        
        # Cập nhật thống kê tổng
        total_stats["depth"] += stats["depth"]
        total_stats["framebuffer"] += stats["framebuffer"]
        total_stats["skipped"] += stats["skipped"]
        
        print(f"✓ Depth: {stats['depth']}, FrameBuffer: {stats['framebuffer']}")
    
    # In tổng kết
    print()
    print("=" * 70)
    print("HOÀN THÀNH!")
    print("=" * 70)
    print(f" THỐNG KÊ TỔNG HỢP:")
    print(f"    Tổng số ảnh Depth: {total_stats['depth']:,}")
    print(f"    Tổng số ảnh FrameBuffer: {total_stats['framebuffer']:,}")
    print(f"    Số file bỏ qua: {total_stats['skipped']}")
    print()
    print(f" Kết quả được lưu tại: {OUTPUT_ROOT}")
    print(f" Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
 


if __name__ == "__main__":
    main()
