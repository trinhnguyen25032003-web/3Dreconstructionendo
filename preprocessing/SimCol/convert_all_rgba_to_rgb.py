import os
from PIL import Image
from pathlib import Path
from datetime import datetime

# Đường dẫn đến folder đã xử lý
PROCESSED_ROOT = Path(r"c:\Users\Trinh Nguyen\Downloads\simcold3d\SyntheticColon_I\Processed_SyntheticColon_I")

# HÀM XỬ LÝ
def convert_folder_rgba_to_rgb(framebuffer_folder: Path) -> dict:
    """
    Chuyển đổi tất cả ảnh RGBA trong folder sang RGB.
    
    Args:
        framebuffer_folder: Đường dẫn đến folder FrameBuffer
    
    Returns:
        dict: Thống kê số lượng file đã xử lý
    """
    stats = {"converted": 0, "already_rgb": 0, "errors": 0}
    
    # Lấy danh sách tất cả file PNG
    files = sorted(list(framebuffer_folder.glob("*.png")))
    
    for file_path in files:
        try:
            # Mở ảnh
            img = Image.open(file_path)
            
            # Kiểm tra mode
            if img.mode == "RGBA":
                # Chuyển sang RGB
                img_rgb = img.convert("RGB")
                # Ghi đè file gốc
                img_rgb.save(file_path, "PNG")
                stats["converted"] += 1
            elif img.mode == "RGB":
                stats["already_rgb"] += 1
            else:
                # Chuyển các mode khác sang RGB
                img_rgb = img.convert("RGB")
                img_rgb.save(file_path, "PNG")
                stats["converted"] += 1
                
        except Exception as e:
            stats["errors"] += 1
            print(f"   ⚠ Lỗi xử lý {file_path.name}: {e}")
    
    return stats


def main():
    """Hàm chính để xử lý toàn bộ dataset."""
    
    print("=" * 70)
    print("SCRIPT CHUYỂN ĐỔI RGBA -> RGB CHO TẤT CẢ FRAMEBUFFER")
    print("=" * 70)
    print(f"Thời gian bắt đầu: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Folder gốc: {PROCESSED_ROOT}")
    print()
    
    # Kiểm tra folder gốc tồn tại
    if not PROCESSED_ROOT.exists():
        print(f" LỖI: Folder không tồn tại: {PROCESSED_ROOT}")
        return
    
    # Lấy danh sách tất cả subfolder Frames_*
    frame_folders = sorted([
        f for f in PROCESSED_ROOT.iterdir() 
        if f.is_dir() and f.name.startswith("Frames_")
    ])
    
    if not frame_folders:
        print(" LỖI: Không tìm thấy folder Frames_* nào!")
        return
    
    print(f"📁 Tìm thấy {len(frame_folders)} folder cần xử lý:")
    for folder in frame_folders:
        print(f"   - {folder.name}")
    print()
    
    # Thống kê tổng hợp
    total_stats = {"converted": 0, "already_rgb": 0, "errors": 0}
    
    # Xử lý từng folder
    for idx, frame_folder in enumerate(frame_folders, 1):
        folder_name = frame_folder.name
        framebuffer_folder = frame_folder / "FrameBuffer"
        
        if not framebuffer_folder.exists():
            print(f"[{idx}/{len(frame_folders)}] {folder_name}: ⚠ Không tìm thấy folder FrameBuffer")
            continue
        
        print(f"[{idx}/{len(frame_folders)}] Đang xử lý: {folder_name}/FrameBuffer...", end=" ", flush=True)
        
        stats = convert_folder_rgba_to_rgb(framebuffer_folder)
        
        # Cập nhật thống kê tổng
        total_stats["converted"] += stats["converted"]
        total_stats["already_rgb"] += stats["already_rgb"]
        total_stats["errors"] += stats["errors"]
        
        print(f"✓ Converted: {stats['converted']}, Already RGB: {stats['already_rgb']}")
    
    # In tổng kết
    print()
    print("=" * 70)
    print("HOÀN THÀNH!")
    print("=" * 70)
    print(f" THỐNG KÊ TỔNG HỢP:")
    print(f"    Số ảnh đã chuyển đổi RGBA -> RGB: {total_stats['converted']:,}")
    print(f"    Số ảnh đã là RGB (bỏ qua): {total_stats['already_rgb']:,}")
    print(f"    Số lỗi: {total_stats['errors']}")
    print()
    print(f" Thời gian kết thúc: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
