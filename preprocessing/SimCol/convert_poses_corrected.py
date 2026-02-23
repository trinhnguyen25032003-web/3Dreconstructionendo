import numpy as np
from scipy.spatial.transform import Rotation as R

def load_and_convert_poses(position_file_path, rotation_file_path, output_txt_path=None):

    # 1. Đọc dữ liệu từ file txt
    try:
        # Dùng np.loadtxt để đọc nhanh hơn vòng lặp for
        locations = np.loadtxt(position_file_path)
        rotations = np.loadtxt(rotation_file_path)
    except Exception as e:
        print(f"Lỗi khi đọc file: {e}")
        return None

    # Kiểm tra số lượng dòng có khớp nhau không
    if locations.shape[0] != rotations.shape[0]:
        print("Cảnh báo: Số lượng dòng giữa file vị trí và rotation không khớp nhau!")
        return None

    # 2. Chuyển đổi Quaternion sang Ma trận xoay (Rotation Matrix)
    # Lưu ý: Scipy giả định quaternion input là (x, y, z, w), khớp với format của dataset
    r_matrices = R.from_quat(rotations).as_matrix()

    # 3. Tạo ma trận chuyển đổi hệ tọa độ (Transformation Matrix)
    # Ma trận này đảo ngược trục Y (y = -y) để chuyển từ Left-handed (Unity) sang Right-handed
    TM = np.eye(4)
    TM[1, 1] = -1

    standard_poses = []

    # 4. Ghép thành ma trận 4x4 và áp dụng chuyển đổi
    for i in range(locations.shape[0]):
        # Tạo ma trận gốc (Raw Pose từ Unity)
        # QUAN TRỌNG: Theo file read_poses.py gốc, cần concatenate theo cột
        # Tức là [R | t] với R là 3x3, t là 3x1
        ri = r_matrices[i]  # 3x3 rotation matrix
        ti = locations[i].reshape((3, 1))  # 3x1 translation vector
        
        # Ghép R và t theo cột: [R | t] -> shape (3, 4)
        Pi_top = np.concatenate((ri, ti), axis=1)
        
        # Thêm hàng [0, 0, 0, 1] ở dưới -> shape (4, 4)
        Pi_bottom = np.array([[0.0, 0.0, 0.0, 1.0]])
        Pi = np.concatenate((Pi_top, Pi_bottom), axis=0)

        # Chuyển đổi hệ tọa độ theo công thức của tác giả: P_new = TM * P_old * TM
        # Phép nhân này thực hiện việc chuyển đổi không gian từ Unity sang chuẩn tính toán
        Pi_standard = TM @ Pi @ TM
        
        standard_poses.append(Pi_standard)

    poses_array = np.array(standard_poses)
    
    # 5. Xuất ra file txt nếu được chỉ định
    if output_txt_path is not None:
        save_poses_to_txt(poses_array, output_txt_path)
        print(f"Đã lưu {len(poses_array)} poses vào file: {output_txt_path}")
    
    return poses_array


def save_poses_to_txt(poses, output_path):
    """
    Format: col1_row1,col2_row1,col3_row1,col4_row1,col1_row2,col2_row2,...
    
    :param poses: Numpy array shape (N, 4, 4)
    :param output_path: Đường dẫn file output
    """
    with open(output_path, 'w') as f:
        for pose in poses:
            # Flatten theo cột (Fortran-style, column-major order)
            # Này sẽ đọc theo cột: [cột1, cột2, cột3, cột4]
            flattened = pose.flatten(order='F')
            # Chuyển thành chuỗi với dấu phẩy phân cách
            line = ','.join([f"{val:.6f}" for val in flattened])
            f.write(line + '\n')


if __name__ == "__main__":
    # Đường dẫn đến các file input
    path_pos = r'C:\Users\Trinh Nguyen\Desktop\simcol3dcolon\SavedPosition_S1.txt' 
    path_rot = r'C:\Users\Trinh Nguyen\Desktop\simcol3dcolon\SavedRotationQuaternion_S1.txt'
    
    # Đường dẫn file output
    path_output = r'C:\Users\Trinh Nguyen\Desktop\simcol3dcolon\ConvertedPoses_S1.txt'
    
    # Gọi hàm chuyển đổi và lưu kết quả
    poses = load_and_convert_poses(path_pos, path_rot, path_output)
    
    if poses is not None:
        print(f"\n✓ Đã tải và chuyển đổi thành công {poses.shape[0]} poses.")
        print(f"✓ Kích thước output: {poses.shape}")
        print(f"\nPose đầu tiên (Matrix 4x4):")
        print(poses[0])
        print(f"\nPose cuối cùng (Matrix 4x4):")
        print(poses[-1])
