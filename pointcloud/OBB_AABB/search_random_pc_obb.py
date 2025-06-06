import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt

class BoundingBox:
    def __init__(self):
        self.center = None     # 中心点 (x, y, z)
        self.rotation = None  # 旋转矩阵 (3x3)
        self.extent = None    # 尺寸 (width, height, depth)
        self.min_angle = None # 最小面积对应的角度

def compute_vertical_obb_rotation_search(point_cloud):
    """
    通过旋转搜索计算垂直于Z轴的定向包围盒(OBB)
    
    参数:
        point_cloud: 目标物体的点云 (numpy数组, Nx3)
        
    返回:
        BoundingBox 对象
    """
    # 1. 提取XY点集
    xy_points = point_cloud[:, :2]
    center_xy = np.mean(xy_points, axis=0)
    
    # 2. 初始化搜索参数
    min_area = float('inf')
    best_angle = 0
    best_bbox_2d = None
    
    # 3. 在0-180度范围内搜索最小包围面积
    for angle_deg in np.arange(0, 181, 1.):  # 0到180度
        # 计算旋转矩阵（绕Z轴）
        angle_rad = np.radians(angle_deg)
        rotation_2d = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 旋转点集
        rotated_points = (rotation_2d @ (xy_points - center_xy).T).T + center_xy
        
        # 计算轴对齐包围盒
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)
        width = max_vals[0] - min_vals[0]
        height = max_vals[1] - min_vals[1]
        area = width * height
        
        # 更新最小面积和最佳角度
        if area < min_area:
            min_area = area
            best_angle = angle_deg
            best_bbox_2d = (min_vals, max_vals, width, height)
    
    print(f"找到最小面积 {min_area:.2f} 在角度 {best_angle}°")
    
    # 4. 构建3D OBB
    # 提取最小Z和最大Z值
    min_z = np.min(point_cloud[:, 2])
    max_z = np.max(point_cloud[:, 2])
    depth = max_z - min_z
    
    # 计算2D中心
    min_vals, max_vals, width, height = best_bbox_2d
    center_2d = (min_vals + max_vals) / 2
    
    # 构建3D旋转矩阵 (绕Z轴)
    best_angle_rad = np.radians(-best_angle)
    rotation_3d = np.array([
        [np.cos(best_angle_rad), -np.sin(best_angle_rad), 0],
        [np.sin(best_angle_rad), np.cos(best_angle_rad), 0],
        [0, 0, 1]
    ])
    
    # 3D中心点
    center_3d = np.array([center_2d[0], center_2d[1], (min_z + max_z) / 2])
    
    # 5. 创建包围盒对象
    bbox = BoundingBox()
    bbox.rotation = rotation_3d
    bbox.min_angle = best_angle
    
    # 6. 计算包围盒的尺寸和中心
    # 将点云转换到OBB坐标系
    points_local = (rotation_3d.T @ (point_cloud - center_3d).T).T
    
    # 计算局部坐标系下的边界框
    min_vals_local = np.min(points_local, axis=0)
    max_vals_local = np.max(points_local, axis=0)
    
    # 计算包围盒的尺寸（在原始坐标系中）
    bbox.extent = np.array([width, height, depth])
    
    # 计算包围盒的中心（在原始坐标系中）
    local_center = (max_vals_local + min_vals_local) / 2
    bbox.center = center_3d + rotation_3d @ local_center
    
    return bbox

# 可视化搜索过程
def visualize_search(point_cloud):
    xy_points = point_cloud[:, :2]
    center_xy = np.mean(xy_points, axis=0)
    
    areas = []
    
    plt.figure(figsize=(10, 6))
    plt.title("Rotation Angle vs. Bounding Box Area")
    plt.xlabel("Rotation Angle (degrees)")
    plt.ylabel("Bounding Box Area")
    
    for angle_deg in range(0, 181, 5):  # 每5度采样一次
        angle_rad = np.radians(angle_deg)
        rotation_2d = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ])
        
        # 旋转点集
        rotated_points = (rotation_2d @ (xy_points - center_xy).T).T + center_xy
        
        # 计算包围盒面积
        min_vals = np.min(rotated_points, axis=0)
        max_vals = np.max(rotated_points, axis=0)
        area = (max_vals[0] - min_vals[0]) * (max_vals[1] - min_vals[1])
        areas.append(area)
    
    plt.plot(range(0, 181, 5), areas, 'b-')
    plt.grid(True)
    plt.show()

# 使用示例
if __name__ == "__main__":
    # 1. 创建示例点云 (矩形)
    def generate_rectangle(width, height, translation=(0, 0, 0), rotation=30, n_points=1000):
        # 生成随机点
        # 生成主要点云
        x = np.random.uniform(-width/2, width/2, n_points)
        y = np.random.uniform(-height/2, height/2, n_points)
        z = np.random.uniform(0, 1, n_points)  # 高度范围0-1
        points = np.column_stack((x, y, z))
        
        # 添加一些异常点
        n_outliers = int(n_points * 0.05)  # 5%的异常点
        outlier_x = np.random.uniform(-width*10, width*2, n_outliers)  # 扩大范围
        outlier_y = np.random.uniform(-height*5, height*5, n_outliers)
        outlier_z = np.random.uniform(-0.5, 1.5, n_outliers)  # 扩大高度范围
        outliers = np.column_stack((outlier_x, outlier_y, outlier_z))
        
        # 合并主要点云和异常点
        points =  np.vstack((outliers,points))
        
        # 应用旋转
        theta = np.radians(rotation)
        rot_mat = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        points = points @ rot_mat.T + translation
        
        return points
    
    # 2. 生成点云 (长宽比2:1的矩形)
    rectangle_points = generate_rectangle(width=2.0, height=1.0, translation=(0, 1., 2.0), rotation=30)
    
    # 3. 添加噪声
    np.random.seed(42)
    rectangle_points += np.random.normal(0, 0.02, rectangle_points.shape)
    
    # 可视化搜索过程
    visualize_search(rectangle_points)
    
    # 4. 计算OBB (使用旋转搜索法)
    obb = compute_vertical_obb_rotation_search(rectangle_points)
    
    # 5. 打印结果
    print(f"最佳旋转角度: {obb.min_angle}°")
    print("OBB中心点:", obb.center)
    print("旋转矩阵:\n", obb.rotation)
    print("尺寸 (宽, 高, 深度):", obb.extent)
    
    # 6. 可视化
    # 创建点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rectangle_points)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色点云
    
    # 创建OBB可视化
    obb_box = o3d.geometry.OrientedBoundingBox(
        center=obb.center, R=obb.rotation, extent=obb.extent
    )
    
    # 设置OBB颜色和透明度
    obb_box.color = [1, 0, 0]  # 红色边框
   
    
    # 创建坐标轴
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.)
    
    # 可视化
    o3d.visualization.draw_geometries([pcd, obb_box, coord_frame])
