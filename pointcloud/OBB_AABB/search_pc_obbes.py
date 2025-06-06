import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import copy

import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


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
    for angle_deg in range(0, 181):  # 0到180度
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

def load_and_process_ply(file_path, align_ground=False):
    """
    读取PLY文件并处理点云
    :param file_path: PLY文件路径
    """
    # 1. 读取PLY文件
    pcd = o3d.io.read_point_cloud(file_path)
    print(f"原始点云点数: {len(pcd.points)}")

    # 2. 预处理：移除无效点
    pcd = pcd.remove_non_finite_points()
    print(f"有效点云点数: {len(pcd.points)}")

    # 3. 检测地平面
    if align_ground:
        pcd, transform = detect_and_align_ground_plane(pcd)
        # 将点云移动到坐标原点
        # points = np.asarray(pcd.points)
        # center = np.mean(points, axis=0)
        # transform_matrix = np.eye(4)
        # transform_matrix[:3, 3] = -center
        # transform[:3, 3] = -center
        # pcd.transform(transform_matrix)
    
        # 保存旋转后的点云
        # o3d.io.write_point_cloud(file_path.replace('.ply','_tran.ply'), pcd)
    else:
        transform = np.eye(4)
    
    
    # 4. 按颜色聚类
    color_clusters, color_indices = cluster_points_by_color(pcd)
    
    # 5. 为每个颜色类别计算OBB
    bbox_list = visualize_color_clusters_with_obb(pcd, color_indices, vis=False)
    
    # 6. 将bbox坐标还原到原始坐标系
    transform_inv = np.linalg.inv(transform)
    bbox_list_original = []
    for bbox in bbox_list:
        # 将8个顶点坐标转换回原始坐标系
        bbox_original = []
        for point in bbox:
            # 将3D点转换为齐次坐标
            point_homo = np.append(point, 1)
            # 应用逆变换
            point_original = transform_inv @ point_homo
            # 转回3D坐标
            bbox_original.append(point_original[:3])
        bbox_list_original.append(np.array(bbox_original))
    import os
    output_path = os.path.dirname(file_path)
    scene_name = output_path.split('/')[-1]
    print('shape',np.array(bbox_list_original).shape)
    np.savez(os.path.join(os.path.dirname(output_path), 'bbox', scene_name+'.npz'), bbox_list=np.array(bbox_list_original))
    return np.array(bbox_list_original)

def detect_and_align_ground_plane(pcd):
    """
    检测地平面并校正坐标系使Z轴垂直于地面
    :param pcd: 输入点云
    :return: 校正后的点云, 变换矩阵
    """
    # 使用RANSAC检测地平面
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.02,
                                           ransac_n=3,
                                           num_iterations=1000)
    [a, b, c, d] = plane_model
    print(f"检测到的平面方程: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # 计算旋转矩阵使平面法向量对齐到Z轴
    plane_normal = np.array([a, b, c])
    z_axis = np.array([0, 0, 1])
    
    # 计算旋转轴和角度
    rotation_axis = np.cross(plane_normal, z_axis)
    rotation_angle = np.arccos(np.dot(plane_normal, z_axis) / np.linalg.norm(plane_normal))
    
    # 创建旋转矩阵
    if np.linalg.norm(rotation_axis) > 1e-6:
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * rotation_angle)
    else:
        R = np.eye(3)  # 已经是Z轴向上
    
    # 创建变换矩阵
    transform = np.eye(4)
    transform[:3, :3] = R
    
    # 应用变换
    pcd_aligned = copy.deepcopy(pcd).transform(transform)
    
    return pcd_aligned, transform

def cluster_points_by_color(pcd, n_clusters=5):
    """
    按颜色对点云进行聚类
    :param pcd: 输入点云
    :param n_clusters: 聚类数量
    :return: 聚类结果列表，每个元素是点云索引
    """
    # 获取颜色数据并转换为HSV空间
    colors = np.asarray(pcd.colors)
    unique_colors, color_indices = np.unique(colors, axis=0, return_inverse=True)
    return unique_colors, color_indices

def visualize_color_clusters_with_obb(pcd, clusters, vis=False):
    """
    可视化颜色聚类结果和OBB
    :param pcd: 点云
    :param clusters: 聚类索引列表
    """
    # 创建可视化窗口
    if vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        
        # 只可视化第一个聚类的点云
        pcd_vis = copy.deepcopy(pcd)
        points = np.asarray(pcd_vis.points)
        colors = np.asarray(pcd_vis.colors)
        vis.add_geometry(pcd_vis)
        # 定义颜色映射
        colors = [
            [1, 0, 0],  # 红
            [0, 1, 0],  # 绿
            [0, 0, 1],  # 蓝
            [1, 1, 0],  # 黄
            [1, 0, 1],  # 紫
        ]
    
    # 为每个聚类创建OBB
    bbox_list = []
    for i in range(clusters.max()):
        cluster_indices = np.where(clusters == i)[0]  # 获取当前聚类的点索引
        # 提取当前聚类的点
        cluster_points = np.asarray(pcd.points)[cluster_indices]
        
        # 创建当前聚类的点云
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        # visualize_search(cluster_points)
        obb = compute_vertical_obb_rotation_search(cluster_points)
        # 5. 打印结果
        print(f"最佳旋转角度: {obb.min_angle}°")
        print("OBB中心点:", obb.center)
        print("旋转矩阵:\n", obb.rotation)
        print("尺寸 (宽, 高, 深度):", obb.extent)
        obb_box = o3d.geometry.OrientedBoundingBox(
            center=obb.center, R=obb.rotation, extent=obb.extent
        )
        
        if vis:
            obb_box.color = colors[i % len(colors)]
            vis.add_geometry(obb_box)
        
        # 打印OBB信息
        print(f"聚类 {i} OBB信息:")
        print(f"中心点: {obb_box.center}")
        print(f"尺寸: {obb_box.extent}")
        print(f"旋转矩阵:\n{obb_box.R}")
        bbox_corners = np.asarray(obb_box.get_box_points())
        bbox_list.append(bbox_corners)

    if vis:
        render_opt = vis.get_render_option()
        render_opt.background_color = [1, 1, 1]  # 白色背景
        render_opt.point_size = 2.0
        
        # 设置视角
        view_control = vis.get_view_control()
        view_control.set_front([0, 0, -1])
        view_control.set_up([0, 1, 0])  # 确保Z轴向上
        
        # 添加坐标轴
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=2.,  # 坐标轴长度
            origin=[0, 0, 0]  # 坐标原点
        )
        vis.add_geometry(coordinate_frame)
        vis.run()
        vis.destroy_window()
    return bbox_list

    

if __name__ == "__main__":
    # 替换为你的PLY文件路径
    import os
    data_path = '/media/gzr/新加卷/dataset/CA-1M-slam'

    scene_names = os.listdir(data_path)
    for scene_name in scene_names:
        print(scene_name)
        try:
            # ply_file = f"/home/gzr/workspace/slam_bbox/OASeg/output/{scene_name}/seg_-1.ply"
            ply_file = f"/home/gzr/workspace/slam_bbox/ESAM/vis_demo/results/{scene_name}/point_cloud.ply"
            load_and_process_ply(ply_file)
        except:
            print('error')

