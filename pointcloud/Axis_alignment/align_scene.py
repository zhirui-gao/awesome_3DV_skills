import open3d as o3d
import numpy as np
import os
from sklearn.decomposition import PCA
# 加载点云数据

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA

def align_scene_to_z_up(pcd, save_pcd_path=None, save_transform_path=None, visualize=False, fit_ground=True):
    """
    将点云扫描转换为Z轴向上对齐，并尝试将墙壁对齐到X-Y平面
    
    参数:
        points (np.ndarray): (N, 3) 点云数据
        visualize (bool): 是否可视化结果
        
    返回:
        aligned_points (np.ndarray): 对齐后的点云
        transform_matrix (np.ndarray): 应用的4x4变换矩阵
    """
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    distance_threshold = 0.02  # 动态距离阈值
    if fit_ground:
        plane_model, ground_inliers = pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        [a, b, c, d] = plane_model
        ground_normal = np.array([a, b, c])
        print('ground_normal:', ground_normal)
        # 可视化地面点云
        ground_cloud = pcd.select_by_index(ground_inliers)
        ground_cloud.paint_uniform_color([1, 0, 0])  # 红色表示地面
        if visualize:
            o3d.visualization.draw_geometries([ground_cloud])
        
        # 3. 计算旋转矩阵使地面法向量对齐到Z轴
        z_axis = np.array([0, 0, 1])
        ground_normal = ground_normal / np.linalg.norm(ground_normal)
        
        # 计算旋转轴和角度
        rotation_axis = np.cross(ground_normal, z_axis)
        if np.linalg.norm(rotation_axis) < 1e-6:
            rotation_axis = np.array([0, 1, 0])  # 避免零向量
        else:
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        cos_theta = np.dot(ground_normal, z_axis)
        angle = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # 使用罗德里格斯公式计算旋转矩阵
        K = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        R_ground = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
        
        # 4. 应用地面旋转
        centered_points = points - centroid
        ground_rotated = (R_ground @ centered_points.T).T
        
        # 5. 检测墙壁平面（垂直平面）
        all_indices = np.arange(len(points))
        non_ground_indices = np.setdiff1d(all_indices, ground_inliers)
        non_ground_points = ground_rotated[non_ground_indices]
        
        # 创建临时点云用于墙壁检测
        temp_pcd = o3d.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(non_ground_points)
        remaining_pcd = temp_pcd
    else:
        # already z up
        remaining_pcd = pcd
    
    wall_planes = []
    wall_directions = []
    max_points = 0
    best_direction = None
    R_walls = np.eye(3)
    # 检测多个墙壁平面
    for _ in range(6):
        if len(remaining_pcd.points) < 100:
            break
            
        plane_model, inliers = remaining_pcd.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        wall_cloud = remaining_pcd.select_by_index(inliers)
        wall_cloud.paint_uniform_color([1, 0, 0])  # 红色表示地面
        if visualize:
            o3d.visualization.draw_geometries([wall_cloud])

        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        
        # 检查是否为垂直平面（法向量的Z分量接近0）
        if abs(normal[2]) < 0.1 and np.linalg.norm(normal[:2]) > 0.5:
            # 提取水平方向
            horizontal_dir = normal[:2] / np.linalg.norm(normal[:2])
            wall_directions = horizontal_dir
            print('wall direction',normal)
            if max_points < len(inliers):
                max_points = len(inliers)
                print(max_points)
                best_direction = wall_directions
                 # 计算最佳方向的旋转角度
                angle = np.arctan2(best_direction[1], best_direction[0])
                
                # 创建最终的旋转矩阵
                R_walls = np.array([
                    [np.cos(-angle), -np.sin(-angle), 0],
                    [np.sin(-angle), np.cos(-angle), 0],
                    [0, 0, 1]
                ])
        
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    
    # 8. 创建变换矩阵（旋转 + 平移）
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = R_walls
    transform_matrix[:3, 3] = -R_walls @ centroid  # 平移使中心到原点
    
    # 9. 应用变换
    aligned_points = (transform_matrix[:3, :3] @ points.T + transform_matrix[:3, 3:4]).T
    
    # 10. 创建对齐后的点云
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_points)

    if pcd.colors:
        aligned_pcd.colors = pcd.colors
    
    # 11. 保存结果
    if save_pcd_path:
        o3d.io.write_point_cloud(save_pcd_path, aligned_pcd)
    
    if save_transform_path:
        np.savetxt(save_transform_path, transform_matrix, fmt='%.8f')
    

    
    return aligned_pcd, transform_matrix


    """创建4x4变换矩阵"""
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    return transform

def visualize_alignment(pcd_orig, pcd_aligned):
    """可视化原始点云和对齐后的点云"""
    pcd_orig.paint_uniform_color([1, 0, 0])  # 红色为原始点云
    pcd_aligned.paint_uniform_color([0, 1, 0])  # 绿色为对齐后点云
    
    # 创建坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    
    o3d.visualization.draw_geometries([pcd_orig, pcd_aligned, coord_frame])



ply_dir = '/media/gzr/新加卷/dataset/CA-1M-slam'
scene_names = os.listdir(ply_dir)
for scene_name in scene_names:
    print(scene_name)
    # if scene_name !='42897501':
    #     continue
    if not os.path.isdir(os.path.join(ply_dir, scene_name)):
        continue
        
    pcd = o3d.io.read_point_cloud(os.path.join(ply_dir, scene_name, "mesh.ply"))
    
    # 执行对齐
    transformed_pcd, transform = align_scene_to_z_up(pcd, save_transform_path=os.path.join(ply_dir, scene_name, "transform.txt"),
                                             visualize=False, fit_ground=False)

    aligned_points = np.asarray(transformed_pcd.points)


    # scale point cloud
    min_z = np.min(aligned_points[:, 2])
    max_z = np.max(aligned_points[:, 2])
    height = max_z - min_z
    print(f"Original height: {height}")
    estimated_height = 2.5
    scale = estimated_height / height
    
    # 保存变换后的点云到原路径
    o3d.io.write_point_cloud(os.path.join(ply_dir, scene_name, "mesh_align.ply"), transformed_pcd)
 
