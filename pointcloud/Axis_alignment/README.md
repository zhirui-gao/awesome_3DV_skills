### Robust axis aligner for scanned .ply file
Transforms scans(ply) to z-up alignment for scans and tries to align walls to x-y planes.  

许多点云处理的代码需要点云满足轴对齐,这个代码可以很好把非轴对齐的点云变为轴对齐
#### 算法步骤:
1. 检测地平面,地面可以认为找到点云中最大的平面,也可以用地面检测算法,例如[link](https://github.com/AlaricYZB/patchwork-Ground_pointcloud_seg)
2. 把点云z轴和地面垂直
3. 接着拟合剩余的平面,找到和z轴垂直的最大的平面(maxpooling),作为墙面,把这个面和x轴平行即可
4. 得到轴对齐的点云


![image](https://github.com/user-attachments/assets/0ddad994-4237-4eb4-9461-f555df1e31a6)
