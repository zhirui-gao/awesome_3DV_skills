### Get OBB from the segmented point cloud

> Here is an algorithm to get the bounding box with direction from the segmented point cloud

open3d的 obb检测算法无法满足obb垂直于地面,于是需要一个实用的obb检测算法
1. 输入一个点云,不同颜色表示不同的分割
2. 点云变成Z轴垂直于地面
3. 把点云投影到xy平面,枚举点云的选择角度,保存一个最小面积的包围框
4. 最小面积的包围框对应的旋转角度的逆即为bbox的旋转方向

![35837c5864df1eac0ab217227c733784](https://github.com/user-attachments/assets/b57c6283-59b6-4bbb-8f3c-ceaf57d3c625)
