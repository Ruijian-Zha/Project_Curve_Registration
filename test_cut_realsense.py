# initial import
import open3d as o3d
import numpy as np
import copy
import os
import sys
import matplotlib.pyplot as plt

# monkey patches visualization and provides helpers to load geometries
sys.path.append('..')
import open3d_tutorial as o3dtut
# change to True if you want to interact with the visualization windows
o3dtut.interactive = not "CI" in os.environ

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.05, origin=[0,0,0])

pcd = o3d.io.read_point_cloud("C:/Users/zhar2/Desktop/4.11.1.ply")
pcd = pcd.voxel_down_sample(0.001)

plane_model, inliers = pcd.segment_plane(distance_threshold=0.001,
                                         ransac_n=3,
                                         num_iterations=1000)
[a, b, c, d] = plane_model
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([1.0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                  zoom=0.8,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

pcd = outlier_cloud

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd.cluster_dbscan(eps=0.03, min_points=10, print_progress=True)) # recommend 0.8 and 30 for "tie_biao"
#         # pcd.cluster_dbscan(eps=0.8, min_points=30, print_progress=True)) # recommend 0.8 and 30 for "tie_biao"

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

# o3d.visualization.draw_geometries([mesh_frame, pcd],
#                                       zoom=1.4559,
#                                       front=[0.6452, -0.3036, -0.7011],
#                                       lookat=[1.9892, 2.0208, 1.8945],
#                                       up=[-0.2779, -0.9482, 0.1556])

out_pc = o3d.geometry.PointCloud()
out_pc.points = o3d.utility.Vector3dVector()
for i in range(len(pcd.points)):
    # if labels[i] == label_largest:
    #     out_pc.points.append(pcd.points[i])

    import math
    # print the absolute value of -0.5

    if math.fabs(pcd.points[i][0])>0.2 or math.fabs(pcd.points[i][1])>0.2:
        continue

    # # if the z value of pcd.points[i] is less than 0.5, then it is an outlier
    if pcd.points[i][2] > -d/c+ 0.2:
        out_pc.points.append(pcd.points[i])

pcd = out_pc

with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    labels = np.array(
        pcd.cluster_dbscan(eps=0.03, min_points=10, print_progress=True)) # recommend 0.8 and 30 for "tie_biao"
        # pcd.cluster_dbscan(eps=0.8, min_points=30, print_progress=True)) # recommend 0.8 and 30 for "tie_biao"

max_label = labels.max()
print(f"point cloud has {max_label + 1} clusters")
colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
colors[labels < 0] = 0
pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

o3d.visualization.draw_geometries([mesh_frame, pcd],
                                      zoom=1.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

# zhar2: find the largest cluster in the picture
label_largest = np.argmax([np.sum(labels[1] == k) for k in range(labels[1].max() + 1)])
out_pc = o3d.geometry.PointCloud()
# print(pcd.points[0])

# if lable[i] == label_largest, use o3d.utility.Vector3dVector to set out_pc.points
# else, use o3d.utility.Vector3dVector to set out_pc.colors
out_pc.points = o3d.utility.Vector3dVector()
for i in range(len(pcd.points)):
    if labels[i] == label_largest:
        out_pc.points.append(pcd.points[i])

o3d.visualization.draw_geometries([mesh_frame, out_pc],
                                      zoom=1.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])
o3d.io.write_point_cloud("C:/Users/zhar2/Desktop/cutted.ply", out_pc)