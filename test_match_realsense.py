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

# o3d.visualization.draw_geometries([mesh_frame, source, target],
#                                   zoom=0.8,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])

def draw_registration_result(source, target, sphere, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    # zhar2: sphere deepcopy
    sphere_temp = copy.deepcopy(sphere)

    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)

    # zhar2: transform the sphere
    sphere_temp.transform(transformation)

    # zhar2: draw a bounding box
    max_bound = source_temp.get_max_bound()
    min_bound = source_temp.get_min_bound()
    bounding_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    bounding_box.color = (0, 1, 0)

    # zhar2: prepare the coordiantes for this dataset
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.05, origin=[0,0,0])
    o3d.visualization.draw_geometries([sphere_temp, mesh_frame, source_temp, target_temp, bounding_box],
                                      zoom=0.4559,
                                      front=[0.6452, -0.3036, -0.7011],
                                      lookat=[1.9892, 2.0208, 1.8945],
                                      up=[-0.2779, -0.9482, 0.1556])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def prepare_dataset(voxel_size):
    print(":: Load two point clouds and disturb initial pose.")

    source = o3d.io.read_point_cloud("./pattern.ply")
    # target = o3d.io.read_point_cloud("./curve.ply")
    target = o3d.io.read_point_cloud("./cutted.ply")
    source = source.voxel_down_sample(1)

    center=source.get_center()

    # translate source to the origin of frame
    source.translate(-center)
    source.scale(0.001, center=source.get_center())

    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

    # zhar2: rotate the target
    import math
    trans_matrix = np.asarray([ [1.0, 0.0, 0.0, 0.0], 
                            [0.0, math.cos(math.radians(30)), -math.sin(math.radians(30)), 0.0],
                            [0.0, math.sin(math.radians(30)), math.cos(math.radians(30)), 0.0], 
                            [0.0, 0.0, 0.0, 1.0]])
    target.transform(trans_matrix)
    
    source.transform(trans_init)

    # zhar2: create sphere
    # find x y z position of the point with largest z value in the target point cloud
    points = np.asarray(source.points)
    max_z = np.max(points[:, 2])
    max_z_index = np.argmax(points[:, 2])
    max_point = points[max_z_index]

    sphere = o3d.geometry.TriangleMesh.create_sphere(0.005)
    sphere.translate((max_point[0], max_point[1], max_point[2]))


    draw_registration_result(source, target, sphere, np.identity(4))

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh, sphere

voxel_size = 2  # means 5cm for this dataset 0.05
source, target, source_down, target_down, source_fpfh, target_fpfh, sphere = prepare_dataset(
    voxel_size)

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.99999))
    return result

result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
print(result_ransac.transformation)
draw_registration_result(source_down, target_down, sphere, result_ransac.transformation)

def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result

source.estimate_normals()
target.estimate_normals()
result_icp = refine_registration(source, target, source_fpfh, target_fpfh,
                                 voxel_size)
print(result_icp)
draw_registration_result(source, target, sphere, result_icp.transformation)