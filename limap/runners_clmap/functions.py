import os
import warnings
import numpy as np
from tqdm import tqdm
import copy

import limap.util.io as limapio
import limap.structures as _structures


def eval_support_relations_consistency(line_mapper, best_proposal_type):
    th_angle_list = [1, 5, 10]  # in degrees
    for th_angle in th_angle_list:
        (A, num_valid_best_props, num_supports_sum) = line_mapper.ComputeAngleConsistencyPercentage(
            best_proposal_type, th_angle)
        print("th_angle = {0}, A = {1}, num_valid_best_props = {2}, num_supports_sum = {3}".format(
            th_angle, A, num_valid_best_props, num_supports_sum), flush=True)
    th_dist_list = [0.001, 0.005, 0.01]  # in meters for Hypersim dataset, depends on the scale of the scene
    for th_dist in th_dist_list:
        (D, num_valid_best_props, num_supports_sum) = line_mapper.ComputeDistanceConsistencyPercentage(
            best_proposal_type, th_dist)
        print("th_dist = {0}, D = {1}, num_valid_best_props = {2}, num_supports_sum = {3}".format(
            th_dist, D, num_valid_best_props, num_supports_sum), flush=True)
    th_angle_list = [1, 5, 10]  # in degrees
    th_dist_list = [0.005, 0.01, 0.05]  # in meters for Hypersim dataset, depends on the scale of the scene
    for th_angle in th_angle_list:
        for th_dist in th_dist_list:
            (AD, num_valid_best_props, num_supports_sum) = line_mapper.ComputeAngleDistanceConsistencyPercentage(
                best_proposal_type, th_angle, th_dist)
            print("th_angle = {0}, th_dist = {1}, AD = {2}, num_valid_best_props = {3}, num_supports_sum = {4}".format(
                th_angle, th_dist, AD, num_valid_best_props, num_supports_sum), flush=True)
    th_angle_th_dist_list = [(1, 0.005), (5, 0.01), (10, 0.05)]
    for th_angle_th_dist in th_angle_th_dist_list:
        (AD, num_valid_best_props, num_supports_sum) = line_mapper.ComputeAngleDistanceConsistencyPercentage(
            best_proposal_type, th_angle_th_dist[0], th_angle_th_dist[1])
        print("th_angle = {0}, th_dist = {1}, AD = {2}, num_valid_best_props = {3}, num_supports_sum = {4}".format(
            th_angle_th_dist[0], th_angle_th_dist[1], AD, num_valid_best_props, num_supports_sum), flush=True)


def compute_2d_bipartites_from_colmap(reconstruction, imagecols, all_2d_lines, cfg=dict(), min_support_images=3):
    all_bpt2ds = {}
    cfg_bpt2d = _structures.PL_Bipartite2dConfig(cfg)
    reconstruction_copy = copy.deepcopy(reconstruction)
    colmap_cameras, colmap_images, colmap_points = reconstruction_copy[
        "cameras"], reconstruction_copy["images"], reconstruction_copy["points"]
    print("Start computing 2D bipartites...")

    filtered_point_ids = set()

    for img_id, colmap_image in tqdm(colmap_images.items()):
        n_points = colmap_image.xys.shape[0]
        indexes = np.arange(0, n_points)
        xys = colmap_image.xys
        point3D_ids = colmap_image.point3D_ids
        mask = []
        for point3D_id in point3D_ids:
            if point3D_id >= 0:
                if colmap_points[point3D_id].image_ids.shape[0] >= min_support_images:
                    mask.append(True)
                else:
                    filtered_point_ids.add(point3D_id)
                    mask.append(False)
            else:
                mask.append(False)

        # resize xys if needed
        cam_id = imagecols.camimage(img_id).cam_id
        orig_size = (colmap_cameras[cam_id].width, colmap_cameras[cam_id].height)
        cam = imagecols.cam(cam_id)
        new_size = (cam.w(), cam.h())
        if orig_size != new_size:
            xys[:, 0] = xys[:, 0] * new_size[0] / orig_size[0]
            xys[:, 1] = xys[:, 1] * new_size[1] / orig_size[1]

        # init bpt2d
        bpt2d = _structures.PL_Bipartite2d(cfg_bpt2d)
        bpt2d.init_lines(all_2d_lines[img_id])
        bpt2d.add_keypoints_with_point3D_ids(xys[mask], point3D_ids[mask], indexes[mask])
        all_bpt2ds[img_id] = bpt2d
        
    points = {}
    for point3d_id, p in tqdm(colmap_points.items()):
        if p.image_ids.shape[0] >= min_support_images:
            points[point3d_id] = p.xyz

    print(f"number of original 3D points: {len(colmap_points)}")
    print(
        f"number of remained 3D points (# support images >= {min_support_images}): {len(colmap_points) - len(filtered_point_ids)}")

    return all_bpt2ds, points


def print_heading1(msg):
    print("=" * 78, flush=True)
    print(msg)
    print("=" * 78, flush=True)
