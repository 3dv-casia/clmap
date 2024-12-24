import os
import warnings
import numpy as np
from tqdm import tqdm
import copy

import limap.util.io as limapio
import limap.structures as _structures


def setup(cfg):
    folder_save = cfg["output_dir"]
    if folder_save is None:
        folder_save = 'tmp'
    limapio.check_makedirs(folder_save)
    folder_load = cfg["load_dir"]
    if cfg["use_tmp"]:
        folder_load = "tmp"
    if folder_load is None:
        folder_load = folder_save
    cfg["dir_save"] = folder_save
    cfg["dir_load"] = folder_load
    print("[LOG] Output dir: {0}".format(cfg["dir_save"]))
    print("[LOG] Loading dir: {0}".format(cfg["dir_load"]))
    if "weight_path" in cfg and cfg["weight_path"] is not None:
        cfg["weight_path"] = os.path.expanduser(cfg["weight_path"])
        print("[LOG] weight dir: {0}".format(cfg["weight_path"]))
    return cfg


def compute_matches(cfg, descinfo_folder, image_ids, neighbors):
    """
    Match lines for each image with its visual neighbors

    Args:
        cfg (dict): Configuration
        descinfo_folder (str): path to store the descriptors
        image_ids (list[int]): list of image ids
        neighbors (dict[int -> list[int]]): visual neighbors for each image
    Returns:
        matches_folder (str): path to store the computed matches
    """
    weight_path = None if "weight_path" not in cfg else cfg["weight_path"]
    print("[LOG] Start matching 2D lines... (extractor = {0}, matcher = {1}, n_images = {2}, n_neighbors = {3})".format(
        cfg["line2d"]["extractor"]["method"], cfg["line2d"]["matcher"]["method"], len(image_ids), cfg["n_neighbors"]))
    import limap.line2d
    basedir = os.path.join("line_matchings", cfg["line2d"]["detector"]
                           ["method"], "feats_{0}".format(cfg["line2d"]["extractor"]["method"]))
    extractor = limap.line2d.get_extractor(cfg["line2d"]["extractor"], weight_path=weight_path)
    se_match = cfg["skip_exists"] or cfg["line2d"]["matcher"]["skip_exists"]
    matcher = limap.line2d.get_matcher(cfg["line2d"]["matcher"], extractor,
                                       n_neighbors=cfg["n_neighbors"], weight_path=weight_path)
    folder_save = os.path.join(cfg["dir_save"], basedir)
    if not cfg["load_match"]:
        matches_folder = matcher.match_all_neighbors(
            folder_save, image_ids, neighbors, descinfo_folder, skip_exists=se_match)
    else:
        folder_load = os.path.join(cfg["dir_load"], basedir)
        matches_folder = matcher.get_matches_folder(folder_load)
    if ("save_l3dpp_matches" in cfg["line2d"]) and cfg["line2d"]["save_l3dpp_matches"]:
        limapio.save_l3dpp_matches(os.path.join(folder_save, "l3dpp_matches_format"), image_ids, matches_folder)
    return matches_folder


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
            # print(f"orig_size = {orig_size}, new_size = {new_size}, img_id = {img_id}")

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
