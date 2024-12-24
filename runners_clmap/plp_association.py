import os
import sys
import numpy as np
import math
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from limap.runners_clmap.plp_association import plp_association
from limap.runners_clmap.visualize_3d_planes_bpt import create_geometries_from_lp_bpt3d_pp_bpt3d, get_textural_and_structural_lines
from limap.runners_clmap.plane_detection import detect_planes
import _limap._clmap as _clmap
import limap.structures as _structures
import limap.vplib as _vplib
import limap.pointsfm as _psfm
import limap.base as _base
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis
import limap.runners_clmap as _runners_clmap
import limap.runners as _runners


def report_vp(vpresults, vptracks, print_pairs=False):
    vptracks_list = []
    if type(vptracks) == dict:
        for k, v in vptracks.items():
            vptracks_list.append(v)
    else:
        vptracks_list = vptracks
    n_pairs_parallel, n_pairs_orthogonal = 0, 0
    for i in range(len(vptracks_list) - 1):
        for j in np.arange(i + 1, len(vptracks_list)):
            cosine = abs(vptracks_list[i].direction @ vptracks_list[j].direction)
            angle = math.acos(min(cosine, 1.0)) * 180.0 / math.pi
            if angle <= 1.0:
                n_pairs_parallel += 1
                if print_pairs:
                    print("[LOG] Parallel pair detected: {0} / {1}, angle = {2:.2f}".format(i, j, angle))
            if angle >= 87.0:
                n_pairs_orthogonal += 1
                if print_pairs:
                    print("[LOG] Orthogonal pair detected: {0} / {1}, angle = {2:.2f}".format(i, j, angle))
    print("[LOG] number of VP tracks: {0}".format(len(vptracks_list)))
    print("[LOG]", [track.length() for track in vptracks_list])
    print("[LOG] parallel pairs: {0}, orthogonal pairs: {1}".format(n_pairs_parallel, n_pairs_orthogonal))


def run_plp_association(cfg, input_folder, colmap_model_path=None):
    """
    Args:
        input_folder: CLMAP/LIMAP folder-to-linetracks
        colmap_model_path: the corresponding COLMAP model path
    """
    print("=" * 78)
    print("Joint optimization with points, lines, planes and VPs")
    print("=" * 78)

    cfg = _runners.setup(cfg)

    ############################################################
    # [1] initialize
    ############################################################

    imagecols = None
    all_2d_lines = dict()
    vpresults = dict()
    pointtracks = dict()
    linetracks = dict()
    all_bpt2ds = dict()
    vptracks = list()
    all_bpt2ds_vp = dict()
    inf_planes3d = dict()
    pp_bpt3d = _clmap.PP_Bipartite3d()
    lp_bpt3d = _clmap.LP_Bipartite3d()

    time_dict = dict()

    limapio.check_makedirs(cfg["output_dir"])
    print("line map input_folder =", input_folder)
    print("line map output_folder =", os.path.join(cfg["output_dir"], cfg["output_folder"]))
    print("plane detection output_folder =", os.path.join(cfg["output_dir"], "plane_detection"), "\n")
    limapio.check_makedirs(os.path.join(cfg["output_dir"], "plane_detection"))

    linetracks, _, imagecols, all_2d_segs = limapio.read_folder_linetracks_with_info(input_folder)
    all_2d_lines = _base.get_all_lines_2d(all_2d_segs)

    lines = [track.line for track in linetracks]

    use_ranges = False
    if use_ranges:
        ranges = limapvis.compute_robust_range_lines(lines)
    else:
        ranges = None

    ############################################################
    # [2] build point tracks
    ############################################################

    t1 = time.time()
    use_plane = cfg["plp_association"]["use_pointplane"] or cfg["plp_association"]["use_lineplane"] or cfg[
        "plp_association"]["use_plane_orthogonality"] or cfg["plp_association"]["use_plane_parallelism"]
    use_point = cfg["plp_association"]["use_point_geometry"] or cfg["plp_association"]["use_pointline"] or use_plane
    if use_point:
        print("colmap_model_path =", colmap_model_path)
        if colmap_model_path is None:
            # given poses from line tracks, perform point sfm triangulation to get point tracks
            colmap_output_path = os.path.join(cfg["output_dir"], "colmap_outputs")
            _psfm.run_colmap_sfm_with_known_poses(
                cfg["sfm"], imagecols, output_path=colmap_output_path, skip_exists=cfg["skip_exists"], neighbors=None)
            colmap_model_path = os.path.join(colmap_output_path, "sparse")
        reconstruction = _psfm.PyReadCOLMAP(colmap_model_path)
        pointtracks_tmp = _psfm.ReadPointTracks(reconstruction, imagecols)  # dict
        # filter pointracks
        pointtracks = {}
        min_support_images = 3
        for ptrack_id, ptrack in pointtracks_tmp.items():
            if ptrack.count_images() >= min_support_images:
                pointtracks[ptrack_id] = ptrack
        # initialize point-line bipartites on 2d for each image
        all_bpt2ds, _ = _runners_clmap.compute_2d_bipartites_from_colmap(
            reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"], min_support_images=min_support_images)
    t2 = time.time()
    time_dict["build point tracks"] = t2 - t1

    ##########################################################
    # [3] detect VPs and build VP tracks
    ##########################################################

    t1 = time.time()
    use_vp = cfg["plp_association"]["use_vpline"] or cfg["plp_association"]["use_vp_orthogonality"] or cfg["plp_association"]["use_vp_collinearity"]
    if use_vp:
        if not cfg["load_vpdet"]:
            print("Detect VPs", flush=True)
            vpdetector = _vplib.get_vp_detector(cfg["plp_association"]["vpdet"],
                                                n_jobs=cfg["plp_association"]["vpdet"]["n_jobs"])
            vpresults = vpdetector.detect_vp_all_images(all_2d_lines, imagecols.get_map_camviews())
            vp_save_dir = os.path.join(cfg["dir_save"], "vp_results", cfg["line2d"]["detector"]["method"])
            limapio.check_makedirs(vp_save_dir)
            limapio.write_pkl(vpresults, os.path.join(vp_save_dir, "vp_results.pkl"))
        else:
            print("Load VPs", flush=True)
            vp_load_dir = os.path.join(cfg["dir_load"], "vp_results", cfg["line2d"]["detector"]["method"])
            vpresults = limapio.read_pkl(os.path.join(vp_load_dir, "vp_results.pkl"))
        print("Build VP tracks", flush=True)
        vptrack_constructor = _vplib.GlobalVPTrackConstructor()
        vptrack_constructor.Init(vpresults)
        vptracks = vptrack_constructor.ClusterLineTracks(linetracks, imagecols)
        all_bpt2ds_vp = _structures.GetAllBipartites_VPLine2d(all_2d_lines, vpresults, vptracks)
    t2 = time.time()
    time_dict["detect VPs and build VPtracks"] = t2 - t1

    ############################################################
    # [4] (a) detect planes from point cloud
    #     (b) build initial point-plane and line-plane associations
    ############################################################

    linetracks_map = {}
    for track_id, line_track in enumerate(linetracks):
        linetracks_map[track_id] = line_track
    linetracks = linetracks_map

    time_dict["detect 3D planes"] = 0  # init

    if use_plane:
        point_cloud_output_path = os.path.join(cfg["output_dir"], "plane_detection", "point_cloud.txt")
        planes_output_path = os.path.join(cfg["output_dir"], "plane_detection", "planes.txt")

        t1 = time.time()
        inf_planes3d, pp_bpt3d, lp_bpt3d = detect_planes(
            cfg, imagecols, pointtracks, linetracks, point_cloud_output_path, planes_output_path, visualize_point_cloud=cfg["visualize"])
        # filter planes by supporting lines
        num_all_planes, num_filtered_planes, inf_planes3d, pp_bpt3d, lp_bpt3d = _clmap.filter_inf_plane3d_with_PP_LP_Bipartite3d(
            inf_planes3d, pp_bpt3d, lp_bpt3d, cfg["plp_association"]["min_support_line"])
        print("[before plane filtering] number of 3D planes =", num_all_planes)
        print("[after plane filtering] number of 3D planes =", num_all_planes - num_filtered_planes, flush=True)
        t2 = time.time()
        time_dict["detect 3D planes"] = t2 - t1

        # visualization before optimization
        if cfg["visualize"]:
            plane_ids = list(inf_planes3d.keys())
            plane_id_to_rgb = limapvis.get_id_to_rgb(plane_ids)
            plane_meshes, line_set, point_cloud = create_geometries_from_lp_bpt3d_pp_bpt3d(
                pp_bpt3d, lp_bpt3d, plane_id_to_rgb)
            print("[before optimization] ratio of 3D lines that have associated 3D planes =",
                  len(line_set.lines) / len(linetracks), f"({len(line_set.lines)} / {len(linetracks)})")
            print("[before optimization] number of 3D planes =", len(plane_meshes), flush=True)

            import open3d as o3d
            # visualize point-plane and line-plane associations
            open3d_geometries = plane_meshes + [line_set, point_cloud]
            o3d.visualization.draw_geometries(open3d_geometries, mesh_show_back_face=True)

            # visualize textural (green) and structural (red) lines
            textural_line_set, structural_line_set = get_textural_and_structural_lines(lp_bpt3d)
            open3d_geometries = [textural_line_set, structural_line_set]
            o3d.visualization.draw_geometries(open3d_geometries)

    if cfg["visualize"]:
        line_tracks_list = [v for k, v in linetracks.items()]
        # visualize points and lines
        VisTrack = limapvis.Open3DTrackVisualizer(line_tracks_list)
        VisTrack.vis_reconstruction_point_line_plane_Visualizer(
            imagecols=None, pointtracks=pointtracks, plane_meshes=None, n_visible_views=cfg['n_visible_views'], ranges=ranges, scale=1.0, cam_scale=2.0)

    ############################################################
    # [5] joint optimization
    ############################################################

    t1 = time.time()
    imagecols, pointtracks, linetracks, vptracks, pl_bpt3d, vpline_bpt3d, inf_planes3d, pp_bpt3d, lp_bpt3d, pl_bpt3d_before_ba, vpline_bpt3d_before_ba = plp_association(
        cfg["plp_association"], imagecols, all_2d_lines, vpresults, pointtracks, linetracks, all_bpt2ds, vptracks, all_bpt2ds_vp, inf_planes3d, pp_bpt3d, lp_bpt3d)
    t2 = time.time()
    time_dict["optimization"] = t2 - t1

    ############################################################
    # [6] output & visualize
    ############################################################

    if cfg["visualize"] and cfg["plp_association"]["use_pointline"]:
        print("show point-line associations before optimization", flush=True)
        limapvis.open3d_draw_bipartite3d_pointline_Visualizer(
            pl_bpt3d_before_ba, ranges=ranges, draw_planes=False, draw_edges=False, imagecols=None, cam_scale=2.0)

    if cfg["visualize"] and cfg["plp_association"]["use_pointline"]:
        print("show point-line associations after optimization", flush=True)
        limapvis.open3d_draw_bipartite3d_pointline_Visualizer(
            pl_bpt3d, ranges=ranges, draw_planes=False, draw_edges=False, imagecols=None, cam_scale=2.0)

    if cfg["visualize"] and cfg["plp_association"]["use_vpline"]:
        report_vp(vpresults, vptracks, print_pairs=True)
        print("show line-vp associations before optimization", flush=True)
        limapvis.open3d_draw_bipartite3d_vpline_Visualizer(
            vpline_bpt3d_before_ba, ranges=ranges, imagecols=None, cam_scale=2.0)

    if cfg["visualize"] and cfg["plp_association"]["use_vpline"]:
        report_vp(vpresults, vptracks, print_pairs=True)
        print("show line-vp associations after optimization", flush=True)
        limapvis.open3d_draw_bipartite3d_vpline_Visualizer(vpline_bpt3d, ranges=ranges, imagecols=None, cam_scale=2.0)

    if use_plane:
        # filter planes
        num_all_planes, num_filtered_planes, inf_planes3d, pp_bpt3d, lp_bpt3d = _clmap.filter_inf_plane3d_with_PP_LP_Bipartite3d(
            inf_planes3d, pp_bpt3d, lp_bpt3d, cfg["plp_association"]["min_support_line"])
        print("[before plane filtering] number of planes =", num_all_planes)
        print("[after plane filtering] number of planes =", num_all_planes - num_filtered_planes)

        # visualization after optimization
        if cfg["visualize"]:
            plane_meshes, line_set, point_cloud = create_geometries_from_lp_bpt3d_pp_bpt3d(
                pp_bpt3d, lp_bpt3d, plane_id_to_rgb)
            print("[after optimization] ratio of 3D lines that have associated 3D planes =",
                  len(line_set.lines) / len(linetracks), f"({len(line_set.lines)} / {len(linetracks)})")
            print("[after optimization] number of 3D planes =", len(plane_meshes))

            import open3d as o3d
            # visualize point-plane and line-plane associations
            open3d_geometries = plane_meshes + [line_set, point_cloud]
            o3d.visualization.draw_geometries(open3d_geometries, mesh_show_back_face=True)

            # visualize textural (green) and structural (red) lines
            textural_line_set, structural_line_set = get_textural_and_structural_lines(lp_bpt3d)
            open3d_geometries = [textural_line_set, structural_line_set]
            o3d.visualization.draw_geometries(open3d_geometries)

            # visualize sfm points
            VisTrack = limapvis.Open3DTrackVisualizer(line_tracks_list)
            VisTrack.vis_reconstruction_point_line_vp_Visualizer(
                imagecols=imagecols, pointtracks=pointtracks, line_vp_bpt3d=None, n_visible_views=cfg['n_visible_views'], ranges=ranges, scale=1.0, cam_scale=2.0, vp_id_to_rgb=None)

            # visualize sfm points, lines and planes
            line_tracks_list = [v for k, v in linetracks.items()]
            VisTrack = limapvis.Open3DTrackVisualizer(line_tracks_list)
            VisTrack.report()
            VisTrack.vis_reconstruction_point_line_plane_Visualizer(
                imagecols=imagecols, pointtracks=pointtracks, plane_meshes=plane_meshes, n_visible_views=cfg['n_visible_views'], ranges=ranges, scale=1.0, cam_scale=2.0)

            # visualize sfm points, lines, and VPs
            VisTrack = limapvis.Open3DTrackVisualizer(line_tracks_list)
            VisTrack.vis_reconstruction_point_line_vp_Visualizer(
                imagecols=imagecols, pointtracks=pointtracks, line_vp_bpt3d=vpline_bpt3d, n_visible_views=cfg['n_visible_views'], ranges=ranges, scale=1.0, cam_scale=2.0, vp_id_to_rgb=None)

            # visualize lines and VPs
            VisTrack = limapvis.Open3DTrackVisualizer(line_tracks_list)
            VisTrack.vis_reconstruction_point_line_vp_Visualizer(
                imagecols=imagecols, pointtracks=None, line_vp_bpt3d=vpline_bpt3d, n_visible_views=cfg['n_visible_views'], ranges=ranges, scale=1.0, cam_scale=2.0, vp_id_to_rgb=None)

    line_tracks_list = [v for k, v in linetracks.items()]
    linetracks = line_tracks_list

    # save filtered line tracks according to cfg["n_visible_views"]
    limapio.save_txt_linetracks(os.path.join(cfg["output_dir"], "alltracks_nv{0}.txt".format(
        cfg["n_visible_views"])), linetracks, n_visible_views=cfg["n_visible_views"])

    # save folder-to-linetracks
    # Note: none of the line tracks contained in linetracks_folder are filtered by n_visible_views
    linetracks_folder = os.path.join(cfg["output_dir"], cfg["output_folder"])
    limapio.save_folder_linetracks_with_info(linetracks_folder, linetracks,
                                             config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)

    print("Report running times:")
    for k, v in time_dict.items():
        print(f"{k} : {v} seconds")

    return imagecols, pointtracks, linetracks, vptracks, pl_bpt3d, vpline_bpt3d, inf_planes3d, pp_bpt3d, lp_bpt3d


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description="Joint optimization using points, lines, planes and VPs.")
    arg_parser.add_argument("-i", "--input_folder", type=str, default=None, help="CLMAP/LIMAP folder-to-linetracks")
    arg_parser.add_argument("--colmap_model_path", type=str, default=None,
                            help="colmap model path storing point tracks (camera.txt/bin, points3D.txt/bin, images.txt/bin)")
    arg_parser.add_argument("-c", "--config_file", type=str,
                            default="cfgs_clmap/plp_association/default.yaml", help="config file")
    arg_parser.add_argument("--default_config_file", type=str,
                            default="cfgs_clmap/triangulation/default.yaml", help="config file")
    arg_parser.add_argument("--output_dir", type=str, default="tmp/plp", help="folder to save")
    arg_parser.add_argument("--output_folder", type=str, default="finaltracks", help="output filename")
    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts["-nv"] = "--n_visible_views"
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["input_folder"] = args.input_folder
    cfg["colmap_model_path"] = args.colmap_model_path
    cfg["output_dir"] = args.output_dir
    if cfg["output_dir"] is None:
        cfg["output_dir"] = os.path.dirname(cfg["input_folder"])
    cfg["output_folder"] = args.output_folder
    print(f"plp association cfg: {cfg}")
    return cfg


if __name__ == "__main__":
    cfg = parse_config()
    imagecols, pointtracks, linetracks, vptracks, pl_bpt3d, vpline_bpt3d, inf_planes3d, pp_bpt3d, lp_bpt3d = run_plp_association(
        cfg, cfg["input_folder"], cfg["colmap_model_path"])
