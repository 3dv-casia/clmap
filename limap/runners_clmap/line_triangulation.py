import os
from tqdm import tqdm

import time

import limap.base as _base
import limap.merging as _mrg
import limap.vplib as _vplib
import limap.pointsfm as _psfm
import limap.util.io as limapio
import limap.visualize as limapvis

import _limap._clmap as _clmap
import limap.runners as _runners
import limap.runners_clmap as _runners_clmap


def line_triangulation(cfg, imagecols, neighbors=None, ranges=None, colmap_model_path=None):
    '''
    [CLMAP] Main interface of line triangulation over multi-view images.

    Args:
        cfg (dict): Configuration. Fields refer to :file:`cfgs_clmap/triangulation/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): The image collection corresponding to all the images of interest
        neighbors (dict[int -> list[int]], optional): visual neighbors for each image. By default we compute neighbor information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): robust 3D ranges for the scene. By default we compute range information from the COLMAP triangulation.
        colmap_model_path: undistorted colmap reconstruction path
    Returns:
        linetracks (list[:class:`limap.base.LineTrack`]): list of output 3D line tracks
        colmap_model_path: the colmap reconstruction path offering point sfm information
    '''
    time_dict = dict()
    print("[LOG] Number of images: {0}".format(imagecols.NumImages()))
    cfg = _runners.setup(cfg)
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]

    if not imagecols.IsUndistorted():
        raise RuntimeError("input imagecols must be undistorted")

    # resize cameras
    assert imagecols.IsUndistorted() == True
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    limapio.save_txt_imname_dict(os.path.join(cfg["dir_save"], 'image_list.txt'), imagecols.get_image_name_dict())
    limapio.save_npy(os.path.join(cfg["dir_save"], 'imagecols.npy'), imagecols.as_dict())

    ##########################################################
    # [1] sfm metainfos (neighbors, ranges)
    ##########################################################

    t1 = time.time()
    sfminfos_colmap_folder = None
    if neighbors is None:
        sfminfos_colmap_folder, neighbors, ranges = _runners.compute_sfminfos(cfg, imagecols)
    else:
        limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges)
        neighbors = imagecols.update_neighbors(neighbors)
        for img_id, neighbor in neighbors.items():
            neighbors[img_id] = neighbors[img_id][:cfg["n_neighbors"]]
    limapio.save_txt_metainfos(os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges)
    t2 = time.time()
    time_dict["sfm"] = t2 - t1

    ##########################################################
    # [2] get 2D line segments for each image
    ##########################################################

    t1 = time.time()
    compute_descinfo = (not cfg["triangulation"]["use_exhaustive_matcher"])
    compute_descinfo = (compute_descinfo and (not cfg["load_match"]) and (
        not cfg["load_det"])) or cfg["line2d"]["compute_descinfo"]
    all_2d_segs, descinfo_folder = _runners.compute_2d_segs(cfg, imagecols, compute_descinfo=compute_descinfo)
    all_2d_lines = _base.get_all_lines_2d(all_2d_segs)
    t2 = time.time()
    time_dict["detect 2D lines"] = t2 - t1

    ##########################################################
    # [3] get line matches
    ##########################################################

    t1 = time.time()
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        if not cfg["triangulation"]['match_lines_by_epipolar_IoU']:
            matches_dir = _runners_clmap.compute_matches(cfg, descinfo_folder, imagecols.get_img_ids(), neighbors)
    t2 = time.time()
    time_dict["match 2D lines"] = t2 - t1

    ##########################################################
    # [4] detect VPs
    ##########################################################

    t1 = time.time()
    vpresults = None
    if cfg["triangulation"]["use_vp"]:
        if not cfg["load_vpdet"]:
            print("Detect VPs", flush=True)
            vpdetector = _vplib.get_vp_detector(cfg["triangulation"]["vpdet_config"],
                                                n_jobs=cfg["triangulation"]["vpdet_config"]["n_jobs"])
            vpresults = vpdetector.detect_vp_all_images(all_2d_lines, imagecols.get_map_camviews())
            vp_save_dir = os.path.join(cfg["dir_save"], "vp_results", cfg["line2d"]["detector"]["method"])
            limapio.check_makedirs(vp_save_dir)
            limapio.write_pkl(vpresults, os.path.join(vp_save_dir, "vp_results.pkl"))
        else:
            print("Load VPs", flush=True)
            vp_load_dir = os.path.join(cfg["dir_load"], "vp_results", cfg["line2d"]["detector"]["method"])
            vpresults = limapio.read_pkl(os.path.join(vp_load_dir, "vp_results.pkl"))
    t2 = time.time()
    time_dict["detect VPs"] = t2 - t1

    ##########################################################
    # [5] get 2D point-line bipartites from pointsfm model
    ##########################################################

    t1 = time.time()
    all_bpt2ds, sfm_points = None, None
    if cfg["triangulation"]["use_pointsfm"]:
        # print("colmap_model_path =", colmap_model_path)
        if colmap_model_path is None:
            if cfg["triangulation"]["pointsfm_config"]["colmap_folder"] is None:
                # check if colmap model exists from sfminfos computation
                if cfg["triangulation"]["pointsfm_config"]["reuse_sfminfos_colmap"] and sfminfos_colmap_folder is not None:
                    colmap_model_path = os.path.join(sfminfos_colmap_folder, "sparse")
                    if not _psfm.check_exists_colmap_model(colmap_model_path):
                        colmap_model_path = None
                # retriangulate
                if colmap_model_path is None:
                    colmap_output_path = os.path.join(cfg["dir_save"], "colmap_outputs_junctions")
                    input_neighbors = None
                    if cfg["triangulation"]["pointsfm_config"]["use_neighbors"]:
                        input_neighbors = neighbors
                    _psfm.run_colmap_sfm_with_known_poses(
                        cfg["sfm"], imagecols, output_path=colmap_output_path, skip_exists=cfg["skip_exists"], neighbors=input_neighbors)
                    colmap_model_path = os.path.join(colmap_output_path, "sparse")
            else:
                colmap_model_path = cfg["triangulation"]["pointsfm_config"]["colmap_folder"]
        elif cfg["triangulation"]["pointsfm_config"]["retriangulation"]:
            colmap_output_path = os.path.join(cfg["dir_save"], "colmap_retriangulation_outputs")
            input_neighbors = None
            if cfg["triangulation"]["pointsfm_config"]["use_neighbors"]:
                input_neighbors = neighbors
            _psfm.run_colmap_sfm_with_known_poses(
                cfg["sfm"], imagecols, output_path=colmap_output_path, skip_exists=cfg["skip_exists"], neighbors=input_neighbors)
            colmap_model_path = os.path.join(colmap_output_path, "sparse")
        reconstruction = _psfm.PyReadCOLMAP(colmap_model_path)
        all_bpt2ds, sfm_points = _runners_clmap.compute_2d_bipartites_from_colmap(
            reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"], min_support_images=3)
    t2 = time.time()
    time_dict["compute 2D point-line bipartites"] = t2 - t1

    ##########################################################
    # [6] initialize line mapper
    ##########################################################

    t_begin = time.time()

    line_mapper = _clmap.LineMapper(cfg["triangulation"])
    line_mapper.SetImageCollection(imagecols)
    line_mapper.SetAllLine2D(all_2d_lines)
    line_mapper.SetRanges(ranges)
    if cfg["triangulation"]["use_pointsfm"]:
        line_mapper.SetPLBipartite2d(all_bpt2ds)
        line_mapper.SetSfMPoints(sfm_points)
    if cfg["triangulation"]["use_vp"]:
        line_mapper.SetVPResults(vpresults)

    line_mapper.Initialize()

    # matching
    t1 = time.time()
    if cfg["triangulation"]['match_lines_by_epipolar_IoU']:
        print("Match 2D lines by epipolar IoU", flush=True)
        line_mapper.MatchLines2dByEpipolarIoU(
            neighbors, cfg["triangulation"]['n_matches'], cfg["triangulation"]['th_IoU'])
    t2 = time.time()
    time_dict["match 2D lines by epipolar IoU"] = t2 - t1

    ##########################################################
    # [7] proposal generation
    ##########################################################

    t1, t2 = 0, 0
    print('Generate 3D line segment proposals for each 2D line segment')
    if cfg["triangulation"]['match_lines_by_epipolar_IoU']:
        t1 = time.time()
        line_mapper.TriangulateAllImages()
        t2 = time.time()
        time_dict["generate proposals"] = t2 - t1
    elif cfg["triangulation"]['bi-matching']:
        matches = {}
        for img_id in tqdm(imagecols.get_img_ids()):
            matches[img_id] = limapio.read_npy(os.path.join(
                matches_dir, "matches_{0}.npy".format(img_id))).item()
        line_mapper.LoadBidirectionalMatches(matches)
        t1 = time.time()
        line_mapper.TriangulateAllImages()
        t2 = time.time()
        time_dict["generate proposals"] = t2 - t1
    else:
        t1 = time.time()
        for img_id in tqdm(imagecols.get_img_ids()):
            if cfg["triangulation"]["use_exhaustive_matcher"]:
                line_mapper.TriangulateImageWithExhaustiveMatches(
                    img_id, neighbors[img_id])
            else:
                matches = limapio.read_npy(os.path.join(
                    matches_dir, "matches_{0}.npy".format(img_id))).item()
                line_mapper.TriangulateImageWithInitialMatches(img_id, matches)
        t2 = time.time()
        time_dict["generate proposals"] = t2 - t1
    print(f"number of proposals = {line_mapper.GetNumProposal()}", flush=True)

    ##########################################################
    # [8] best proposal selection
    ##########################################################

    compute_local_best_proposal = (cfg["triangulation"]["initial_global_best_proposal"] ==
                                   "local") or cfg["triangulation"]["use_local_valid_lines2d"] or cfg["triangulation"]["debug_mode"]
    t1 = time.time()
    if compute_local_best_proposal:
        line_mapper.SelectLocalBestProposal()
    t2 = time.time()
    time_dict["select local best proposals"] = t2 - t1

    t1 = time.time()
    line_mapper.InitializeGlobalBestProposal(
        cfg["triangulation"]["initial_global_best_proposal"], cfg["triangulation"]["th_global_score"])

    if cfg["triangulation"]["debug_mode"]:
        print("Evaluate consistency of support relations of local best proposal", flush=True)
        _runners_clmap.eval_support_relations_consistency(line_mapper, "local")
        print("Evaluate consistency of support relations of initial global best proposal", flush=True)
        _runners_clmap.eval_support_relations_consistency(line_mapper, "global")

    num_lines2d_having_prop = line_mapper.GetNumLines2dHavingProposal()

    save_iter_best_prop = False
    if cfg["triangulation"]["debug_mode"]:
        if save_iter_best_prop:
            linetracks = line_mapper.BuildGlobalLineTrack(cfg["triangulation"]["show_valid"])
            linetracks_folder = os.path.join(cfg["dir_save"], "init", cfg["output_folder"])
            limapio.save_folder_linetracks_with_info(
                linetracks_folder, linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)

    _runners_clmap.print_heading1("Select global best proposals iteratively")

    for num_iter in range(cfg["triangulation"]["max_iter_num"]):
        print("-" * 78, flush=True)
        print("iter =", num_iter)
        print("-" * 78, flush=True)

        num_changed_lines2d = line_mapper.SelectGlobalBestProposal(cfg["triangulation"]["th_global_score"])

        ratio = num_changed_lines2d / num_lines2d_having_prop
        print(f"changed ratio = {num_changed_lines2d} / {num_lines2d_having_prop} = {ratio}", flush=True)

        if cfg["triangulation"]["debug_mode"]:
            print(
                "Evaluate consistency of support relations of optimizing global best proposal", flush=True)
            _runners_clmap.eval_support_relations_consistency(line_mapper, "global")

            if save_iter_best_prop:
                linetracks = line_mapper.BuildGlobalLineTrack(cfg["triangulation"]["show_valid"])
                linetracks_folder = os.path.join(cfg["dir_save"], str(num_iter), cfg["output_folder"])
                limapio.save_folder_linetracks_with_info(
                    linetracks_folder, linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)

        if ratio < cfg["triangulation"]["min_change_ratio"]:
            break
    t2 = time.time()
    time_dict["select global best proposals"] = t2 - t1

    ##########################################################
    # [9] line track building
    ##########################################################

    t1 = time.time()
    linetracks = line_mapper.BuildGlobalLineTrack(cfg["triangulation"]["show_valid"])
    t2 = time.time()
    time_dict["build line track"] = t2 - t1

    t1 = time.time()
    if cfg['triangulation']['line_track_building_method'] == "cluster":
        # filtering 2d supports
        linetracks = _mrg.filtertracksbyreprojection(
            linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
        # remerging
        if not cfg["triangulation"]["remerging"]["disable"]:
            linker3d = _base.LineLinker3d(cfg["triangulation"]["remerging"]["linker3d"])
            linetracks = _runners_clmap.remerge_clmap(linker3d, linetracks, num_outliers=cfg["triangulation"]["num_outliers_aggregator"], use_collinearity2D=cfg["triangulation"]["remerging"][
                "use_collinearity2D"], overlap_th=cfg["triangulation"]["max_same_line3D_overlap"], angle_th=cfg["triangulation"]["min_same_line3D_angle"], perp_dist_th=cfg["triangulation"]["min_same_line3D_perp_dist"])
            linetracks = _mrg.filtertracksbyreprojection(
                linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_angular_2d"], cfg["triangulation"]["filtering2d"]["th_perp_2d"])
        linetracks = _mrg.filtertracksbysensitivity(
            linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"], cfg["triangulation"]["filtering2d"]["th_sv_num_supports"])
        linetracks = _mrg.filtertracksbyoverlap(
            linetracks, imagecols, cfg["triangulation"]["filtering2d"]["th_overlap"], cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"])
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
    t2 = time.time()
    time_dict["filter line tracks"] = t2 - t1

    t_end = time.time()
    time_dict["total line triangulation time"] = t_end - t_begin

    ##########################################################
    # [10] output & visualize
    ##########################################################

    # save filtered line tracks according to cfg["n_visible_views"]
    limapio.save_txt_linetracks(os.path.join(cfg["dir_save"], "alltracks_nv{0}.txt".format(
        cfg["n_visible_views"])), linetracks, n_visible_views=cfg["n_visible_views"])

    # save folder-to-linetracks
    # Note: none of the line tracks contained in linetracks_folder are filtered by n_visible_views
    linetracks_folder = os.path.join(cfg["dir_save"], cfg["output_folder"])
    limapio.save_folder_linetracks_with_info(
        linetracks_folder, linetracks, config=cfg, imagecols=imagecols, all_2d_segs=all_2d_segs)

    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(os.path.join(cfg["dir_save"], 'triangulated_lines_nv{0}.obj'.format(
        cfg["n_visible_views"])), VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]))

    # visualize
    if cfg["visualize"]:
        validtracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

        def report_track(track_id):
            limapvis.visualize_line_track(
                imagecols, validtracks[track_id], prefix="track.{0}".format(track_id))
        import pdb
        pdb.set_trace()
        VisTrack.vis_reconstruction(
            imagecols, n_visible_views=cfg["n_visible_views"], width=2)
        pdb.set_trace()

    _runners_clmap.print_heading1("Report running times")
    for k, v in time_dict.items():
        print(f"{k} : {v} seconds")

    return linetracks, colmap_model_path
