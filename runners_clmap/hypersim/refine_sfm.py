import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Hypersim import Hypersim
from loader import read_scene_hypersim

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
import limap.util.config as cfgutils
import limap.util.evaluation as limapeval
import limap.util.io as limapio
import limap.runners
import limap.optimize
import limap.pointsfm

from runners_clmap.colmap_triangulation import run_colmap_triangulation
from runners_clmap.plp_association import run_plp_association


def run_scene_hypersim(imagecols_output_dir, tri_cfg, plp_cfg, hypersim_dataset, scene_id, cam_id=0):
    imagecols_gt = read_scene_hypersim(
        tri_cfg, hypersim_dataset, scene_id, cam_id=cam_id, load_depth=False)

    ###########################################################################
    # [1] run colmap to obtain point tracks
    ###########################################################################

    import limap.pointsfm as _psfm
    tri_cfg = limap.runners.setup(tri_cfg)
    # global_dir_save = tri_cfg["dir_save"]
    limapio.check_makedirs(imagecols_output_dir)
    limapio.save_npy(os.path.join(
        imagecols_output_dir, "imagecols_gt.npy"), imagecols_gt)
    colmap_path = os.path.join(tri_cfg["dir_save"], "colmap_sfm")

    # if the following code had been executed, you can comment it to use the last COLMAP results
    _psfm.run_colmap_sfm(tri_cfg["sfm"], imagecols_gt, output_path=colmap_path,
                         skip_exists=tri_cfg["skip_exists"], map_to_original_image_names=False)

    imagecols, _, _ = _psfm.read_infos_colmap(tri_cfg["sfm"], colmap_path, model_path="sparse/0", image_path="images")
    limapio.save_npy(os.path.join(imagecols_output_dir, "imagecols_sfm.npy"), imagecols)

    ###########################################################################
    # [2] given colmap sfm poses, run CLMAP to obtain line tracks
    ###########################################################################

    colmap_folder = os.path.join(colmap_path, "sparse/0")

    # run CLMAP
    tri_cfg["info_path"] = None
    linetracks, undistort_model_path = run_colmap_triangulation(
        tri_cfg, colmap_path, model_path="sparse/0", image_path="images")
    plp_input_dir = os.path.join(tri_cfg["output_dir"], tri_cfg["output_folder"])
    _, _, linetracks, _, _, _, _, _, _ = run_plp_association(plp_cfg, plp_input_dir, colmap_folder)

    ###########################################################################
    # [3] run point-line joint BA to optimize poses
    ###########################################################################

    reconstruction = limap.pointsfm.PyReadCOLMAP(colmap_folder)
    pointtracks = limap.pointsfm.ReadPointTracks(reconstruction, imagecols)
    cfg_ba = limap.optimize.HybridBAConfig()
    ba_engine = limap.optimize.solve_hybrid_bundle_adjustment(cfg_ba, imagecols, pointtracks, linetracks)
    new_imagecols = ba_engine.GetOutputImagecols()

    # save optimized poses
    limapio.save_npy(os.path.join(
        imagecols_output_dir, "imagecols_optimized.npy"), new_imagecols)

    ###########################################################################
    # [4] evaluate absolute and relative pose error using GT poses
    ###########################################################################

    print("=" * 78)
    print("Evaluate absolute and relative pose error using GT poses")
    print("=" * 78)
    print("eval scene_id =", scene_id)
    val_image_ids = limapeval.get_valid_image_ids(imagecols, imagecols_gt, 1, max_error=0.01)
    print("len(val_image_ids) =", len(val_image_ids))
    print("len(ori_image_ids) =", len(imagecols.get_img_ids()))

    # evaluate absolute pose error
    trans_errs_orig, rot_errs_orig = limapeval.eval_valid_imagecols(val_image_ids, imagecols, imagecols_gt)
    trans_errs, rot_errs = limapeval.eval_valid_imagecols(val_image_ids, new_imagecols, imagecols_gt)

    print("(median) original: trans: {0:.6f}, rot: {1:.6f}".format(
        np.median(trans_errs_orig), np.median(rot_errs_orig)))
    print("(mean) original: trans: {0:.6f}, rot: {1:.6f}\n".format(
        np.mean(trans_errs_orig), np.mean(rot_errs_orig)))
    print("(median) optimized: trans: {0:.6f}, rot: {1:.6f}".format(
        np.median(trans_errs), np.median(rot_errs)))
    print("(mean) optimized: trans: {0:.6f}, rot: {1:.6f}\n".format(
        np.mean(trans_errs), np.mean(rot_errs)))

    # evaluate relative pose error
    rel_errs_orig, rel_rot_errs_orig, rel_t_angle_orig = limapeval.eval_valid_imagecols_relpose(
        val_image_ids, imagecols, imagecols_gt, enable_logging=True)
    rel_errs, rel_rot_errs, rel_t_angle = limapeval.eval_valid_imagecols_relpose(
        val_image_ids, new_imagecols, imagecols_gt, enable_logging=True)

    print("(median) original: rel_errs: {0:.6f}".format(
        np.median(rel_errs_orig)))
    print("(mean) original: rel_errs: {0:.6f}\n".format(
        np.mean(rel_errs_orig)))
    print("(median) original: rel_rot_errs: {0:.6f}".format(
        np.median(rel_rot_errs_orig)))
    print("(mean) original: rel_rot_errs: {0:.6f}\n".format(
        np.mean(rel_rot_errs_orig)))
    print("(median) original: rel_t_angle: {0:.6f}".format(
        np.median(rel_t_angle_orig)))
    print("(mean) original: rel_t_angle: {0:.6f}\n".format(
        np.mean(rel_t_angle_orig)))

    print("(median) optimized: rel_errs: {0:.6f}".format(np.median(rel_errs)))
    print("(mean) optimized: rel_errs: {0:.6f}\n".format(np.mean(rel_errs)))
    print("(median) optimized: rel_rot_errs: {0:.6f}".format(
        np.median(rel_rot_errs)))
    print("(mean) optimized: rel_rot_errs: {0:.6f}\n".format(
        np.mean(rel_rot_errs)))
    print("(median) optimized: rel_t_angle: {0:.6f}".format(
        np.median(rel_t_angle)))
    print("(mean) optimized: rel_t_angle: {0:.6f}\n".format(
        np.mean(rel_t_angle)))

    trans_errs_orig, rot_errs_orig, acc_orig = limapeval.eval_imagecols_with_acc_percentage(imagecols, imagecols_gt)
    trans_errs_new, rot_errs_new, acc_new = limapeval.eval_imagecols_with_acc_percentage(new_imagecols, imagecols_gt)
    print(f"[median] rot_errs_orig = {np.median(rot_errs_orig)}, trans_errs_orig = {np.median(trans_errs_orig)}")
    print(f"[median] rot_errs_new = {np.median(rot_errs_new)}, trans_errs_new = {np.median(trans_errs_new)}")


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description='refining COLMAP SfM poses using line tracks from CLMAP')
    arg_parser.add_argument('-c', '--tri_config_file', type=str,
                            default='cfgs_clmap/triangulation/hypersim.yaml', help='triangulation config file')
    arg_parser.add_argument('--tri_default_config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='triangulation default config file')
    arg_parser.add_argument('--plp_config_file', type=str,
                            default='cfgs_clmap/plp_association/default.yaml', help='plp association (joint optimization) config file')
    arg_parser.add_argument('--plp_default_config_file', type=str,
                            default='cfgs_clmap/plp_association/default.yaml', help='plp association (joint optimization) default config file')
    arg_parser.add_argument('--npyfolder', type=str, default=None,
                            help='folder to load precomputed results')
    arg_parser.add_argument('--tri_output_dir', type=str,
                            default="tmp/tri", help='triangulation output dir (unoptimized line map)')
    arg_parser.add_argument('--plp_output_dir', type=str,
                            default="tmp/plp", help='plp association output dir (final optimized line map)')
    arg_parser.add_argument('--imagecols_output_dir', type=str,
                            default="tmp/imagecols", help='imagecols output dir')

    args, unknown = arg_parser.parse_known_args()
    tri_cfg = cfgutils.load_config(
        args.tri_config_file, default_path=args.tri_default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-sid'] = '--scene_id'
    tri_cfg = cfgutils.update_config(tri_cfg, unknown, shortcuts)
    tri_cfg["folder_to_load"] = args.npyfolder
    if tri_cfg["folder_to_load"] is None:
        tri_cfg["folder_to_load"] = os.path.join(
            "precomputed", "hypersim", tri_cfg["scene_id"])
    tri_cfg["output_dir"] = args.tri_output_dir

    plp_cfg = cfgutils.load_config(
        args.plp_config_file, default_path=args.plp_default_config_file)
    plp_cfg = cfgutils.update_config(plp_cfg, unknown, shortcuts)
    plp_cfg["output_dir"] = args.plp_output_dir

    return args.imagecols_output_dir, tri_cfg, plp_cfg


def main():
    imagecols_output_dir, tri_cfg, plp_cfg = parse_config()
    dataset = Hypersim(tri_cfg["data_dir"])
    run_scene_hypersim(imagecols_output_dir, tri_cfg, plp_cfg, dataset,
                       tri_cfg["scene_id"], cam_id=tri_cfg["cam_id"])


if __name__ == '__main__':
    main()
