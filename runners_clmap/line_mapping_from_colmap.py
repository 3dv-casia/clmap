import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import limap.base as _base
import limap.pointsfm as _psfm
import limap.util.io as limapio
import limap.util.config as cfgutils
from scripts_clmap.undistort_colmap import undistort_colmap

import limap.runners_clmap
from runners_clmap.plp_association import run_plp_association


def read_scene_colmap(cfg, colmap_path, model_path="sparse", image_path="images", n_neighbors=20):
    metainfos_filename = "infos_colmap.npy"
    output_dir = "tmp" if cfg["output_dir"] is None else cfg["output_dir"]
    limapio.check_makedirs(output_dir)

    # undistort colmap reconstruction
    undistort_model_path = os.path.join(output_dir, "undistort_colmap")
    undistort_colmap(os.path.join(colmap_path, image_path), os.path.join(colmap_path, model_path), undistort_model_path)

    imagecols, neighbors, ranges = _psfm.read_infos_colmap(
        cfg["sfm"], undistort_model_path, model_path="sparse", image_path="images", n_neighbors=n_neighbors)
    with open(os.path.join(output_dir, metainfos_filename), 'wb') as f:
        np.savez(f, imagecols_np=imagecols.as_dict(), neighbors=neighbors, ranges=ranges)
    return imagecols, neighbors, ranges, undistort_model_path


def run_colmap_triangulation(cfg, colmap_path, model_path="sparse", image_path="images"):
    '''
    Run triangulation from COLMAP input
    '''
    # undistort colmap reconstruction
    imagecols, neighbors, ranges, undistort_model_path = read_scene_colmap(
        cfg, colmap_path, model_path=model_path, image_path=image_path, n_neighbors=cfg["n_neighbors"])

    # run triangulation
    linetracks, colmap_model_path = limap.runners_clmap.line_triangulation(
        cfg, imagecols, neighbors=neighbors, ranges=ranges, colmap_model_path=os.path.join(undistort_model_path, "sparse"))

    # note:
    # if cfg["triangulation"]["pointsfm_config"]["retriangulation"] is True, the output `colmap_model_path` will be a retriangulated sfm model.
    # if cfg["triangulation"]["pointsfm_config"]["retriangulation"] is False, the output `colmap_model_path` will be `os.path.join(undistort_model_path, "sparse")`.
    return linetracks, colmap_model_path


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description='triangulate and joint optimize 3d lines from COLMAP')
    arg_parser.add_argument('-c', '--tri_config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='triangulation config file')
    arg_parser.add_argument('--tri_default_config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='triangulation default config file')
    arg_parser.add_argument('--plp_config_file', type=str,
                            default='cfgs_clmap/plp_association/default.yaml', help='plp association (i.e., joint optimization) config file')
    arg_parser.add_argument('--plp_default_config_file', type=str,
                            default='cfgs_clmap/plp_association/default.yaml', help='plp association (i.e., joint optimization) default config file')
    arg_parser.add_argument('-a', '--colmap_path',
                            type=str, default=None, help='colmap path')
    arg_parser.add_argument('-m', '--model_path', type=str,
                            default='sparse', help='model path')
    arg_parser.add_argument('-i', '--image_path', type=str,
                            default='images', help='image path')
    arg_parser.add_argument('--npyfolder', type=str, default="tmp",
                            help='folder to load precomputed results')
    arg_parser.add_argument('--max_image_dim', type=int,
                            default=None, help='max image dim')
    arg_parser.add_argument('--info_path', type=str,
                            default=None, help='load precomputed info')
    arg_parser.add_argument('--tri_output_dir', type=str,
                            default="tmp/tri", help='triangulation output dir (unoptimized line map)')
    arg_parser.add_argument('--plp_output_dir', type=str,
                            default="tmp/plp", help='plp association output dir (final optimized line map)')
    arg_parser.add_argument(
        '--use_optim', type=bool, default=True, help="whether to use joint optimization (i.e., plp association)")

    args, unknown = arg_parser.parse_known_args()
    tri_cfg = cfgutils.load_config(
        args.tri_config_file, default_path=args.tri_default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    tri_cfg = cfgutils.update_config(tri_cfg, unknown, shortcuts)
    print(f"unknown = {unknown}")
    print(f"shortcuts = {shortcuts}")
    tri_cfg["colmap_path"] = args.colmap_path
    tri_cfg["image_path"] = args.image_path
    tri_cfg["model_path"] = args.model_path
    tri_cfg["folder_to_load"] = args.npyfolder
    tri_cfg["info_path"] = args.info_path
    if tri_cfg["colmap_path"] is None and tri_cfg["info_path"] is None:
        raise ValueError("Error! colmap_path unspecified.")
    if ("max_image_dim" not in tri_cfg.keys()) or args.max_image_dim is not None:
        tri_cfg["max_image_dim"] = args.max_image_dim
    tri_cfg["output_dir"] = args.tri_output_dir

    plp_cfg = cfgutils.load_config(
        args.plp_config_file, default_path=args.plp_default_config_file)
    plp_cfg = cfgutils.update_config(plp_cfg, unknown, shortcuts)
    plp_cfg["output_dir"] = args.plp_output_dir
    print(f"tri cfg = {tri_cfg}")
    print(f"plp cfg = {plp_cfg}")
    print(f"use_optim = {args.use_optim}")
    return tri_cfg, plp_cfg, args.use_optim


def main():
    tri_cfg, plp_cfg, use_optim = parse_config()

    # triangulation: line detection & matching, proposal generation, best proposal selection, line track building
    linetracks, colmap_model_path = run_colmap_triangulation(
        tri_cfg, tri_cfg["colmap_path"], tri_cfg["model_path"], tri_cfg["image_path"])

    # plp_cfg["visualize"] = False
    plp_cfg["load_dir"] = tri_cfg["output_dir"]
    plp_cfg["load_vpdet"] = True

    # joint optimization
    if use_optim:
        plp_input_dir = os.path.join(tri_cfg["output_dir"], tri_cfg["output_folder"])
        run_plp_association(plp_cfg, plp_input_dir, colmap_model_path)


if __name__ == '__main__':
    main()
