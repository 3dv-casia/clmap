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
        description='triangulate 3d lines from COLMAP (without joint optimization)')
    arg_parser.add_argument('-c', '--config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='default config file')
    arg_parser.add_argument('-a', '--colmap_path', type=str, default=None, help='colmap path')
    arg_parser.add_argument('-m', '--model_path', type=str, default='sparse', help='model path')
    arg_parser.add_argument('-i', '--image_path', type=str, default='images', help='image path')
    arg_parser.add_argument('--npyfolder', type=str, default="tmp", help='folder to load precomputed results')
    arg_parser.add_argument('--max_image_dim', type=int, default=None, help='max image dim')
    arg_parser.add_argument('--info_path', type=str, default=None, help='load precomputed info')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["colmap_path"] = args.colmap_path
    cfg["image_path"] = args.image_path
    cfg["model_path"] = args.model_path
    cfg["folder_to_load"] = args.npyfolder
    cfg["info_path"] = args.info_path
    if cfg["colmap_path"] is None and cfg["info_path"] is None:
        raise ValueError("Error! colmap_path unspecified.")
    if ("max_image_dim" not in cfg.keys()) or args.max_image_dim is not None:
        cfg["max_image_dim"] = args.max_image_dim
    print(f"tri cfg = {cfg}")
    return cfg


def main():
    cfg = parse_config()
    run_colmap_triangulation(cfg, cfg["colmap_path"], cfg["model_path"], cfg["image_path"])


if __name__ == '__main__':
    main()
