
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import limap.runners_clmap
import limap.util.config as cfgutils
from loader import read_scene_hypersim
from Hypersim import Hypersim


def run_scene_hypersim(cfg, dataset, scene_id, cam_id=0, image_ids=None):
    imagecols = read_scene_hypersim(
        cfg, dataset, scene_id, cam_id=cam_id, load_depth=False, image_ids=image_ids)
    linetracks, colmap_model_path = limap.runners_clmap.line_triangulation(
        cfg, imagecols)
    return linetracks, colmap_model_path


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(description='triangulate 3d lines')
    arg_parser.add_argument('-c', '--config_file', type=str,
                            default='cfgs_clmap/triangulation/hypersim.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str,
                            default='cfgs_clmap/triangulation/default.yaml', help='default config file')
    arg_parser.add_argument('--npyfolder', type=str, default=None,
                            help='folder to load precomputed results')
    arg_parser.add_argument('--image_ids', nargs='*', default=None, help='image ids')

    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(
        args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    shortcuts['-nn'] = '--n_neighbors'
    shortcuts['-sid'] = '--scene_id'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["folder_to_load"] = args.npyfolder
    if cfg["folder_to_load"] is None:
        cfg["folder_to_load"] = os.path.join(
            "precomputed", "hypersim", cfg["scene_id"])
    image_ids = args.image_ids
    if image_ids is not None:
        image_ids = list(map(int, args.image_ids))
    print(f"triangulation cfg: {cfg}")
    print(f"image_ids: {image_ids}")
    return cfg, image_ids


def main():
    cfg, image_ids = parse_config()
    dataset = Hypersim(cfg["data_dir"])
    run_scene_hypersim(cfg, dataset, cfg["scene_id"], cam_id=cfg["cam_id"], image_ids=image_ids)


if __name__ == '__main__':
    main()
