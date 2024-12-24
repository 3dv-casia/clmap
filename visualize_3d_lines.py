import os
import sys
import numpy as np
import copy

import limap.base as _base
import limap.util.io as limapio
import limap.visualize as limapvis
import limap.pointsfm as _psfm


from limap.visualize.vis_lines import open3d_get_cameras


def read_ply(fname):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points = np.stack([x, y, z], axis=1)
    print("number of points: {0}".format(points.shape[0]))
    return points


def parse_args():
    import argparse
    arg_parser = argparse.ArgumentParser(description='visualize 3d lines')
    arg_parser.add_argument('-i', '--input_dir', type=str,
                            help='input line file. Format supported now: .obj, .npy, linetrack folder.')
    arg_parser.add_argument('-nv', '--n_visible_views',
                            type=int, default=2, help='number of visible views')
    arg_parser.add_argument('--imagecols', type=str,
                            default=None, help=".npy file for imagecols")
    arg_parser.add_argument("--metainfos", type=str, default=None,
                            help=".txt file for neighbors and ranges")
    arg_parser.add_argument(
        '--mode', type=str, default="open3d", help="[pyvista, open3d]")
    arg_parser.add_argument('--use_robust_ranges', action='store_true',
                            help="whether to use computed robust ranges")
    arg_parser.add_argument('--scale', type=float, default=1.0,
                            help="scaling both the lines and the camera geometry")
    arg_parser.add_argument('--cam_scale', type=float,
                            default=3.0, help="scale of the camera geometry")
    arg_parser.add_argument('--output_dir', type=str, default=None,
                            help="if set, save the scaled lines in obj format")
    arg_parser.add_argument('--colmap_model_dir', type=str,
                            default=None, help="colmap_model_dir")
    arg_parser.add_argument('--ply_path', type=str,
                            default=None, help="point cloud ply path")
    arg_parser.add_argument('--colmap_sfm_log', type=str,
                            default=None, help="COLMAP SfM log from official tnt website")
    args = arg_parser.parse_args()
    return args


def vis_3d_lines(lines, mode="open3d", ranges=None, scale=1.0):
    if mode == "pyvista":
        limapvis.pyvista_vis_3d_lines(lines, ranges=ranges, scale=scale)
    elif mode == "open3d":
        limapvis.open3d_vis_3d_lines(lines, ranges=ranges, scale=scale)
    else:
        raise NotImplementedError


def vis_reconstruction(linetracks, imagecols, mode="open3d", n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError(
            "Error! Visualization with cameras is only supported with open3d.")
    VisTrack.report()
    VisTrack.vis_reconstruction(
        imagecols, n_visible_views=n_visible_views, ranges=ranges, scale=scale, cam_scale=cam_scale)


def vis_reconstruction_point_line(linetracks, point_tracks, imagecols, mode="open3d", n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError(
            "Error! Visualization with cameras is only supported with open3d.")
    VisTrack.report()
    VisTrack.vis_reconstruction_point_line(
        point_tracks, imagecols, n_visible_views=n_visible_views, ranges=ranges, scale=scale, cam_scale=cam_scale)


def vis_reconstruction_plypoint_line(linetracks, points, imagecols, mode="open3d", n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0):
    if mode == "open3d":
        VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    else:
        raise ValueError(
            "Error! Visualization with cameras is only supported with open3d.")
    VisTrack.report()
    VisTrack.vis_reconstruction_plypoint_line(
        points, imagecols, n_visible_views=n_visible_views, ranges=ranges, scale=scale, cam_scale=cam_scale)


def vis_colmap_sfm_log(log_path):
    with open(log_path, "r") as file:
        Rs = []
        Ts = []
        i = 0
        R = np.zeros((3, 3), dtype=float)
        T = np.zeros((3,), dtype=float)
        for line in file:
            words = line.strip().split(' ')
            if len(words) == 3:
                continue
            if i == 0:
                R[0, 0] = float(words[0])
                R[0, 1] = float(words[1])
                R[0, 2] = float(words[2])
                T[0] = float(words[3])
            elif i == 1:
                R[1, 0] = float(words[0])
                R[1, 1] = float(words[1])
                R[1, 2] = float(words[2])
                T[1] = float(words[3])
            elif i == 2:
                R[2, 0] = float(words[0])
                R[2, 1] = float(words[1])
                R[2, 2] = float(words[2])
                T[2] = float(words[3])
                Rs.append(copy.deepcopy(R))
                Ts.append(copy.deepcopy(T))
            else:
                i = 0
                continue
            i += 1
    # print(Ts)
    # create imagecols
    default_h, default_w = 768, 1024
    h, w = default_h, default_w
    # fov_x = 60 * np.pi / 180  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    fov_x = np.pi / 3  # set fov_x to pi/3 to match DIODE dataset (60 degrees)
    f = w / (2 * np.tan(fov_x / 2))
    default_K = np.array([[f, 0, w / 2],
                          [0, f, h / 2],
                          [0, 0, 1]])
    cameras, camimages = {}, {}
    cameras[0] = _base.Camera("SIMPLE_PINHOLE", default_K, cam_id=0, hw=(h, w))
    for i in range(len(Rs)):
        pose = _base.CameraPose(Rs[i].T, -Rs[i].T @ Ts[i])
        camimage = _base.CameraImage(0, pose, image_name=f"test_{i}.jpg")
        camimages[i] = camimage
    imagecols = _base.ImageCollection(cameras, camimages)
    # vis
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    camera_set = open3d_get_cameras(imagecols, scale_cam_geometry=10.0)
    vis.add_geometry(camera_set)
    vis.run()
    vis.destroy_window()


def main(args):
    if args.colmap_sfm_log is not None:
        vis_colmap_sfm_log(args.colmap_sfm_log)
        return
    lines, linetracks = limapio.read_lines_from_input(args.input_dir)
    ranges = None
    if args.metainfos is not None:
        _, ranges = limapio.read_txt_metainfos(args.metainfos)
    if args.use_robust_ranges:
        ranges = limapvis.compute_robust_range_lines(lines)
    if args.n_visible_views > 2 and linetracks is None:
        raise ValueError("Error! Track information is not available.")
    if args.imagecols is None:
        if linetracks is not None:
            lines = []  # reset
            for track in linetracks:
                if track.count_images() < args.n_visible_views:
                    continue
                lines.append(track.line)
        vis_3d_lines(lines, mode=args.mode, ranges=ranges, scale=args.scale)
    else:
        if (not os.path.exists(args.imagecols)) or (not args.imagecols.endswith('.npy')):
            raise ValueError(
                "Error! Input file {0} is not valid".format(args.imagecols))
        imagecols = _base.ImageCollection(limapio.read_npy(args.imagecols).item())
        if args.colmap_model_dir is None:
            if args.ply_path is not None:
                points = read_ply(args.ply_path)
                vis_reconstruction_plypoint_line(linetracks, points, imagecols, mode=args.mode, n_visible_views=args.n_visible_views,
                                                 ranges=ranges, scale=args.scale, cam_scale=args.cam_scale)
            else:
                vis_reconstruction(linetracks, imagecols, mode=args.mode, n_visible_views=args.n_visible_views,
                                   ranges=ranges, scale=args.scale, cam_scale=args.cam_scale)
        else:
            reconstruction = _psfm.PyReadCOLMAP(args.colmap_model_dir, model_path=None)
            vis_reconstruction_point_line(linetracks, reconstruction["points"], imagecols, mode=args.mode, n_visible_views=args.n_visible_views,
                                          ranges=ranges, scale=args.scale, cam_scale=args.cam_scale)

    if args.output_dir is not None:
        limapio.save_obj(args.output_dir, lines)


if __name__ == '__main__':
    args = parse_args()
    main(args)
