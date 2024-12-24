import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
import ast

import limap.evaluation as _eval
import limap.util.config as cfgutils
import limap.util.io as limapio

from runners_clmap.hypersim.Hypersim import Hypersim

import matplotlib.pyplot as plt


# hypersim
MPAU = 0.02539999969303608


def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')


def plot_curve(fname, thresholds, data):
    plt.plot(thresholds, data)
    plt.savefig(fname)


def visualize_error_to_GT(evaluator, lines, threshold):
    # get inlier and outlier segments
    inlier_lines = evaluator.ComputeInlierSegs(lines, threshold)
    outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)

    # visualize
    import limap.visualize as limapvis
    import open3d as o3d
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)
    inlier_line_set = limapvis.open3d_get_line_set(
        inlier_lines, color=[0., 1., 0.])
    vis.add_geometry(inlier_line_set)
    outlier_line_set = limapvis.open3d_get_line_set(
        outlier_lines, color=[1., 0., 0.])
    vis.add_geometry(outlier_line_set)
    vis.run()
    vis.destroy_window()


def report_error_to_GT(evaluator, lines, vis_err_th=None):
    # [optional] visualize
    if vis_err_th is not None:
        visualize_error_to_GT(evaluator, lines, vis_err_th)
    lengths = np.array([line.length() for line in lines])
    sum_length = lengths.sum()
    thresholds = [0.001, 0.005, 0.01]
    list_recall, list_precision = [], []
    for threshold in thresholds:
        ratios = np.array([evaluator.ComputeInlierRatio(line, threshold) for line in lines])
        length_recall = (lengths * ratios).sum()
        list_recall.append(length_recall)
        precision = 100 * (ratios > 0).astype(int).sum() / ratios.shape[0]
        list_precision.append(precision)
    print("R: recall, P: precision")
    for idx, threshold in enumerate(thresholds):
        print("R / P at {0}mm: {1:.2f} / {2:.2f}".format(int(threshold * 1000), list_recall[idx], list_precision[idx]))
    return evaluator, thresholds, list_recall, list_precision


def read_ply(fname):
    from plyfile import PlyData, PlyElement
    plydata = PlyData.read(fname)
    x = np.asarray(plydata.elements[0].data['x'])
    y = np.asarray(plydata.elements[0].data['y'])
    z = np.asarray(plydata.elements[0].data['z'])
    points = np.stack([x, y, z], axis=1)
    print("number of points: {0}".format(points.shape[0]))
    return points


def write_ply(fname, points):
    from plyfile import PlyData, PlyElement
    points = [(points[i, 0], points[i, 1], points[i, 2])
              for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=True).write(fname)


def report_error_to_mesh(mesh_fname, lines, vis_err_th=None):
    evaluator = _eval.MeshEvaluator(mesh_fname, MPAU)
    return report_error_to_GT(evaluator, lines, vis_err_th=vis_err_th)


def report_error_to_point_cloud(points, lines, kdtree_dir=None, vis_err_th=None):
    # CLMAP: fix possible bugs
    # evaluator = _eval.PointCloudEvaluator(points, vis_err_th=vis_err_th)
    evaluator = _eval.PointCloudEvaluator(points)
    if kdtree_dir is None:
        evaluator.Build()
        evaluator.Save('tmp/kdtree.bin')
    else:
        evaluator.Load(kdtree_dir)
    # CLMAP: fix possible bugs.
    # return report_error_to_GT(evaluator, lines)
    return report_error_to_GT(evaluator, lines, vis_err_th=vis_err_th)


def eval_hypersim_RP(cfg, lines, dataset_hypersim, scene_id, cam_id=0, vis_err_th=None, visualize_path="tmp"):
    """
    Returns:
        thresholds: [0.001, 0.005, 0.01]
        list_recall: [R at 1mm, R at 5mm, R at 10mm]
        list_precision: [P at 1mm, P at 5mm, P at 10mm]
    """
    # set scene id
    dataset_hypersim.set_scene_id(scene_id)
    dataset_hypersim.set_max_dim(cfg["max_image_dim"])
    dataset_hypersim.load_cameras(cam_id=cam_id)

    # generate image indexes
    index_list = np.arange(0, cfg["input_n_views"], cfg["input_stride"]).tolist()
    index_list = dataset_hypersim.filter_index_list(index_list, cam_id=cam_id)

    if cfg["mesh_dir"] is not None:
        # eval w.r.t mesh
        print("evaluate w.r.t mesh")
        evaluator, thresholds, list_recall, list_precision = report_error_to_mesh(
            cfg["mesh_dir"], lines, vis_err_th=vis_err_th)
    elif cfg["pc_dir"] is not None:
        print("evaluate w.r.t provided point cloud")
        points = read_ply(cfg["pc_dir"])
        evaluator, thresholds, list_recall, list_precision = report_error_to_point_cloud(
            points, lines, kdtree_dir=cfg["kdtree_dir"], vis_err_th=vis_err_th)
    else:  # eval w.r.t point cloud
        print("evaluate w.r.t visible GT point cloud (generated from GT depth map and GT poses)")
        points = dataset_hypersim.get_point_cloud_from_list(index_list, cam_id=cam_id)
        evaluator, thresholds, list_recall, list_precision = report_error_to_point_cloud(
            points, lines, kdtree_dir=cfg["kdtree_dir"], vis_err_th=vis_err_th)
        # visualize GT points and reconstructed lines together
        if cfg["visualize"]:
            import open3d as o3d
            from limap.runners_clmap.visualize_3d_planes_bpt import create_line_set, create_point_cloud
            point_list = [points[i, :] for i in range(points.shape[0])]
            print("number of points =", len(point_list))
            print("number of lines =", len(lines))
            point_cloud = create_point_cloud(point_list, colors=[0.8, 0.8, 0.8])
            line_set = create_line_set(lines, colors=[0.0, 1.0, 0.0])
            o3d.visualization.draw_geometries([point_cloud, line_set])
    if cfg["visualize"]:
        vis_thresholds = np.arange(1, 11, 1) * 0.001
        for threshold in tqdm(vis_thresholds.tolist()):
            inlier_lines = evaluator.ComputeInlierSegs(lines, threshold)
            inlier_lines_np = np.array([line.as_array() for line in inlier_lines])
            limapio.check_makedirs(visualize_path)
            # limapio.save_obj("tmp/inliers_th_{0:.4f}.obj".format(threshold), inlier_lines_np)
            visualize_obj_file = os.path.join(visualize_path, "inliers_th_{0:.4f}.obj".format(threshold))
            limapio.save_obj(visualize_obj_file, inlier_lines_np)
            outlier_lines = evaluator.ComputeOutlierSegs(lines, threshold)
            outlier_lines_np = np.array([line.as_array() for line in outlier_lines])
            # limapio.save_obj("tmp/outliers_th_{0:.4f}.obj".format(threshold), outlier_lines_np)
            visualize_obj_file = os.path.join(visualize_path, "outliers_th_{0:.4f}.obj".format(threshold))
            limapio.save_obj(visualize_obj_file, outlier_lines_np)

    return thresholds, list_recall, list_precision


def evaluate_single_hypersim(cfg, vis_err_th=None, visualize_path="tmp"):
    print("-" * 78, flush=True)
    print("Start evaluating a 3D line map on Hypersim dataset")
    print("-" * 78, flush=True)
    print("scene_id =", cfg["scene_id"])
    print("cam_id =", cfg["cam_id"])
    print("data_dir =", cfg["data_dir"])
    print("kdtree_dir =", cfg["kdtree_dir"])
    print("input_path =", cfg["input_path"])
    print("nv =", cfg["n_visible_views"])
    print("vis_err_th =", vis_err_th)
    print("visualize =", cfg["visualize"])
    print("visualize_path =", visualize_path, flush=True)

    init_workspace()
    dataset_hypersim = Hypersim(cfg["data_dir"])

    # read lines and tracks
    lines, linetracks = limapio.read_lines_from_input(cfg["input_path"])
    num_lines = len(lines)
    if linetracks is not None:
        lines = [track.line for track in linetracks if track.count_images() >= cfg["n_visible_views"]]
        linetracks = [track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    # eval recall and precision
    thresholds, list_recall, list_precision = eval_hypersim_RP(
        cfg, lines, dataset_hypersim, cfg["scene_id"], cam_id=cfg["cam_id"], vis_err_th=vis_err_th, visualize_path=visualize_path)

    # report track quality
    sup_image_counts_mean = None
    sup_line_counts_mean = None
    num_nv_lines = None
    if linetracks is not None:
        sup_image_counts = np.array([track.count_images() for track in linetracks])
        sup_line_counts = np.array([track.count_lines() for track in linetracks])
        sup_image_counts_mean = sup_image_counts.mean()
        sup_line_counts_mean = sup_line_counts.mean()
        num_nv_lines = len(lines)
        print("supporting images / lines: ({0:.2f} / {1:.2f})".format(sup_image_counts_mean, sup_line_counts_mean))
        print(f"number of 3D line segments (nv >= {cfg['n_visible_views']}) = {num_nv_lines}")
    print("number of 3D line segments (nv >= 0) =", num_lines)
    return thresholds, list_recall, list_precision, sup_image_counts_mean, sup_line_counts_mean, num_nv_lines, num_lines


def evaluate_multiple_hypersim(cfg, input_path_list, vis_err_th=None, visualize_dir="tmp"):
    print("=" * 78, flush=True)
    print("Start evaluating multiple 3D line maps on Hypersim dataset")
    print("=" * 78, flush=True)

    list_recall_list = []
    list_precision_list = []
    sup_image_counts_mean_list = []
    sup_line_counts_mean_list = []
    num_nv_lines_list = []
    num_lines_list = []
    thresholds = None
    for input_idx, input in enumerate(input_path_list):
        cfg["scene_id"] = input[0]
        cfg["input_path"] = input[1]
        visualize_path = os.path.join(visualize_dir, str(input_idx)) if cfg["visualize"] else None

        thresholds, list_recall, list_precision, sup_image_counts_mean, sup_line_counts_mean, num_nv_lines, num_lines = evaluate_single_hypersim(
            cfg, vis_err_th=vis_err_th, visualize_path=visualize_path)
        list_recall_list.append(list_recall)
        list_precision_list.append(list_precision)
        num_lines_list.append(num_lines)
        if sup_image_counts_mean is not None:  # if cfg["input_path"] has line tracks information
            assert sup_line_counts_mean is not None
            assert num_nv_lines is not None
            sup_image_counts_mean_list.append(sup_image_counts_mean)
            sup_line_counts_mean_list.append(sup_line_counts_mean)
            num_nv_lines_list.append(num_nv_lines)

    # compute average results
    list_recall_mean = np.mean(np.array(list_recall_list), axis=0)
    list_precision_mean = np.mean(np.array(list_precision_list), axis=0)
    num_lines_mean = np.mean(np.array(num_lines_list))
    if len(sup_image_counts_mean_list) != 0:
        assert len(sup_line_counts_mean_list) != 0
        assert len(num_nv_lines_list) != 0
        sup_image_counts_mean_mean = np.mean(np.array(sup_image_counts_mean_list))
        sup_line_counts_mean_mean = np.mean(np.array(sup_line_counts_mean_list))
        num_nv_lines_mean = np.mean(np.array(num_nv_lines_list))

    print("-" * 78, flush=True)
    print("Report the average results of the multiple 3D line maps on Hypersim dataset")
    print("-" * 78, flush=True)
    print("R: recall, P: precision")
    for idx, threshold in enumerate(thresholds):
        print("mean R / P at {0}mm: {1:.2f} / {2:.2f}".format(int(threshold * 1000),
              list_recall_mean[idx], list_precision_mean[idx]))
    if len(sup_image_counts_mean_list) != 0:
        print(
            "mean supporting images / lines: ({0:.2f} / {1:.2f})".format(sup_image_counts_mean_mean, sup_line_counts_mean_mean))
        print(f"mean number of 3D line segments (nv >= {cfg['n_visible_views']}) = {num_nv_lines_mean}")
    print("mean number of 3D line segments (nv >= 0) =", num_lines_mean)


def evaluate_multiple_hypersim_visible_GT_point_cloud(cfg):
    # add additional config
    cfg["mesh_dir"] = None
    cfg["pc_dir"] = None
    cfg["kdtree_dir"] = None
    visualize_error_threshold = None if not cfg["visualize"] else cfg["visualize_error_threshold"]

    evaluate_multiple_hypersim(cfg, cfg["input_path_list"],
                               vis_err_th=visualize_error_threshold, visualize_dir=cfg["visualize_dir"])


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Evaluate multiple 3d line maps on Hypersim dataset for CLMAP, LIMAP or L3D++. (only support visible GT point cloud (generated from GT depth map and GT poses) now)")
    arg_parser.add_argument("-i", "--input_path_list", type=str, required=True, help="scene id & 3D line map path list")
    arg_parser.add_argument('-c', '--config_file', type=str,
                            default='cfgs_clmap/eval/hypersim.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str,
                            default='cfgs_clmap/eval/default.yaml', help='default config file')
    arg_parser.add_argument('--visualize_error_threshold',
                            type=float, default=0.005, help="visualize threshold")
    arg_parser.add_argument('--visualize_dir', type=str,
                            default='tmp', help='If cfg["visualize"] == True, the evaluation visualization of i-th input line map in cfg["input_path_list"] will be saved in "visualize_dir/(i-1)".')
    args, unknown = arg_parser.parse_known_args()
    cfg = cfgutils.load_config(args.config_file, default_path=args.default_config_file)
    shortcuts = dict()
    shortcuts['-nv'] = '--n_visible_views'
    cfg = cfgutils.update_config(cfg, unknown, shortcuts)
    cfg["visualize_error_threshold"] = args.visualize_error_threshold
    cfg["visualize_dir"] = args.visualize_dir
    cfg["input_path_list"] = ast.literal_eval(args.input_path_list)
    print(f"eval hypersim cfg: {cfg}")
    return cfg


def main():
    # line map format:
    #   CLMAP/LIMAP: *.npy (no line tracks) or *.obj (no line tracks) or folder-to-linetracks (contains line tracks).
    #   L3D++: .txt (contains line tracks).
    # Note:
    #   If you want to evaluate the quality of line tracks, the format of input line map must be folder-to-linetracks for CLMAP/LIMAP or .txt for L3D++.

    cfg = parse_config()

    # evaluate 3d line maps in `cfg["input_path_list"]` and report the average results using visible GT point cloud generated from GT depth map
    evaluate_multiple_hypersim_visible_GT_point_cloud(cfg)

    # TODO: evaluation using GT mesh and GT point cloud.


if __name__ == '__main__':
    main()
