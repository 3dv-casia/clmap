import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tqdm import tqdm
import ast

import limap.base as _base
import limap.evaluation as _eval
import limap.util.config as cfgutils
import limap.util.io as limapio
import limap.visualize as limapvis

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_curve(fname, thresholds, data):
    plt.plot(thresholds, data)
    plt.savefig(fname)


def transform_lines(fname, lines):
    with open(fname, 'r') as f:
        flines = f.readlines()
    mat = []
    for fline in flines:
        fline = fline.strip().split()
        mat.append([float(k) for k in fline])
    trans = np.array(mat)
    new_lines = []
    for line in lines:
        newstart = trans[:3, :3] @ line.start + trans[:3, 3]
        newend = trans[:3, :3] @ line.end + trans[:3, 3]
        newline = _base.Line3d(newstart, newend)
        new_lines.append(newline)
    return new_lines


def init_workspace():
    if not os.path.exists('tmp'):
        os.makedirs('tmp')


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
    thresholds = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    list_recall, list_precision = [], []
    for threshold in thresholds:
        ratios = np.array([evaluator.ComputeInlierRatio(
            line, threshold) for line in lines])
        length_recall = (lengths * ratios).sum()
        list_recall.append(length_recall)
        precision = 100 * (ratios > 0).astype(int).sum() / ratios.shape[0]
        list_precision.append(precision)
    for idx, threshold in enumerate(thresholds):
        print("R / P at {0}mm: {1:.2f} / {2:.2f}".format(int(threshold * 1000), list_recall[idx], list_precision[idx]))
    return evaluator, thresholds, list_recall, list_precision


def report_pc_recall_for_GT(evaluator, lines):
    '''
    To compute invert point recall
    '''
    thresholds = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
    point_dists = evaluator.ComputeDistsforEachPoint(lines)
    # point_dists = evaluator.ComputeDistsforEachPoint_KDTree(lines)
    point_dists = np.array(point_dists)
    n_points = point_dists.shape[0]
    print("Compute point recall metrics.")
    for threshold in thresholds.tolist():
        num_inliers = (point_dists < threshold).sum()
        point_recall = 100 * num_inliers / n_points
        print("{0:.0f}mm, inliers = {1}, point recall = {2:.2f}".format(
            int(threshold * 1000), num_inliers, point_recall))
    return evaluator


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
    evaluator = _eval.PointCloudEvaluator(points)
    if kdtree_dir is None:
        evaluator.Build()
        evaluator.Save('tmp/kdtree.bin')
    else:
        evaluator.Load(kdtree_dir)
    # evaluator = report_pc_recall_for_GT(evaluator, lines)
    evaluator = report_error_to_GT(evaluator, lines, vis_err_th=vis_err_th)
    return evaluator


def eval_tnt_RP(cfg, lines, ref_lines=None, vis_err_th=None, visualize_path='tmp'):
    # eval w.r.t psuedo gt lines
    if ref_lines is not None:
        pass
    if cfg["mesh_dir"] is not None:
        # eval w.r.t mesh
        print("evaluate w.r.t mesh")
        evaluator, thresholds, list_recall, list_precision = report_error_to_mesh(
            cfg["mesh_dir"], lines, vis_err_th=vis_err_th)
    elif cfg["pc_dir"] is not None:
        print("evaluate w.r.t point cloud")
        points = read_ply(cfg["pc_dir"])
        if cfg["use_ranges"]:
            n_lines = len(lines)
            ranges = [points.min(0) - 0.1, points.max(0) + 0.1]
            lines = [line for line in lines if limapvis.test_line_inside_ranges(line, ranges)]
            print("Filtering by range: {0} / {1}".format(len(lines), n_lines))
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
    else:
        raise NotImplementedError
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


def evaluate_single_tnt(cfg, vis_err_th=None, visualize_path="tmp"):
    print("-" * 78, flush=True)
    print("Start evaluating a 3D line map on Tanks and Temples dataset")
    print("-" * 78, flush=True)
    print("pc_dir =", cfg["pc_dir"])
    print("transform_txt =", cfg["transform_txt"])
    print("kdtree_dir =", cfg["kdtree_dir"])
    print("input_path =", cfg["input_path"])
    print("nv =", cfg["n_visible_views"])
    print("vis_err_th =", vis_err_th)
    print("visualize =", cfg["visualize"])
    print("visualize_path =", visualize_path, flush=True)

    init_workspace()

    # read lines and tracks
    lines, linetracks = limapio.read_lines_from_input(cfg["input_path"])
    num_lines = len(lines)
    if linetracks is not None:
        lines = [track.line for track in linetracks if track.count_images()
                 >= cfg["n_visible_views"]]
        linetracks = [
            track for track in linetracks if track.count_images() >= cfg["n_visible_views"]]

    # align linetracks to GT poses
    if (cfg["transform_txt"] is not None) and (len(cfg["transform_txt"]) > 0):
        lines = transform_lines(cfg["transform_txt"], lines)
        # limapio.save_obj('tmp/lines_transform.obj', lines)

    # eval recall and precision
    thresholds, list_recall, list_precision = eval_tnt_RP(
        cfg, lines, vis_err_th=vis_err_th, visualize_path=visualize_path)

    # report track quality
    sup_image_counts_mean = None
    sup_line_counts_mean = None
    num_nv_lines = None
    if linetracks is not None:
        sup_image_counts = np.array([track.count_images()
                                    for track in linetracks])
        sup_line_counts = np.array([track.count_lines()
                                   for track in linetracks])
        sup_image_counts_mean = sup_image_counts.mean()
        sup_line_counts_mean = sup_line_counts.mean()
        num_nv_lines = len(lines)
        print("supporting images / lines: ({0:.2f} / {1:.2f})".format(sup_image_counts_mean, sup_line_counts_mean))
        print("number of 3D line segments ( nv >=", cfg["n_visible_views"], ") =", num_nv_lines)
    print("number of 3D line segments (nv >= 0) =", num_lines)
    return thresholds, list_recall, list_precision, sup_image_counts_mean, sup_line_counts_mean, num_nv_lines, num_lines


def evaluate_multiple_tnt(cfg, input_path_list, vis_err_th=None, visualize_dir="tmp"):
    print("=" * 78, flush=True)
    print("Start evaluating multiple 3D line maps on Tanks and Temples dataset")
    print("=" * 78, flush=True)

    # evaluation
    list_recall_list = []
    list_precision_list = []
    sup_image_counts_mean_list = []
    sup_line_counts_mean_list = []
    num_nv_lines_list = []
    num_lines_list = []
    thresholds = None
    for input_idx, input in enumerate(input_path_list):
        cfg["pc_dir"] = input[0]
        cfg["transform_txt"] = input[1]
        cfg["input_path"] = input[2]

        visualize_path = os.path.join(visualize_dir, str(input_idx))

        thresholds, list_recall, list_precision, sup_image_counts_mean, sup_line_counts_mean, num_nv_lines, num_lines = evaluate_single_tnt(
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
        sup_image_counts_mean_mean = np.mean(
            np.array(sup_image_counts_mean_list))
        sup_line_counts_mean_mean = np.mean(
            np.array(sup_line_counts_mean_list))
        num_nv_lines_mean = np.mean(np.array(num_nv_lines_list))

    print("-" * 78, flush=True)
    print("Report the average results of the multiple 3D line maps on Tanks and Temples dataset")
    print("-" * 78, flush=True)
    print("R: recall, P: precision")
    for idx, threshold in enumerate(thresholds):
        print("mean R / P at {0}mm: {1:.2f} / {2:.2f}".format(int(threshold * 1000),
              list_recall_mean[idx], list_precision_mean[idx]))
    if len(sup_image_counts_mean_list) != 0:
        print(
            "mean supporting images / lines: ({0:.2f} / {1:.2f})".format(sup_image_counts_mean_mean, sup_line_counts_mean_mean))
        print("mean number of 3D line segments ( nv >=", cfg["n_visible_views"], ") =", num_nv_lines_mean)
    print("mean number of 3D line segments (nv >= 0) =", num_lines_mean)


def evaluate_multiple_tnt_GT_point_cloud(cfg):
    # add additional config
    cfg["mesh_dir"] = None
    cfg["kdtree_dir"] = None
    visualize_error_threshold = None if not cfg["visualize"] else cfg["visualize_error_threshold"]

    evaluate_multiple_tnt(
        cfg, cfg["input_path_list"], vis_err_th=visualize_error_threshold, visualize_dir=cfg["visualize_dir"])


def parse_config():
    import argparse
    arg_parser = argparse.ArgumentParser(
        description="Evaluate multiple 3d line maps on TanksTemples dataset for CLMAP, LIMAP or L3D++. (only support GT point cloud now)")
    arg_parser.add_argument("-i", "--input_path_list", type=str, required=True,
                            help="GT point cloud (.ply) & transform txt & 3D line map path list")
    arg_parser.add_argument('-c', '--config_file', type=str,
                            default='cfgs_clmap/eval/tnt.yaml', help='config file')
    arg_parser.add_argument('--default_config_file', type=str,
                            default='cfgs_clmap/eval/default.yaml', help='default config file')
    arg_parser.add_argument('--use_ranges', action='store_true', help='use ranges for testing')
    arg_parser.add_argument('--visualize_error_threshold',
                            type=float, default=0.01, help="visualize threshold")
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
    cfg["use_ranges"] = args.use_ranges
    print(f"eval tnt cfg = {cfg}")
    return cfg


def main():
    # line map format:
    #   CLMAP/LIMAP: *.npy (no line tracks) or *.obj (no line tracks) or folder-to-linetracks (contains line tracks).
    #   L3D++: .txt (contains line tracks).
    # Note:
    #   If you want to evaluate the quality of line tracks, the format of input line map must be folder-to-linetracks for CLMAP/LIMAP or .txt for L3D++.

    cfg = parse_config()

    # evaluate 3d line maps in `cfg["input_path_list"]` and report the average results using GT point cloud
    evaluate_multiple_tnt_GT_point_cloud(cfg)


if __name__ == '__main__':
    main()
