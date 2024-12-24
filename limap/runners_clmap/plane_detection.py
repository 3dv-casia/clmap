import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import math
import _limap._clmap as _clmap
from limap.runners_clmap.visualize_3d_planes_bpt import create_point_cloud


def create_point_cloud_from_line_tracks(line_tracks, med_n_sample, min_n_sample, init_point_id):
    """create point cloud from line tracks
    Returns:
        points (list): sampled points on lines
        p_id_to_lt_id (dict): point_id in `points` to linetrack_id
    """
    assert med_n_sample >= 2
    assert min_n_sample >= 2

    # compute median length of all 3D line segments
    lines_length = []
    for k, v in line_tracks.items():
        lines_length.append(v.line.length())
    median_length = np.median(np.array(lines_length))

    # compute median interval
    med_interval = median_length / (med_n_sample - 1)

    # sample points from lines
    points = []
    p_id_to_lt_id = {}
    point_id = init_point_id
    for k, v in line_tracks.items():
        line_id = k
        start = np.array(v.line.start)
        direction = np.array(v.line.direction())  # start to end
        length = v.line.length()
        n_sample = math.ceil(length / med_interval)

        if n_sample < min_n_sample:
            n_sample = min_n_sample
        assert n_sample >= 2

        interval = length / (n_sample - 1)
        for i in range(n_sample):
            points.append(start + i * interval * direction)
            p_id_to_lt_id[point_id] = line_id
            point_id += 1

    return points, p_id_to_lt_id


def create_point_cloud_from_point_tracks(point_tracks, init_point_id=0):
    """create point from point tracks
    Returns:
        points (list): points of pointtracks (point cloud)
        p_id_to_pt_id (dict): point_id in point cloud to pointtrack_id
    """
    points = []
    p_id_to_pt_id = {}
    point_id = init_point_id  # point_id in point cloud
    for k, v in point_tracks.items():
        point_track_id = k
        points.append(v.p)
        p_id_to_pt_id[point_id] = point_track_id
        point_id += 1
    return points, p_id_to_pt_id


def create_point_cloud_from_point_and_line_tracks(point_tracks, line_tracks, med_n_sample, min_n_sample, visualize=False):
    """create point cloud from point and line tracks
    Returns:
        points (list): point cloud
        p_id_to_pt_id (dict): point_id in point cloud to pointtrack_id map
        p_id_to_lt_id (dict): point_id in point cloud to linetrack_id map
    """
    points1, p_id_to_pt_id = create_point_cloud_from_point_tracks(point_tracks, 0)

    max_point_id = np.max(np.array(list(p_id_to_pt_id.keys())))
    print("len(points1) =", len(points1))
    print("max_point_id =", max_point_id)
    assert len(points1) == (max_point_id + 1)

    points2, p_id_to_lt_id = create_point_cloud_from_line_tracks(
        line_tracks, med_n_sample, min_n_sample, max_point_id + 1)

    points = points1 + points2  # point cloud

    max_point_id = np.max(np.array(list(p_id_to_lt_id.keys())))
    assert len(points) == (max_point_id + 1)

    print("number of sfm points: ", len(points1))
    print("number of sampled points on lines: ", len(points2))
    print("number of all points: ", len(points), flush=True)

    # visualize sfm points (green) and sampled 3D points on 3D lines (red)
    if visualize:
        import open3d as o3d
        point_cloud_sfm = create_point_cloud(points1, [0.0, 1.0, 0.0])
        point_cloud_line = create_point_cloud(points2, [1.0, 0.0, 0.0])
        open3d_geometries = [point_cloud_sfm, point_cloud_line]
        o3d.visualization.draw_geometries(open3d_geometries)

    return points, p_id_to_pt_id, p_id_to_lt_id


def write_point_cloud_txt(output_path, points, colors=None):
    """
    Args:
        points: [[x, y, z], ...]
        colors: [[r, g, b], ...]  r, g, b \in [0, 1]
    """
    if colors is not None:
        assert len(points) == len(colors)

    with open(output_path, "w") as f:
        n = len(points)
        for i in range(n):
            f.write(str(points[i][0]) + ' ' + str(points[i][1]) + ' ' + str(points[i][2]))
            if colors is not None:
                f.write(' ' + str(colors[i][0] * 255) + ' ' + str(colors[i][1] * 255) + ' ' + str(colors[i][2] * 255))
            f.write('\n')


def read_planes(plane_path):
    planes = {}
    with open(plane_path, 'r') as f:
        n = int(f.readline().strip('\n').split()[-1])
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                plane_idx = int(elems[0])
                normal = np.array(tuple(map(float, elems[1:4])))
                center = np.array(tuple(map(float, elems[4:7])))
                v1 = np.array(tuple(map(float, elems[7:10])))
                v2 = np.array(tuple(map(float, elems[10:13])))
                v3 = np.array(tuple(map(float, elems[13:16])))
                v4 = np.array(tuple(map(float, elems[16:19])))
                point_ids = np.array(tuple(map(int, elems[19:])))
                planes[plane_idx] = {'normal': normal, 'center': center, 'v1': v1,
                                     'v2': v2, 'v3': v3, 'v4': v4, 'point_ids': point_ids}
        assert len(planes) == n
        return planes


def create_inf_planes3d(planes):
    inf_planes3d = {}
    for k, v in planes.items():
        plane_id = k
        assert np.linalg.norm(v['normal']) > 0
        n0 = v['normal'] / np.linalg.norm(v['normal'])
        d = np.dot(n0, v['center'])
        inf_planes3d[plane_id] = _clmap.InfinitePlane3d(n0, d)
    return inf_planes3d


def detect_planes(cfg, imagecols, pointtracks, linetracks, point_cloud_output_path, planes_output_path, visualize_point_cloud=False):
    # create point cloud from sfm points and sampled 3D points on 3D lines
    med_n_sample = 10
    min_n_sample = 5
    points, p_id_to_pt_id, p_id_to_lt_id = create_point_cloud_from_point_and_line_tracks(
        pointtracks, linetracks, med_n_sample, min_n_sample, visualize=visualize_point_cloud)
    write_point_cloud_txt(point_cloud_output_path, points)

    # default hyperparameters in `PlaneDetecion` open-source code
    normal_variance_threshold_deg = 60
    coplanarity_deg = 75
    minNormalDiff = np.cos(normal_variance_threshold_deg * np.pi / 180)
    maxDist = np.cos(coplanarity_deg * np.pi / 180)
    outlierRatio = 0.75
    normalsNeighborSize = 30

    # detect planes
    _clmap.RobustStatisticsApproach(inputFileName=point_cloud_output_path, outputFileName=planes_output_path, minNormalDiff=minNormalDiff,
                                    maxDist=maxDist, outlierRatio=outlierRatio, normalsNeighborSize=normalsNeighborSize)

    # read plane
    planes = read_planes(planes_output_path)

    # create inf_planes3d
    inf_planes3d = create_inf_planes3d(planes)

    # build candidate pointtrack/linetrack and plane bipartite3d
    c_pp_bpt = {}
    c_lp_bpt = {}
    for k, v in planes.items():
        plane_id = k
        point_ids = list(v['point_ids'])
        for point_id in point_ids:
            if point_id in p_id_to_pt_id.keys():
                point_track_id = p_id_to_pt_id[point_id]
                c_pp_bpt[point_track_id] = plane_id
            else:
                assert point_id in p_id_to_lt_id.keys()
                line_track_id = p_id_to_lt_id[point_id]
                if line_track_id not in c_lp_bpt.keys():
                    c_lp_bpt[line_track_id] = set()
                c_lp_bpt[line_track_id].add(plane_id)

    # build initial bipartite3d
    point_var2d = cfg['plp_association']['point_var2d']
    point_u_method = cfg['plp_association']['point_u_method']
    th_hard_pointplane_si_dist3d = cfg['plp_association']['th_hard_pointplane_si_dist3d']
    pp_bpt3d = _clmap.build_init_PP_Bipartite3d(pointtracks, inf_planes3d, c_pp_bpt, imagecols,
                                                point_var2d, point_u_method, th_hard_pointplane_si_dist3d)

    line_var2d = cfg['plp_association']['line_var2d']
    line_u_method = cfg['plp_association']['line_u_method']
    th_hard_lineplane_si_dist3d = cfg['plp_association']["th_hard_lineplane_si_dist3d"]
    th_hard_lineplane_angle = cfg['plp_association']["th_hard_lineplane_angle"]
    lp_bpt3d = _clmap.build_init_LP_Bipartite3d(linetracks, inf_planes3d, c_lp_bpt, imagecols,
                                                line_var2d, line_u_method, th_hard_lineplane_si_dist3d,
                                                th_hard_lineplane_angle)

    return inf_planes3d, pp_bpt3d, lp_bpt3d
