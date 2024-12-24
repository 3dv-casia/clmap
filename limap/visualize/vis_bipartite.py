import os
import copy
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .vis_utils import compute_robust_range_lines, draw_segments, draw_points, test_point_inside_ranges, test_line_inside_ranges
from .vis_lines import open3d_add_points, open3d_get_cameras, open3d_add_line_set, open3d_add_cameras, open3d_get_line_set
import limap.runners_clmap

import _limap._base as _base


def draw_bipartite2d(image, bpt2d):
    image = copy.deepcopy(image)
    lines = bpt2d.get_all_lines()
    image = draw_segments(image, [line.as_array().reshape(-1) for line in lines], (0, 255, 0))
    junctions = bpt2d.get_all_junctions()
    image = draw_points(image, [junc.p.p for junc in junctions if junc.degree() == 1], (255, 0, 0), 1)
    image = draw_points(image, [junc.p.p for junc in junctions if junc.degree() == 2], (0, 0, 255), 1)
    image = draw_points(image, [junc.p.p for junc in junctions if junc.degree() > 2], (0, 0, 0), 1)
    return image


def open3d_draw_bipartite3d_pointline(bpt3d, ranges=None, draw_edges=True, imagecols=None, draw_planes=False):
    '''
    Visualize point-line bipartite on 3D with Open3D
    '''
    points, degrees = [], []
    for idx, ptrack in bpt3d.get_dict_points().items():
        p = ptrack.p
        deg = bpt3d.pdegree(idx)
        if ranges is not None:
            if not test_point_inside_ranges(p, ranges):
                continue
        points.append(p)
        degrees.append(deg)
    points_deg0 = [p for p, deg in zip(points, degrees) if deg == 0]
    points_deg1 = [p for p, deg in zip(points, degrees) if deg == 1]
    points_deg2 = [p for p, deg in zip(points, degrees) if deg == 2]
    points_deg3p = [p for p, deg in zip(points, degrees) if deg >= 3]
    lines = bpt3d.get_line_cloud()
    if ranges is not None:
        lines = [l for l in lines if test_line_inside_ranges(l, ranges)]

    # optionally draw edges
    edges = None
    if draw_edges:
        edges = []
        for p_id, ptrack in bpt3d.get_dict_points().items():
            if bpt3d.pdegree(p_id) == 0:
                continue
            p = ptrack.p
            if ranges is not None:
                if not test_point_inside_ranges(p, ranges):
                    continue
            for line_id in bpt3d.neighbor_lines(p_id):
                line = bpt3d.line(line_id).line
                p_proj = line.point_projection(p)
                edges.append(_base.Line3d(p, p_proj))

    # optiionally draw planes
    planes = None
    if draw_planes:
        planes = []
        scale = 0.2  # TODO: port properties
        for point_id in bpt3d.get_point_ids():
            # TODO: now we only consider degree-2 points
            if bpt3d.pdegree(point_id) != 2:
                continue

            point = bpt3d.point(point_id).p
            line_ids = bpt3d.neighbor_lines(point_id)
            direc1 = bpt3d.line(line_ids[0]).line.direction()
            direc2 = bpt3d.line(line_ids[1]).line.direction()
            normal = np.cross(direc1, direc2)
            # test if it is a psuedo junction
            if np.linalg.norm(normal) < 0.80:
                continue
            normal /= np.linalg.norm(normal)

            # draw plane
            base1 = direc1
            base2 = np.cross(normal, base1)

            vert1 = point + base1 * scale * 0.5
            vert2 = point + base2 * scale * 0.5
            vert3 = point + base1 * scale * (-0.5)
            vert4 = point + base2 * scale * (-0.5)
            planes.append([vert1, vert2, vert3, vert4])

    import open3d as o3d
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
    w.show_skybox(False)
    w = open3d_add_points(w, points_deg1, color=(0.0, 0.0, 1.0), psize=1, name="pcd_deg1")
    w = open3d_add_points(w, points_deg2, color=(1.0, 0.0, 0.0), psize=3, name="pcd_deg2")
    w = open3d_add_points(w, points_deg3p, color=(1.0, 0.0, 0.0), psize=5, name="pcd_deg3p")
    w = open3d_add_line_set(w, lines, color=(0.0, 1.0, 0.0), width=-1, name="line_set")
    if edges is not None:
        w = open3d_add_line_set(w, edges, color=(0.0, 0.0, 0.0), width=-1, name="line_set_constraints")

    if planes is not None:
        for plane_id, plane in enumerate(planes):
            plane_c = (0.5, 0.4, 0.6)
            mesh = o3d.geometry.TriangleMesh()
            np_vertices = np.array(plane)
            np_triangles = np.array([[0, 1, 2], [0, 2, 3]]).astype(np.int32)
            mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
            w.add_geometry("plane_{0}".format(plane_id), mesh)

    # optionally draw cameras
    if imagecols is not None:
        w = open3d_add_cameras(w, imagecols)
    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()


def open3d_draw_bipartite3d_pointline_Visualizer(bpt3d, ranges=None, draw_edges=True, draw_planes=False, imagecols=None, cam_scale=1.0):
    '''
    Visualize point-line bipartite on 3D with Open3D
    '''
    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)

    points, degrees = [], []
    for idx, ptrack in bpt3d.get_dict_points().items():
        p = ptrack.p
        deg = bpt3d.pdegree(idx)
        if ranges is not None:
            if not test_point_inside_ranges(p, ranges):
                continue
        points.append(p)
        degrees.append(deg)
    points_deg0 = [p for p, deg in zip(points, degrees) if deg == 0]
    points_deg1 = [p for p, deg in zip(points, degrees) if deg == 1]
    points_deg2 = [p for p, deg in zip(points, degrees) if deg == 2]
    points_deg3 = [p for p, deg in zip(points, degrees) if deg >= 3]
    points_deg1 = limap.runners_clmap.create_point_cloud(points_deg1, colors=[0.0, 0.0, 1.0])
    points_deg2 = limap.runners_clmap.create_point_cloud(points_deg2, colors=[1.0, 0.0, 0.0])
    points_deg3 = limap.runners_clmap.create_point_cloud(points_deg3, colors=[1.0, 0.0, 0.0])
    vis.add_geometry(points_deg1)
    vis.add_geometry(points_deg2)
    vis.add_geometry(points_deg3)

    lines_deg0, lines_deg1 = [], []
    for line_id, ltrack in bpt3d.get_dict_lines().items():
        line = ltrack.line
        deg = bpt3d.ldegree(line_id)
        if deg == 0:
            lines_deg0.append(line)
        else:
            lines_deg1.append(line)
    # line_set = open3d_get_line_set(lines_deg0, color=[0.0, 0.0, 0.0], ranges=ranges)
    # vis.add_geometry(line_set)
    line_set = open3d_get_line_set(lines_deg1, color=[0.0, 1.0, 0.0], ranges=ranges)
    vis.add_geometry(line_set)

    # optionally draw edges
    edges = None
    if draw_edges:
        edges = []
        for p_id, ptrack in bpt3d.get_dict_points().items():
            if bpt3d.pdegree(p_id) == 0:
                continue
            p = ptrack.p
            if ranges is not None:
                if not test_point_inside_ranges(p, ranges):
                    continue
            for line_id in bpt3d.neighbor_lines(p_id):
                line = bpt3d.line(line_id).line
                p_proj = line.point_projection(p)
                edges.append(_base.Line3d(p, p_proj))

    if edges is not None:
        line_set = open3d_get_line_set(edges, color=[0.0, 0.0, 0.0])
        vis.add_geometry(line_set)

    # optiionally draw planes
    planes = None
    if draw_planes:
        planes = []
        scale = 0.2  # TODO: port properties
        for point_id in bpt3d.get_point_ids():
            # TODO: now we only consider degree-2 points
            if bpt3d.pdegree(point_id) != 2:
                continue

            point = bpt3d.point(point_id).p
            line_ids = bpt3d.neighbor_lines(point_id)
            direc1 = bpt3d.line(line_ids[0]).line.direction()
            direc2 = bpt3d.line(line_ids[1]).line.direction()
            normal = np.cross(direc1, direc2)
            # test if it is a psuedo junction
            if np.linalg.norm(normal) < 0.80:
                continue
            normal /= np.linalg.norm(normal)

            # draw plane
            base1 = direc1
            base2 = np.cross(normal, base1)

            vert1 = point + base1 * scale * 0.5
            vert2 = point + base2 * scale * 0.5
            vert3 = point + base1 * scale * (-0.5)
            vert4 = point + base2 * scale * (-0.5)
            planes.append([vert1, vert2, vert3, vert4])

    if planes is not None:
        for plane_id, plane in enumerate(planes):
            plane_c = (0.5, 0.4, 0.6)
            mesh = o3d.geometry.TriangleMesh()
            np_vertices = np.array(plane)
            np_triangles = np.array([[0, 1, 2], [0, 2, 3]]).astype(np.int32)
            mesh.vertices = o3d.utility.Vector3dVector(np_vertices)
            mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
            vis.add_geometry(mesh)

    if imagecols is not None:
        if ranges is not None:
            lranges = ranges
        else:
            lranges = compute_robust_range_lines(bpt3d.get_line_cloud())
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
        camera_set = open3d_get_cameras(
            imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale)
        vis.add_geometry(camera_set)

    vis.run()
    vis.destroy_window()


def open3d_draw_bipartite3d_vpline(bpt3d, ranges=None):
    '''
    Visualize point-line bipartite on 3D with Open3D
    '''
    import seaborn as sns
    n_vps = bpt3d.count_points()
    colors = sns.color_palette("husl", n_colors=n_vps)

    vp_ids = bpt3d.get_point_ids()
    vp_id_to_color = {vp_id: colors[idx] for idx, vp_id in enumerate(vp_ids)}
    vp_line_sets = {vp_id: [] for vp_id in vp_ids}
    nonvp_line_set = []
    for line_id, ltrack in bpt3d.get_dict_lines().items():
        if ranges is not None:
            if not test_line_inside_ranges(ltrack.line, ranges):
                continue
        labels = bpt3d.neighbor_points(line_id)
        if len(labels) == 0:
            nonvp_line_set.append(ltrack.line)
            continue
        assert len(labels) == 1
        label = labels[0]
        vp_line_sets[label].append(ltrack.line)

    # open3d
    import open3d as o3d
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
    w.show_skybox(False)
    for vp_id in vp_ids:
        if len(vp_line_sets[vp_id]) == 0:
            continue
        w = open3d_add_line_set(w, vp_line_sets[vp_id], color=vp_id_to_color[vp_id],
                                width=2, name="lineset_vp_{0}".format(vp_id))
    w = open3d_add_line_set(w, nonvp_line_set, color=(0.0, 0.0, 0.0), width=2, name="lineset_nonvp")
    w.reset_camera_to_default()
    w.scene_shader = w.UNLIT
    w.enable_raw_mode(True)
    app.add_window(w)
    app.run()


def open3d_draw_bipartite3d_vpline_Visualizer(bpt3d, vp_id_to_rgb=None, ranges=None, imagecols=None, cam_scale=1.0):
    '''
    Visualize point-line bipartite on 3D with Open3D
    '''

    import seaborn as sns
    n_vps = bpt3d.count_points()
    colors = sns.color_palette("husl", n_colors=n_vps)

    import open3d as o3d

    vis = o3d.visualization.Visualizer()
    vis.create_window(height=1080, width=1920)

    vp_ids = bpt3d.get_point_ids()
    if vp_id_to_rgb is None:
        import seaborn as sns
        n_vps = bpt3d.count_points()
        colors = sns.color_palette("husl", n_colors=n_vps)
        vp_id_to_color = {vp_id: colors[idx] for idx, vp_id in enumerate(vp_ids)}
    else:
        vp_id_to_color = {vp_id: np.array(rgb) / 255 for vp_id, rgb in vp_id_to_rgb.items()}
    vp_line_sets = {vp_id: [] for vp_id in vp_ids}
    nonvp_line_set = []
    lines = []
    for line_id, ltrack in bpt3d.get_dict_lines().items():
        lines.append(ltrack.line)
        if ranges is not None:
            if not test_line_inside_ranges(ltrack.line, ranges):
                continue
        labels = bpt3d.neighbor_points(line_id)
        if len(labels) == 0:
            nonvp_line_set.append(ltrack.line)
            continue
        assert len(labels) == 1
        label = labels[0]
        vp_line_sets[label].append(ltrack.line)
    # add 3D lines where the lines associating with the same 3D VP have the same color
    for vp_id in vp_ids:
        if len(vp_line_sets[vp_id]) == 0:
            continue
        line_set = open3d_get_line_set(vp_line_sets[vp_id], color=vp_id_to_color[vp_id], ranges=ranges)
        vis.add_geometry(line_set)
    # # add 3D lines without accociating with 3D VP
    # line_set = open3d_get_line_set(nonvp_line_set, color=[0.0, 0.0, 0.0], ranges=ranges)
    # vis.add_geometry(line_set)

    if imagecols is not None:
        if ranges is not None:
            lranges = ranges
        else:
            lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
        camera_set = open3d_get_cameras(
            imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale)
        vis.add_geometry(camera_set)

    vis.run()
    vis.destroy_window()
