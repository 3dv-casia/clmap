import numpy as np
from scipy.spatial import ConvexHull
import open3d as o3d
import _limap._clmap as _clmap


def create_line_set(line3d_list, colors=[0.0, 1.0, 0.0]):
    o3d_points, o3d_lines, o3d_colors = [], [], []

    for counter, line3d in enumerate(line3d_list):
        o3d_points.append(line3d.start)
        o3d_points.append(line3d.end)
        o3d_lines.append([2 * counter, 2 * counter + 1])
        o3d_colors.append(colors)

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(o3d_points)
    line_set.lines = o3d.utility.Vector2iVector(o3d_lines)
    line_set.colors = o3d.utility.Vector3dVector(o3d_colors)
    return line_set


def create_point_cloud(points, colors=[0.0, 0.0, 1.0]):
    o3d_colors = []
    for i in range(len(points)):
        o3d_colors.append(colors)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(o3d_colors)
    return point_cloud


def create_convex_hull_mesh(points2d_in_plane_corrds, proj_points, color):
    import time

    hull = ConvexHull(points2d_in_plane_corrds)

    # add vertices
    vertices = []
    for v_idx in hull.vertices:
        vertices.append(proj_points[v_idx, :])
    vertices = np.array(vertices)

    # add indices
    indices = []
    num_v = hull.vertices.shape[0]
    for idx in range(1, num_v - 1):
        indices.append([0, idx, idx + 1])
    indices = np.array(indices)

    # add colors
    colors = []
    for i in range(len(vertices)):
        colors.append(color)
    colors = np.array(colors)

    # create an Open3D triangle mesh object
    mesh = o3d.geometry.TriangleMesh()

    # set vertices and faces for the mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(indices)

    # assign the face color to each vertex
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    return mesh


def get_projected_points(inf_plane3d, points):
    # `proj_points` is obtained by projecting `points` onto `inf_plane3d`
    proj_points = []

    n0 = np.array(inf_plane3d.n0)
    d = inf_plane3d.d

    for point in points:
        dist = d - np.dot(point, n0)  # signed distance
        proj_point = point + dist * n0
        proj_points.append(proj_point)

    return proj_points


def estimate_plane_transformation(inf_plane3d, points_on_plane):
    n0 = np.array(inf_plane3d.n0)

    # t^{w}_{pORG}
    t = np.mean(points_on_plane, axis=0)

    u, s, vh = np.linalg.svd(points_on_plane - t.reshape(1, 3))
    V = vh.T
    plane_x_axis = V[:, 0]
    plane_y_axis = V[:, 1]
    plane_z_axis = np.cross(plane_x_axis, plane_y_axis)

    # debug
    if 1 - np.abs(plane_z_axis.dot(np.array(n0))) >= 0.001:
        # if points_on_plane.shpae[0] == 2, i.e., there is only one line associated with this plane, PCA will fail.
        print("[error] points_on_plane.shpae =", points_on_plane.shpae)
        print("[error] np.abs(plane_z_axis.dot(np.array(n0))) = ", np.abs(plane_z_axis.dot(np.array(n0))))
    assert 1 - np.abs(plane_z_axis.dot(np.array(n0))) < 0.001

    # R^{w}_{p}
    R = np.concatenate([plane_x_axis.reshape(3, 1), plane_y_axis.reshape(3, 1), plane_z_axis.reshape(3, 1)], axis=1)

    return R, t


def transform_world_point_to_plane(R, t, world_points):
    """
    Args:
        world_points: (N, 3)
    Returns: (N, 3)
    """
    return (R.T @ (world_points.T - t.reshape(3, 1))).T


def create_geometries_from_lp_bpt3d_pp_bpt3d(pp_bpt3d, lp_bpt3d, plane_id_to_rgb):
    # create o3d.geometry.TriangleMesh for each plane
    plane_meshes = []
    for k, v in lp_bpt3d.n_2_to_1.items():
        plane_id = k
        inf_plane3d = lp_bpt3d.obj2_map[plane_id]

        # process associated lines
        endpoints = []
        for linetrack_id in v:
            linetrack = lp_bpt3d.obj1_map[linetrack_id]
            endpoints.append(np.array(linetrack.line.start))
            endpoints.append(np.array(linetrack.line.end))

        # process associated sfm points
        sfm_points = [np.array(pointtrack.p) for pointtrack in pp_bpt3d.GetObj2Neighbors(plane_id)]

        # create a mesh
        points = endpoints + sfm_points
        proj_points = np.array(get_projected_points(inf_plane3d, points))
        R, t = estimate_plane_transformation(inf_plane3d, proj_points)
        points_in_plane_corrds = transform_world_point_to_plane(R, t, proj_points)
        points2d_in_plane_corrds = points_in_plane_corrds[:, :2]
        mesh = create_convex_hull_mesh(points2d_in_plane_corrds, proj_points, plane_id_to_rgb[plane_id])
        plane_meshes.append(mesh)

    import time
    t1 = time.time()
    # create o3d.geometry.LineSet using associated 3D lines
    lines3d = _clmap.GetInlierLine3dsFromLP_Bipartite3d(lp_bpt3d)
    line_set = create_line_set(lines3d)
    t2 = time.time()
    print(f"time of create_line_set: {t2 - t1} s", flush=True)

    # create o3d.geometry.PointCloud using associated SfM points
    t1 = time.time()
    sfm_points = _clmap.GetInlierPoint3dsFromPP_Bipartite3d(pp_bpt3d)
    t2 = time.time()
    print(f"time of append sfm_points: {t2 - t1} s", flush=True)

    t1 = time.time()
    point_cloud = create_point_cloud(sfm_points)
    t2 = time.time()
    print(f"time of create_point_cloud: {t2 - t1} s", flush=True)

    return plane_meshes, line_set, point_cloud


def get_textural_and_structural_lines(lp_bpt3d):
    textural_lines3d, structural_lines3d = [], []

    for linetrack_id, plane_ids in lp_bpt3d.n_1_to_2.items():
        if len(plane_ids) == 1:
            textural_lines3d.append(lp_bpt3d.obj1_map[linetrack_id].line)
        elif len(plane_ids) > 1:
            structural_lines3d.append(lp_bpt3d.obj1_map[linetrack_id].line)

    textural_line_set = create_line_set(textural_lines3d, colors=[0.0, 1.0, 0.0])
    structural_line_set = create_line_set(structural_lines3d, colors=[1.0, 0.0, 0.0])
    return textural_line_set, structural_line_set
