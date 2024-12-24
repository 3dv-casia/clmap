import open3d as o3d
from .base import BaseTrackVisualizer
from ..vis_utils import compute_robust_range_lines, test_line_inside_ranges, test_point_inside_ranges
from ..vis_lines import open3d_get_line_set, open3d_get_cameras, open3d_add_cameras, open3d_add_line_set, open3d_add_points

import limap.runners_clmap
import numpy as np


class Open3DTrackVisualizer(BaseTrackVisualizer):
    def __init__(self, tracks):
        super(Open3DTrackVisualizer, self).__init__(tracks)

    def reset(self):
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        return app

    def vis_all_lines(self, n_visible_views=4, width=2, scale=1.0):
        lines = self.get_lines_n_visible_views(n_visible_views)
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(
            lines, width=width, ranges=None, scale=scale)
        vis.add_geometry(line_set)
        vis.run()
        vis.destroy_window()

    def vis_reconstruction(self, imagecols, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        lines = self.get_lines_n_visible_views(n_visible_views)
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(
            lines, width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        camera_set = open3d_get_cameras(
            imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
        vis.add_geometry(camera_set)
        vis.run()
        vis.destroy_window()

    def vis_reconstruction_point_line(self, point_tracks, imagecols, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        """
        point_tracks: from colmap model
        """

        lines = self.get_lines_n_visible_views(n_visible_views)
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(
            lines, color=[0.0, 1.0, 0.0], width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        camera_set = open3d_get_cameras(
            imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
        vis.add_geometry(camera_set)

        # plot point3d
        point3d_xyz_list = []
        for point3d_id, point_track in point_tracks.items():
            if len(point_track.image_ids) < 3:
                continue
            point3d_xyz_list.append(point_track.xyz)
        point_cloud = limap.runners_clmap.create_point_cloud(
            point3d_xyz_list, colors=[0.7, 0.7, 0.7])
        vis.add_geometry(point_cloud)

        vis.run()
        vis.destroy_window()

    def vis_reconstruction_plypoint_line(self, points, imagecols, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        """
        point_tracks: from colmap modelplanes_mesh
        """

        lines = self.get_lines_n_visible_views(n_visible_views)
        lranges = compute_robust_range_lines(lines)
        scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)
        line_set = open3d_get_line_set(lines, color=[0.0, 1.0, 0.0], width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)
        camera_set = open3d_get_cameras(
            imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
        vis.add_geometry(camera_set)

        # plot point3d
        point_cloud = limap.runners_clmap.create_point_cloud(points, colors=[0.7, 0.7, 0.7])
        vis.add_geometry(point_cloud)

        vis.run()
        vis.destroy_window()

    def vis_reconstruction_point_line_plane(self, imagecols=None, pointtracks=None, plane_meshes=None, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        """
        pointtracks: pointtracks[point3d_id] = class limap::PointTrack
        """
        import open3d as o3d
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
        w.show_skybox(False)

        lines = self.get_lines_n_visible_views(n_visible_views)

        # add cameras
        if imagecols is not None:
            lranges = compute_robust_range_lines(lines)
            scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
            w = open3d_add_cameras(w, imagecols, color=[1.0, 0.0, 0.0], ranges=None,
                                   scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)

        # add points
        if pointtracks is not None:
            point3d_xyz_list = []
            for point3d_id, pointtrack in pointtracks.items():
                point3d_xyz_list.append(np.array(pointtrack.p))
            w = open3d_add_points(w, point3d_xyz_list, color=[0.6, 0.6, 0.6],
                                  psize=3.0, name="pcd", ranges=ranges, scale=scale)

        # add lines
        w = open3d_add_line_set(w, lines, color=[0.0, 1.0, 0.0], width=2, name="line_set")

        # add planes
        for plane_idx, plane_mesh in enumerate(plane_meshes):
            w.add_geometry(f"plane_{plane_idx}", plane_mesh)

        w.reset_camera_to_default()
        w.scene_shader = w.UNLIT
        w.enable_raw_mode(True)
        app.add_window(w)
        app.run()

    def vis_reconstruction_point_line_plane_Visualizer(self, imagecols=None, pointtracks=None, plane_meshes=None, n_visible_views=4, width=2, ranges=None, scale=1.0, cam_scale=1.0):
        """
        pointtracks: pointtracks[point3d_id] = class limap::PointTrack
        """
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)

        lines = self.get_lines_n_visible_views(n_visible_views)
        line_set = open3d_get_line_set(
            lines, color=[0.0, 1.0, 0.0], width=width, ranges=ranges, scale=scale)
        vis.add_geometry(line_set)

        if imagecols is not None:
            lranges = compute_robust_range_lines(lines)
            scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
            camera_set = open3d_get_cameras(
                imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
            vis.add_geometry(camera_set)

        # plot point3d
        point3d_xyz_list = []
        for point3d_id, pointtrack in pointtracks.items():
            if ranges is not None:
                if not test_point_inside_ranges(pointtrack.p, ranges):
                    continue
            point3d_xyz_list.append(np.array(pointtrack.p))
        point_cloud = limap.runners_clmap.create_point_cloud(point3d_xyz_list, colors=[0.5, 0.5, 0.5])
        vis.add_geometry(point_cloud)

        # plot planes
        if plane_meshes is not None:
            for plane_mesh in plane_meshes:
                vis.add_geometry(plane_mesh)

        vis.get_render_option().mesh_show_back_face = True

        vis.run()
        vis.destroy_window()

    def vis_reconstruction_point_line_vp(self, imagecols=None, pointtracks=None, line_vp_bpt3d=None, n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0):
        import open3d as o3d
        app = o3d.visualization.gui.Application.instance
        app.initialize()
        w = o3d.visualization.O3DVisualizer(height=1080, width=1920)
        w.show_skybox(False)

        # add cameras
        if imagecols is not None:
            lines = self.get_lines_n_visible_views(n_visible_views)
            lranges = compute_robust_range_lines(lines)
            scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
            w = open3d_add_cameras(w, imagecols, color=[1.0, 0.0, 0.0], ranges=None,
                                   scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)

        # add points
        if pointtracks is not None:
            point3d_xyz_list = []
            for point3d_id, pointtrack in pointtracks.items():
                point3d_xyz_list.append(np.array(pointtrack.p))
            w = open3d_add_points(w, point3d_xyz_list, color=[0.6, 0.6, 0.6],
                                  psize=3.0, name="pcd", ranges=ranges, scale=scale)

        # add lines and VPs
        if line_vp_bpt3d is not None:
            import seaborn as sns
            n_vps = line_vp_bpt3d.count_points()
            colors = sns.color_palette("husl", n_colors=n_vps)
            vp_ids = line_vp_bpt3d.get_point_ids()
            vp_id_to_color = {vp_id: colors[idx] for idx, vp_id in enumerate(vp_ids)}
            vp_line_sets = {vp_id: [] for vp_id in vp_ids}
            nonvp_line_set = []
            for line_id, ltrack in line_vp_bpt3d.get_dict_lines().items():
                if ranges is not None:
                    if not test_line_inside_ranges(ltrack.line, ranges):
                        continue
                labels = line_vp_bpt3d.neighbor_points(line_id)
                if len(labels) == 0:
                    nonvp_line_set.append(ltrack.line)
                    continue
                assert len(labels) == 1
                label = labels[0]
                vp_line_sets[label].append(ltrack.line)

            mat = o3d.visualization.rendering.MaterialRecord()
            mat.shader = "unlitLine"
            mat.line_width = 2
            # add 3D lines where the lines associating with the same 3D VP have the same color
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

    def vis_reconstruction_point_line_vp_Visualizer(self, imagecols=None, pointtracks=None, line_vp_bpt3d=None, n_visible_views=4, ranges=None, scale=1.0, cam_scale=1.0, vp_id_to_rgb=None):
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=1080, width=1920)

        # add cameras
        if imagecols is not None:
            lines = self.get_lines_n_visible_views(n_visible_views)
            lranges = compute_robust_range_lines(lines)
            scale_cam_geometry = abs(lranges[1, :] - lranges[0, :]).max()
            camera_set = open3d_get_cameras(
                imagecols, ranges=None, scale_cam_geometry=scale_cam_geometry * cam_scale, scale=scale)
            vis.add_geometry(camera_set)

        # add points
        if pointtracks is not None:
            point3d_xyz_list = []
            for point3d_id, pointtrack in pointtracks.items():
                if ranges is not None:
                    if not test_point_inside_ranges(pointtrack.p, ranges):
                        continue
                point3d_xyz_list.append(np.array(pointtrack.p))
            point_cloud = limap.runners_clmap.create_point_cloud(point3d_xyz_list, colors=[0.5, 0.5, 0.5])
            vis.add_geometry(point_cloud)

        # add lines and VPs
        if line_vp_bpt3d is not None:
            vp_ids = line_vp_bpt3d.get_point_ids()
            if vp_id_to_rgb is None:
                import seaborn as sns
                n_vps = line_vp_bpt3d.count_points()
                colors = sns.color_palette("husl", n_colors=n_vps)
                vp_id_to_color = {vp_id: colors[idx] for idx, vp_id in enumerate(vp_ids)}
            else:
                vp_id_to_color = {vp_id: np.array(rgb) / 255 for vp_id, rgb in vp_id_to_rgb.items()}
            vp_line_sets = {vp_id: [] for vp_id in vp_ids}
            nonvp_line_set = []
            for line_id, ltrack in line_vp_bpt3d.get_dict_lines().items():
                if ranges is not None:
                    if not test_line_inside_ranges(ltrack.line, ranges):
                        continue
                labels = line_vp_bpt3d.neighbor_points(line_id)
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
                line_set = open3d_get_line_set(
                    vp_line_sets[vp_id], color=vp_id_to_color[vp_id], ranges=ranges, scale=scale)
                vis.add_geometry(line_set)
            # add 3D lines without accociating with 3D VP
            line_set = open3d_get_line_set(nonvp_line_set, color=[0.0, 0.0, 0.0], ranges=ranges, scale=scale)
            vis.add_geometry(line_set)

        vis.run()
        vis.destroy_window()
