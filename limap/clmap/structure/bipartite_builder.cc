
#include "clmap/structure/bipartite_builder.h"

#include "clmap/base/geometry.h"

namespace limap {

PP_Bipartite3d BuildInitPP_Bipartite3d(
    const std::unordered_map<int, PointTrack>& point_tracks,
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const std::unordered_map<int, int>& candidate_point_plane_bpt,
    const ImageCollection& img_cols, double var2d, const std::string& u_method,
    double th_hard_pointplane_si_dist3d) {
  // init
  PP_Bipartite3d pp_bpt3d;

  // add obj1 and obj2
  for (auto it = point_tracks.begin(); it != point_tracks.end(); it++) {
    pp_bpt3d.AddObj1(it->second, it->first);
  }

  for (auto it = inf_planes3d.begin(); it != inf_planes3d.end(); it++) {
    pp_bpt3d.AddObj2(it->second, it->first);
  }

  // compute point3d uncertainty
  std::unordered_map<int, double> uncertainties;  // (point3d_id, uncertainty)
  geometry::ComputePoints3dUncertainty(point_tracks, img_cols, &uncertainties,
                                       var2d, u_method);

  size_t num_original_associations = candidate_point_plane_bpt.size();
  size_t num_final_associations = 0;

  // test whether the 3D point on the infinite 3D plane
  for (auto it = candidate_point_plane_bpt.begin();
       it != candidate_point_plane_bpt.end(); it++) {
    int point_id = it->first;
    int plane_id = it->second;
    double scale_inv_dist;
    if (geometry::TestPointOnInfPlane3d(
            point_tracks.at(point_id).p, inf_planes3d.at(plane_id),
            uncertainties.at(point_id), th_hard_pointplane_si_dist3d,
            &scale_inv_dist)) {
      THROW_CHECK_GE(scale_inv_dist, 0)
      // add connection
      pp_bpt3d.AddEdge(point_id, plane_id, -scale_inv_dist);
      num_final_associations++;
    }
  }

  std::cout << "[build initial point-plane associations] number of original "
               "associations = "
            << num_original_associations << std::endl;
  std::cout << "[build initial point-plane associations] number of final "
               "associations = "
            << num_final_associations << std::endl;

  return pp_bpt3d;
}

LP_Bipartite3d BuildInitLP_Bipartite3d(
    const std::unordered_map<int, LineTrack>& line_tracks,
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const std::unordered_map<int, std::unordered_set<int>>&
        candidate_line_planes_bpt,
    const ImageCollection& img_cols, double var2d, const std::string& u_method,
    double th_hard_lineplane_si_dist3d, double th_hard_lineplane_angle) {
  // init
  LP_Bipartite3d lp_bpt3d;

  // add obj1 and obj2
  for (auto it = line_tracks.begin(); it != line_tracks.end(); it++) {
    lp_bpt3d.AddObj1(it->second, it->first);
  }
  for (auto it = inf_planes3d.begin(); it != inf_planes3d.end(); it++) {
    lp_bpt3d.AddObj2(it->second, it->first);
  }

  // compute lines3d uncertainty
  std::unordered_map<int, std::pair<V3D, double>> u_start;
  std::unordered_map<int, std::pair<V3D, double>> u_end;
  std::unordered_map<int, std::pair<V3D, double>> u_median;
  geometry::ComputeLines3dUncertainty(line_tracks, img_cols, &u_start, &u_end,
                                      &u_median, var2d, u_method);

  size_t num_original_associations = 0;
  size_t num_final_associations = 0;

  // test whether the 3D line segment on the 3D infinite plane
  for (auto it = candidate_line_planes_bpt.begin();
       it != candidate_line_planes_bpt.end(); it++) {
    int line_id = it->first;
    const auto& plane_ids = it->second;
    num_original_associations += plane_ids.size();

    const Line3d& line3d = line_tracks.at(line_id).line;
    double start_uncertainty = u_start.at(line_id).second;
    double end_uncertainty = u_end.at(line_id).second;

    for (const auto& plane_id : plane_ids) {
      const auto& inf_plane3d = inf_planes3d.at(plane_id);

      double scale_inv_dist;
      if (geometry::TestLine3dOnInfPlane3d(
              line3d, inf_plane3d, start_uncertainty, end_uncertainty,
              th_hard_lineplane_si_dist3d, th_hard_lineplane_angle,
              &scale_inv_dist)) {
        THROW_CHECK_GE(scale_inv_dist, 0)
        // add connection
        lp_bpt3d.AddEdge(line_id, plane_id, -scale_inv_dist);
        num_final_associations++;
      }
    }
  }

  std::cout << "[build initial line-plane associations] number of original "
               "associations = "
            << num_original_associations << std::endl;
  std::cout << "[build initial line-plane associations] number of final "
               "associations = "
            << num_final_associations << std::endl;

  return lp_bpt3d;
}

std::tuple<size_t, size_t, std::unordered_map<int, InfinitePlane3d>,
           PP_Bipartite3d, LP_Bipartite3d>
FilterInfPlane3dWithPP_LP_Bipartite3d(
    const std::unordered_map<int, InfinitePlane3d>& inf_planes3d,
    const PP_Bipartite3d& pp_bpt3d, const LP_Bipartite3d& lp_bpt3d,
    int min_support_lines) {
  std::unordered_map<int, InfinitePlane3d> new_inf_planes3d = inf_planes3d;
  PP_Bipartite3d new_pp_bpt3d = pp_bpt3d;
  LP_Bipartite3d new_lp_bpt3d = lp_bpt3d;

  const size_t num_all_planes = inf_planes3d.size();
  size_t num_filtered_planes = 0;

  // check (inf_planes3d, pp_bpt3d, lp_bpt3d) have the same plane ids
  std::vector<int> plane_ids1;
  for (auto it = inf_planes3d.begin(); it != inf_planes3d.end(); it++) {
    plane_ids1.push_back(it->first);
  }
  std::vector<int> plane_ids2 = pp_bpt3d.GetObj2Ids();
  std::vector<int> plane_ids3 = lp_bpt3d.GetObj2Ids();
  std::sort(plane_ids1.begin(), plane_ids1.end());
  std::sort(plane_ids2.begin(), plane_ids2.end());
  std::sort(plane_ids3.begin(), plane_ids3.end());
  for (size_t i = 0; i < plane_ids1.size(); i++) {
    THROW_CHECK_EQ(plane_ids1[i], plane_ids2.at(i));
    THROW_CHECK_EQ(plane_ids1[i], plane_ids3.at(i));
  }

  // delete planes
  for (const int& plane_id : plane_ids1) {
    if (lp_bpt3d.n_2_to_1.at(plane_id).size() < min_support_lines) {
      num_filtered_planes++;
      new_inf_planes3d.erase(plane_id);
      new_pp_bpt3d.DeleteObj2(plane_id);
      new_lp_bpt3d.DeleteObj2(plane_id);
    }
  }

  // re-check (inf_planes3d, pp_bpt3d, lp_bpt3d) have the same plane ids
  plane_ids1.clear();
  for (auto it = new_inf_planes3d.begin(); it != new_inf_planes3d.end(); it++) {
    plane_ids1.push_back(it->first);
  }
  plane_ids2.clear();
  plane_ids3.clear();
  plane_ids2 = new_pp_bpt3d.GetObj2Ids();
  plane_ids3 = new_lp_bpt3d.GetObj2Ids();
  std::sort(plane_ids1.begin(), plane_ids1.end());
  std::sort(plane_ids2.begin(), plane_ids2.end());
  std::sort(plane_ids3.begin(), plane_ids3.end());
  for (size_t i = 0; i < plane_ids1.size(); i++) {
    THROW_CHECK_EQ(plane_ids1[i], plane_ids2.at(i));
    THROW_CHECK_EQ(plane_ids1[i], plane_ids3.at(i));
  }

  THROW_CHECK_EQ(num_all_planes - num_filtered_planes, new_inf_planes3d.size());

  return {num_all_planes, num_filtered_planes, new_inf_planes3d, new_pp_bpt3d,
          new_lp_bpt3d};
}

std::vector<V3D> GetInlierPoint3dsFromPP_Bipartite3d(
    const PP_Bipartite3d& pp_bpt3d) {
  std::vector<V3D> points;
  for (const auto& pair : pp_bpt3d.n_1_to_2) {
    if (!pair.second.empty()) {  // The point has an associated plane
      points.push_back(pp_bpt3d.obj1_map.at(pair.first).p);
    }
  }
  return points;
}

std::vector<Line3d> GetInlierLine3dsFromLP_Bipartite3d(
    const LP_Bipartite3d& lp_bpt3d) {
  std::vector<Line3d> lines;
  for (const auto& pair : lp_bpt3d.n_1_to_2) {
    if (!pair.second.empty()) {  // The line has an associated plane
      lines.push_back(lp_bpt3d.obj1_map.at(pair.first).line);
    }
  }
  return lines;
}

}  // namespace limap
