#pragma once

#include "clmap/base/proposal.h"
#include "clmap/mapping/line_triangulation.h"
#include "clmap/util/types.h"

#include <map>
#include <unordered_map>
#include <vector>

#include "base/graph.h"
#include "base/image_collection.h"
#include "base/line_linker.h"
#include "base/linebase.h"
#include "structures/pl_bipartite.h"
#include "vplib/vpbase.h"

namespace limap {

struct LineMapperConfig {
 public:
  LineMapperConfig() {}
  LineMapperConfig(py::dict dict);

  //////////////////////////////////////////////////////////////////////////////
  // Basic Config
  //////////////////////////////////////////////////////////////////////////////

  // enable debug
  bool debug_mode = false;

  // Offset half pixel for each line.
  bool add_halfpix = false;

  // The min 2D line segment length.
  double min_length_2d = 0.0;  // in pixels

  // Uncertainty of line detection.
  double var2d = 4.0;  // in pixels

  // Line linker.
  LineLinker2dConfig linker2d_config;
  LineLinker3dConfig linker3d_config;

  //////////////////////////////////////////////////////////////////////////////
  // Proposal generation
  //////////////////////////////////////////////////////////////////////////////

  // Whether to use unsharded 3D points.
  bool use_unshared_points = true;

  // Whether to use range filter.
  bool use_range_filter = true;

  // Whether to use two 2D endpoints matches for line triangulation.
  bool use_endpoints_triangulation = false;

  // It is used to check whether 2D line is close to the epipolar line.
  double line_tri_angle_threshold = 1.0;  // in degrees

  // The threshold for line match filtering using weak epipolar constraint.
  double IoU_threshold = 0.1;

  // Sensitivity threshold for 3D line filtering.
  double sensitivity_threshold = 70.0;  // in degrees

  // Max reprojection error for identifying valid unshared 3D points.
  double max_point_line_reproject_distance = 1.0;  // in pixels

  // Whether to use VPs for line triangulation.
  bool use_vp = true;

  // Whether to use SfM points for line triangulation.
  bool use_pointsfm = true;

  // Disable "Line + Line" line triangulation method.
  bool disable_algebraic_triangulation = false;

  // Disable "Multiple Points" line triangulation method.
  bool disable_many_points_triangulation = false;

  // Disable "Line + Point" line triangulation method.
  bool disable_one_point_triangulation = false;

  // Disable "Line + VP" line triangulation method.
  bool disable_vp_triangulation = false;

  //////////////////////////////////////////////////////////////////////////////
  // Best proposal selection
  //////////////////////////////////////////////////////////////////////////////

  // Whether to use local valid 2D lines as the neighbor 2D lines to be
  // searched. If used, it will save a lot of time, but it will lose performance
  // and need more memery.
  bool use_local_valid_lines2d = false;

  //////////////////////////////////////////////////////////////////////////////
  // Line track building
  //////////////////////////////////////////////////////////////////////////////

  // The method of generating line track.
  // "cluster": cluster 2D line segments to form line tracks according to the
  //            similarity of their best proposals and the initial line matches.
  // "best_prop": output all best proposals directly, each best proposal forms
  //              a line track.
  std::string line_track_building_method =
      "cluster";  // ["cluster", "best_prop"]

  // The method of clustering.
  std::string clustering_method =
      "collinearity2D";  // ["greedy", "exhaustive", "avg", "collinearity2D"]

  // Choose the (`num_outliers_aggregator` + 1)th far projected 3D endpoint as
  // the final 3D endpoint.
  int num_outliers_aggregator = 2;

  // Thresholds for determining whether two 2D lines are collinear.
  double max_same_line3D_overlap = 0.0;
  double min_same_line3D_angle = 2.0;      // in degrees
  double min_same_line3D_perp_dist = 2.0;  // in pixels

  // The method of getting a new 3D line by aggregating some known 3D lines.
  std::string aggregate_method =
      "direction_PCA";  // ["point_PCA", "point_weighted_PCA", "direction_PCA",
                        // "direction_weighted_PCA"]
};

class LineMapper {
 public:
  LineMapper() {}
  LineMapper(const LineMapperConfig& config);
  LineMapper(py::dict dict);

  // Setter.
  inline void SetImageCollection(const ImageCollection& image_collection);
  inline void SetAllLine2D(
      const std::unordered_map<image_t, std::vector<Line2d>>& all_line2D);
  inline void SetRanges(const std::pair<V3D, V3D>& ranges);
  inline void SetPLBipartite2d(
      const std::unordered_map<image_t, limap::structures::PL_Bipartite2d>&
          all_pl_bipartite2d);
  inline void SetSfMPoints(
      const std::unordered_map<point3d_t, V3D>& sfm_points);
  inline void SetVPResults(
      const std::unordered_map<image_t, limap::vplib::VPResult>& vp_results);

  // Getter.
  inline const std::unordered_map<image_t, std::vector<int>>&
  GetLocalBestProposalIdx() const;
  inline const std::unordered_map<image_t, std::vector<int>>&
  GetGlobalBestProposalIdx() const;
  size_t GetNumProposal() const;
  size_t GetNumLines2dHavingProposal() const;

  // Initialize 3D line mapping.
  void Initialize();

  //
  void MatchLines2dByEpipolarIoU(
      const std::unordered_map<image_t, std::vector<image_t>>& all_neighbors,
      const int n_matches, const double th_IoU);

  //
  void TriangulateAllImages();

  //
  void LoadBidirectionalMatches(
      const std::unordered_map<
          image_t, std::unordered_map<image_t, Eigen::MatrixXi>>& matches);

  // Triangulate an image with initial line matches.
  void TriangulateImageWithInitialMatches(
      const image_t& ref_image_id,
      const std::unordered_map<image_t, Eigen::MatrixXi>& init_ng_matches);

  // Triangulate an image with exhaustive line matches.
  void TriangulateImageWithExhaustiveMatches(
      const image_t& ref_image_id, const std::vector<image_t>& ng_image_ids);

  // Select local best proposals.
  void SelectLocalBestProposal();

  // Initialize global best proposals.
  void InitializeGlobalBestProposal(const std::string& method,
                                    double global_score_th);

  // Select global best proposals iteratively.
  size_t SelectGlobalBestProposal(const double& global_score_th);

  // Compute mean angle consistency percentage of all support relations.
  //
  // @param best_proposal_type    "local", "global".
  // @param th_angle              angle threshold.
  //
  // @return     (A, num_valid_line2d, num_supports_sum)
  //
  // A: the mean angle consistency percentage at `th_angle` degree of all
  //    support relations.
  // num_valid_line2d: the number of 2D line segments whose best proposal has
  //                   at least one supporting 2D line segment.
  // num_supports_sum: the number of all supporting 2D line segments.
  std::tuple<double, size_t, size_t> ComputeAngleConsistencyPercentage(
      const std::string& best_prop_type, double th_angle) const;

  // Compute mean distance consistency percentage of all support relations.
  //
  // @param best_proposal_type    "local", "global".
  // @param th_dist               3D endpoints max-max perpendicular distance
  //                              threshold.
  //
  // @return     (D, num_valid_line2d, num_supports_sum)
  //
  // D: the mean distance consistency percentage at `th_dist` mm of all support
  //    relations.
  // num_valid_line2d: the number of 2D line segments whose best proposal has
  //                   at least one supporting 2D line segment.
  // num_supports_sum: the number of all supporting 2D line segments.
  std::tuple<double, size_t, size_t> ComputeDistanceConsistencyPercentage(
      const std::string& best_prop_type, double th_dist) const;

  // Compute mean angle and distance consistency percentage of all support
  // relations.
  //
  // @param best_proposal_type    "local", "global".
  // @param th_angle              angle threshold.
  // @param th_dist               3D endpoints max-max perpendicular distance
  //                              threshold.
  //
  // @return     (AD, num_valid_line2d, num_supports_sum)
  //
  // AD: the mean angle and distance consistency percentage at `th_angle`
  //     degrees and `th_dist` mm of all support relations.
  // num_valid_line2d: the number of 2D line segments whose best proposal has
  //                   at least one supporting 2D line segment.
  // num_supports_sum: the number of all supporting 2D line segments.
  std::tuple<double, size_t, size_t> ComputeAngleDistanceConsistencyPercentage(
      const std::string& best_prop_type, double th_angle, double th_dist) const;

  // Build line tracks.
  std::vector<LineTrack> BuildGlobalLineTrack(bool show_valid = false);

 private:
  void PreStartCheck() const;
  void OffsetHalfPixel();
  void TriangulateImage(const image_t& ref_image_id, const image_t& ng_image_id,
                        const std::vector<std::vector<line2d_t>>& matches);
  void CreateMatchGraph(Graph* graph);
  void BuildTracksFromClusters(Graph* graph);

  //////////////////////////////////////////////////////////////////////////////
  // Inputs
  //////////////////////////////////////////////////////////////////////////////

  // Config of LineMapper.
  const LineMapperConfig config_;

  // Measure the similarity of two 2D lines or 3D lines.
  const LineLinker linker_;

  // Class that hold all cameras and images.
  ImageCollection image_collection_;

  // All Line2d for each image.
  std::unordered_map<image_t, std::vector<Line2d>> all_line2d_;

  // Scene ranges.
  std::pair<V3D, V3D> ranges_;

  // 2D bipartite graph associating 2D point and 2D line segment.
  std::shared_ptr<
      std::unordered_map<image_t, limap::structures::PL_Bipartite2d>>
      all_pl_bipartite2d_;

  // SfM points.
  std::unordered_map<point3d_t, V3D> sfm_points_;

  // VP results for each image.
  std::unordered_map<image_t, limap::vplib::VPResult> vp_results_;

  // matches_.at(image_id)[line2d_idx]: (ng_image_id, (ng_line2d_idx))
  std::unordered_map<
      image_t, std::vector<std::unordered_map<image_t, std::vector<line2d_t>>>>
      matches_;

  std::unordered_map<image_t, std::unordered_set<image_t>> neighbors_;

  //////////////////////////////////////////////////////////////////////////////
  // Outputs
  //////////////////////////////////////////////////////////////////////////////

  // 3D line segment proposals for each 2D line segment.
  std::unordered_map<image_t, std::vector<std::vector<Proposal>>> proposals_;

  // Indices of best propsoal selected by local consistency.
  std::unordered_map<image_t, std::vector<int>> local_best_proposal_indices_;

  // Indices of best propsoal selected by global consistency.
  std::unordered_map<image_t, std::vector<int>> global_best_proposal_indices_;

  // Only the global best proposal of the visible 2D line segment can guide the
  // global best proposal selection of other 2D line segment in the next
  // iteration.
  std::unordered_map<image_t, std::vector<char>> visible_line2d_flags_;

  // 3D line segment tracks
  std::vector<LineTrack> tracks_;

  // Number of all detected 2D line segments.
  size_t num_lines2d_;

  // true: if the function SelectLocalBestProposal() has been called.
  bool select_local_best_prop_;
};

////////////////////////////////////////////////////////////////////////////////
// Implementation
////////////////////////////////////////////////////////////////////////////////

void LineMapper::SetImageCollection(const ImageCollection& image_collection) {
  image_collection_ = image_collection;
}

void LineMapper::SetAllLine2D(
    const std::unordered_map<image_t, std::vector<Line2d>>& all_line2D) {
  all_line2d_ = all_line2D;
}

void LineMapper::SetRanges(const std::pair<V3D, V3D>& ranges) {
  ranges_ = ranges;
}

void LineMapper::SetPLBipartite2d(
    const std::unordered_map<image_t, limap::structures::PL_Bipartite2d>&
        all_pl_bipartite2d) {
  all_pl_bipartite2d_ = std::make_shared<
      std::unordered_map<image_t, limap::structures::PL_Bipartite2d>>(
      all_pl_bipartite2d);
}

void LineMapper::SetSfMPoints(
    const std::unordered_map<point3d_t, V3D>& sfm_points) {
  sfm_points_ = sfm_points;
}

void LineMapper::SetVPResults(
    const std::unordered_map<image_t, limap::vplib::VPResult>& vp_results) {
  vp_results_ = vp_results;
}

const std::unordered_map<image_t, std::vector<int>>&
LineMapper::GetLocalBestProposalIdx() const {
  return local_best_proposal_indices_;
}

const std::unordered_map<image_t, std::vector<int>>&
LineMapper::GetGlobalBestProposalIdx() const {
  return global_best_proposal_indices_;
}

}  // namespace limap
