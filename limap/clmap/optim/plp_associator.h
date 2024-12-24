#pragma once

#include "clmap/base/bipartite.h"
#include "clmap/base/infinite_plane3d.h"

#include "optimize/global_pl_association/global_associator.h"

namespace limap {

class PLPAssociatorConfig
    : public limap::optimize::global_pl_association::GlobalAssociatorConfig {
 public:
  PLPAssociatorConfig();
  PLPAssociatorConfig(py::dict dict);

  // Max number of BA iterations.
  int max_ba_num_iteration = 50;

  // Whether to optimize 3D planes.
  bool constant_plane = false;

  bool use_point_geometry = true;
  bool use_line_geometry = true;
  bool use_pointline = true;
  bool use_vpline = true;
  bool use_pointplane = true;
  bool use_lineplane = true;
  bool use_vp_orthogonality = true;
  bool use_vp_collinearity = true;
  bool use_plane_orthogonality = true;
  bool use_plane_parallelism = true;

  // Whether to replace 3D distance with uncertainty
  bool use_uncertainty = false;

  // point-plane association
  std::shared_ptr<ceres::LossFunction> point_plane_association_3d_loss_function;
  double lw_pointplane_association = 1.0;

  // line-plane association
  std::shared_ptr<ceres::LossFunction>
      line_plane_distance_association_3d_loss_function;
  std::shared_ptr<ceres::LossFunction>
      line_plane_angle_association_3d_loss_function;
  std::shared_ptr<ceres::LossFunction>
      plane_orthogonality_association_3d_loss_function;
  std::shared_ptr<ceres::LossFunction>
      plane_parallelism_association_3d_loss_function;
  double lw_lineplane_distance_association = 1.0;
  double lw_lineplane_angle_association = 1.0;
  double lw_plane_orthogonality = 1.0;
  double lw_plane_parallelism = 1.0;

  // plane
  double th_plane_orthogonality_angle = 87.0;
  double th_plane_parallelism_angle = 1.0;
  MinimalInfinitePlane3d::MinimalType mini_plane_type =
      MinimalInfinitePlane3d::MinimalType::CP;
  double plane_intersection_min_angle = 30.0;
  double point_var2d = 1.0;
  double line_var2d = 4.0;
  std::string point_u_method = "median";  // ["median", "average", "min"]
  std::string line_u_method = "median";   // ["median", "average", "min"]
  int num_points_on_line =
      1;  //  # [0, 1, 2], 0: closet-point method, 1: set the median point on
          //  the 3D line segment as the unique fixed point, 2: set the start
          //  and end point on the 3D line segment as two fixed points.

  // output
  double th_hard_pl_si_dist3d = 2.0;
  // Note: th_hard_vpline_angle3d will be set in
  //       `limap::optimize::global_pl_association::GlobalAssociatorConfig`.
  double th_hard_pointplane_si_dist3d = 2.0;
  double th_hard_lineplane_si_dist3d = 2.0;
  double th_hard_lineplane_angle = 30.0;

 private:
  void InitConfig() {
    point_plane_association_3d_loss_function.reset(new ceres::HuberLoss(0.01));
    line_plane_distance_association_3d_loss_function.reset(
        new ceres::HuberLoss(0.01));
    line_plane_angle_association_3d_loss_function.reset(
        new ceres::HuberLoss(0.01));
    plane_orthogonality_association_3d_loss_function.reset(
        new ceres::TrivialLoss());
    plane_parallelism_association_3d_loss_function.reset(
        new ceres::TrivialLoss());
  }
};

class PLPAssociator
    : public limap::optimize::global_pl_association::GlobalAssociator {
 public:
  PLPAssociator() {}
  ~PLPAssociator() {}
  PLPAssociator(const PLPAssociatorConfig& config);

  // init
  void InitPlanes(const std::map<int, InfinitePlane3d>& planes);
  void InitPP_Bipartite3d(const PP_Bipartite3d& pp_bpt3d);
  void InitLP_Bipartite3d(const LP_Bipartite3d& lp_bpt3d);

  // compute uncertainty
  void ComputePointsUncertainty(const std::string& point_u_method = "median");
  void ComputePointsOnLineUncertainty(
      const std::string& line_u_method = "median");

  // get initial cost
  double GetPointInitialCost() const;
  double GetLineInitialCost() const;
  double GetPointLineInitialCost() const;
  double GetVPLineInitialCost() const;
  double GetVPOrthogonalityInitialCost() const;
  double GetVPCollinearityInitialCost() const;
  double GetPointPlaneInitialCost() const;
  double GetLinePlaneDistanceInitialCost() const;
  double GetLinePlaneAngleInitialCost() const;
  double GetPlaneOrthogonalityInitialCost() const;
  double GetPlaneParallelismInitialCost() const;

  // compute adaptive loss weight according to initial cost
  void ComputeAdaptiveLossWeight();

  // output all loss weights
  void PrintLossWeight() const;
  std::unordered_map<std::string, double> GetAllLossWeight() const;

  // setup and solver
  void SetUp();
  bool Solve();

  // update results, these functions should be called in the following order
  void UpdatePointTracks();
  void UpdateLineTracks(int num_outliers_aggregate = 2);
  void UpdateUncertainty(const std::string& point_u_method = "median",
                         const std::string& line_u_method = "median");
  void UpdatePL_Bipartite3d(double th_hard_pl_si_dist3d);
  void UpdateVPLine_Bipartite3d(double th_hard_vpline_angle3d);
  void UpdatePlanes();
  void UpdatePP_Bipartite3d();
  void UpdateLP_Bipartite3d();

  // output
  std::map<int, PointTrack> GetPointTracks() const;
  std::map<int, LineTrack> GetLineTracks() const;
  structures::PL_Bipartite3d GetPL_Bipartite3d() const;
  structures::VPLine_Bipartite3d GetVPLine_Bipartite3d() const;
  std::map<int, InfinitePlane3d> GetPlanes() const;
  PP_Bipartite3d GetPP_Bipartite3d() const;
  LP_Bipartite3d GetLP_Bipartite3d() const;

 protected:
  // parameterization
  void ParameterizePlanes();

  // add residuals
  void AddPointLineAssociationResiduals(
      const std::map<std::pair<int, int>, double>& weights);
  void AddPointPlaneAssociationResiduals(
      const std::unordered_map<int, int>& point_plane_pairs);
  void AddLinePlaneAssociationResiduals(
      const std::unordered_map<int, std::vector<int>>& line_planes_pairs);
  void AddPlaneOrthogonalityResiduals();
  void AddPlaneParallelismResiduals();

  // find associations
  void FindAllPointPlanePairs(
      std::unordered_map<int, int>* point_plane_pairs) const;
  void FindAllLinePlanesPairs(
      std::unordered_map<int, std::vector<int>>* line_planes_pairs) const;
  void FindOrthogonalityPlanePairs(
      double th_orthogonality_angle,
      std::vector<std::pair<int, int>>* planes_pairs) const;
  void FindParallelismPlanePairs(
      double th_parallelism_angle,
      std::vector<std::pair<int, int>>* planes_pairs) const;

  // config
  PLPAssociatorConfig config_;

  // point-line association
  structures::PL_Bipartite3d pl_bpt3d_;

  // vp-line association
  structures::VPLine_Bipartite3d vpline_bpt3d_;

  // planes
  std::map<int, MinimalInfinitePlane3d> mini_planes_;
  std::map<int, InfinitePlane3d> planes_;

  // point-plane association
  PP_Bipartite3d pp_bpt3d_;

  // line-plane association
  LP_Bipartite3d lp_bpt3d_;

  // (point3d_id, uncertainty of points3d)
  std::map<int, double> un_points_;

  // (line3d_id, ((a point3d on line3d, uncertainty of the point3d)))
  std::map<int, std::vector<std::pair<V3D, double>>> un_points_on_lines_;

  // (line3d_id, uncertainty of the start of the line3d)
  std::map<int, double> un_start_on_lines_;

  // (line3d_id, uncertainty of the end of the line3d)
  std::map<int, double> un_end_on_lines_;
};

}  // namespace limap
