#include "clmap/optim/plp_associator.h"

#include "clmap/base/geometry.h"
#include "clmap/mapping/line_triangulation.h"
#include "clmap/optim/cost_function.h"

#include <unordered_set>

#include "base/camera_models.h"
#include "base/camera_view.h"
#include "ceresbase/parameterization.h"
#include "optimize/global_pl_association/cost_functions.h"
#include "optimize/hybrid_bundle_adjustment/cost_functions.h"
#include "optimize/hybrid_bundle_adjustment/hybrid_bundle_adjustment.h"
#include "optimize/line_refinement/cost_functions.h"
#include <colmap/optim/bundle_adjustment.h>
#include <colmap/util/misc.h>
#include <colmap/util/threading.h>

namespace limap {

PLPAssociatorConfig::PLPAssociatorConfig()
    : limap::optimize::global_pl_association::GlobalAssociatorConfig() {
  InitConfig();
}

PLPAssociatorConfig::PLPAssociatorConfig(py::dict dict)
    : limap::optimize::global_pl_association::GlobalAssociatorConfig(dict) {
  InitConfig();
  ASSIGN_PYDICT_ITEM(dict, max_ba_num_iteration, int)
  solver_options.max_num_iterations = max_ba_num_iteration;
  ASSIGN_PYDICT_ITEM(dict, constant_plane, bool)
  ASSIGN_PYDICT_ITEM(dict, use_point_geometry, bool)
  ASSIGN_PYDICT_ITEM(dict, use_line_geometry, bool)
  ASSIGN_PYDICT_ITEM(dict, use_pointline, bool)
  ASSIGN_PYDICT_ITEM(dict, use_vpline, bool)
  ASSIGN_PYDICT_ITEM(dict, use_pointplane, bool)
  ASSIGN_PYDICT_ITEM(dict, use_lineplane, bool)
  ASSIGN_PYDICT_ITEM(dict, use_vp_orthogonality, bool)
  ASSIGN_PYDICT_ITEM(dict, use_vp_collinearity, bool)
  ASSIGN_PYDICT_ITEM(dict, use_plane_orthogonality, bool)
  ASSIGN_PYDICT_ITEM(dict, use_plane_parallelism, bool)
  ASSIGN_PYDICT_ITEM(dict, use_uncertainty, bool)
  ASSIGN_PYDICT_ITEM(dict, lw_pointplane_association, double)
  ASSIGN_PYDICT_ITEM(dict, lw_lineplane_distance_association, double)
  ASSIGN_PYDICT_ITEM(dict, lw_lineplane_angle_association, double)
  ASSIGN_PYDICT_ITEM(dict, lw_plane_orthogonality, double)
  ASSIGN_PYDICT_ITEM(dict, lw_plane_parallelism, double)
  ASSIGN_PYDICT_ITEM(dict, th_plane_orthogonality_angle, double)
  ASSIGN_PYDICT_ITEM(dict, th_plane_parallelism_angle, double)
  std::string mini_plane_type_str;
  ASSIGN_PYDICT_ITEM(dict, mini_plane_type_str, std::string)
  if (mini_plane_type_str == "CP") {
    mini_plane_type = MinimalInfinitePlane3d::MinimalType::CP;
  } else if (mini_plane_type_str == "QUATERNION") {
    mini_plane_type = MinimalInfinitePlane3d::MinimalType::QUATERNION;
  } else if (mini_plane_type_str == "HOMOGENEOUS") {
    mini_plane_type = MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS;
  } else {
    throw std::invalid_argument("Error: Plane minimal type is invalid.");
  }
  ASSIGN_PYDICT_ITEM(dict, plane_intersection_min_angle, double)
  ASSIGN_PYDICT_ITEM(dict, point_var2d, double)
  ASSIGN_PYDICT_ITEM(dict, line_var2d, double)
  ASSIGN_PYDICT_ITEM(dict, point_u_method, std::string)
  ASSIGN_PYDICT_ITEM(dict, line_u_method, std::string)
  ASSIGN_PYDICT_ITEM(dict, num_points_on_line, int)
  ASSIGN_PYDICT_ITEM(dict, th_hard_pl_si_dist3d, double)
  ASSIGN_PYDICT_ITEM(dict, th_hard_pointplane_si_dist3d, double)
  ASSIGN_PYDICT_ITEM(dict, th_hard_lineplane_si_dist3d, double)
  ASSIGN_PYDICT_ITEM(dict, th_hard_lineplane_angle, double)
}

PLPAssociator::PLPAssociator(const PLPAssociatorConfig& config)
    : limap::optimize::global_pl_association::GlobalAssociator(config),
      config_(config) {}

void PLPAssociator::InitPlanes(const std::map<int, InfinitePlane3d>& planes) {
  mini_planes_.clear();
  planes_.clear();
  planes_ = planes;
  for (auto it = planes.begin(); it != planes.end(); it++) {
    const InfinitePlane3d& inf_plane3d = it->second;
    mini_planes_.emplace(it->first, MinimalInfinitePlane3d(
                                        inf_plane3d, config_.mini_plane_type));
  }
}

void PLPAssociator::InitPP_Bipartite3d(const PP_Bipartite3d& pp_bpt3d) {
  pp_bpt3d_ = pp_bpt3d;
}

void PLPAssociator::InitLP_Bipartite3d(const LP_Bipartite3d& lp_bpt3d) {
  lp_bpt3d_ = lp_bpt3d;
}

void PLPAssociator::ComputePointsUncertainty(
    const std::string& point_u_method) {
  un_points_.clear();
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); it++) {
    int point_id = it->first;
    const auto& point_track = it->second;
    double uncertainty;
    geometry::ComputePoint3dUncertainty(point_track, imagecols_, &uncertainty,
                                        config_.point_var2d, point_u_method);
    un_points_.emplace(point_id, uncertainty);
  }
}

void PLPAssociator::ComputePointsOnLineUncertainty(
    const std::string& line_u_method) {
  un_points_on_lines_.clear();
  un_start_on_lines_.clear();
  un_end_on_lines_.clear();

  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); it++) {
    int line_id = it->first;
    const auto& line_track = it->second;

    std::pair<V3D, double> u_start;
    std::pair<V3D, double> u_end;
    std::pair<V3D, double> u_median;

    geometry::ComputeLine3dUncertainty(line_track, imagecols_, &u_start, &u_end,
                                       &u_median, config_.line_var2d,
                                       line_u_method);

    un_start_on_lines_.emplace(line_id, u_start.second);
    un_end_on_lines_.emplace(line_id, u_end.second);

    if (config_.num_points_on_line == 0) {
      un_points_on_lines_.emplace(line_id,
                                  std::vector<std::pair<V3D, double>>());
    } else if (config_.num_points_on_line == 1) {
      std::vector<std::pair<V3D, double>> points;
      points.push_back(u_median);
      un_points_on_lines_.emplace(line_id, points);
    } else if (config_.num_points_on_line == 2) {
      std::vector<std::pair<V3D, double>> points;
      points.push_back(u_start);
      points.push_back(u_end);
      un_points_on_lines_.emplace(line_id, points);
    } else {
      throw std::runtime_error("Error: num_points_on_line is invalid");
    }
  }
}

double PLPAssociator::GetPointInitialCost() const {
  if (points_.empty()) return 0.0;
  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto points = points_;
  auto imagecols = imagecols_;

  // set up
  for (auto it = points.begin(); it != points.end(); ++it) {
    int track_id = it->first;
    const PointTrack& track = point_tracks_.at(track_id);
    ceres::LossFunction* loss_function =
        config_.point_geometric_loss_function.get();
    for (size_t i = 0; i < track.count_images(); ++i) {
      int img_id = track.image_id_list[i];
      int model_id = imagecols_.camview(img_id).cam.ModelId();
      V2D p2d = track.p2d_list[i];

      ceres::CostFunction* cost_function = nullptr;
      switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                       \
  case CameraModel::kModelId:                                                \
    cost_function =                                                          \
        optimize::hybrid_bundle_adjustment::PointGeometricRefinementFunctor< \
            CameraModel>::Create(p2d, NULL, NULL, NULL);                     \
    break;
        LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
      }

      ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
          loss_function, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
      ceres::ResidualBlockId block_id = problem->AddResidualBlock(
          cost_function, scaled_loss_function, points.at(track_id).data(),
          imagecols.params_data(img_id), imagecols.qvec_data(img_id),
          imagecols.tvec_data(img_id));
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetPointInitialCost] initial cost = " << summary.initial_cost
            << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetLineInitialCost() const {
  if (lines_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto lines = lines_;
  auto imagecols = imagecols_;

  // set up
  for (auto it = lines.begin(); it != lines.end(); ++it) {
    int track_id = it->first;
    const LineTrack& track = line_tracks_.at(track_id);
    ceres::LossFunction* loss_function =
        config_.line_geometric_loss_function.get();

    // compute line weights
    auto idmap = track.GetIdMap();
    std::vector<double> weights;
    ComputeLineWeights(track, weights);

    // add to problem for each supporting image (for each supporting line)
    auto& minimal_line = lines.at(track_id);
    std::vector<int> image_ids = track.GetSortedImageIds();
    for (auto it1 = image_ids.begin(); it1 != image_ids.end(); ++it1) {
      int img_id = *it1;
      int model_id = imagecols_.camview(img_id).cam.ModelId();
      const auto& ids = idmap.at(img_id);
      for (auto it2 = ids.begin(); it2 != ids.end(); ++it2) {
        const Line2d& line = track.line2d_list[*it2];
        double weight = weights[*it2];
        ceres::CostFunction* cost_function = nullptr;

        switch (model_id) {
#define CAMERA_MODEL_CASE(CameraModel)                                         \
  case CameraModel::kModelId:                                                  \
    cost_function = optimize::line_refinement::GeometricRefinementFunctor<     \
        CameraModel>::Create(line, NULL, NULL, NULL, config_.geometric_alpha); \
    break;
          LIMAP_UNDISTORTED_CAMERA_MODEL_SWITCH_CASES
#undef CAMERA_MODEL_CASE
        }

        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
            loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function, minimal_line.uvec.data(),
            minimal_line.wvec.data(), imagecols.params_data(img_id),
            imagecols.qvec_data(img_id), imagecols.tvec_data(img_id));
      }
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetLineInitialCost] initial cost = " << summary.initial_cost
            << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetPointLineInitialCost() const {
  if (points_.empty() || lines_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto points = points_;
  auto lines = lines_;

  // set up
  auto weights = construct_weights_pointline(config_.th_weight_pointline);
  ceres::LossFunction* loss_function =
      config_.point_line_association_3d_loss_function.get();
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    double weight = it->second;

    double uncertainty =
        config_.use_uncertainty ? un_points_.at(point3d_id) : 1.0;

    ceres::CostFunction* cost_function =
        PointLineAssociation3dFunctor::Create(uncertainty);
    ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem->AddResidualBlock(
        cost_function, scaled_loss_function, points.at(point3d_id).data(),
        lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data());
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetPointLineInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetVPLineInitialCost() const {
  if (vps_.empty() || lines_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto vps = vps_;
  auto lines = lines_;

  // set up
  auto weights = construct_weights_vpline(config_.th_count_vpline);
  ceres::LossFunction* loss_function =
      config_.vp_line_association_3d_loss_function.get();
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int vp3d_id = it->first.first;
    int line3d_id = it->first.second;
    int weight = it->second;

    ceres::CostFunction* cost_function =
        optimize::global_pl_association::VPLineAssociation3dFunctor::Create();
    ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
        loss_function, double(weight), ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem->AddResidualBlock(
        cost_function, scaled_loss_function, vps.at(vp3d_id).data(),
        lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data());
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetVPLineInitialCost] initial cost = " << summary.initial_cost
            << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetVPOrthogonalityInitialCost() const {
  if (vps_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto vps = vps_;

  // set up
  ceres::LossFunction* loss_function =
      config_.vp_orthogonality_loss_function.get();
  auto pairs =
      construct_pairs_vp_orthogonality(vps, config_.th_angle_orthogonality);
  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
    int id1 = it->first;
    int id2 = it->second;

    ceres::CostFunction* cost_function =
        optimize::global_pl_association::VPOrthogonalityFunctor::Create();
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id =
        problem->AddResidualBlock(cost_function, scaled_loss_function,
                                  vps.at(id1).data(), vps.at(id2).data());
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetVPOrthogonalityInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetVPCollinearityInitialCost() const {
  if (vps_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto vps = vps_;

  // set up
  ceres::LossFunction* loss_function =
      config_.vp_collinearity_loss_function.get();
  auto pairs =
      construct_pairs_vp_collinearity(vps, config_.th_angle_collinearity);

  for (auto it = pairs.begin(); it != pairs.end(); ++it) {
    int id1 = it->first;
    int id2 = it->second;

    ceres::CostFunction* cost_function =
        optimize::global_pl_association::VPCollinearityFunctor::Create();
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id =
        problem->AddResidualBlock(cost_function, scaled_loss_function,
                                  vps.at(id1).data(), vps.at(id2).data());
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetVPCollinearityInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetPointPlaneInitialCost() const {
  if (points_.empty() || mini_planes_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto points = points_;
  auto mini_planes = mini_planes_;

  // set up
  std::unordered_map<int, int> point_plane_pairs;
  FindAllPointPlanePairs(&point_plane_pairs);
  ceres::LossFunction* loss_function =
      config_.point_plane_association_3d_loss_function.get();

  for (const auto& point_plane_pair : point_plane_pairs) {
    int point3d_id = point_plane_pair.first;
    int plane3d_id = point_plane_pair.second;

    THROW_CHECK(point_tracks_.find(point3d_id) != point_tracks_.end());
    THROW_CHECK(points_.find(point3d_id) != points_.end());
    THROW_CHECK(planes_.find(plane3d_id) != planes_.end())
    THROW_CHECK(mini_planes_.find(plane3d_id) != mini_planes_.end())

    // number of images in point track.
    double weight = point_tracks_.at(point3d_id).count_images();

    double uncertainty =
        config_.use_uncertainty ? un_points_.at(point3d_id) : 1.0;

    ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);

    ceres::CostFunction* cost_function = PointPlaneAssociation3dFunctor::Create(
        config_.mini_plane_type, uncertainty);

    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id = problem->AddResidualBlock(
          cost_function, scaled_loss_function, points.at(point3d_id).data(),
          mini_planes.at(plane3d_id).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id = problem->AddResidualBlock(
          cost_function, scaled_loss_function, points.at(point3d_id).data(),
          mini_planes.at(plane3d_id).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id = problem->AddResidualBlock(
          cost_function, scaled_loss_function, points.at(point3d_id).data(),
          mini_planes.at(plane3d_id).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetPointPlaneInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetLinePlaneDistanceInitialCost() const {
  if (lines_.empty() || mini_planes_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto lines = lines_;
  auto mini_planes = mini_planes_;

  // set up
  std::unordered_map<int, std::vector<int>> line_planes_pairs;
  FindAllLinePlanesPairs(&line_planes_pairs);

  for (const auto& line_planes_pair : line_planes_pairs) {
    int line3d_id = line_planes_pair.first;
    const auto& plane3d_ids = line_planes_pair.second;

    THROW_CHECK(line_tracks_.find(line3d_id) != line_tracks_.end());
    THROW_CHECK(lines_.find(line3d_id) != lines_.end());
    for (const auto& plane3d_id : plane3d_ids) {
      THROW_CHECK(planes_.find(plane3d_id) != planes_.end());
      THROW_CHECK(mini_planes_.find(plane3d_id) != mini_planes_.end());
    }

    // number of images in point track
    double weight = line_tracks_.at(line3d_id).count_images();

    std::vector<std::pair<V3D, double>> un_points_on_line_tmp =
        un_points_on_lines_.at(line3d_id);

    if (!config_.use_uncertainty) {
      for (auto& kv : un_points_on_line_tmp) {
        kv.second = 1.0;
      }
    }

    for (const auto& plane3d_id : plane3d_ids) {
      ceres::LossFunction* loss_function =
          config_.line_plane_distance_association_3d_loss_function.get();
      ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);

      ceres::CostFunction* cost_function =
          LinePlaneAssociation3dDistanceFunctor::Create(config_.mini_plane_type,
                                                        un_points_on_line_tmp);

      if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).cpvec.data());
      } else if (config_.mini_plane_type ==
                 MinimalInfinitePlane3d::MinimalType::QUATERNION) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).qvec.data());
      } else if (config_.mini_plane_type ==
                 MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).hvec.data());
      } else {
        throw std::invalid_argument("Error: minimal_type is invalid");
      }
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetLinePlaneDistanceInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetLinePlaneAngleInitialCost() const {
  if (lines_.empty() || mini_planes_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto lines = lines_;
  auto mini_planes = mini_planes_;

  // set up
  std::unordered_map<int, std::vector<int>> line_planes_pairs;
  FindAllLinePlanesPairs(&line_planes_pairs);

  for (const auto& line_planes_pair : line_planes_pairs) {
    int line3d_id = line_planes_pair.first;
    const auto& plane3d_ids = line_planes_pair.second;

    THROW_CHECK(line_tracks_.find(line3d_id) != line_tracks_.end());
    THROW_CHECK(lines_.find(line3d_id) != lines_.end());
    for (const auto& plane3d_id : plane3d_ids) {
      THROW_CHECK(planes_.find(plane3d_id) != planes_.end());
      THROW_CHECK(mini_planes_.find(plane3d_id) != mini_planes_.end());
    }

    // number of images in point track
    double weight = line_tracks_.at(line3d_id).count_images();

    for (const auto& plane3d_id : plane3d_ids) {
      ceres::LossFunction* loss_function =
          config_.line_plane_angle_association_3d_loss_function.get();
      ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
          loss_function, weight, ceres::DO_NOT_TAKE_OWNERSHIP);

      ceres::CostFunction* cost_function =
          LinePlaneAssociation3dAngleFunctor::Create(config_.mini_plane_type);

      if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).cpvec.data());
      } else if (config_.mini_plane_type ==
                 MinimalInfinitePlane3d::MinimalType::QUATERNION) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).qvec.data());
      } else if (config_.mini_plane_type ==
                 MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
        ceres::ResidualBlockId block_id = problem->AddResidualBlock(
            cost_function, scaled_loss_function,
            lines.at(line3d_id).uvec.data(), lines.at(line3d_id).wvec.data(),
            mini_planes.at(plane3d_id).hvec.data());
      } else {
        throw std::invalid_argument("Error: minimal_type is invalid");
      }
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetLinePlaneAngleInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetPlaneOrthogonalityInitialCost() const {
  if (mini_planes_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto mini_planes = mini_planes_;

  // set up
  ceres::LossFunction* loss_function =
      config_.plane_orthogonality_association_3d_loss_function.get();

  std::vector<std::pair<int, int>> planes_pairs;
  FindOrthogonalityPlanePairs(config_.th_plane_orthogonality_angle,
                              &planes_pairs);

  for (auto it = planes_pairs.begin(); it != planes_pairs.end(); ++it) {
    int plane_id1 = it->first;
    int plane_id2 = it->second;

    ceres::CostFunction* cost_function =
        PlaneOrthogonalityFunctor::Create(config_.mini_plane_type);
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);

    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).cpvec.data(),
                                    mini_planes.at(plane_id2).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).qvec.data(),
                                    mini_planes.at(plane_id2).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).hvec.data(),
                                    mini_planes.at(plane_id2).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetPlaneOrthogonalityInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

double PLPAssociator::GetPlaneParallelismInitialCost() const {
  if (mini_planes_.empty()) return 0.0;

  std::unique_ptr<ceres::Problem> problem;
  problem.reset(new ceres::Problem(config_.problem_options));
  ceres::Solver::Summary summary;

  // copy
  auto mini_planes = mini_planes_;

  // set up
  ceres::LossFunction* loss_function =
      config_.plane_parallelism_association_3d_loss_function.get();

  std::vector<std::pair<int, int>> planes_pairs;
  FindParallelismPlanePairs(config_.th_plane_parallelism_angle, &planes_pairs);

  for (auto it = planes_pairs.begin(); it != planes_pairs.end(); ++it) {
    int plane_id1 = it->first;
    int plane_id2 = it->second;

    ceres::CostFunction* cost_function =
        PlaneParallelismFunctor::Create(config_.mini_plane_type);
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, 1.0, ceres::DO_NOT_TAKE_OWNERSHIP);
    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).cpvec.data(),
                                    mini_planes.at(plane_id2).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).qvec.data(),
                                    mini_planes.at(plane_id2).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id =
          problem->AddResidualBlock(cost_function, scaled_loss_function,
                                    mini_planes.at(plane_id1).hvec.data(),
                                    mini_planes.at(plane_id2).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }

  // solve
  ceres::Solver::Options solver_options = config_.solver_options;
  solver_options.max_num_iterations = 0;

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR
  ceres::Solve(solver_options, problem.get(), &summary);
  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[GetPlaneParallelismInitialCost] initial cost = "
            << summary.initial_cost << std::endl;
  CHECK_GE(summary.initial_cost, 0);
  return summary.initial_cost;
}

void PLPAssociator::ComputeAdaptiveLossWeight() {
  // compute initial cost for each term
  double point_init_cost = GetPointInitialCost();
  double line_init_cost = GetLineInitialCost();
  double point_line_init_cost = GetPointLineInitialCost();
  double vp_line_init_cost = GetVPLineInitialCost();
  double vp_orthogonality_init_cost = GetVPOrthogonalityInitialCost();
  double vp_collinearity_init_cost = GetVPCollinearityInitialCost();
  double point_plane_init_cost = GetPointPlaneInitialCost();
  double line_plane_distance_init_cost = GetLinePlaneDistanceInitialCost();
  double line_plane_angle_init_cost = GetLinePlaneAngleInitialCost();
  double plane_orthogonality_init_cost = GetPlaneOrthogonalityInitialCost();
  double plane_parallelism_init_cost = GetPlaneParallelismInitialCost();

  // print initial cost
  auto print = [](const std::string& info, double value) {
    std::cout << info << value << std::endl;
  };
  print("point_init_cost = ", point_init_cost);
  print("line_init_cost (reference initial cost) = ", line_init_cost);
  print("point_line_init_cost = ", point_line_init_cost);
  print("vp_line_init_cost = ", vp_line_init_cost);
  print("vp_orthogonality_init_cost = ", vp_orthogonality_init_cost);
  print("vp_collinearity_init_cost = ", vp_collinearity_init_cost);
  print("point_plane_init_cost = ", point_plane_init_cost);
  print("line_plane_distance_init_cost = ", line_plane_distance_init_cost);
  print("line_plane_angle_init_cost = ", line_plane_angle_init_cost);
  print("plane_orthogonality_init_cost = ", plane_orthogonality_init_cost);
  print("plane_parallelism_init_cost = ", plane_parallelism_init_cost);

  // set `line_init_cost` as the reference initial cost
  const double ref_init_cost = (line_init_cost > 0)
                                   ? line_init_cost
                                   : 1000.0;  // 1000.0 is an empirical value

  // for E_{L\Pi}, first pull the angle and distance to the same order of
  // magnitude, the `line_plane_distance_init_cost` is set as the reference
  const double lw_inner_line_plane_angle =
      (line_plane_distance_init_cost > 0 && line_plane_angle_init_cost > 0)
          ? line_plane_distance_init_cost / line_plane_angle_init_cost
          : 1.0;

  config_.lw_point =
      (point_init_cost > 0) ? ref_init_cost / point_init_cost : 1.0;
  config_.lw_pointline_association =
      (point_line_init_cost > 0) ? ref_init_cost / point_line_init_cost : 1.0;
  config_.lw_vpline_association =
      (vp_line_init_cost > 0) ? ref_init_cost / vp_line_init_cost : 1.0;
  config_.lw_pointplane_association =
      (point_plane_init_cost > 0) ? ref_init_cost / point_plane_init_cost : 1.0;
  config_.lw_lineplane_distance_association =
      (line_plane_distance_init_cost > 0)
          ? ref_init_cost /
                (line_plane_distance_init_cost +
                 lw_inner_line_plane_angle * line_plane_angle_init_cost)
          : 1.0;
  config_.lw_lineplane_angle_association =
      (line_plane_angle_init_cost > 0)
          ? lw_inner_line_plane_angle * ref_init_cost /
                (line_plane_distance_init_cost +
                 lw_inner_line_plane_angle * line_plane_angle_init_cost)
          : 1.0;

  // regularization terms
  const double ref_reg_init_cost = 1.0 * ref_init_cost;
  config_.lw_vp_orthogonality =
      (vp_orthogonality_init_cost > 0)
          ? ref_reg_init_cost / vp_orthogonality_init_cost
          : 1.0;
  config_.lw_vp_collinearity =
      (vp_collinearity_init_cost > 0)
          ? ref_reg_init_cost / vp_collinearity_init_cost
          : 1.0;
  config_.lw_plane_orthogonality =
      (plane_orthogonality_init_cost > 0)
          ? ref_reg_init_cost / plane_orthogonality_init_cost
          : 1.0;
  config_.lw_plane_parallelism =
      (plane_parallelism_init_cost > 0)
          ? ref_reg_init_cost / plane_parallelism_init_cost
          : 1.0;
}

void PLPAssociator::PrintLossWeight() const {
  auto print = [](std::string info, double value) {
    std::cout << info << value << std::endl;
  };
  print("lw_point = ", config_.lw_point);
  print("lw_line (the fixed parameter, always equals 1) = ", 1.0);
  print("lw_pointline_association = ", config_.lw_pointline_association);
  print("lw_vpline_association = ", config_.lw_vpline_association);
  print("lw_vp_orthogonality = ", config_.lw_vp_orthogonality);
  print("lw_vp_collinearity = ", config_.lw_vp_collinearity);
  print("lw_pointplane_association = ", config_.lw_pointplane_association);
  print("lw_lineplane_distance_association = ",
        config_.lw_lineplane_distance_association);
  print("lw_lineplane_angle_association = ",
        config_.lw_lineplane_angle_association);
  print("lw_plane_orthogonality = ", config_.lw_plane_orthogonality);
  print("lw_plane_parallelism = ", config_.lw_plane_parallelism);
}

std::unordered_map<std::string, double> PLPAssociator::GetAllLossWeight()
    const {
  std::unordered_map<std::string, double> loss_weight_dict;
  loss_weight_dict["lw_point"] = config_.lw_point;
  loss_weight_dict["lw_pointline_association"] =
      config_.lw_pointline_association;
  loss_weight_dict["lw_vpline_association"] = config_.lw_vpline_association;
  loss_weight_dict["lw_vp_orthogonality"] = config_.lw_vp_orthogonality;
  loss_weight_dict["lw_vp_collinearity"] = config_.lw_vp_collinearity;
  loss_weight_dict["lw_pointplane_association"] =
      config_.lw_pointplane_association;
  loss_weight_dict["lw_lineplane_distance_association"] =
      config_.lw_lineplane_distance_association;
  loss_weight_dict["lw_lineplane_angle_association"] =
      config_.lw_lineplane_angle_association;
  loss_weight_dict["lw_plane_orthogonality"] = config_.lw_plane_orthogonality;
  loss_weight_dict["lw_plane_parallelism"] = config_.lw_plane_parallelism;
  return loss_weight_dict;
}

void PLPAssociator::SetUp() {
  // reset problem
  problem_.reset(new ceres::Problem(config_.problem_options));

  // add residuals
  // R1.1: point geometric residual
  if (config_.use_point_geometry &&
      (!config_.constant_point || !config_.constant_intrinsics ||
       !config_.constant_pose)) {
    std::cout << "[set up] use point geometry" << std::endl;
    for (auto it = points_.begin(); it != points_.end(); ++it) {
      AddPointGeometricResiduals(it->first);
    }
  }

  // R1.2: line geometric residual
  if (config_.use_line_geometry &&
      (!config_.constant_line || !config_.constant_intrinsics ||
       !config_.constant_pose)) {
    std::cout << "[set up] use line geometry" << std::endl;
    for (auto it = lines_.begin(); it != lines_.end(); ++it) {
      AddLineGeometricResiduals(it->first);
    }
  }

  // R2.1: point line association residual
  if (config_.use_pointline &&
      (!config_.constant_point || !config_.constant_line)) {
    std::cout << "[set up] use point-line" << std::endl;
    auto weights = construct_weights_pointline(config_.th_weight_pointline);
    AddPointLineAssociationResiduals(weights);
  }

  // R2.2: vp line association residual
  if (config_.use_vpline && (!config_.constant_vp || !config_.constant_line)) {
    std::cout << "[set up] use line-VP" << std::endl;
    auto weights = construct_weights_vpline(config_.th_count_vpline);
    AddVPLineAssociationResiduals(weights);
  }
  // R2.3: vp orthogonality residual
  if (config_.use_vp_orthogonality && !config_.constant_vp) {
    std::cout << "[set up] use VP orthogonality" << std::endl;
    AddVPOrthogonalityResiduals();
  }
  // R2.4: vp collinearity residual
  if (config_.use_vp_collinearity && !config_.constant_vp) {
    std::cout << "[set up] use VP collinearity" << std::endl;
    AddVPCollinearityResiduals();
  }

  // R3.1: point plane association residual
  if (config_.use_pointplane &&
      (!config_.constant_point || !config_.constant_plane)) {
    std::cout << "[set up] use point-plane" << std::endl;
    std::unordered_map<int, int> point_plane_pairs;
    FindAllPointPlanePairs(&point_plane_pairs);
    AddPointPlaneAssociationResiduals(point_plane_pairs);
  }

  // R3.2: line plane association residual
  if (config_.use_lineplane &&
      (!config_.constant_line || !config_.constant_plane)) {
    std::cout << "[set up] use line-plane" << std::endl;
    std::unordered_map<int, std::vector<int>> line_planes_pairs;
    FindAllLinePlanesPairs(&line_planes_pairs);
    AddLinePlaneAssociationResiduals(line_planes_pairs);
  }

  // R3.3: plane orthogonality residual
  if (config_.use_plane_orthogonality && !config_.constant_plane) {
    std::cout << "[set up] use plane orthogonality" << std::endl;
    AddPlaneOrthogonalityResiduals();
  }
  // R3.4: plane parallelism residual
  if (config_.use_plane_parallelism && !config_.constant_plane) {
    std::cout << "[set up] use plane parallelism" << std::endl;
    AddPlaneParallelismResiduals();
  }

  auto print = [](std::string info, double value) {
    std::cout << info << value << std::endl;
  };
  print("[set up] constant_intrinsics = ", config_.constant_intrinsics);
  print("[set up] constant_principal_point  = ",
        config_.constant_principal_point);
  print("[set up] constant_pose = ", config_.constant_pose);
  print("[set up] constant_point = ", config_.constant_point);
  print("[set up] constant_line = ", config_.constant_line);
  print("[set up] constant_vp = ", config_.constant_vp);
  print("[set up] constant_plane = ", config_.constant_plane);

  // parameterize
  ParameterizeCameras();
  ParameterizePoints();
  ParameterizeLines();
  ParameterizeVPs();
  ParameterizePlanes();
}

bool PLPAssociator::Solve() {
  if (problem_->NumParameterBlocks() == 0) return false;
  if (problem_->NumResiduals() == 0) return false;
  ceres::Solver::Options solver_options = config_.solver_options;

  // Empirical choice.
  const size_t kMaxNumImagesDirectDenseSolver = 50;
  const size_t kMaxNumImagesDirectSparseSolver = 900;
  const size_t num_images = imagecols_.NumImages();
  if (num_images <= kMaxNumImagesDirectDenseSolver) {
    solver_options.linear_solver_type = ceres::DENSE_SCHUR;
  } else if (num_images <= kMaxNumImagesDirectSparseSolver) {
    solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
  } else {  // Indirect sparse (preconditioned CG) solver.
    solver_options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    solver_options.preconditioner_type = ceres::SCHUR_JACOBI;
  }

  solver_options.num_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_threads);
#if CERES_VERSION_MAJOR < 2
  solver_options.num_linear_solver_threads =
      colmap::GetEffectiveNumThreads(solver_options.num_linear_solver_threads);
#endif  // CERES_VERSION_MAJOR

  std::string solver_error;
  CHECK(solver_options.IsValid(&solver_error)) << solver_error;

  ceres::Solve(solver_options, problem_.get(), &summary_);
  if (solver_options.minimizer_progress_to_stdout) {
    std::cout << std::endl;
  }

  if (config_.print_summary) {
    colmap::PrintHeading2("Optimization report");
    colmap::PrintSolverSummary(
        summary_);  // We need to replace this with our own Printer!!!
  }
  std::cout << "[solve] total initial cost = " << summary_.initial_cost
            << std::endl;
  return true;
}

void PLPAssociator::UpdatePointTracks() {
  std::unordered_set<int> removed_point3d_ids;
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
    int point3d_id = it->first;
    (it->second).p = points_.at(point3d_id);

    // TODO: filter pointtrack
  }
  // filter
  for (const int id : removed_point3d_ids) {
    points_.erase(id);
    point_tracks_.erase(id);
  }
}

void PLPAssociator::UpdateLineTracks(int num_outliers_aggregate) {
  std::unordered_set<int> removed_line3d_ids;
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    int line3d_id = it->first;
    auto& line_track = it->second;

    
    Line3d line3d;
    if (line_track.line3d_list.empty()) {
      // collect views for each observed 2D line segment
      size_t n_views = line_track.image_id_list.size();
      std::vector<CameraView> views(n_views);
      for (size_t i = 0; i < n_views; i++) {
        views[i] = imagecols_.camview(line_track.image_id_list[i]);
      }
      // collect line2ds for each observed 2D line segment
      std::vector<Line2d> line2ds(n_views);
      for (size_t i = 0; i < n_views; i++) {
        line2ds[i] = line_track.line2d_list[i];
      }
      line3d = GetLineSegmentFromInfiniteLine3d(
          lines_.at(line3d_id).GetInfiniteLine(), views, line2ds,
          num_outliers_aggregate);
    } else {
      line3d = GetLineSegmentFromInfiniteLine3d(
          lines_.at(line3d_id).GetInfiniteLine(), line_track.line3d_list,
          num_outliers_aggregate);
    }

    line_track.line = line3d;

    // TODO: filter linetrack
  }
  // filter
  for (const int id : removed_line3d_ids) {
    lines_.erase(id);
    line_tracks_.erase(id);
  }
}

void PLPAssociator::UpdateUncertainty(const std::string& point_u_method,
                                      const std::string& line_u_method) {
  // Note: point tracks and line tracks should be updated in advance.
  ComputePointsUncertainty(point_u_method);
  ComputePointsOnLineUncertainty(line_u_method);
}

void PLPAssociator::UpdatePL_Bipartite3d(const double th_hard_pl_si_dist3d) {
  if (all_bpt2ds_.empty()) return;

  // init bipartite
  structures::PL_Bipartite3d bpt;

  // get pointtracks
  // for (auto it = points_.begin(); it != points_.end(); ++it) {
  //   int point3d_id = it->first;
  //   PointTrack ptrack = PointTrack(point_tracks_.at(point3d_id));
  //   ptrack.p = points_.at(point3d_id);
  //   bpt.add_point(ptrack, point3d_id);
  // }
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
    int point3d_id = it->first;
    bpt.add_point(it->second, point3d_id);
  }

  // get linetracks
  // for (auto it = lines_.begin(); it != lines_.end(); ++it) {
  //   int line3d_id = it->first;
  //   LineTrack ltrack = LineTrack(line_tracks_.at(line3d_id));
  //   InfiniteLine3d inf_line = lines_.at(line3d_id).GetInfiniteLine();

  //   // std::vector<CameraView> views;
  //   // for (size_t i = 0; i < ltrack.count_lines(); ++i) {
  //   //   views.push_back(imagecols_.camview(ltrack.image_id_list[i]));
  //   // }
  //   // Line3d line = GetLineSegmentFromInfiniteLine3d(
  //   //     inf_line, views, ltrack.line2d_list,
  //   config_.num_outliers_aggregate);

  //   Line3d line = GetLineSegmentFromInfiniteLine3d(
  //       lines_.at(line3d_id).GetInfiniteLine(), ltrack.line3d_list,
  // config_.num_outliers_aggregate);
  //   ltrack.line = line;
  //   // if (!test_linetrack_validity(ltrack))
  //   //   ltrack.line = line_tracks_.at(line3d_id).line;
  //   bpt.add_line(ltrack, line3d_id);
  // }
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    int line3d_id = it->first;
    bpt.add_line(it->second, line3d_id);
  }

  auto weights = construct_weights_pointline(config_.th_weight_pointline);

  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    if (!bpt.exist_point(point3d_id) || !bpt.exist_line(line3d_id)) continue;
    double dist =
        bpt.line(line3d_id).line.point_distance(bpt.point(point3d_id).p);
    double si_dist = dist / un_points_.at(point3d_id);
    // test on 3d
    if (si_dist > th_hard_pl_si_dist3d) continue;
    bpt.add_edge(point3d_id, line3d_id);
  }

  pl_bpt3d_ = bpt;
}

void PLPAssociator::UpdateVPLine_Bipartite3d(
    const double th_hard_vpline_angle3d) {
  if (all_bpt2ds_vp_.empty()) return;

  // init bipartite
  structures::VPLine_Bipartite3d bpt;

  // get vptracks
  for (auto it = vps_.begin(); it != vps_.end(); ++it) {
    int vp3d_id = it->first;
    vplib::VPTrack track = vplib::VPTrack(vp_tracks_.at(vp3d_id));
    track.direction = vps_.at(vp3d_id);
    bpt.add_point(track, vp3d_id);
  }

  // get linetracks
  // for (auto it = lines_.begin(); it != lines_.end(); ++it) {
  //   int line3d_id = it->first;
  //   LineTrack ltrack = LineTrack(line_tracks_.at(line3d_id));
  //   InfiniteLine3d inf_line = lines_.at(line3d_id).GetInfiniteLine();
  //   std::vector<CameraView> views;
  //   for (size_t i = 0; i < ltrack.count_lines(); ++i) {
  //     views.push_back(imagecols_.camview(ltrack.image_id_list[i]));
  //   }
  //   // Line3d line =
  //   //     GetLineSegmentFromInfiniteLine3d(inf_line, views,
  //   //     ltrack.line2d_list);
  //   Line3d line = GetLineSegmentFromInfiniteLine3d(
  //       lines_.at(line3d_id).GetInfiniteLine(), ltrack.line3d_list,
  //       config_.num_outliers_aggregate);
  //   ltrack.line = line;
  //   // if (!test_linetrack_validity(ltrack))
  //   //   ltrack.line = line_tracks_.at(line3d_id).line;
  //   bpt.add_line(ltrack, line3d_id);
  // }
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    int line3d_id = it->first;
    bpt.add_line(it->second, line3d_id);
  }

  // build connections
  auto weights = construct_weights_vpline(config_.th_count_vpline);

  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int vp3d_id = it->first.first;
    int line3d_id = it->first.second;
    if (!bpt.exist_point(vp3d_id) || !bpt.exist_line(line3d_id)) continue;
    // test on 3d
    double cosine = std::abs(
        bpt.point(vp3d_id).direction.dot(bpt.line(line3d_id).line.direction()));
    if (cosine > 1.0) cosine = 1.0;
    double angle = acos(cosine) * 180.0 / M_PI;
    if (angle > th_hard_vpline_angle3d) continue;
    bpt.add_edge(vp3d_id, line3d_id);
  }

  // delete unassociated vp
  for (const int& vp3d_id : bpt.get_point_ids()) {
    if (bpt.pdegree(vp3d_id) == 0) bpt.delete_point(vp3d_id);
  }

  vpline_bpt3d_ = bpt;
}

void PLPAssociator::UpdatePlanes() {
  for (auto it = planes_.begin(); it != planes_.end(); ++it) {
    int plane_id = it->first;
    it->second = mini_planes_.at(plane_id).GetInfinitePlane3d();
  }
}

void PLPAssociator::UpdatePP_Bipartite3d() {
  if (pp_bpt3d_.obj1_map.empty() || pp_bpt3d_.obj2_map.empty()) return;

  // init
  PP_Bipartite3d new_pp_bpt3d;
  for (auto it = point_tracks_.begin(); it != point_tracks_.end(); ++it) {
    new_pp_bpt3d.AddObj1(it->second, it->first);
  }
  for (auto it = planes_.begin(); it != planes_.end(); ++it) {
    new_pp_bpt3d.AddObj2(it->second, it->first);
  }

  // re-build association according to original association
  for (auto it = pp_bpt3d_.n_2_to_1.begin(); it != pp_bpt3d_.n_2_to_1.end();
       it++) {
    int plane_id = it->first;
    THROW_CHECK(planes_.find(plane_id) != planes_.end())

    // associate inlier points3d
    const InfinitePlane3d& inf_plane3d = planes_.at(plane_id);
    const auto& points = it->second;
    for (const auto& point : points) {
      int point_id = point.first;
      // point_id has been filtered
      if (point_tracks_.find(point_id) == point_tracks_.end()) continue;

      const V3D& point3d_xyz = point_tracks_.at(point_id).p;
      double d =
          geometry::ComputeDistancePoint3dToPlane3d(point3d_xyz, inf_plane3d);
      double uncertainty = un_points_.at(point_id);
      THROW_CHECK_GT(uncertainty, 0);
      double si_d = d / uncertainty;  // >= 0.0
      THROW_CHECK_GE(si_d, 0);

      if (si_d < config_.th_hard_pointplane_si_dist3d) {
        // add connection
        new_pp_bpt3d.AddEdge(point_id, plane_id, -si_d);
      }
    }
  }

  pp_bpt3d_ = new_pp_bpt3d;
}

void PLPAssociator::UpdateLP_Bipartite3d() {
  if (lp_bpt3d_.obj1_map.empty() || lp_bpt3d_.obj2_map.empty()) return;

  // init
  LP_Bipartite3d new_lp_bpt3d;
  for (auto it = line_tracks_.begin(); it != line_tracks_.end(); ++it) {
    new_lp_bpt3d.AddObj1(it->second, it->first);
  }
  for (auto it = planes_.begin(); it != planes_.end(); ++it) {
    new_lp_bpt3d.AddObj2(it->second, it->first);
  }

  // re-build association according to original association
  for (auto it = lp_bpt3d_.n_2_to_1.begin(); it != lp_bpt3d_.n_2_to_1.end();
       it++) {
    int plane_id = it->first;
    THROW_CHECK(planes_.find(plane_id) != planes_.end())

    const InfinitePlane3d& inf_plane3d = planes_.at(plane_id);

    // associate inlier lines3d
    const auto& lines = it->second;
    for (const auto& line : lines) {
      int line_id = line.first;
      // line_id has been filtered
      if (line_tracks_.find(line_id) == line_tracks_.end()) continue;

      const Line3d& line3d = line_tracks_.at(line_id).line;
      double start_uncertainty = un_start_on_lines_.at(line_id);
      double end_uncertainty = un_end_on_lines_.at(line_id);

      double scale_inv_dist;
      if (geometry::TestLine3dOnInfPlane3d(
              line3d, inf_plane3d, start_uncertainty, end_uncertainty,
              config_.th_hard_lineplane_si_dist3d,
              config_.th_hard_lineplane_angle, &scale_inv_dist)) {
        THROW_CHECK_GE(scale_inv_dist, 0)
        // add connection
        new_lp_bpt3d.AddEdge(line_id, plane_id, -scale_inv_dist);
      }
    }
  }

  lp_bpt3d_ = new_lp_bpt3d;
}

void PLPAssociator::FindAllPointPlanePairs(
    std::unordered_map<int, int>* point_plane_pairs) const {
  CHECK_NOTNULL(point_plane_pairs);
  point_plane_pairs->clear();
  const auto& n_1_to_2 = pp_bpt3d_.n_1_to_2;
  for (auto it = n_1_to_2.begin(); it != n_1_to_2.end(); it++) {
    int point3d_id = it->first;
    const auto& planes = it->second;
    for (const auto& plane : planes) {
      point_plane_pairs->emplace(point3d_id, plane.first);
    }
  }
}

void PLPAssociator::FindAllLinePlanesPairs(
    std::unordered_map<int, std::vector<int>>* line_planes_pairs) const {
  CHECK_NOTNULL(line_planes_pairs);
  line_planes_pairs->clear();
  const auto& n_1_to_2 = lp_bpt3d_.n_1_to_2;
  for (auto it = n_1_to_2.begin(); it != n_1_to_2.end(); it++) {
    int line3d_id = it->first;
    const auto& planes = it->second;
    std::vector<int> planes_vec;
    for (const auto& kv : planes) {
      planes_vec.push_back(kv.first);
    }
    line_planes_pairs->emplace(line3d_id, planes_vec);
  }
}

void PLPAssociator::FindOrthogonalityPlanePairs(
    double th_orthogonality_angle,
    std::vector<std::pair<int, int>>* planes_pairs) const {
  CHECK_NOTNULL(planes_pairs);
  planes_pairs->clear();

  for (auto it1 = planes_.begin(); it1 != planes_.end(); it1++) {
    int plane_id1 = it1->first;
    const InfinitePlane3d& inf_plane3d1 = it1->second;
    for (auto it2 = planes_.begin(); it2 != planes_.end(); it2++) {
      int plane_id2 = it2->first;
      if (plane_id1 >= plane_id2) continue;
      const InfinitePlane3d& inf_plane3d2 = it2->second;
      if (geometry::ComputeAngle(inf_plane3d1.n0, inf_plane3d2.n0) >
          th_orthogonality_angle) {
        planes_pairs->emplace_back(plane_id1, plane_id2);
      }
    }
  }
}

void PLPAssociator::FindParallelismPlanePairs(
    double th_parallelism_angle,
    std::vector<std::pair<int, int>>* planes_pairs) const {
  CHECK_NOTNULL(planes_pairs);
  planes_pairs->clear();

  for (auto it1 = planes_.begin(); it1 != planes_.end(); it1++) {
    int plane_id1 = it1->first;
    const InfinitePlane3d& inf_plane3d1 = it1->second;
    for (auto it2 = planes_.begin(); it2 != planes_.end(); it2++) {
      int plane_id2 = it2->first;
      if (plane_id1 >= plane_id2) continue;
      const InfinitePlane3d& inf_plane3d2 = it2->second;
      if (geometry::ComputeAngle(inf_plane3d1.n0, inf_plane3d2.n0) <
          th_parallelism_angle) {
        planes_pairs->emplace_back(plane_id1, plane_id2);
      }
    }
  }
}

void PLPAssociator::ParameterizePlanes() {
  for (auto it = planes_.begin(); it != planes_.end(); ++it) {
    int plane_id = it->first;
    MinimalInfinitePlane3d& mini_planes = mini_planes_.at(plane_id);
    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      double* cpvec_data = mini_planes.cpvec.data();
      if (!problem_->HasParameterBlock(cpvec_data)) continue;
      if (config_.constant_plane) {
        problem_->SetParameterBlockConstant(cpvec_data);
      } else {
        // do not need additional parameterization
      }
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      double* qvec_data = mini_planes.qvec.data();
      if (!problem_->HasParameterBlock(qvec_data)) continue;
      if (config_.constant_plane) {
        problem_->SetParameterBlockConstant(qvec_data);
      } else {
        SetQuaternionManifold(problem_.get(), qvec_data);
      }
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      double* hvec_data = mini_planes.hvec.data();
      if (!problem_->HasParameterBlock(hvec_data)) continue;
      if (config_.constant_plane) {
        problem_->SetParameterBlockConstant(hvec_data);
      } else {
        SetSphereManifold<4>(problem_.get(), hvec_data);
      }
    } else {
      THROW_EXCEPTION(std::invalid_argument, "Error: minimal_type is invalid");
    }
  }
}

void PLPAssociator::AddPointLineAssociationResiduals(
    const std::map<std::pair<int, int>, double>& weights) {
  if (config_.lw_pointline_association <= 0) return;
  ceres::LossFunction* loss_function =
      config_.point_line_association_3d_loss_function.get();
  for (auto it = weights.begin(); it != weights.end(); ++it) {
    int point3d_id = it->first.first;
    int line3d_id = it->first.second;
    double weight = it->second;

    double uncertainty =
        config_.use_uncertainty ? un_points_.at(point3d_id) : 1.0;

    ceres::CostFunction* cost_function =
        PointLineAssociation3dFunctor::Create(uncertainty);
    ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight * config_.lw_pointline_association,
        ceres::DO_NOT_TAKE_OWNERSHIP);
    ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
        cost_function, scaled_loss_function, points_.at(point3d_id).data(),
        lines_.at(line3d_id).uvec.data(), lines_.at(line3d_id).wvec.data());
  }
}

void PLPAssociator::AddPointPlaneAssociationResiduals(
    const std::unordered_map<int, int>& point_plane_pairs) {
  if (config_.lw_pointplane_association <= 0) return;
  ceres::LossFunction* loss_function =
      config_.point_plane_association_3d_loss_function.get();

  for (const auto& point_plane_pair : point_plane_pairs) {
    const int point3d_id = point_plane_pair.first;
    const int plane3d_id = point_plane_pair.second;

    THROW_CHECK(point_tracks_.find(point3d_id) != point_tracks_.end());
    THROW_CHECK(points_.find(point3d_id) != points_.end());
    THROW_CHECK(planes_.find(plane3d_id) != planes_.end())
    THROW_CHECK(mini_planes_.find(plane3d_id) != mini_planes_.end())

    // number of images in point track
    double weight = point_tracks_.at(point3d_id).count_images();

    double uncertainty =
        config_.use_uncertainty ? un_points_.at(point3d_id) : 1.0;

    ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
        loss_function, weight * config_.lw_pointplane_association,
        ceres::DO_NOT_TAKE_OWNERSHIP);

    ceres::CostFunction* cost_function = PointPlaneAssociation3dFunctor::Create(
        config_.mini_plane_type, uncertainty);

    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, points_.at(point3d_id).data(),
          mini_planes_.at(plane3d_id).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, points_.at(point3d_id).data(),
          mini_planes_.at(plane3d_id).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
          cost_function, scaled_loss_function, points_.at(point3d_id).data(),
          mini_planes_.at(plane3d_id).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }
}

void PLPAssociator::AddLinePlaneAssociationResiduals(
    const std::unordered_map<int, std::vector<int>>& line_planes_pairs) {
  if (config_.lw_lineplane_distance_association <= 0 &&
      config_.lw_lineplane_angle_association <= 0)
    return;

  for (const auto& line_planes_pair : line_planes_pairs) {
    int line3d_id = line_planes_pair.first;
    const auto& plane3d_ids = line_planes_pair.second;

    THROW_CHECK(line_tracks_.find(line3d_id) != line_tracks_.end());
    THROW_CHECK(lines_.find(line3d_id) != lines_.end());
    for (const auto& plane3d_id : plane3d_ids) {
      THROW_CHECK(planes_.find(plane3d_id) != planes_.end());
      THROW_CHECK(mini_planes_.find(plane3d_id) != mini_planes_.end());
    }

    // number of images in line track
    double weight = line_tracks_.at(line3d_id).count_images();

    std::vector<std::pair<V3D, double>> un_points_on_line_tmp =
        un_points_on_lines_.at(line3d_id);

    if (!config_.use_uncertainty) {
      for (auto& kv : un_points_on_line_tmp) {
        kv.second = 1.0;
      }
    }

    // if (plane3d_ids.size() == 0) {
    //   std::cout << "no associated plane to the line " << line3d_id <<
    //   std::endl;
    // } else if (plane3d_ids.size() == 1) {
    //   std::cout << "associate a plane " << plane3d_ids[0]
    //             << " to the line " << line3d_id << std::endl;
    // } else {
    //   std::cout << "associate multiple planes ";
    //   for (const auto& id : plane3d_ids) {
    //     std::cout << id << " ";
    //   }
    //   std::cout << "to the line " << line3d_id << std::endl;
    // }

    for (const auto& plane3d_id : plane3d_ids) {
      if (config_.lw_lineplane_distance_association > 0) {
        ceres::LossFunction* loss_function =
            config_.line_plane_distance_association_3d_loss_function.get();
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
            loss_function, weight * config_.lw_lineplane_distance_association,
            ceres::DO_NOT_TAKE_OWNERSHIP);

        ceres::CostFunction* cost_function =
            LinePlaneAssociation3dDistanceFunctor::Create(
                config_.mini_plane_type, un_points_on_line_tmp);

        if (config_.mini_plane_type ==
            MinimalInfinitePlane3d::MinimalType::CP) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).cpvec.data());
        } else if (config_.mini_plane_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).qvec.data());
        } else if (config_.mini_plane_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).hvec.data());
        } else {
          throw std::invalid_argument("Error: minimal_type is invalid");
        }
      }

      if (config_.lw_lineplane_angle_association > 0) {
        ceres::LossFunction* loss_function =
            config_.line_plane_angle_association_3d_loss_function.get();
        ceres::LossFunction* scaled_loss_function = new ceres::ScaledLoss(
            loss_function, weight * config_.lw_lineplane_angle_association,
            ceres::DO_NOT_TAKE_OWNERSHIP);

        ceres::CostFunction* cost_function =
            LinePlaneAssociation3dAngleFunctor::Create(config_.mini_plane_type);

        if (config_.mini_plane_type ==
            MinimalInfinitePlane3d::MinimalType::CP) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).cpvec.data());
        } else if (config_.mini_plane_type ==
                   MinimalInfinitePlane3d::MinimalType::QUATERNION) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).qvec.data());
        } else if (config_.mini_plane_type ==
                   MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
          ceres::ResidualBlockId block_id = problem_->AddResidualBlock(
              cost_function, scaled_loss_function,
              lines_.at(line3d_id).uvec.data(),
              lines_.at(line3d_id).wvec.data(),
              mini_planes_.at(plane3d_id).hvec.data());
        } else {
          throw std::invalid_argument("Error: minimal_type is invalid");
        }
      }
    }
  }
}

void PLPAssociator::AddPlaneOrthogonalityResiduals() {
  if (config_.lw_plane_orthogonality <= 0) return;
  ceres::LossFunction* loss_function =
      config_.plane_orthogonality_association_3d_loss_function.get();

  std::vector<std::pair<int, int>> planes_pairs;
  FindOrthogonalityPlanePairs(config_.th_plane_orthogonality_angle,
                              &planes_pairs);

  for (auto it = planes_pairs.begin(); it != planes_pairs.end(); ++it) {
    int plane_id1 = it->first;
    int plane_id2 = it->second;

    ceres::CostFunction* cost_function =
        PlaneOrthogonalityFunctor::Create(config_.mini_plane_type);
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, config_.lw_plane_orthogonality,
                              ceres::DO_NOT_TAKE_OWNERSHIP);

    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).cpvec.data(),
                                     mini_planes_.at(plane_id2).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).qvec.data(),
                                     mini_planes_.at(plane_id2).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).hvec.data(),
                                     mini_planes_.at(plane_id2).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }
}

void PLPAssociator::AddPlaneParallelismResiduals() {
  if (config_.lw_plane_parallelism <= 0) return;
  ceres::LossFunction* loss_function =
      config_.plane_parallelism_association_3d_loss_function.get();

  std::vector<std::pair<int, int>> planes_pairs;
  FindParallelismPlanePairs(config_.th_plane_parallelism_angle, &planes_pairs);

  for (auto it = planes_pairs.begin(); it != planes_pairs.end(); ++it) {
    int plane_id1 = it->first;
    int plane_id2 = it->second;

    ceres::CostFunction* cost_function =
        PlaneParallelismFunctor::Create(config_.mini_plane_type);
    ceres::LossFunction* scaled_loss_function =
        new ceres::ScaledLoss(loss_function, config_.lw_plane_parallelism,
                              ceres::DO_NOT_TAKE_OWNERSHIP);
    if (config_.mini_plane_type == MinimalInfinitePlane3d::MinimalType::CP) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).cpvec.data(),
                                     mini_planes_.at(plane_id2).cpvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::QUATERNION) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).qvec.data(),
                                     mini_planes_.at(plane_id2).qvec.data());
    } else if (config_.mini_plane_type ==
               MinimalInfinitePlane3d::MinimalType::HOMOGENEOUS) {
      ceres::ResidualBlockId block_id =
          problem_->AddResidualBlock(cost_function, scaled_loss_function,
                                     mini_planes_.at(plane_id1).hvec.data(),
                                     mini_planes_.at(plane_id2).hvec.data());
    } else {
      throw std::invalid_argument("Error: minimal_type is invalid");
    }
  }
}

std::map<int, PointTrack> PLPAssociator::GetPointTracks() const {
  return point_tracks_;
}

std::map<int, LineTrack> PLPAssociator::GetLineTracks() const {
  return line_tracks_;
}

structures::PL_Bipartite3d PLPAssociator::GetPL_Bipartite3d() const {
  return pl_bpt3d_;
}

structures::VPLine_Bipartite3d PLPAssociator::GetVPLine_Bipartite3d() const {
  return vpline_bpt3d_;
}

std::map<int, InfinitePlane3d> PLPAssociator::GetPlanes() const {
  return planes_;
}

PP_Bipartite3d PLPAssociator::GetPP_Bipartite3d() const { return pp_bpt3d_; }

LP_Bipartite3d PLPAssociator::GetLP_Bipartite3d() const { return lp_bpt3d_; }

}  // namespace limap
