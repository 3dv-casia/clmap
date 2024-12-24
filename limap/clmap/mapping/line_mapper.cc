#include "clmap/mapping/line_mapper.h"

#include "clmap/base/geometry.h"
#include "clmap/mapping/merging.h"
#include "clmap/util/math.h"
#include "clmap/util/misc.h"

#include <numeric>
#include <queue>
#include <random>
#include <typeinfo>
#include <unordered_set>

#include "base/line_dists.h"
#include "merging/merging.h"
#include "progressbar.hpp"
#include "triangulation/functions.h"

namespace limap {

LineMapperConfig::LineMapperConfig(py::dict dict) {
  ASSIGN_PYDICT_ITEM(dict, debug_mode, bool)
  ASSIGN_PYDICT_ITEM(dict, add_halfpix, bool)
  ASSIGN_PYDICT_ITEM(dict, min_length_2d, double)
  ASSIGN_PYDICT_ITEM(dict, var2d, double)
  if (dict.contains("linker2d_config"))
    linker2d_config = LineLinker2dConfig(dict["linker2d_config"]);
  if (dict.contains("linker3d_config"))
    linker3d_config = LineLinker3dConfig(dict["linker3d_config"]);
  ASSIGN_PYDICT_ITEM(dict, use_unshared_points, bool)
  ASSIGN_PYDICT_ITEM(dict, use_range_filter, bool)
  ASSIGN_PYDICT_ITEM(dict, use_endpoints_triangulation, bool)
  ASSIGN_PYDICT_ITEM(dict, line_tri_angle_threshold, double)
  ASSIGN_PYDICT_ITEM(dict, IoU_threshold, double)
  ASSIGN_PYDICT_ITEM(dict, sensitivity_threshold, double)
  ASSIGN_PYDICT_ITEM(dict, max_point_line_reproject_distance, double)
  ASSIGN_PYDICT_ITEM(dict, use_vp, bool)
  ASSIGN_PYDICT_ITEM(dict, use_pointsfm, bool)
  ASSIGN_PYDICT_ITEM(dict, disable_many_points_triangulation, bool)
  ASSIGN_PYDICT_ITEM(dict, disable_one_point_triangulation, bool)
  ASSIGN_PYDICT_ITEM(dict, disable_algebraic_triangulation, bool)
  ASSIGN_PYDICT_ITEM(dict, disable_vp_triangulation, bool)
  ASSIGN_PYDICT_ITEM(dict, use_local_valid_lines2d, bool)
  ASSIGN_PYDICT_ITEM(dict, line_track_building_method, std::string)
  ASSIGN_PYDICT_ITEM(dict, clustering_method, std::string)
  ASSIGN_PYDICT_ITEM(dict, num_outliers_aggregator, int)
  ASSIGN_PYDICT_ITEM(dict, max_same_line3D_overlap, double)
  ASSIGN_PYDICT_ITEM(dict, min_same_line3D_angle, double)
  ASSIGN_PYDICT_ITEM(dict, min_same_line3D_perp_dist, double)
  ASSIGN_PYDICT_ITEM(dict, aggregate_method, std::string)
}

LineMapper::LineMapper(const LineMapperConfig& config)
    : config_(config),
      ranges_(std::make_pair(V3D::Zero(), V3D::Zero())),
      linker_(config.linker2d_config, config.linker3d_config) {}

LineMapper::LineMapper(py::dict dict) : LineMapper((LineMapperConfig(dict))) {}

void LineMapper::Initialize() {
  PreStartCheck();

  if (config_.add_halfpix) OffsetHalfPixel();

  // initialize empty containers
  size_t num_image = image_collection_.NumImages();
  proposals_.reserve(num_image);
  local_best_proposal_indices_.reserve(num_image);
  visible_line2d_flags_.reserve(num_image);
  matches_.reserve(num_image);
  num_lines2d_ = 0;
  for (image_t image_id : image_collection_.get_img_ids()) {
    size_t num_lines2d = all_line2d_.at(image_id).size();
    num_lines2d_ += num_lines2d;
    proposals_.emplace(image_id,
                       std::vector<std::vector<Proposal>>(num_lines2d));
    local_best_proposal_indices_.emplace(image_id,
                                         std::vector<int>(num_lines2d, -1));
    visible_line2d_flags_.emplace(image_id, std::vector<char>(num_lines2d, 0));
    matches_.emplace(
        image_id,
        std::vector<std::unordered_map<image_t, std::vector<line2d_t>>>(
            num_lines2d));
  }
  select_local_best_prop_ = false;
  global_best_proposal_indices_ = local_best_proposal_indices_;
}

void LineMapper::PreStartCheck() const {
  if (image_collection_.NumImages() == 0 ||
      image_collection_.NumCameras() == 0) {
    throw std::logic_error(
        "Error: image_collection_ was not set, please set it in advance.");
  }
  if (all_line2d_.empty()) {
    throw std::logic_error(
        "Error: all_line2d_ was not set, please set it in advance.");
  }
  if (config_.use_range_filter && (ranges_.first.isApprox(V3D::Zero()) &&
                                   ranges_.second.isApprox(V3D::Zero()))) {
    throw std::logic_error(
        "Error: config_.use_range_filter is true, but ranges_ was not set, "
        "please set it in advance.");
  }
}

void LineMapper::OffsetHalfPixel() {
  std::vector<int> image_ids = image_collection_.get_img_ids();
  for (auto it = image_ids.begin(); it != image_ids.end(); ++it) {
    image_t image_id = *it;
    for (line2d_t line2d_idx = 0; line2d_idx < all_line2d_.at(image_id).size();
         ++line2d_idx) {
      auto& line = all_line2d_.at(image_id)[line2d_idx];
      line.start = line.start + V2D(0.5, 0.5);
      line.end = line.end + V2D(0.5, 0.5);
    }
  }
}

void LineMapper::MatchLines2dByEpipolarIoU(
    const std::unordered_map<image_t, std::vector<image_t>>& all_neighbors,
    const int n_matches, const double th_IoU) {
  neighbors_.clear();
  for (const auto& pair : all_neighbors) {
    image_t ref_image_id = pair.first;
    for (const image_t ng_image_id : pair.second) {
      neighbors_[ref_image_id].insert(ng_image_id);
      neighbors_[ng_image_id].insert(ref_image_id);
    }
  }

  std::vector<image_t> image_ids;
  image_ids.reserve(all_line2d_.size());
  for (const auto& pair : all_line2d_) {
    image_ids.emplace_back(pair.first);
  }

  // compute fundamental matrix
  std::unordered_map<std::pair<image_t, image_t>, M3D> Fs;
  for (int i = 0; i < image_ids.size(); i++) {
    image_t image_id = image_ids[i];
    const auto& view = image_collection_.camview(image_id);
    const auto& neighbors = all_neighbors.at(image_id);
    for (const auto& ng_image_id : neighbors) {
      const auto& ng_view = image_collection_.camview(ng_image_id);
      if (image_id == ng_image_id) continue;
      if (image_id < ng_image_id) {
        if (Fs.find(std::make_pair(image_id, ng_image_id)) != Fs.end()) {
          continue;
        }
        M3D F = triangulation::compute_fundamental_matrix(view, ng_view);
        Fs.emplace(std::make_pair(image_id, ng_image_id), F);
      } else {
        if (Fs.find(std::make_pair(ng_image_id, image_id)) != Fs.end()) {
          continue;
        }
        M3D F = triangulation::compute_fundamental_matrix(ng_view, view);
        Fs.emplace(std::make_pair(ng_image_id, image_id), F);
      }
    }
  }

  // init
  std::unordered_map<
      image_t,
      std::vector<std::unordered_map<image_t, std::unordered_set<line2d_t>>>>
      matches_set;
  for (image_t image_id : image_collection_.get_img_ids()) {
    matches_set.emplace(
        image_id,
        std::vector<
            std::unordered_map<image_t, std::unordered_set<line2d_t>>>());
  }

  // matching
  progressbar bar(image_ids.size());
#pragma omp parallel for
  for (size_t i = 0; i < image_ids.size(); i++) {
    bar.update();
    const image_t image_id = image_ids[i];
    const auto& all_lines2d = all_line2d_.at(image_id);
    const auto& neighbors = all_neighbors.at(image_id);

    std::vector<std::unordered_map<image_t, std::unordered_set<line2d_t>>>
        all_matches(all_lines2d.size());

    for (const auto& ng_image_id : neighbors) {
      if (image_id == ng_image_id) continue;
      const auto& all_ng_lines2d = all_line2d_.at(ng_image_id);

      M3D F;
      if (image_id < ng_image_id) {
        F = Fs.at(std::make_pair(image_id, ng_image_id));
      } else {
        F = (Fs.at(std::make_pair(ng_image_id, image_id))).transpose();
      }

      for (line2d_t line2d_idx = 0; line2d_idx < all_lines2d.size();
           line2d_idx++) {
        auto& matches = all_matches.at(line2d_idx);

        // init
        matches[ng_image_id] = std::unordered_set<line2d_t>();
        auto& m = matches.at(ng_image_id);

        // compute epipolar line
        const auto& line2d = all_lines2d[line2d_idx];
        V3D coor_epline_start =
            (F * V3D(line2d.start[0], line2d.start[1], 1)).normalized();
        V3D coor_epline_end =
            (F * V3D(line2d.end[0], line2d.end[1], 1)).normalized();

        // init priority queue
        auto cmp = [](const std::pair<line2d_t, double>& a,
                      const std::pair<line2d_t, double>& b) {
          return a.second < b.second;
        };
        std::priority_queue<std::pair<line2d_t, double>,
                            std::vector<std::pair<line2d_t, double>>,
                            decltype(cmp)>
            pq(cmp);

        std::vector<line2d_t> valid_ng_lines2d;

        for (line2d_t ng_line2d_idx = 0; ng_line2d_idx < all_ng_lines2d.size();
             ng_line2d_idx++) {
          const auto& ng_line2d = all_ng_lines2d[ng_line2d_idx];
          // compute epipolar IoU
          V3D coor_ng_line = ng_line2d.coords();
          V3D homo_c_start = coor_ng_line.cross(coor_epline_start);
          V2D c_start = dehomogeneous(homo_c_start);
          V3D homo_c_end = coor_ng_line.cross(coor_epline_end);
          V2D c_end = dehomogeneous(homo_c_end);
          V2D ng_line2d_dir = ng_line2d.direction();
          double ng_line2d_length = ng_line2d.length();
          double c1 =
              (c_start - ng_line2d.start).dot(ng_line2d_dir) / ng_line2d_length;
          double c2 =
              (c_end - ng_line2d.start).dot(ng_line2d_dir) / ng_line2d_length;
          if (c1 > c2) std::swap(c1, c2);
          double IoU = (std::min(c2, 1.0) - std::max(c1, 0.0)) /
                       (std::max(c2, 1.0) - std::min(c1, 0.0));

          if (IoU >= th_IoU) {
            if (n_matches > 0) {  // use top K matching
              pq.push(std::make_pair(ng_line2d_idx, IoU));
            } else {
              valid_ng_lines2d.push_back(ng_line2d_idx);
            }
          }
        }

        // store matches
        if (n_matches > 0) {
          if (pq.empty()) {  // no matching
            matches.erase(ng_image_id);
          } else {
            while (!pq.empty() && (m.size() < n_matches)) {
              m.insert(pq.top().first);
              pq.pop();
            }
          }
        } else {
          if (valid_ng_lines2d.empty()) {  // no matching
            matches.erase(ng_image_id);
          } else {
            for (const auto& ng_line2d_idx : valid_ng_lines2d) {
              m.insert(ng_line2d_idx);
            }
          }
        }
      }
    }
#pragma omp critical
    { matches_set.at(image_id) = all_matches; }
  }

  // make matching graph bidirectional
  for (const auto& pair1 : matches_set) {
    image_t image_id1 = pair1.first;
    for (line2d_t line2d_idx1 = 0; line2d_idx1 < pair1.second.size();
         line2d_idx1++) {
      for (const auto& pair2 : pair1.second[line2d_idx1]) {
        image_t image_id2 = pair2.first;
        auto& matches_2 = matches_set.at(image_id2);
        for (const line2d_t line2d_idx2 : pair2.second) {
          auto& m_2 = matches_2.at(line2d_idx2);
          if (m_2.find(image_id1) == m_2.end()) {
            m_2[image_id1] = std::unordered_set<line2d_t>();
          }
          auto& m_22 = m_2.at(image_id1);
          if (m_22.find(line2d_idx1) == m_22.end()) {
            m_22.insert(line2d_idx1);
          }
        }
      }
    }
  }

  // convert matches_set to matches_
  for (const auto& pair1 : matches_set) {
    image_t image_id1 = pair1.first;
    auto& matches = matches_.at(image_id1);
    for (line2d_t line2d_idx1 = 0; line2d_idx1 < pair1.second.size();
         line2d_idx1++) {
      auto& matches_per_line2d = matches.at(line2d_idx1);
      matches_per_line2d.clear();
      for (const auto& pair2 : pair1.second[line2d_idx1]) {
        image_t image_id2 = pair2.first;
        THROW_CHECK(matches_per_line2d.find(image_id2) ==
                    matches_per_line2d.end());
        matches_per_line2d[image_id2].assign(pair2.second.begin(),
                                             pair2.second.end());
      }
    }
  }
}

void LineMapper::LoadBidirectionalMatches(
    const std::unordered_map<
        image_t, std::unordered_map<image_t, Eigen::MatrixXi>>& matches) {
  // init
  std::unordered_map<
      image_t,
      std::vector<std::unordered_map<image_t, std::unordered_set<line2d_t>>>>
      matches_set;
  for (image_t image_id : image_collection_.get_img_ids()) {
    size_t num_lines2d = all_line2d_.at(image_id).size();
    matches_set.emplace(
        image_id,
        std::vector<std::unordered_map<image_t, std::unordered_set<line2d_t>>>(
            num_lines2d));
  }
  neighbors_.clear();

  // load matches
  for (const auto& pair : matches) {
    const image_t ref_image_id = pair.first;
    size_t num_ref_line2d = all_line2d_.at(ref_image_id).size();
    if (num_ref_line2d == 0) return;

    auto& all_matches = matches_set.at(ref_image_id);

    for (const auto& match : pair.second) {
      image_t ng_image_id = match.first;
      if (ref_image_id == ng_image_id) continue;

      neighbors_[ref_image_id].insert(ng_image_id);
      neighbors_[ng_image_id].insert(ref_image_id);

      const Eigen::MatrixXi& match_info = match.second;

      // convert original match_info to vector-like matches
      if (match_info.rows() != 0) {
        THROW_CHECK_EQ(match_info.cols(), 2);
      }
      size_t num_matches = match_info.rows();
      // (ref_line2d_idx, (ng_line2d_idx))
      std::vector<std::unordered_set<line2d_t>> m_set(num_ref_line2d);
      for (size_t i = 0; i < num_matches; i++) {
        line2d_t ref_line2D_idx = match_info(i, 0);
        line2d_t ng_line2D_idx = match_info(i, 1);
        if (ref_line2D_idx >= num_ref_line2d || match_info(i, 0) < 0) {
          throw std::runtime_error(
              "Error: Out-of-index matches exist between image (img_id = " +
              std::to_string(ref_image_id) + ") and neighbor image (img_id = " +
              std::to_string(ng_image_id) + ").");
        }
        m_set[ref_line2D_idx].insert(ng_line2D_idx);
      }

      for (size_t i = 0; i < num_ref_line2d; i++) {
        all_matches[i].emplace(ng_image_id, m_set[i]);
      }
    }
  }

  // make matching graph bidirectional
  for (const auto& pair1 : matches_set) {
    image_t image_id1 = pair1.first;
    for (line2d_t line2d_idx1 = 0; line2d_idx1 < pair1.second.size();
         line2d_idx1++) {
      for (const auto& pair2 : pair1.second[line2d_idx1]) {
        image_t image_id2 = pair2.first;
        auto& matches_2 = matches_set.at(image_id2);
        for (const line2d_t line2d_idx2 : pair2.second) {
          auto& m_2 = matches_2.at(line2d_idx2);
          if (m_2.find(image_id1) == m_2.end()) {
            m_2[image_id1] = std::unordered_set<line2d_t>();
          }
          auto& m_22 = m_2.at(image_id1);
          if (m_22.find(line2d_idx1) == m_22.end()) {
            m_22.insert(line2d_idx1);
          }
        }
      }
    }
  }

  // convert matches_set to matches_
  for (const auto& pair1 : matches_set) {
    image_t image_id1 = pair1.first;
    auto& matches = matches_.at(image_id1);
    for (line2d_t line2d_idx1 = 0; line2d_idx1 < pair1.second.size();
         line2d_idx1++) {
      auto& matches_per_line2d = matches.at(line2d_idx1);
      matches_per_line2d.clear();
      for (const auto& pair2 : pair1.second[line2d_idx1]) {
        image_t image_id2 = pair2.first;
        THROW_CHECK(matches_per_line2d.find(image_id2) ==
                    matches_per_line2d.end());
        matches_per_line2d[image_id2].assign(pair2.second.begin(),
                                             pair2.second.end());
      }
    }
  }
}

void LineMapper::TriangulateAllImages() {
  std::vector<image_t> image_ids;
  image_ids.reserve(matches_.size());
  for (const auto& pair : matches_) {
    image_ids.emplace_back(pair.first);
  }

  progressbar bar(image_ids.size());
  for (size_t i = 0; i < image_ids.size(); i++) {
    bar.update();
    image_t ref_image_id = image_ids[i];
    const auto& matches = matches_.at(ref_image_id);
    size_t num_ref_line2d = all_line2d_.at(ref_image_id).size();
    if (num_ref_line2d == 0) continue;
    for (const auto& ng_image_id : neighbors_.at(ref_image_id)) {
      size_t num_ng_line2d = all_line2d_.at(ng_image_id).size();
      if (num_ng_line2d == 0) continue;

      // (ref_line2d_idx, (ng_line2d_idx))
      std::vector<std::vector<line2d_t>> matches_vec(num_ref_line2d);

      bool has_at_least_one_match = false;
      for (size_t ref_line2d_idx = 0; ref_line2d_idx < num_ref_line2d;
           ref_line2d_idx++) {
        const auto& m = matches.at(ref_line2d_idx);
        if (m.find(ng_image_id) != m.end()) {
          matches_vec[ref_line2d_idx] = m.at(ng_image_id);
          if (!has_at_least_one_match && !matches_vec[ref_line2d_idx].empty()) {
            has_at_least_one_match = true;
          }
        }
      }
      if (has_at_least_one_match) {
        // proposal generation
        TriangulateImage(ref_image_id, ng_image_id, matches_vec);
      }
    }
  }
}

void LineMapper::TriangulateImageWithInitialMatches(
    const image_t& ref_image_id,
    const std::unordered_map<image_t, Eigen::MatrixXi>& init_matches) {
  size_t num_ref_line2d = all_line2d_.at(ref_image_id).size();
  if (num_ref_line2d == 0) return;

  auto& all_matches = matches_.at(ref_image_id);

  for (const auto& init_match : init_matches) {
    image_t ng_image_id = init_match.first;
    if (ref_image_id == ng_image_id) continue;

    const Eigen::MatrixXi& match_info = init_match.second;

    // convert original match_info to vector-like matches
    if (match_info.rows() != 0) {
      THROW_CHECK_EQ(match_info.cols(), 2);
    }
    size_t num_matches = match_info.rows();
    // (ref_line2d_idx, (ng_line2d_idx))
    std::vector<std::unordered_set<line2d_t>> matches_set(num_ref_line2d);
    for (size_t i = 0; i < num_matches; i++) {
      line2d_t ref_line2D_idx = match_info(i, 0);
      line2d_t ng_line2D_idx = match_info(i, 1);
      if (ref_line2D_idx >= num_ref_line2d || match_info(i, 0) < 0) {
        throw std::runtime_error(
            "Error: Out-of-index matches exist between image (img_id = " +
            std::to_string(ref_image_id) + ") and neighbor image (img_id = " +
            std::to_string(ng_image_id) + ").");
      }
      matches_set[ref_line2D_idx].insert(ng_line2D_idx);
    }

    // (ref_line2d_idx, (ng_line2d_idx))
    std::vector<std::vector<line2d_t>> matches_vec(num_ref_line2d);
    // note: matches[i] may be empty
    for (size_t i = 0; i < num_ref_line2d; i++) {
      matches_vec[i].assign(matches_set[i].begin(), matches_set[i].end());
      // store matches
      all_matches[i].emplace(ng_image_id, matches_vec[i]);
    }
    matches_set.clear();
    matches_set.shrink_to_fit();

    // proposal generation
    TriangulateImage(ref_image_id, ng_image_id, matches_vec);
  }
}

void LineMapper::TriangulateImageWithExhaustiveMatches(
    const image_t& ref_image_id, const std::vector<image_t>& ng_image_ids) {
  size_t num_ref_line2d = all_line2d_.at(ref_image_id).size();
  if (num_ref_line2d == 0) return;
  auto& all_matches = matches_.at(ref_image_id);

  for (const image_t& ng_image_id : ng_image_ids) {
    if (ref_image_id == ng_image_id) continue;

    // create exhaustive vector-like matches
    size_t num_ng_line2d = all_line2d_.at(ng_image_id).size();
    std::vector<line2d_t> ng_line2d_indices_vec(num_ng_line2d);
    std::iota(ng_line2d_indices_vec.begin(), ng_line2d_indices_vec.end(), 0);
    std::vector<std::vector<line2d_t>> matches_vec(num_ref_line2d,
                                                   ng_line2d_indices_vec);

    // store matches
    for (size_t i = 0; i < num_ref_line2d; i++) {
      all_matches[i].emplace(ng_image_id, ng_line2d_indices_vec);
    }

    // proposal generation
    TriangulateImage(ref_image_id, ng_image_id, matches_vec);
  }
}

void LineMapper::TriangulateImage(
    const image_t& ref_image_id, const image_t& ng_image_id,
    const std::vector<std::vector<line2d_t>>& matches) {
  const CameraView& ref_view = image_collection_.camview(ref_image_id);
  const CameraView& ng_view = image_collection_.camview(ng_image_id);
  const std::vector<Line2d>& ref_lines2D = all_line2d_.at(ref_image_id);
  const std::vector<Line2d>& ng_lines2D = all_line2d_.at(ng_image_id);

  const limap::structures::PL_Bipartite2d* ref_bpt_ptr = nullptr;
  const limap::structures::PL_Bipartite2d* ng_bpt_ptr = nullptr;
  if (config_.use_pointsfm) {
    ref_bpt_ptr = &(all_pl_bipartite2d_->at(ref_image_id));
    ng_bpt_ptr = &(all_pl_bipartite2d_->at(ng_image_id));
  }

  std::vector<std::vector<Proposal>>& proposals = proposals_.at(ref_image_id);

  // precomputation to avoid repetitive computation
  M3D F;
  if (!config_.disable_algebraic_triangulation) {
    F = limap::triangulation::compute_fundamental_matrix(ref_view, ng_view);
  }

#pragma omp parallel for
  for (size_t ref_line2D_idx = 0; ref_line2D_idx < ref_lines2D.size();
       ref_line2D_idx++) {
    const Line2d& l1 = ref_lines2D[ref_line2D_idx];
    if (l1.length() <= config_.min_length_2d) continue;

    std::unordered_set<int> ref_point3D_ids;
    if (config_.use_pointsfm && config_.use_unshared_points) {
      // collect the ID of 3D sfm points whose 2D observations are on the
      // reference 2D line segment.
      FindPoints3D(ref_line2D_idx, *ref_bpt_ptr, &ref_point3D_ids);
    }

    // precomputation to avoid repetitive computation
    V3D ray1_start, ray1_end, coor_epline_start, coor_epline_end;
    if (!config_.disable_algebraic_triangulation) {
      ray1_start = ref_view.ray_direction(l1.start);  // x_1^r
      ray1_end = ref_view.ray_direction(l1.end);      // x_2^r
      coor_epline_start = (F * V3D(l1.start[0], l1.start[1], 1)).normalized();
      coor_epline_end = (F * V3D(l1.end[0], l1.end[1], 1)).normalized();
    }

    for (const line2d_t& ng_matched_line2D_idx : matches[ref_line2D_idx]) {
      const Line2d& l2 = ng_lines2D[ng_matched_line2D_idx];
      if (l2.length() <= config_.min_length_2d) continue;

      //////////////////////////////////////////////////////////////////////////
      // Two-View Line Triangulation.
      //////////////////////////////////////////////////////////////////////////

      std::vector<Proposal> proposals_tmp;

      // step 1.1: Multiple Points triangulation
      // step 1.2: Line + Point triangulation
      if (config_.use_pointsfm && (!config_.disable_many_points_triangulation ||
                                   !config_.disable_one_point_triangulation)) {
        THROW_CHECK(!sfm_points_.empty());

        std::unordered_set<int> shared_point_ids;
        FindSharedPoints3DId(ref_view, ng_view, ref_line2D_idx,
                             ng_matched_line2D_idx, *ref_bpt_ptr, *ng_bpt_ptr,
                             sfm_points_, &shared_point_ids);

        std::unordered_set<int> valid_point3D_ids = shared_point_ids;
        shared_point_ids.clear();

        if (config_.use_unshared_points) {
          std::unordered_set<int> ng_point3D_ids;
          FindPoints3D(ng_matched_line2D_idx, *ng_bpt_ptr, &ng_point3D_ids);

          // Filter point3D according to the distance between the reprojected
          // point2D to reference or neighboring 2D infinite line.
          for (const int& point3D_id : ng_point3D_ids) {
            if (point3D_id == -1) continue;
            if (valid_point3D_ids.find(point3D_id) != valid_point3D_ids.end())
              continue;
            const V3D& point3D = sfm_points_.at(point3D_id);
            V2D point2D = ref_view.projection(point3D);
            V3D point2D_homo = homogeneous(point2D);
            double d = geometry::ComputeDistance(l1.coords(), point2D_homo);
            if (d < config_.max_point_line_reproject_distance) {
              valid_point3D_ids.insert(point3D_id);
            }
          }
          ng_point3D_ids.clear();
          for (const int& point3D_id : ref_point3D_ids) {
            if (point3D_id == -1) continue;
            if (valid_point3D_ids.find(point3D_id) != valid_point3D_ids.end())
              continue;
            const V3D& point3D = sfm_points_.at(point3D_id);
            V2D ng_point2D = ng_view.projection(point3D);
            V3D ng_point2D_homo = homogeneous(ng_point2D);
            double d = geometry::ComputeDistance(l2.coords(), ng_point2D_homo);
            if (d < config_.max_point_line_reproject_distance) {
              valid_point3D_ids.insert(point3D_id);
            }
          }
        }

        std::vector<V3D> valid_points3D;
        for (const int& point3D_id : valid_point3D_ids) {
          valid_points3D.push_back(sfm_points_.at(point3D_id));
        }
        valid_point3D_ids.clear();

        // step 1.1: Multiple Points triangulation
        if (!config_.disable_many_points_triangulation &&
            valid_points3D.size() >= 2) {
          Line3d line;
          bool success =
              TriangulateLineWithPoints(ref_view, l1, valid_points3D, &line);
          if (success) {
            // compute line uncertainty
            double u1 = line.computeUncertainty(ref_view, config_.var2d);
            double u2 = line.computeUncertainty(ng_view, config_.var2d);
            line.uncertainty = std::min(u1, u2);
            // store 3D line segment proposal
            proposals_tmp.emplace_back(line, ng_image_id,
                                       ng_matched_line2D_idx);
          }
        }

        // step 1.2: Line + Point triangulation
        if (!config_.disable_one_point_triangulation &&
            !valid_points3D.empty()) {
          for (const V3D& p : valid_points3D) {
            Line3d line = limap::triangulation::triangulate_with_one_point(
                l1, ref_view, l2, ng_view, p);
            if (line.score > 0) {
              // compute line uncertainty
              double u1 = line.computeUncertainty(ref_view, config_.var2d);
              double u2 = line.computeUncertainty(ng_view, config_.var2d);
              line.uncertainty = std::min(u1, u2);
              // store 3D line segment proposal
              proposals_tmp.emplace_back(line, ng_image_id,
                                         ng_matched_line2D_idx);
            }
          }
        }

        valid_points3D.clear();
        valid_points3D.shrink_to_fit();
      }

      // step 2: Line + VP triangulation
      if (config_.use_vp && !config_.disable_vp_triangulation) {
        // vp1
        if (vp_results_[ref_image_id].HasVP(ref_line2D_idx)) {
          V3D direc = limap::triangulation::getDirectionFromVP(
              vp_results_[ref_image_id].GetVP(ref_line2D_idx), ref_view);
          Line3d line = limap::triangulation::triangulate_with_direction(
              l1, ref_view, l2, ng_view, direc);
          if (line.score > 0) {
            // compute line uncertainty
            double u1 = line.computeUncertainty(ref_view, config_.var2d);
            double u2 = line.computeUncertainty(ng_view, config_.var2d);
            line.uncertainty = std::min(u1, u2);
            // store 3D line segment proposal
            proposals_tmp.emplace_back(line, ng_image_id,
                                       ng_matched_line2D_idx);
          }
        }
        // vp2
        if (vp_results_[ng_image_id].HasVP(ng_matched_line2D_idx)) {
          V3D direc = limap::triangulation::getDirectionFromVP(
              vp_results_[ng_image_id].GetVP(ng_matched_line2D_idx), ref_view);
          Line3d line = limap::triangulation::triangulate_with_direction(
              l1, ref_view, l2, ng_view, direc);
          if (line.score > 0) {
            // compute line uncertainty
            double u1 = line.computeUncertainty(ref_view, config_.var2d);
            double u2 = line.computeUncertainty(ng_view, config_.var2d);
            line.uncertainty = std::min(u1, u2);
            // store 3D line segment proposal
            proposals_tmp.emplace_back(line, ng_image_id,
                                       ng_matched_line2D_idx);
          }
        }
      }

      // step 3: Line + Line triangulation
      if (!config_.disable_algebraic_triangulation) {
        // test degeneracy by ray-plane angles
        V3D n2 = limap::triangulation::getNormalDirection(
            l2, ng_view);  // ((R^m)^{T}l_m)^T
        double angle_start =
            90 - acos(std::abs(n2.dot(ray1_start))) * 180.0 / M_PI;
        if (angle_start < config_.line_tri_angle_threshold) continue;
        double angle_end = 90 - acos(std::abs(n2.dot(ray1_end))) * 180.0 / M_PI;
        if (angle_end < config_.line_tri_angle_threshold) continue;

        // test weak epipolar constraints
        V3D coor_l2 = l2.coords();
        V3D homo_c_start = coor_l2.cross(coor_epline_start);
        V2D c_start = dehomogeneous(homo_c_start);
        V3D homo_c_end = coor_l2.cross(coor_epline_end);
        V2D c_end = dehomogeneous(homo_c_end);
        double c1 = (c_start - l2.start).dot(l2.direction()) / l2.length();
        double c2 = (c_end - l2.start).dot(l2.direction()) / l2.length();
        if (c1 > c2) std::swap(c1, c2);
        double IoU = (std::min(c2, 1.0) - std::max(c1, 0.0)) /
                     (std::max(c2, 1.0) - std::min(c1, 0.0));
        if (IoU < config_.IoU_threshold) continue;

        // triangulation
        Line3d line;
        if (!config_.use_endpoints_triangulation)
          line = limap::triangulation::triangulate(l1, ref_view, l2, ng_view);
        else
          line = limap::triangulation::triangulate_endpoints(l1, ref_view, l2,
                                                             ng_view);
        if (line.sensitivity(ref_view) > config_.sensitivity_threshold &&
            line.sensitivity(ng_view) > config_.sensitivity_threshold)
          line.score = -1;
        if (line.score > 0) {
          // compute line uncertainty
          double u1 = line.computeUncertainty(ref_view, config_.var2d);
          double u2 = line.computeUncertainty(ng_view, config_.var2d);
          line.uncertainty = std::min(u1, u2);
          // store 3D line segment proposal
          proposals_tmp.emplace_back(line, ng_image_id, ng_matched_line2D_idx);
        }
      }

      // store valid 3D line segment proposals
      for (const auto& proposal : proposals_tmp) {
        if (config_.use_range_filter) {
          // filter out proposals that are out of range
          if (!limap::triangulation::test_line_inside_ranges(
                  proposal.tri_line3D, ranges_))
            continue;
        }
#pragma omp critical
        { proposals[ref_line2D_idx].push_back(proposal); }
      }
      proposals_tmp.clear();
      proposals_tmp.shrink_to_fit();
    }
  }
}

void LineMapper::SelectLocalBestProposal() {
  PrintHeading1("Select local best proposal");

  sfm_points_.clear();
  vp_results_.clear();

  LineLinker linker = linker_;
  linker.linker_3d.config.set_to_shared_parent_scoring();
  linker.linker_2d.config.set_to_default();
  double score2D_th = linker.linker_2d.config.score_th;
  double score3D_th = linker.linker_3d.config.score_th;

  const auto image_ids = image_collection_.get_img_ids();
  size_t num_images = image_ids.size();
  progressbar bar(num_images);
  for (image_t image_id : image_ids) {
    bar.update();
    const size_t num_lines2d = all_line2d_.at(image_id).size();
    auto all_props = proposals_.at(image_id);
    auto all_local_best_porp_indices =
        local_best_proposal_indices_.at(image_id);

#pragma omp parallel for
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2d; line2d_idx++) {
      // init
      int& local_best_prop_idx = all_local_best_porp_indices[line2d_idx];
      std::vector<Proposal>& proposals = all_props[line2d_idx];
      if (proposals.empty()) {
        local_best_prop_idx = -1;
        continue;
      }
      const size_t num_prop = proposals.size();

      for (size_t i = 0; i < num_prop; ++i) {
        ////////////////////////////////////////////////////////////////////////
        // step1: compute similarity score with neighboring proposals
        ////////////////////////////////////////////////////////////////////////

        Proposal& proposal1 = proposals[i];
        const Line3d& line3D1 = proposal1.tri_line3D;
        const image_t image_id1 = proposal1.ng_image_id;
        std::unordered_map<image_t, std::vector<std::pair<double, size_t>>>
            score_table;  // (ng_image_id, (similarity_score, ng_prop_idx))
        std::unordered_map<image_t, std::unordered_set<line2d_t>>
            local_support_lines2D_set;  // (ng_image_id,
                                        // (valid_ng_line2d_idx))

        for (size_t j = 0; j < num_prop; ++j) {
          if (i == j) continue;
          const Proposal& proposal2 = proposals[j];
          const Line3d& line3D2 = proposal2.tri_line3D;
          const image_t image_id2 = proposal2.ng_image_id;
          if (image_id1 == image_id2) continue;
          line2d_t line2d_idx2 = proposal2.ng_line2D_idx;
          const CameraView& view2 = image_collection_.camview(image_id2);

          // compute the similarity score of proposal1 and proposal2
          double score3D = linker.compute_score_3d(line3D1, line3D2);
          if (score3D == 0) continue;
          double score2D =
              linker.compute_score_2d(line3D1.projection(view2),
                                      all_line2d_.at(image_id2)[line2d_idx2]);
          if (score2D == 0) continue;
          double score = std::min(score2D, score3D);

          THROW_CHECK_GE(score, score2D_th);
          THROW_CHECK_GE(score, score3D_th);
          score_table[image_id2].emplace_back(score, j);
          if (config_.use_local_valid_lines2d) {
            local_support_lines2D_set[image_id2].insert(line2d_idx2);
          }
        }

        proposal1.local_valid_lines2D.clear();

        if (config_.use_local_valid_lines2d) {
          for (const auto& kv : local_support_lines2D_set) {
            proposal1.local_valid_lines2D[kv.first].assign(kv.second.begin(),
                                                           kv.second.end());
          }
        }

        ////////////////////////////////////////////////////////////////////////
        // step2.1: compute local consistency score for each proposal
        // step2.2: find local supporting 2D line segments for each proposal
        ////////////////////////////////////////////////////////////////////////

        // reset
        proposal1.local_score = 0.0;  // local consistency score
        proposal1.local_support_lines2D.clear();

        // Note: One neighboring image contributes at most one supporting
        // proposal, thereby at most one supporting 2D line segment.
        for (auto it = score_table.begin(); it != score_table.end(); it++) {
          const std::vector<std::pair<double, size_t>>& scores = it->second;
          double max_score = -1.0;
          size_t max_score_idx = 0;
          for (size_t idx = 0; idx < scores.size(); idx++) {
            double score = scores[idx].first;
            THROW_CHECK_GE(score, 0.0);
            if (score > max_score) {
              max_score = score;
              max_score_idx = idx;
            }
          }
          THROW_CHECK_GT(max_score, 0)
          proposal1.local_score += max_score;

          if (config_.debug_mode) {
            size_t support_prop_idx = scores.at(max_score_idx).second;
            const auto& local_support_proposal = proposals[support_prop_idx];

            proposal1.local_support_lines2D.emplace_back(
                local_support_proposal.ng_image_id,
                local_support_proposal.ng_line2D_idx);
          }
        }

        if (config_.debug_mode) {
          // check
          if (proposal1.local_score < 0.5) {
            THROW_CHECK(proposal1.local_support_lines2D.empty());
          } else {
            THROW_CHECK(!proposal1.local_support_lines2D.empty());
            if (proposal1.local_score > 1.0) {
              THROW_CHECK_GE(proposal1.local_support_lines2D.size(), 2);
            }
          }
        }
      }

      ////////////////////////////////////////////////////////////////////////
      // step3: select local best proposal which has max local score
      ////////////////////////////////////////////////////////////////////////

      double max_score = -1.0;
      for (int i = 0; i < num_prop; ++i) {
        double local_score = proposals[i].local_score;
        if (local_score > max_score) {
          max_score = local_score;
          local_best_prop_idx = i;
        }
      }

      // clear
      if (config_.debug_mode) {
        for (int i = 0; i < num_prop; ++i) {
          if (i != local_best_prop_idx) {
            proposals[i].local_support_lines2D.clear();
            proposals[i].local_support_lines2D.shrink_to_fit();
          }
        }
      }
    }

    proposals_.at(image_id) = all_props;
    local_best_proposal_indices_.at(image_id) = all_local_best_porp_indices;
  }
  select_local_best_prop_ = true;
}

void LineMapper::InitializeGlobalBestProposal(const std::string& method,
                                              double global_score_th) {
  PrintHeading1("Initialize global best proposal");
  std::cout << "method = " << method << std::endl;
  if (method == "local" && !select_local_best_prop_) {
    throw std::runtime_error(
        "Error: Please call SelectLocalBestProposal() in advance.");
  }

  sfm_points_.clear();
  vp_results_.clear();

  for (image_t image_id : image_collection_.get_img_ids()) {
    auto& all_props = proposals_.at(image_id);
    auto& all_global_best_props = global_best_proposal_indices_.at(image_id);
    const auto& all_local_best_props =
        local_best_proposal_indices_.at(image_id);
    auto& visible_line2d_flags = visible_line2d_flags_.at(image_id);

    for (line2d_t line2d_idx = 0; line2d_idx < all_line2d_.at(image_id).size();
         line2d_idx++) {
      auto& props = all_props[line2d_idx];
      if (props.empty()) {
        all_global_best_props[line2d_idx] = -1;
        visible_line2d_flags[line2d_idx] = 0;
        continue;
      }
      if (method == "random") {
        // initialize global best proposal
        std::random_device seed;
        std::ranlux48 engine(seed());
        std::uniform_int_distribution<> distrib(0, props.size() - 1);
        int random = distrib(engine);
        all_global_best_props[line2d_idx] = random;

        // initialize visible_line2D_flags
        visible_line2d_flags[line2d_idx] = 1;  // to start iteration

        // initialize global score and global supporting 2D line segments
        for (size_t i = 0; i < props.size(); i++) {
          auto& prop = props[i];
          // to be able to build line tracks if there is no iteration (i.e.
          // max_iter_num = 0)
          prop.global_score = std::numeric_limits<double>::max();

          // Note: If there is no iteration (i.e. max_iter_num = 0), the
          // consistency of support relationships for the global method (i.e.
          // the proposed iterative method in paper) cannot be evaluated,
          // because the supporting 2D line segments (i.e.
          // global_support_lines2D) will be empty.
          prop.global_support_lines2D.clear();
        }
      } else if (method == "local") {
        // initialize global best proposal
        int local_best_prop_idx = all_local_best_props[line2d_idx];
        THROW_CHECK_GE(local_best_prop_idx, 0);
        all_global_best_props[line2d_idx] = local_best_prop_idx;

        // initialize visible_line2D_flags
        if (props.at(local_best_prop_idx).local_score < global_score_th) {
          visible_line2d_flags[line2d_idx] = 0;
        } else {
          visible_line2d_flags[line2d_idx] = 1;
        }

        // initialize global score and global supporting 2D line segments
        for (size_t i = 0; i < props.size(); i++) {
          auto& prop = props[i];
          // to be able to build line tracks if there is no iteration (i.e.
          // max_iter_num = 0)
          prop.global_score = prop.local_score;

          // Note: If there is no iteration (i.e. max_iter_num = 0), the
          // consistency of support relationships for the global method (i.e.
          // the proposed iterative method in paper) cannot be evaluated,
          // because the supporting 2D line segments (i.e.
          // global_support_lines2D) will be empty.
          prop.global_support_lines2D.clear();
        }
      } else {
        throw std::invalid_argument("Error: Not Implemented.");
      }
    }
  }
}

size_t LineMapper::SelectGlobalBestProposal(const double& global_score_th) {
  LineLinker linker = linker_;
  linker.linker_3d.config.set_to_spatial_merging();
  linker.linker_2d.config.set_to_default();
  double score3D_th = linker.linker_3d.config.score_th;
  double score2D_th = linker.linker_2d.config.score_th;

  if (config_.debug_mode) {
    std::cout << "-----------------------------------------------" << std::endl;
    std::cout << "[DEBUG] th_angle = " << linker.linker_3d.config.th_angle
              << std::endl;
    std::cout << "[DEBUG] th_overlap = " << linker.linker_3d.config.th_overlap
              << std::endl;
    std::cout << "[DEBUG] th_innerseg = " << linker.linker_3d.config.th_innerseg
              << std::endl;
    std::cout << "[DEBUG] global_score_th = " << global_score_th << std::endl;
  }

  size_t num_visible_lines2D = 0;
  size_t num_changed_lines2D = 0;
  auto new_best_proposal_indices = global_best_proposal_indices_;
  const auto image_ids = image_collection_.get_img_ids();
  const size_t num_images = image_ids.size();

  std::cout << "Selecting global best proposal for each 2D line segment..."
            << std::endl;
  progressbar bar(num_images);
  for (const image_t& image_id : image_ids) {
    bar.update();
    const auto& best_proposal_indices =
        global_best_proposal_indices_.at(image_id);
    const auto& visible_line2d_flags = visible_line2d_flags_.at(image_id);
    size_t num_lines2D = all_line2d_.at(image_id).size();
    const auto& all_matches = matches_.at(image_id);

    auto all_props = proposals_.at(image_id);
    auto all_new_best_proposal_indices = new_best_proposal_indices.at(image_id);

#pragma omp parallel for
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2D; line2d_idx++) {
      std::vector<Proposal>& props = all_props[line2d_idx];

      const int prev_best_prop_idx = best_proposal_indices[line2d_idx];

      if (prev_best_prop_idx == -1) {
        THROW_CHECK(props.empty());
        continue;
      }
      if (props.empty()) {
        THROW_CHECK_EQ(prev_best_prop_idx, -1);
        continue;
      }

      THROW_CHECK_GE(prev_best_prop_idx, 0);

      if (config_.debug_mode) {
#pragma omp critical
        {
          if (visible_line2d_flags[line2d_idx] == 1) {
            num_visible_lines2D++;
          }
        }
      }

      // (ng_image_id, (ng_line2D_idx))
      const std::unordered_map<image_t, std::vector<line2d_t>>* matches_ptr;
      if (!config_.use_local_valid_lines2d) {
        matches_ptr = &(all_matches[line2d_idx]);
      }

      // init
      double max_global_score = -1.0;
      int new_best_prop_idx = prev_best_prop_idx;

      // compute global consistency score for each proposal
      for (size_t prop_idx = 0; prop_idx < props.size(); prop_idx++) {
        Proposal& prop = props[prop_idx];
        const Line3d& line3D1 = prop.tri_line3D;

        // reset
        prop.global_score = 0.0;
        prop.global_support_lines2D.clear();

        if (config_.use_local_valid_lines2d) {
          matches_ptr = &(prop.local_valid_lines2D);
        }
        CHECK_NOTNULL(matches_ptr);

        // Note: One neighboring image contributes at most one supporting 2D
        // line segment and hence at most one global consistency score.
        for (const auto& kv : *matches_ptr) {
          image_t ng_image_id = kv.first;
          THROW_CHECK_NE(image_id, ng_image_id);

          double max_score = -1.0;
          line2d_t support_line2d_idx;
          for (const line2d_t& ng_line2D_idx : kv.second) {
            if (visible_line2d_flags_.at(ng_image_id)[ng_line2D_idx]) {
              const auto& ng_props = proposals_.at(ng_image_id)[ng_line2D_idx];
              const int& ng_best_prop_idx =
                  global_best_proposal_indices_.at(ng_image_id)[ng_line2D_idx];

              THROW_CHECK_GE(ng_best_prop_idx, 0);

              const Proposal& ng_best_prop = ng_props.at(ng_best_prop_idx);

              const Line3d& line3D2 = ng_best_prop.tri_line3D;
              double score3d = linker.compute_score_3d(line3D1, line3D2);
              if (score3d == 0) continue;

              double score2d = linker.compute_score_2d(
                  line3D1.projection(image_collection_.camview(ng_image_id)),
                  all_line2d_.at(ng_image_id)[ng_line2D_idx]);
              if (score2d == 0) continue;

              double score = std::min(score3d, score2d);

              if (score > max_score) {
                max_score = score;
                support_line2d_idx = ng_line2D_idx;
              }
            }
          }

          if (max_score > 0) {
            THROW_CHECK_GE(max_score, score2D_th)
            THROW_CHECK_GE(max_score, score3D_th)
            prop.global_score += max_score;
            if (config_.debug_mode) {
              prop.global_support_lines2D.emplace_back(ng_image_id,
                                                       support_line2d_idx);
            }
          } else {
            // The proposal has no supporting 2D line segment in this
            // neighboring image.
          }
        }

        if (config_.debug_mode) {
          // check
          if (prop.global_score < 0.5) {
            THROW_CHECK(prop.global_support_lines2D.empty());
          } else {
            THROW_CHECK(!prop.global_support_lines2D.empty());
            if (prop.global_score > 1.0) {
              THROW_CHECK_GE(prop.global_support_lines2D.size(), 2);
            }
          }
        }

        // update max_global_score
        if (prop.global_score > max_global_score) {
          max_global_score = prop.global_score;
          new_best_prop_idx = prop_idx;
        }
      }

      // check
      THROW_CHECK_GE(max_global_score, 0.0)
      double new_best_prop_global_score = max_global_score;
      THROW_CHECK_EQ(new_best_prop_global_score,
                     props[new_best_prop_idx].global_score);
      if (std::abs(new_best_prop_global_score) < EPS) {
        THROW_CHECK_EQ(new_best_prop_idx, 0);
      }

      // clear
      if (config_.debug_mode) {
        for (size_t prop_idx = 0; prop_idx < props.size(); prop_idx++) {
          if (prop_idx != new_best_prop_idx) {
            props[prop_idx].global_support_lines2D.clear();
            props[prop_idx].global_support_lines2D.shrink_to_fit();
          }
        }
      }

      // store new_best_prop_idx
      all_new_best_proposal_indices.at(line2d_idx) = new_best_prop_idx;

#pragma omp critical
      {
        if (new_best_prop_idx != prev_best_prop_idx) {
          num_changed_lines2D++;
        }
      }
    }

    proposals_.at(image_id) = all_props;
    new_best_proposal_indices.at(image_id) = all_new_best_proposal_indices;
  }

  // re-assign visibility of line2d
  for (const image_t& image_id : image_ids) {
    const auto& all_props = proposals_.at(image_id);
    auto visible_line2d_flags = visible_line2d_flags_.at(image_id);
    const size_t num_lines2d = all_line2d_.at(image_id).size();
    const auto& all_new_best_proposal_indices =
        new_best_proposal_indices.at(image_id);

#pragma omp parallel for
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2d; line2d_idx++) {
      int new_best_prop_idx = all_new_best_proposal_indices.at(line2d_idx);
      if (new_best_prop_idx == -1) continue;
      THROW_CHECK_GE(new_best_prop_idx, 0);
      const auto& new_best_prop = all_props[line2d_idx].at(new_best_prop_idx);
      if (new_best_prop.global_score < global_score_th) {
        visible_line2d_flags[line2d_idx] = 0;
      } else {
        visible_line2d_flags[line2d_idx] = 1;
      }
    }
    visible_line2d_flags_.at(image_id) = visible_line2d_flags;
  }

  // update all best props
  global_best_proposal_indices_ = new_best_proposal_indices;

  if (config_.debug_mode) {
    std::cout << "[DEBUG] The nunmber of visible 2D line segments in current "
                 "iteration = "
              << num_visible_lines2D << std::endl;
  }
  return num_changed_lines2D;
}

std::tuple<double, size_t, size_t>
LineMapper::ComputeAngleConsistencyPercentage(const std::string& best_prop_type,
                                              double th_angle) const {
  size_t num_valid_line2d = 0;  // i.e. number of valid best proposals
  double angle_ratio_sum = 0.0;
  size_t num_supports_sum = 0;

  for (image_t image_id : image_collection_.get_img_ids()) {
    size_t num_lines2D = all_line2d_.at(image_id).size();
    const auto& all_global_best_proposal_indices =
        global_best_proposal_indices_.at(image_id);
    const auto& all_local_best_proposal_indices =
        local_best_proposal_indices_.at(image_id);

    const auto& all_props = proposals_.at(image_id);
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2D; line2d_idx++) {
      size_t num_consistent_angle_support_lines2D = 0;
      size_t num_support_lines2D = 0;

      if (best_prop_type == "local") {
        int best_idx = all_local_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;
        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.local_support_lines2D.size();

        for (const auto& image_line2d : best_prop.local_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2d_idx = image_line2d.second;

          const int& supp_best_idx =
              local_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2d_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2d_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;

          double angle = limap::compute_angle<Line3d>(line3D1, line3D2);
          if (angle <= th_angle) {
            num_consistent_angle_support_lines2D++;
          }
        }
      } else if (best_prop_type == "global") {
        const int& best_idx = all_global_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;

        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.global_support_lines2D.size();

        for (const auto& image_line2d : best_prop.global_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2D_idx = image_line2d.second;

          const int64_t& supp_best_idx =
              global_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2D_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2D_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;

          double angle = limap::compute_angle<Line3d>(line3D1, line3D2);
          if (angle <= th_angle) {
            num_consistent_angle_support_lines2D++;
          }
        }
      } else {
        throw std::invalid_argument("Error: Not implemented.");
      }

      // Filter best proposal having no supporting 2D line segment.
      if (num_support_lines2D > 0) {
        num_valid_line2d++;
        angle_ratio_sum += (double(num_consistent_angle_support_lines2D) /
                            (double(num_support_lines2D)));
        num_supports_sum += num_support_lines2D;
      }
    }
  }

  if (num_valid_line2d == 0) {
    return {0.0, 0, 0};
  } else {
    return {(angle_ratio_sum / double(num_valid_line2d)) * 100.0,
            num_valid_line2d, num_supports_sum};
  }
}

std::tuple<double, size_t, size_t>
LineMapper::ComputeDistanceConsistencyPercentage(
    const std::string& best_prop_type, double th_dist) const {
  size_t num_valid_line2d = 0;  // i.e. number of valid best proposals
  double dist_ratio_sum = 0.0;
  size_t num_supports_sum = 0;

  for (image_t image_id : image_collection_.get_img_ids()) {
    size_t num_lines2D = all_line2d_.at(image_id).size();
    const auto& all_global_best_proposal_indices =
        global_best_proposal_indices_.at(image_id);
    const auto& all_local_best_proposal_indices =
        local_best_proposal_indices_.at(image_id);

    const auto& all_props = proposals_.at(image_id);
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2D; line2d_idx++) {
      size_t num_consistent_dist_support_lines2D = 0;
      size_t num_support_lines2D = 0;

      if (best_prop_type == "local") {
        int best_idx = all_local_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;
        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.local_support_lines2D.size();

        for (const auto& image_line2d : best_prop.local_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2d_idx = image_line2d.second;

          const int& supp_best_idx =
              local_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2d_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2d_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;

          double max_dist1 = limap::compute_distance<Line3d>(
              line3D1, line3D2, limap::LineDistType::PERPENDICULAR_ONEWAY);

          double max_dist2 = limap::compute_distance<Line3d>(
              line3D2, line3D1, limap::LineDistType::PERPENDICULAR_ONEWAY);

          double max_max_dist = std::max(max_dist1, max_dist2);
          if (max_max_dist <= th_dist) {
            num_consistent_dist_support_lines2D++;
          }
        }
      } else if (best_prop_type == "global") {
        const int& best_idx = all_global_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;

        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.global_support_lines2D.size();

        for (const auto& image_line2d : best_prop.global_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2D_idx = image_line2d.second;

          const int64_t& supp_best_idx =
              global_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2D_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2D_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;

          double max_dist1 = limap::compute_distance<Line3d>(
              line3D1, line3D2, limap::LineDistType::PERPENDICULAR_ONEWAY);

          double max_dist2 = limap::compute_distance<Line3d>(
              line3D2, line3D1, limap::LineDistType::PERPENDICULAR_ONEWAY);

          double max_max_dist = std::max(max_dist1, max_dist2);
          if (max_max_dist <= th_dist) {
            num_consistent_dist_support_lines2D++;
          }
        }
      } else {
        throw std::invalid_argument("Error: Not implemented.");
      }

      // Filter best proposal having no supporting 2D line segment.
      if (num_support_lines2D > 0) {
        num_valid_line2d++;
        dist_ratio_sum += (double(num_consistent_dist_support_lines2D) /
                           (double(num_support_lines2D)));
        num_supports_sum += num_support_lines2D;
      }
    }
  }

  if (num_valid_line2d == 0) {
    return {0.0, 0, 0};
  } else {
    return {(dist_ratio_sum / double(num_valid_line2d)) * 100.0,
            num_valid_line2d, num_supports_sum};
  }
}

std::tuple<double, size_t, size_t>
LineMapper::ComputeAngleDistanceConsistencyPercentage(
    const std::string& best_prop_type, const double th_angle,
    const double th_dist) const {
  size_t num_valid_line2d = 0;  // i.e. number of valid best proposals
  double consistent_ratio_sum = 0.0;
  size_t num_supports_sum = 0;

  for (image_t image_id : image_collection_.get_img_ids()) {
    size_t num_lines2D = all_line2d_.at(image_id).size();
    const auto& all_global_best_proposal_indices =
        global_best_proposal_indices_.at(image_id);
    const auto& all_local_best_proposal_indices =
        local_best_proposal_indices_.at(image_id);

    const auto& all_props = proposals_.at(image_id);
    for (line2d_t line2d_idx = 0; line2d_idx < num_lines2D; line2d_idx++) {
      size_t num_consistent_support_lines2D = 0;
      size_t num_support_lines2D = 0;

      if (best_prop_type == "local") {
        int best_idx = all_local_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;
        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.local_support_lines2D.size();

        for (const auto& image_line2d : best_prop.local_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2d_idx = image_line2d.second;

          const int& supp_best_idx =
              local_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2d_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2d_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;

          double angle = limap::compute_angle<Line3d>(line3D1, line3D2);
          if (angle > th_angle) continue;

          double max_dist1 = limap::compute_distance<Line3d>(
              line3D1, line3D2, limap::LineDistType::PERPENDICULAR_ONEWAY);
          double max_dist2 = limap::compute_distance<Line3d>(
              line3D2, line3D1, limap::LineDistType::PERPENDICULAR_ONEWAY);
          double max_max_dist = std::max(max_dist1, max_dist2);
          if (max_max_dist > th_dist) continue;

          num_consistent_support_lines2D++;
        }
      } else if (best_prop_type == "global") {
        const int& best_idx = all_global_best_proposal_indices[line2d_idx];
        if (best_idx == -1) continue;

        const auto& props = all_props[line2d_idx];
        THROW_CHECK(!props.empty())
        const Proposal& best_prop = props.at(best_idx);
        const Line3d& line3D1 = best_prop.tri_line3D;

        num_support_lines2D = best_prop.global_support_lines2D.size();

        for (const auto& image_line2d : best_prop.global_support_lines2D) {
          image_t supp_image_id = image_line2d.first;
          line2d_t supp_line2D_idx = image_line2d.second;

          const int64_t& supp_best_idx =
              global_best_proposal_indices_.at(supp_image_id)
                  .at(supp_line2D_idx);
          if (supp_best_idx == -1) continue;

          const Line3d& line3D2 = proposals_.at(supp_image_id)[supp_line2D_idx]
                                      .at(supp_best_idx)
                                      .tri_line3D;
          double angle = limap::compute_angle<Line3d>(line3D1, line3D2);
          if (angle > th_angle) continue;

          double max_dist1 = limap::compute_distance<Line3d>(
              line3D1, line3D2, limap::LineDistType::PERPENDICULAR_ONEWAY);
          double max_dist2 = limap::compute_distance<Line3d>(
              line3D2, line3D1, limap::LineDistType::PERPENDICULAR_ONEWAY);
          double max_max_dist = std::max(max_dist1, max_dist2);
          if (max_max_dist > th_dist) continue;

          num_consistent_support_lines2D++;
        }
      } else {
        throw std::invalid_argument("Error: Not implemented.");
      }

      // Filter best proposal having no supporting 2D line segment.
      if (num_support_lines2D > 0) {
        num_valid_line2d++;
        consistent_ratio_sum += (double(num_consistent_support_lines2D) /
                                 (double(num_support_lines2D)));
        num_supports_sum += num_support_lines2D;
      }
    }
  }

  if (num_valid_line2d == 0) {
    return {0.0, 0, 0};
  } else {
    return {(consistent_ratio_sum / double(num_valid_line2d)) * 100.0,
            num_valid_line2d, num_supports_sum};
  }
}

void LineMapper::CreateMatchGraph(Graph* graph) {
  PrintHeading1("Create match graph for clustering");

  LineLinker linker_clustering = linker_;
  linker_clustering.linker_3d.config.set_to_spatial_merging();

  size_t num_best_prop = 0;
  size_t num_valid_best_prop = 0;

  // insert edges one by one to build the match graph
  for (image_t image_id1 : image_collection_.get_img_ids()) {
    size_t num_lines2D = all_line2d_.at(image_id1).size();
    const auto& best_prop_indices = global_best_proposal_indices_.at(image_id1);
    const auto& props = proposals_.at(image_id1);
    const auto& all_matches = matches_.at(image_id1);

    for (line2d_t line2d_idx1 = 0; line2d_idx1 < num_lines2D; ++line2d_idx1) {
      const int best_prop_idx1 = best_prop_indices[line2d_idx1];

      if (best_prop_idx1 == -1) continue;
      num_best_prop++;
      THROW_CHECK_GE(best_prop_idx1, 0);
      const auto& best_prop1 = props[line2d_idx1].at(best_prop_idx1);

      if (best_prop1.global_score < 0.5) {
        // the best proposal has no supporting 2D line
        THROW_CHECK(best_prop1.global_support_lines2D.empty());
        continue;
      }

      num_valid_best_prop++;

      const Line3d& line3D1 = best_prop1.tri_line3D;

      // (ng_image_id, (ng_line2D_idx))
      const auto& matches = all_matches[line2d_idx1];

      for (const auto& kv : matches) {
        image_t image_id2 = kv.first;
        THROW_CHECK_NE(image_id1, image_id2);
        for (const auto& line2d_idx2 : kv.second) {
          int best_prop_idx2 =
              global_best_proposal_indices_.at(image_id2)[line2d_idx2];
          if (best_prop_idx2 == -1) continue;
          THROW_CHECK_GE(best_prop_idx2, 0);

          const Proposal& best_prop2 =
              proposals_.at(image_id2)[line2d_idx2].at(best_prop_idx2);

          if (best_prop2.global_score < 0.5) {
            // the best proposal has no supporting 2D line
            THROW_CHECK(best_prop2.global_support_lines2D.empty());
            continue;
          }

          const Line3d& line3D2 = best_prop2.tri_line3D;

          double score_3d =
              linker_clustering.compute_score_3d(line3D1, line3D2);
          double score = score_3d;

          if (score == 0) continue;

          PatchNode* node1 = graph->FindOrCreateNode(image_id1, line2d_idx1);
          PatchNode* node2 = graph->FindOrCreateNode(image_id2, line2d_idx2);
          graph->AddEdge(node1, node2, score);
        }
      }
    }
  }
  std::cout << "num node = " << graph->nodes.size() << std::endl;
  std::cout << "num best prop = " << num_best_prop << std::endl;
  std::cout << "num valid best prop = " << num_valid_best_prop << std::endl;
  std::cout << "valid ratio (num valid best prop / num best prop)= "
            << double(num_valid_best_prop) / (double(num_best_prop) + EPS)
            << std::endl;
}

void LineMapper::BuildTracksFromClusters(Graph* graph) {
  PrintHeading1("Build line tracks from clusters");

  LineLinker linker = linker_;
  linker.linker_3d.config.set_to_avgtest_merging();

  // collect 3D lines for each node
  std::vector<Line3d> lines_nodes;
  for (auto it = graph->nodes.begin(); it != graph->nodes.end(); ++it) {
    image_t image_id = (*it)->image_idx;
    line2d_t line2d_idx = (*it)->line_idx;
    int best_prop_idx = global_best_proposal_indices_.at(image_id)[line2d_idx];
    THROW_CHECK_GE(best_prop_idx, 0);
    lines_nodes.push_back(
        proposals_.at(image_id)[line2d_idx][best_prop_idx].tri_line3D);
  }

  std::vector<int> track_labels;
  if (config_.clustering_method == "greedy")
    track_labels =
        limap::merging::ComputeLineTrackLabelsGreedy(*graph, lines_nodes);
  else if (config_.clustering_method == "exhaustive")
    track_labels = limap::merging::ComputeLineTrackLabelsExhaustive(
        *graph, lines_nodes, linker.linker_3d);
  else if (config_.clustering_method == "avg")
    track_labels = limap::merging::ComputeLineTrackLabelsAvg(
        *graph, lines_nodes, linker.linker_3d);
  else if (config_.clustering_method == "collinearity2D")
    track_labels = limap::ComputeLineTrackLabelsCollinearity2D(
        *graph, lines_nodes, all_line2d_, config_.max_same_line3D_overlap,
        config_.min_same_line3D_angle, config_.min_same_line3D_perp_dist);
  else
    throw std::runtime_error(
        "Error: The given clustering method is not implemented");

  if (track_labels.empty()) return;
  int n_tracks =
      *std::max_element(track_labels.begin(), track_labels.end()) + 1;
  tracks_.clear();
  tracks_.resize(n_tracks);

  // set all lines into tracks
  size_t n_nodes = graph->nodes.size();
  for (size_t node_id = 0; node_id < n_nodes; ++node_id) {
    PatchNode* node = graph->nodes[node_id];
    image_t image_id = node->image_idx;
    line2d_t line2d_idx = node->line_idx;

    const Line2d& line2d = all_line2d_.at(image_id)[line2d_idx];
    int best_prop_idx = global_best_proposal_indices_.at(image_id)[line2d_idx];
    THROW_CHECK_GE(best_prop_idx, 0);
    const auto& best_prop = proposals_.at(image_id)[line2d_idx][best_prop_idx];

    const Line3d& line3d = best_prop.tri_line3D;
    double score = best_prop.global_score;

    int track_id = track_labels[node_id];
    if (track_id == -1) continue;
    tracks_[track_id].node_id_list.push_back(node_id);
    tracks_[track_id].image_id_list.push_back(image_id);
    tracks_[track_id].line_id_list.push_back(line2d_idx);
    tracks_[track_id].line2d_list.push_back(line2d);
    tracks_[track_id].line3d_list.push_back(line3d);
    tracks_[track_id].score_list.push_back(score);
  }

  // aggregate 3d lines to get final line proposal
  for (auto it = tracks_.begin(); it != tracks_.end(); ++it) {
    it->line = AggregateLine3dList(it->line3d_list, it->score_list,
                                   config_.aggregate_method,
                                   config_.num_outliers_aggregator);
  }
}

std::vector<LineTrack> LineMapper::BuildGlobalLineTrack(bool show_valid) {
  if (config_.line_track_building_method == "cluster") {
    tracks_.clear();
    Graph match_graph;
    CreateMatchGraph(&match_graph);
    BuildTracksFromClusters(&match_graph);
    return tracks_;
  } else if (config_.line_track_building_method == "best_prop") {
    tracks_.clear();
    for (const auto& pair : global_best_proposal_indices_) {
      const image_t image_id = pair.first;
      const auto& global_best_prop_indices = pair.second;

      const auto& props = proposals_.at(image_id);
      const auto& lines2D = all_line2d_.at(image_id);
      const size_t num_lines2D = lines2D.size();

      for (size_t line2d_idx = 0; line2d_idx < num_lines2D; line2d_idx++) {
        size_t best_idx = global_best_prop_indices[line2d_idx];
        if (best_idx == -1) continue;

        THROW_CHECK_GE(best_idx, 0);
        const Proposal& best_prop = props[line2d_idx][best_idx];

        if (show_valid) {
          if (best_prop.global_score < 0.5) continue;
        }

        LineTrack track;
        track.line = best_prop.tri_line3D;
        track.image_id_list.push_back(image_id);
        track.line_id_list.push_back(line2d_idx);
        track.line2d_list.push_back(lines2D[line2d_idx]);
        tracks_.push_back(track);
      }
    }
    return tracks_;
  } else {
    throw std::runtime_error("Error: Not Implemented.");
  }
}

size_t LineMapper::GetNumProposal() const {
  size_t num = 0;
  for (const auto& kv : proposals_) {
    for (const auto& props : kv.second) {
      num += props.size();
    }
  }
  return num;
}

size_t LineMapper::GetNumLines2dHavingProposal() const {
  size_t num = 0;
  for (const auto& kv : proposals_) {
    for (const auto& props : kv.second) {
      if (!props.empty()) num++;
    }
  }
  return num;
}

}  // namespace limap