#pragma once

#include "clmap/util/types.h"

#include "base/linebase.h"

namespace limap {

// 3D line segment proposal.
struct Proposal {
 public:
  Proposal(){};
  Proposal(const Line3d& _tri_line3D, const image_t& _ng_image_id,
           const line2d_t& _ng_line2D_idx);

  // Triangulated 3D line segment.
  Line3d tri_line3D;

  // The neighboring image (ID) generating this propsoal.
  image_t ng_image_id;

  // The neighboring 2D line segment (idx) generating this propsoal.
  line2d_t ng_line2D_idx;

  // Local valid (neighboring) 2D line segments, may be empty.
  // (ng_image_id, (ng_valid_line2d_idx))
  std::unordered_map<image_t, std::vector<line2d_t>> local_valid_lines2D;

  // Local supporting 2D line segements, may be empty.
  std::vector<image_line2d_t> local_support_lines2D;

  // Local consistency score (>= 0.0).
  double local_score;

  // Global supporting 2D line segements, may be empty.
  std::vector<image_line2d_t> global_support_lines2D;

  // Global consistency score (>= 0.0).
  double global_score;
};

}  // namespace limap
