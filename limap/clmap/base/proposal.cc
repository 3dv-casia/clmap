#include "clmap/base/proposal.h"

namespace limap {

Proposal::Proposal(const Line3d& _tri_line3D, const image_t& _ng_image_id,
                   const line2d_t& _ng_line2D_idx)
    : tri_line3D(_tri_line3D),
      ng_image_id(_ng_image_id),
      ng_line2D_idx(_ng_line2D_idx) {}

}  // namespace limap
