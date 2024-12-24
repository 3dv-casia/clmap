#pragma once

#include "util/types.h"

namespace limap {

// Unique identifier for cameras.
typedef uint32_t camera_t;

// Unique identifier for images.
typedef uint32_t image_t;

// Index of a 2D point in an image.
typedef uint32_t point2d_t;

// Unique identifier for 3D points.
typedef uint64_t point3d_t;

// Index of a 2D line segment in an image.
typedef uint32_t line2d_t;

// Unique identifier for 2D line segments.
typedef std::pair<image_t, line2d_t> image_line2d_t;

}  // namespace limap
