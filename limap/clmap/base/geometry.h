#pragma once

#include "clmap/base/infinite_plane3d.h"
#include "clmap/util/types.h"

#include "base/image_collection.h"
#include "base/linetrack.h"
#include "base/pointtrack.h"

namespace limap {

namespace geometry {

double ComputeAngle(const V3D& dir1, const V3D& dir2);

double ComputeDistance(const V3D& line2d, const V3D& point2D);

double ComputeSignedDistancePoint3dToPlane3d(
    const V3D& point_xyz, const InfinitePlane3d& inf_plane3d);

double ComputeDistancePoint3dToPlane3d(const V3D& point_xyz,
                                       const InfinitePlane3d& inf_plane3d);

void ComputeLines3dUncertainty(const std::vector<LineTrack>& line_tracks,
                               const ImageCollection& img_cols,
                               std::vector<std::pair<V3D, double>>* u_start,
                               std::vector<std::pair<V3D, double>>* u_end,
                               std::vector<std::pair<V3D, double>>* u_median,
                               double var2d = 4.0,
                               const std::string& u_method = "median");

void ComputeLines3dUncertainty(
    const std::unordered_map<int, LineTrack>& line_tracks,
    const ImageCollection& img_cols,
    std::unordered_map<int, std::pair<V3D, double>>* u_start,
    std::unordered_map<int, std::pair<V3D, double>>* u_end,
    std::unordered_map<int, std::pair<V3D, double>>* u_median, double var2d,
    const std::string& u_method);

void ComputeLine3dUncertainty(const LineTrack& line_track,
                              const ImageCollection& img_cols,
                              std::pair<V3D, double>* u_start,
                              std::pair<V3D, double>* u_end,
                              std::pair<V3D, double>* u_median,
                              double var2d = 4.0,
                              const std::string& u_method = "median");

void ComputePoints3dUncertainty(const std::vector<PointTrack>& point_tracks,
                                const ImageCollection& img_cols,
                                std::vector<double>* uncertainties,
                                double var2d = 1.0,
                                const std::string& u_method = "median");

void ComputePoints3dUncertainty(
    const std::unordered_map<int, PointTrack>& point_tracks,
    const ImageCollection& img_cols,
    std::unordered_map<int, double>* uncertainties, double var2d = 1.0,
    const std::string& u_method = "median");

void ComputePoint3dUncertainty(const PointTrack& point_track,
                               const ImageCollection& img_cols,
                               double* uncertainty, double var2d = 1.0,
                               const std::string& u_method = "median");

// Return true -> line3d on the infinite plane3d, false -> otherwise.
bool TestLine3dOnInfPlane3d(const Line3d& line3d,
                            const InfinitePlane3d& inf_plane3d,
                            double start_uncertainty, double end_uncertainty,
                            double max_si_dist, double min_angle,
                            double* scale_inv_dist);

// Return true -> point3d on the infinite plane3d, false -> otherwise.
bool TestPointOnInfPlane3d(const V3D& point3d_xyz,
                           const InfinitePlane3d& inf_plane3d,
                           double uncertainty,
                           double th_hard_pointplane_si_dist3d,
                           double* scale_inv_dist);

}  // namespace geometry

}  // namespace limap
