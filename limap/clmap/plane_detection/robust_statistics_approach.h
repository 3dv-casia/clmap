#pragma once

#include <string>

namespace limap {

namespace plane_detection {

// Abner M. C. Araújo et al. A robust statistics approach for plane detection in
// unorganized point clouds. Pattern Recognition. 2020.
//
// The parameters are the default values in the open-source code
// https://github.com/abner-math/PlaneDetection/blob/master/CommandLine/main.cpp
void RobustStatisticsApproach(const std::string& inputFileName,
                              const std::string& outputFileName,
                              float minNormalDiff = 0.5,
                              float maxDist = 0.258819,
                              float outlierRatio = 0.75,
                              size_t normalsNeighborSize = 30);

}  // namespace plane_detection

}  // namespace limap
