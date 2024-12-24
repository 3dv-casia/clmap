#include "clmap/plane_detection/robust_statistics_approach.h"

#include <fstream>
#include <iostream>

#include "PlaneDetection/src/boundaryvolumehierarchy.h"
#include "PlaneDetection/src/connectivitygraph.h"
#include "PlaneDetection/src/normalestimator.h"
#include "PlaneDetection/src/planedetector.h"
#include "PlaneDetection/src/pointcloudio.hpp"

namespace limap {

namespace plane_detection {

void RobustStatisticsApproach(const std::string& inputFileName,
                              const std::string& outputFileName,
                              float minNormalDiff, float maxDist,
                              float outlierRatio, size_t normalsNeighborSize) {
  std::cout << "---------------------------------------------------"
            << std::endl;
  std::cout << "Start detecting planes..." << std::endl;
  std::cout << "Args:" << std::endl;
  std::cout << "    inputFileName: " << inputFileName << std::endl;
  std::cout << "    outputFileName: " << outputFileName << std::endl;
  std::cout << "    minNormalDiff: " << minNormalDiff << std::endl;
  std::cout << "    maxDist: " << maxDist << std::endl;
  std::cout << "    outlierRatio: " << outlierRatio << std::endl;
  std::cout << "    normalsNeighborSize: " << normalsNeighborSize << std::endl;
  std::cout << "---------------------------------------------------"
            << std::endl;

  std::cout << "Reading the point cloud..." << std::endl;
  PointCloudIO pointCloudIO;
  // many formats are supported from this class, but XYZ as chosen for being
  // popular and simple
  PointCloud3d* pointCloud = pointCloudIO.loadFromXYZ(inputFileName);
  std::cout << "Number of original 3D points: " << pointCloud->size()
            << std::endl;

  // you can skip the normal estimation if you point cloud already have normals
  std::cout << "Estimating normals..." << std::endl;
  // size_t normalsNeighborSize = 30;
  Octree octree(pointCloud);
  octree.partition(10, 30);
  ConnectivityGraph* connectivity = new ConnectivityGraph(pointCloud->size());
  pointCloud->connectivity(connectivity);
  NormalEstimator3d estimator(&octree, normalsNeighborSize,
                              NormalEstimator3d::QUICK);
  std::cout << pointCloud->size() << std::endl;
  for (size_t i = 0; i < pointCloud->size(); i++) {
    if (i % 10000 == 0) {
      std::cout << i / float(pointCloud->size()) * 100 << "%..." << std::endl;
    }
    NormalEstimator3d::Normal normal = estimator.estimate(i);
    connectivity->addNode(i, normal.neighbors);
    (*pointCloud)[i].normal(normal.normal);
    (*pointCloud)[i].normalConfidence(normal.confidence);
    (*pointCloud)[i].curvature(normal.curvature);
  }

  std::cout << "Detecting planes..." << std::endl;
  PlaneDetector detector(pointCloud);
  detector.minNormalDiff(minNormalDiff);
  detector.maxDist(maxDist);
  detector.outlierRatio(outlierRatio);
  // detector.minNormalDiff(0.5f);
  // detector.maxDist(0.258819f);
  // detector.outlierRatio(0.75f);

  std::set<Plane*> planes = detector.detect();
  std::cout << planes.size() << " planes detected." << std::endl;

  std::cout << "Saving results..." << std::endl;
  Geometry* geometry = pointCloud->geometry();
  for (Plane* plane : planes) {
    geometry->addPlane(plane);
  }
  // many output formats are allowed. if you want to run our
  // 'compare_plane_detector', uncomment the line below and comment the rest
  // pointCloudIO.saveGeometry(geometry, outputFileName);
  std::ofstream outputFile(outputFileName);

  outputFile << "# Number of 3D planes: " << planes.size() << std::endl;
  outputFile << "# Number of 3D points: " << pointCloud->size() << std::endl;
  size_t n_inlier = 0;
  for (Plane* plane : planes) {
    n_inlier += plane->inliers().size();
  }
  outputFile << "# Number of inlier 3D points: " << n_inlier << std::endl;
  outputFile << "# Inlier rate: "
             << (double)n_inlier / (double)pointCloud->size() << std::endl;
  outputFile << "# Data in each line: Plane_idx Normal_X Normal_Y Normal_Z "
                "Center_X Center_Y Center_Z V1_X V1_Y V1_Z V2_X V2_Y V2_Z V3_X "
                "V3_Y V3_Z V4_X V4_Y V4_Z Inlier_ID1 Inlier_ID2 Inlier_ID3 ..."
             << std::endl;
  size_t plane_idx = 0;
  for (Plane* plane : planes) {
    Eigen::Vector3f v1 = plane->center() + plane->basisU() + plane->basisV();
    Eigen::Vector3f v2 = plane->center() + plane->basisU() - plane->basisV();
    Eigen::Vector3f v3 = plane->center() - plane->basisU() + plane->basisV();
    Eigen::Vector3f v4 = plane->center() - plane->basisU() - plane->basisV();
    // outputFile << "Normal: [" << plane->normal()[0] << ", " <<
    // plane->normal()[1] << ", " << plane->normal()[2] << "]; "
    //            << "Center: [" << plane->center()[0] << ", " <<
    //            plane->center()[1] << ", " << plane->center()[2] << "]; "
    //            << "Vertices: [[" << v1.x() << "," << v1.y() << "," << v1.z()
    //            << "], "
    //            << "[" << v2.x() << "," << v2.y() << "," << v2.z() << "], "
    //            << "[" << v3.x() << "," << v3.y() << "," << v3.z() << "], "
    //            << "[" << v4.x() << "," << v4.y() << "," << v4.z() << "]]" <<
    //            std::endl;
    outputFile << plane_idx++ << " " << plane->normal()[0] << " "
               << plane->normal()[1] << " " << plane->normal()[2] << " "
               << plane->center()[0] << " " << plane->center()[1] << " "
               << plane->center()[2] << " " << v1.x() << " " << v1.y() << " "
               << v1.z() << " " << v2.x() << " " << v2.y() << " " << v2.z()
               << " " << v3.x() << " " << v3.y() << " " << v3.z() << " "
               << v4.x() << " " << v4.y() << " " << v4.z();

    // `inliers` contains indices of the input 3D points stored in
    // `PointCloud3d::mPoints`
    const std::vector<size_t>& inliers = plane->inliers();
    for (const size_t& inlier_idx : inliers) {
      outputFile << " " << inlier_idx;
    }
    outputFile << std::endl;
  }
  delete pointCloud;

  std::cout << "Plane detection completed. " << std::endl;
}

}  // namespace plane_detection

}  // namespace limap
