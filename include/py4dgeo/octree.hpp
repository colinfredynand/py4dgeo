#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <octree/octree.h> // Requires jbehley/octree dependency

namespace py4dgeo {

class Octree
{
public:
  using RadiusSearchResult = std::vector<std::size_t>;
  using RadiusSearchDistanceResult = std::vector<std::pair<std::size_t, double>>;

  Octree(const Eigen::MatrixXd& cloud, std::size_t capacity = 10);
  void insert(const Eigen::Vector3d& point);
  void build_tree();

  std::size_t radius_search(const double* querypoint,
                            double radius,
                            RadiusSearchResult& result) const;

  std::size_t radius_search_with_distances(const double* querypoint,
                                           double radius,
                                           RadiusSearchDistanceResult& result) const;

private:
  struct Point3D {
    float x, y, z;
    Point3D(float x, float y, float z) : x(x), y(y), z(z) {}
  };

  std::vector<Point3D> points_;
  std::size_t capacity_;
  std::unique_ptr<octree::Octree<Point3D>> octree_;
};

} // namespace py4dgeo