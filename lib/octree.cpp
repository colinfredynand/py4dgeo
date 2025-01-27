#include "octree.hpp"
#include <algorithm>

namespace py4dgeo {

Octree::Octree(const Eigen::MatrixXd& cloud, std::size_t capacity)
  : capacity_(capacity)
{
  // Convert Eigen matrix to internal point format
  points_.reserve(cloud.rows());
  for (int i = 0; i < cloud.rows(); ++i) {
    const auto& row = cloud.row(i);
    points_.emplace_back(row[0], row[1], row[2]);
  }
}

void Octree::insert(const Eigen::Vector3d& point) {
  points_.emplace_back(point[0], point[1], point[2]);
}

void Octree::build_tree() {
  octree_ = std::make_unique<octree::Octree<Point3D>>(capacity_);
  octree_->build(points_);
}

std::size_t Octree::radius_search(const double* querypoint,
                                  double radius,
                                  RadiusSearchResult& result) const
{
  if (!octree_) return 0;
  
  std::vector<uint32_t> indices;
  octree_->radiusNeighbors<unibn::L2Distance>(
    {static_cast<float>(querypoint[0]), 
     static_cast<float>(querypoint[1]), 
     static_cast<float>(querypoint[2])},
    static_cast<float>(radius),
    indices
  );

  result.resize(indices.size());
  std::transform(indices.begin(), indices.end(), result.begin(),
                [](uint32_t idx) { return static_cast<std::size_t>(idx); });
  return result.size();
}

std::size_t Octree::radius_search_with_distances(const double* querypoint,
                                                 double radius,
                                                 RadiusSearchDistanceResult& result) const
{
  if (!octree_) return 0;

  std::vector<std::pair<uint32_t, float>> neighbors;
  octree_->radiusNeighbors<unibn::L2Distance>(
    {static_cast<float>(querypoint[0]), 
     static_cast<float>(querypoint[1]), 
     static_cast<float>(querypoint[2])},
    static_cast<float>(radius),
    neighbors
  );

  result.clear();
  result.reserve(neighbors.size());
  for (const auto& [idx, dist] : neighbors) {
    result.emplace_back(static_cast<std::size_t>(idx), 
                       static_cast<double>(dist));
  }
  return result.size();
}

} // namespace py4dgeo