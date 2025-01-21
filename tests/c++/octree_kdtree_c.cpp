#include <iostream>
#include <array>
#include <Eigen/Eigen>

#include "py4dgeo/octree.hpp"
#include "py4dgeo/kdtree.hpp"

int main()
{
  // 1) Create a small 5x3 point cloud in row-major layout
  Eigen::Matrix<double, -1, 3, Eigen::RowMajor> data(5,3);
  data << 1.0,  1.0,  1.0,
          2.0,  2.0,  2.0,
          5.0,  5.0,  5.0,
          10.0, 10.0, 10.0,
          11.0, 10.0,  9.0;

  std::cout << "Testing both Octree and KDTree implementations\n\n";
  
  // Query parameters that will be used for both trees
  std::array<double, 3> query = {2.0, 2.0, 2.0};
  double radius = 3.0;

  // ============ Octree Testing ============
  std::cout << "=== Octree Tests ===\n";
  
  // Create and build the Octree
  py4dgeo::Octree octree = py4dgeo::Octree::create(data);
  octree.build_tree(/*leaf=*/2);

  // Test indices-only search with Octree
  py4dgeo::Octree::RadiusSearchResult octree_indices;
  std::size_t octree_found = octree.radius_search(query.data(), radius, octree_indices);

  std::cout << "[Octree Indices-Only Search]\n";
  std::cout << "Found " << octree_found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto idx : octree_indices)
  {
    std::cout << "  Index: " << idx 
              << "  => Point: [" 
              << data(idx, 0) << ", " 
              << data(idx, 1) << ", " 
              << data(idx, 2) << "]\n";
  }

  // Test search with distances for Octree
  py4dgeo::Octree::RadiusSearchDistanceResult octree_dist_result;
  octree_found = octree.radius_search_with_distances(query.data(), radius, octree_dist_result);

  std::cout << "\n[Octree Indices + Distances Search]\n";
  std::cout << "Found " << octree_found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto& p : octree_dist_result)
  {
    auto idx = p.first;
    auto dist = p.second;
    std::cout << "  Index: " << idx 
              << "  Dist: " << dist
              << "  => Point: [" 
              << data(idx, 0) << ", " 
              << data(idx, 1) << ", " 
              << data(idx, 2) << "]\n";
  }

  // ============ KDTree Testing ============
  std::cout << "\n=== KDTree Tests ===\n";
  
  // Create and build the KDTree
  py4dgeo::KDTree kdtree = py4dgeo::KDTree::create(data);
  kdtree.build_tree(10);  // Using leaf parameter of 10 as seen in kdtree_t.cpp

  // Test indices-only search with KDTree
  py4dgeo::KDTree::RadiusSearchResult kdtree_indices;
  std::size_t kdtree_found = kdtree.radius_search(query.data(), radius, kdtree_indices);

  std::cout << "[KDTree Indices-Only Search]\n";
  std::cout << "Found " << kdtree_found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto idx : kdtree_indices)
  {
    std::cout << "  Index: " << idx 
              << "  => Point: [" 
              << data(idx, 0) << ", " 
              << data(idx, 1) << ", " 
              << data(idx, 2) << "]\n";
  }

  // Test search with distances for KDTree
  py4dgeo::KDTree::RadiusSearchDistanceResult kdtree_dist_result;
  kdtree_found = kdtree.radius_search_with_distances(query.data(), radius, kdtree_dist_result);

  std::cout << "\n[KDTree Indices + Distances Search]\n";
  std::cout << "Found " << kdtree_found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto& p : kdtree_dist_result)
  {
    auto idx = p.first;
    auto dist = p.second;
    std::cout << "  Index: " << idx 
              << "  Dist: " << dist
              << "  => Point: [" 
              << data(idx, 0) << ", " 
              << data(idx, 1) << ", " 
              << data(idx, 2) << "]\n";
  }

  return 0;
}