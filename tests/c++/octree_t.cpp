
// // #include<iostream>
// // #include<py4dgeo/py4dgeo.hpp>
// // int main()
// // {
// //     std::cout<<"Hello world"<<std::endl; 
// // }

// #include "py4dgeo/octree.hpp"
// #include <vector>

// int main() {
//     // Create an octree with bounds from (0,0,0) to (7,7,7)
//     octree::Octree tree(0, 0, 0, 7, 7, 7);
    
//     // Create test points
//     std::vector<octree::Point3D> points = {
//         octree::Point3D(1, 1, 1),
//         octree::Point3D(6, 6, 6),
//         octree::Point3D(2, 5, 3),
//         octree::Point3D(3, 2, 1),
//         octree::Point3D(5, 1, 4),
//         octree::Point3D(7, 7, 7),
//         octree::Point3D(4, 4, 4),
//         octree::Point3D(1, 1, 2)
//     };

//     // Insert points into the octree
//     for (const auto& point : points) {
//         tree.insert(point);
//     }

//     // Print the resulting tree structure
//     tree.print();

//     return 0;
// }

#include <iostream>
#include <array>
#include <Eigen/Eigen>


#include "py4dgeo/octree.hpp"

int main()
{
  // 1) Create a small 5x3 point cloud in row-major layout
  Eigen::Matrix<double, -1, 3, Eigen::RowMajor> data(5,3);
  data << 1.0,  1.0,  1.0,
          2.0,  2.0,  2.0,
          5.0,  5.0,  5.0,
          10.0, 10.0, 10.0,
          11.0, 10.0,  9.0;

  // 2) Create the Octree
  py4dgeo::Octree tree = py4dgeo::Octree::create(data);

  // 3) Build the tree with a leaf (node capacity) of 2
  tree.build_tree(/*leaf=*/2);

  // 4) Perform a radius search for points near (2, 2, 2) with radius = 3
  std::array<double, 3> query = {2.0, 2.0, 2.0};
  double radius = 3.0;

  // 5) RadiusSearchResult (indices only)
  py4dgeo::Octree::RadiusSearchResult indices;
  std::size_t found = tree.radius_search(query.data(), radius, indices);

  std::cout << "[Indices-Only Search]\n";
  std::cout << "Found " << found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto idx : indices)
  {
    std::cout << "  Index: " << idx 
              << "  => Point: [" 
              << data(idx, 0) << ", " 
              << data(idx, 1) << ", " 
              << data(idx, 2) << "]\n";
  }

  // 6) RadiusSearchDistanceResult (indices + distances)
  py4dgeo::Octree::RadiusSearchDistanceResult distResult;
  found = tree.radius_search_with_distances(query.data(), radius, distResult);

  std::cout << "\n[Indices + Distances Search]\n";
  std::cout << "Found " << found << " points within radius " 
            << radius << " of (" << query[0] 
            << ", " << query[1] << ", " << query[2] << ").\n";
  for (auto& p : distResult)
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
