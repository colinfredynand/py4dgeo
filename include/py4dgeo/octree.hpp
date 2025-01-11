// #pragma once

// #include <vector>
// #include <string>
// #include <iostream>

// namespace octree {

// class Point3D {
// public:
//     Point3D(double x = 0, double y = 0, double z = 0);
    
//     // Getters
//     double getX() const;
//     double getY() const;
//     double getZ() const;
    
//     // Setters
//     void setX(double val);
//     void setY(double val);
//     void setZ(double val);

// private:
//     double x, y, z;
// };

// class Node {
// public:
//     // Constructor
//     Node(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax);

//     // Destructor
//     ~Node();

//     // Insert a point into the octree
//     void insert(const Point3D& p);

//     // Print the tree structure
//     void print(const std::string& indent = "") const;

// private:
//     double xMin, yMin, zMin, xMax, yMax, zMax; // Bounding box
//     std::vector<Point3D> points;               // Points in this node (leaf node)
//     Node* children[8];                         // Pointers to children (nullptr if leaf)

//     // Helper method to get octant for a point
//     int getOctant(const Point3D& p) const;

//     // Helper method to create child nodes
//     void subdivide();
// };

// class Octree {
// public:
//     // Constructor
//     Octree(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax);

//     // Destructor
//     ~Octree();

//     // Insert a point
//     void insert(const Point3D& point);

//     // Print the tree
//     void print() const;

// private:
//     Node* root;
// };

// } // namespace octree

#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <ostream>
#include <istream>
#include <vector>

#include "py4dgeo.hpp"

namespace py4dgeo {

/** 
 * @brief An Octree data structure for efficient 3D radius searches.
 *
 * This class is designed to have a structure somewhat similar to KDTree,
 * making it straightforward to integrate into the same project.
 */
class Octree
{
public:
  //! Return type used for radius searches (indices only)
  using RadiusSearchResult = std::vector<IndexType>;

  //! Return type used for radius searches (indices + distances)
  using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;

  /** @brief Construct instance of Octree from a given point cloud
   *
   *  This static function constructs an Octree from the provided
   *  row-major eigen point cloud.
   *
   *  @param cloud The point cloud to construct the search tree for
   *               (Eigen::Ref<Eigen::Matrix<double, -1, 3, Eigen::RowMajor>>)
   */
  static Octree
  create(const EigenPointCloudRef& cloud);

  /** @brief Save the index (tree structure) to a (file) stream */
  std::ostream&
  saveIndex(std::ostream& stream) const;

  /** @brief Load the index (tree structure) from a (file) stream */
  std::istream&
  loadIndex(std::istream& stream);

  /** @brief Build the Octree data structure
   *
   *  @param leaf The threshold parameter (capacity) that defines
   *              how many points a node can hold before splitting.
   */
  void
  build_tree(int leaf);

  /** @brief Invalidate the Octree index
   *
   *  This can be called if the underlying point cloud changes or
   *  if you need to rebuild from scratch.
   */
  void
  invalidate();

  /** @brief Perform a radius search around a given query point
   *
   *  Returns all point indices within the specified radius.
   *
   *  @param[in]  querypoint A pointer to the 3D coordinate of the query point
   *  @param[in]  radius     The search radius
   *  @param[out] result     The vector of found point indices (cleared first)
   *
   *  @return The number of points found
   */
  std::size_t
  radius_search(const double* querypoint,
                double radius,
                RadiusSearchResult& result) const;

  /** @brief Perform a radius search around a given query point exporting distance info
   *
   *  Similar to @ref radius_search but also exports distances in ascending order.
   *
   *  @param[in]  querypoint A pointer to the 3D coordinate of the query point
   *  @param[in]  radius     The search radius
   *  @param[out] result     The vector of (index, distance) pairs
   *
   *  @return The number of points found
   */
  std::size_t
  radius_search_with_distances(const double* querypoint,
                               double radius,
                               RadiusSearchDistanceResult& result) const;

private:
  //! Private constructor; use @ref create() to instantiate
  Octree(const EigenPointCloudRef& cloud);

  /** 
   * @brief Forward declaration of the pImpl 
   *
   * We store all implementation details in a private struct so the header
   * remains minimal. 
   */
  struct Impl;

  //! Pointer to the hidden implementation
  std::shared_ptr<Impl> impl;
};

} // namespace py4dgeo
