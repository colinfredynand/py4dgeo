// #include "octree.hpp"

// namespace octree {

// // Point3D Implementation
// Point3D::Point3D(double x, double y, double z) : x(x), y(y), z(z) {}

// double Point3D::getX() const { return x; }
// double Point3D::getY() const { return y; }
// double Point3D::getZ() const { return z; }

// void Point3D::setX(double val) { x = val; }
// void Point3D::setY(double val) { y = val; }
// void Point3D::setZ(double val) { z = val; }

// // Node Implementation
// Node::Node(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax) 
//     : xMin(xmin), yMin(ymin), zMin(zmin), xMax(xmax), yMax(ymax), zMax(zmax) {
//     for (int i = 0; i < 8; ++i) children[i] = nullptr;
// }

// Node::~Node() {
//     for (int i = 0; i < 8; ++i) {
//         delete children[i];
//     }
// }

// int Node::getOctant(const Point3D& p) const {
//     double midX = (xMin + xMax) / 2;
//     double midY = (yMin + yMax) / 2;
//     double midZ = (zMin + zMax) / 2;
//     int octant = 0;
//     if (p.getX() >= midX) octant |= 4;
//     if (p.getY() >= midY) octant |= 2;
//     if (p.getZ() >= midZ) octant |= 1;
//     return octant;
// }

// void Node::subdivide() {
//     double midX = (xMin + xMax) / 2;
//     double midY = (yMin + yMax) / 2;
//     double midZ = (zMin + zMax) / 2;
    
//     children[0] = new Node(xMin, yMin, zMin, midX, midY, midZ);
//     children[1] = new Node(xMin, yMin, midZ, midX, midY, zMax);
//     children[2] = new Node(xMin, midY, zMin, midX, yMax, midZ);
//     children[3] = new Node(xMin, midY, midZ, midX, yMax, zMax);
//     children[4] = new Node(midX, yMin, zMin, xMax, midY, midZ);
//     children[5] = new Node(midX, yMin, midZ, xMax, midY, zMax);
//     children[6] = new Node(midX, midY, zMin, xMax, yMax, midZ);
//     children[7] = new Node(midX, midY, midZ, xMax, yMax, zMax);

//     // Redistribute existing points
//     for (const auto& point : points) {
//         children[getOctant(point)]->points.push_back(point);
//     }
//     points.clear();
// }

// void Node::insert(const Point3D& p) {
//     if (points.size() < 2) { // Threshold can be adjusted
//         points.push_back(p);
//     } else {
//         if (children[0] == nullptr) {
//             subdivide();
//         }
//         children[getOctant(p)]->insert(p);
//     }
// }

// void Node::print(const std::string& indent) const {
//     std::cout << indent << "[" << xMin << ", " << yMin << ", " << zMin << "] - ["
//              << xMax << ", " << yMax << ", " << zMax << "]: ";
             
//     if (children[0] == nullptr) {
//         std::cout << "{";
//         for (size_t i = 0; i < points.size(); ++i) {
//             std::cout << "(" << points[i].getX() << ", " << points[i].getY() << ", " 
//                      << points[i].getZ() << ")";
//             if (i < points.size() - 1) std::cout << ", ";
//         }
//         std::cout << "}" << std::endl;
//     } else {
//         std::cout << std::endl;
//         for (int i = 0; i < 8; ++i) {
//             children[i]->print(indent + "  ");
//         }
//     }
// }

// // Octree Implementation
// Octree::Octree(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax) {
//     root = new Node(xmin, ymin, zmin, xmax, ymax, zmax);
// }

// Octree::~Octree() {
//     delete root;
// }

// void Octree::insert(const Point3D& point) {
//     root->insert(point);
// }

// void Octree::print() const {
//     std::cout << "Octree Structure:" << std::endl;
//     root->print();
// }

// } // namespace octree

#include "py4dgeo/octree.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>  // for std::iota

namespace py4dgeo {

struct BoundingBox
{
  Eigen::Vector3d minCorner;
  Eigen::Vector3d maxCorner;

  BoundingBox()
    : minCorner(Eigen::Vector3d::Zero())
    , maxCorner(Eigen::Vector3d::Zero())
  {}

  BoundingBox(const Eigen::Vector3d& minC, const Eigen::Vector3d& maxC)
    : minCorner(minC), maxCorner(maxC)
  {}

  bool contains(const Eigen::Vector3d& pt) const
  {
    return (pt.x() >= minCorner.x() && pt.x() <= maxCorner.x() &&
            pt.y() >= minCorner.y() && pt.y() <= maxCorner.y() &&
            pt.z() >= minCorner.z() && pt.z() <= maxCorner.z());
  }

  // Check if bounding box intersects a sphere of radius r around center c
  bool intersectsSphere(const Eigen::Vector3d& c, double r) const
  {
    double distSq = 0.0;
    for (int i = 0; i < 3; ++i)
    {
      if (c[i] < minCorner[i])
      {
        double d = minCorner[i] - c[i];
        distSq += d * d;
      }
      else if (c[i] > maxCorner[i])
      {
        double d = c[i] - maxCorner[i];
        distSq += d * d;
      }
    }
    return distSq <= r * r;
  }
};

/**
 * @brief Implementation struct for Octree   (change Impl to Node)
 */
struct Octree::Impl
{
  // The original point cloud (row-major Nx3)
  EigenPointCloudRef cloud;

  // Indices of the points belonging to this node
  std::vector<IndexType> pointIndices;

  // Children of this node
  std::unique_ptr<Impl> children[8] = {nullptr};

  // Bounding volume of this node
  BoundingBox bbox;

  // Node capacity (leaf threshold)
  std::size_t capacity;

  // Whether we have subdivided this node
  bool subdivided = false;

  
  std::size_t depth = 0;

  // --------------------------------------------------------------------------
  // Constructor
  // --------------------------------------------------------------------------
  Impl(const EigenPointCloudRef& c, std::size_t cap = 10)
    : cloud(c), capacity(cap)
  {}

  // --------------------------------------------------------------------------
  // Subdivide into 8 children
  // --------------------------------------------------------------------------
  void subdivide()
  {
    if (subdivided)
      return;

    Eigen::Vector3d center = 0.5 * (bbox.minCorner + bbox.maxCorner);
    auto& minC = bbox.minCorner;
    auto& maxC = bbox.maxCorner;

    // Precompute mid
    const double midX = center.x();
    const double midY = center.y();
    const double midZ = center.z();

    // 8 bounding boxes
    BoundingBox boxes[8] = {
      // (1)
      BoundingBox({minC.x(), minC.y(), minC.z()},
                  {midX,     midY,     midZ}),
      // (2)
      BoundingBox({midX,     minC.y(), minC.z()},
                  {maxC.x(), midY,     midZ}),
      // (3)
      BoundingBox({minC.x(), midY,     minC.z()},
                  {midX,     maxC.y(), midZ}),
      // (4)
      BoundingBox({midX,     midY,     minC.z()},
                  {maxC.x(), maxC.y(), midZ}),
      // (5)
      BoundingBox({minC.x(), minC.y(), midZ},
                  {midX,     midY,     maxC.z()}),
      // (6)
      BoundingBox({midX,     minC.y(), midZ},
                  {maxC.x(), midY,     maxC.z()}),
      // (7)
      BoundingBox({minC.x(), midY,     midZ},
                  {midX,     maxC.y(), maxC.z()}),
      // (8)
      BoundingBox({midX,     midY,     midZ},
                  {maxC.x(), maxC.y(), maxC.z()})
    };

    // Create children
    for (int i = 0; i < 8; ++i)
    {
      children[i] = std::make_unique<Impl>(cloud, capacity);
      children[i]->bbox = boxes[i];
      children[i]->depth = depth + 1;
    }

    // Distribute existing points into children
    for (auto idx : pointIndices)
    {
      Eigen::Vector3d pt = cloud.row(idx);
      for (int c = 0; c < 8; ++c)
      {
        if (children[c]->bbox.contains(pt))
        {
          children[c]->pointIndices.push_back(idx);
          break;
        }
      }
    }

    // Clear points from the parent node
    pointIndices.clear();
    subdivided = true;
  }

  // --------------------------------------------------------------------------
  // Insert a single index into the (sub)tree
  // --------------------------------------------------------------------------
  void insert(IndexType idx)
  {
    Eigen::Vector3d pt = cloud.row(idx);

    // If not contained in bounding box, ignore
    if (!bbox.contains(pt))
      return;

    // If we have capacity or no further subdivision desired
    if (pointIndices.size() < capacity)
    {
      pointIndices.push_back(idx);
      return;
    }

    // Otherwise, subdivide if not already
    if (!subdivided)
      subdivide();

    // Attempt to pass the point to the correct child
    for (int c = 0; c < 8; ++c)
    {
      if (children[c]->bbox.contains(pt))
      {
        children[c]->insert(idx);
        return;
      }
    }
  }

  // --------------------------------------------------------------------------
  // Recursive radius search (indices only)
  // --------------------------------------------------------------------------
  void radiusSearch(const Eigen::Vector3d& center,
                    double radius,
                    Octree::RadiusSearchResult& indices) const
  {
    // If bounding box doesn't intersect the search sphere, skip,
    if (!bbox.intersectsSphere(center, radius))
      return;

    // Check all points in this node
    double r2 = radius * radius;
    for (auto idx : pointIndices)
    {
      Eigen::Vector3d pt = cloud.row(idx);
      double dist2 = (pt - center).squaredNorm();
      if (dist2 <= r2)
        indices.push_back(idx);
    }

    // If subdivided, recurse
    if (subdivided)
    {
      for (int c = 0; c < 8; ++c)
        children[c]->radiusSearch(center, radius, indices);
    }
  }

  // --------------------------------------------------------------------------
  // Recursive radius search (indices + distances)
  // --------------------------------------------------------------------------
  void radiusSearchWithDistances(const Eigen::Vector3d& center,
                                 double radius,
                                 Octree::RadiusSearchDistanceResult& results) const
  {
    if (!bbox.intersectsSphere(center, radius))
      return;

    double r2 = radius * radius;
    for (auto idx : pointIndices)
    {
      Eigen::Vector3d pt = cloud.row(idx);
      double dist2 = (pt - center).squaredNorm();
      if (dist2 <= r2)
      {
        double dist = std::sqrt(dist2);
        results.push_back({ idx, dist });
      }
    }

    if (subdivided)
    {
      for (int c = 0; c < 8; ++c)
        children[c]->radiusSearchWithDistances(center, radius, results);
    }
  }
};

// --------------------------------------------------------------------------
// Octree constructor (private)
// --------------------------------------------------------------------------
Octree::Octree(const EigenPointCloudRef& cloud)
  : impl(std::make_shared<Impl>(cloud))
{}

// --------------------------------------------------------------------------
// Static create function
// --------------------------------------------------------------------------
Octree
Octree::create(const EigenPointCloudRef& cloud)
{
  // Just call the private constructor
  return Octree(cloud);
}

// --------------------------------------------------------------------------
// build_tree
// --------------------------------------------------------------------------
void
Octree::build_tree(int leaf)
{
  // interpret 'leaf' as capacity
  impl->capacity = static_cast<std::size_t>(leaf);

  // Compute bounding box over entire cloud
  Eigen::Vector3d minC = impl->cloud.colwise().minCoeff();
  Eigen::Vector3d maxC = impl->cloud.colwise().maxCoeff();
  impl->bbox = BoundingBox(minC, maxC);

  // Fill in initial indices
  std::size_t n = static_cast<std::size_t>(impl->cloud.rows());
  impl->pointIndices.resize(n);
  std::iota(impl->pointIndices.begin(), impl->pointIndices.end(), 0);

  // Subdivide if needed
  if (n > impl->capacity)
    impl->subdivide();
}

// --------------------------------------------------------------------------
// invalidate
// --------------------------------------------------------------------------
void
Octree::invalidate()
{
  impl->pointIndices.clear();
  for (auto& child : impl->children)
    child.reset(nullptr);
  impl->subdivided = false;
  impl->depth = 0;
  // We keep the bounding box, but you can reset it if you wish.
}

// --------------------------------------------------------------------------
// radius_search
// --------------------------------------------------------------------------
std::size_t
Octree::radius_search(const double* querypoint,
                      double radius,
                      RadiusSearchResult& result) const
{
  result.clear();
  Eigen::Vector3d center(querypoint[0], querypoint[1], querypoint[2]);
  impl->radiusSearch(center, radius, result);
  return result.size();
}

// --------------------------------------------------------------------------
// radius_search_with_distances
// --------------------------------------------------------------------------
std::size_t
Octree::radius_search_with_distances(const double* querypoint,
                                     double radius,
                                     RadiusSearchDistanceResult& result) const
{
  result.clear();
  Eigen::Vector3d center(querypoint[0], querypoint[1], querypoint[2]);
  impl->radiusSearchWithDistances(center, radius, result);

  // Sort in ascending order by distance
  std::sort(result.begin(), result.end(),
            [](auto& a, auto& b){ return a.second < b.second; });
  return result.size();
}

// --------------------------------------------------------------------------
// saveIndex
// --------------------------------------------------------------------------
std::ostream&
Octree::saveIndex(std::ostream& stream) const
{
  // Minimal example: store bounding box, capacity, and root’s pointIndices
  // Real implementation should recursively handle children as well.

  // Store bounding box
  stream.write(reinterpret_cast<const char*>(&impl->bbox.minCorner[0]), sizeof(double)*3);
  stream.write(reinterpret_cast<const char*>(&impl->bbox.maxCorner[0]), sizeof(double)*3);

  // Store capacity
  std::size_t cap = impl->capacity;
  stream.write(reinterpret_cast<const char*>(&cap), sizeof(std::size_t));

  // Store root’s point indices
  std::size_t n = impl->pointIndices.size();
  stream.write(reinterpret_cast<const char*>(&n), sizeof(std::size_t));
  stream.write(reinterpret_cast<const char*>(impl->pointIndices.data()), sizeof(IndexType)*n);

  return stream;
}

// --------------------------------------------------------------------------
// loadIndex
// --------------------------------------------------------------------------
std::istream&
Octree::loadIndex(std::istream& stream)
{
  // Read bounding box
  stream.read(reinterpret_cast<char*>(&impl->bbox.minCorner[0]), sizeof(double)*3);
  stream.read(reinterpret_cast<char*>(&impl->bbox.maxCorner[0]), sizeof(double)*3);

  // Read capacity
  std::size_t cap;
  stream.read(reinterpret_cast<char*>(&cap), sizeof(std::size_t));
  impl->capacity = cap;

  // Read root’s point indices
  std::size_t n;
  stream.read(reinterpret_cast<char*>(&n), sizeof(std::size_t));
  impl->pointIndices.resize(n);
  stream.read(reinterpret_cast<char*>(impl->pointIndices.data()), sizeof(IndexType)*n);

  // Real solution: also read subdivided state + children if needed.

  return stream;
}

} // namespace py4dgeo
