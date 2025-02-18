#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <cmath>
#include <array>
#include <algorithm>

namespace py4dgeo {

class Octree {
public:
    using IndexType = uint32_t;
    using RadiusSearchResult = std::vector<IndexType>;
    using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;
    using EigenPointCloudRef = Eigen::Ref<const Eigen::Matrix<double, -1, 3, Eigen::RowMajor>>;

    struct CachedPoint {
        double x, y, z;
        IndexType idx;
    };

    static Octree create(const EigenPointCloudRef& cloud);
    void build_tree(int leaf);
    std::size_t radius_search(const double* querypoint,
                              double radius,
                              RadiusSearchResult& result) const;
    std::size_t radius_search_with_distances(const double* querypoint,
                                             double radius,
                                             RadiusSearchDistanceResult& result) const;
    void invalidate();

private:
    explicit Octree(const EigenPointCloudRef& cloud);

    // Ultra-compact node structure
    struct alignas(32) Node {
        double x, y, z, extent;  // Center and half edge length.
        uint32_t pointStart;     // Offset into points array (for leaf nodes).
        uint16_t pointCount;     // Number of points in the leaf.
        uint8_t childMask;       // Bitmask indicating which children exist.
        uint8_t isLeaf;          // 1 if leaf, 0 otherwise.
        Node* children[8];       // Raw pointers to children.

        Node(double cx, double cy, double cz, double e)
            : x(cx), y(cy), z(cz), extent(e),
              pointStart(0), pointCount(0),
              childMask(0), isLeaf(1) {
            std::fill(std::begin(children), std::end(children), nullptr);
        }

        // Check if the node's axis-aligned bounding box intersects a sphere.
        inline bool intersectsSphere(double qx, double qy, double qz, double radius) const {
            double dx = std::abs(qx - x);
            double dy = std::abs(qy - y);
            double dz = std::abs(qz - z);
            double maxDist = std::max({dx, dy, dz});
            return maxDist <= extent + radius;
        }

        // Compute a Morton code (0-7) for a point relative to this node.
        inline uint32_t getMortonCode(double px, double py, double pz) const {
            return ((px > x) ? 1 : 0) |
                   ((py > y) ? 2 : 0) |
                   ((pz > z) ? 4 : 0);
        }
    };

    // Memory management
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<CachedPoint> points;  // Cached points for faster access.
    Node* root_{nullptr};

    // Helper functions
    Node* allocateNode(double x, double y, double z, double extent);
    void buildRecursive(Node* node, uint32_t* indices, uint32_t count);

    // Member variables
    EigenPointCloudRef cloud_;
    int leafSize_;

    // Constants
    static constexpr size_t STACK_SIZE = 32;
    static constexpr size_t INITIAL_POINTS = 1024;
    static constexpr size_t INITIAL_NODES = 128;

    // Portable helper: Returns the index of the highest set bit in an 8-bit value.
    static inline int highestSetBit(uint8_t x) {
        for (int i = 7; i >= 0; --i) {
            if (x & (1 << i))
                return i;
        }
        return -1;
    }
};

} // namespace py4dgeo
