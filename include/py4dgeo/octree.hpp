#pragma once

#include <Eigen/Eigen>
#include <memory>
#include <vector>
#include <cmath>
#include <array>

namespace py4dgeo {

class Octree {
public:
    using IndexType = uint32_t;
    using RadiusSearchResult = std::vector<IndexType>;
    using RadiusSearchDistanceResult = std::vector<std::pair<IndexType, double>>;
    using EigenPointCloudRef = Eigen::Ref<const Eigen::Matrix<double, -1, 3, Eigen::RowMajor>>;

    static Octree create(const EigenPointCloudRef& cloud);
    void build_tree(int leaf);
    std::size_t radius_search(const double* querypoint,
                             double radius,
                             RadiusSearchResult& result) const;
    std::size_t radius_search_with_distances(const double* querypoint,
                                           double radius,
                                           RadiusSearchDistanceResult& result) const;
    void invalidate();
    std::ostream& saveIndex(std::ostream& stream) const;
    std::istream& loadIndex(std::istream& stream);

private:
    explicit Octree(const EigenPointCloudRef& cloud);

    // Cache-optimized node structure
    struct alignas(32) Node {
        // Packed spatial data for better cache line usage
        struct alignas(16) SpatialData {
            double x, y, z;  // Center coordinates
            double extent;
            uint32_t start;  // Start index in points array
            uint32_t size;   // Number of points
            uint8_t childMask;
            bool isLeaf;
            uint16_t padding;  // Maintain alignment
        } spatial;

        // Separate child pointers to avoid false sharing
        alignas(32) std::array<Node*, 8> children{};

        Node() = default;
        Node(const Eigen::Vector3d& c, double e) {
            spatial.x = c.x();
            spatial.y = c.y();
            spatial.z = c.z();
            spatial.extent = e;
            spatial.start = 0;
            spatial.size = 0;
            spatial.childMask = 0;
            spatial.isLeaf = true;
        }

        // Fast Morton code computation
        inline uint32_t getMortonCode(const Eigen::Vector3d& point) const {
            return ((point.x() > spatial.x) ? 1 : 0) |
                   ((point.y() > spatial.y) ? 2 : 0) |
                   ((point.z() > spatial.z) ? 4 : 0);
        }

        // Fast distance check
        inline bool intersectsSphere(const Eigen::Vector3d& query, double radius) const {
            double dx = std::abs(query.x() - spatial.x);
            double dy = std::abs(query.y() - spatial.y);
            double dz = std::abs(query.z() - spatial.z);
            return std::max({dx, dy, dz}) <= spatial.extent + radius;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    };

    // Memory and data management
    std::vector<std::unique_ptr<Node>> nodes;
    std::vector<IndexType> pointIndices;  // Contiguous storage for all point indices
    Node* root_{nullptr};

    // Tree building helpers
    Node* allocateNode(const Eigen::Vector3d& center, double extent);
    void buildRecursive(Node* node, IndexType* indices, size_t size);

    // Member variables
    EigenPointCloudRef cloud_;
    int leafSize_;

    // Constants
    static constexpr size_t STACK_SIZE = 32;
    static constexpr size_t INITIAL_NODE_CAPACITY = 1024;
};

} // namespace py4dgeo