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

    // Aligned node structure for better cache performance
    struct alignas(32) Node {
        // Point indices for leaf nodes
        std::vector<IndexType> pointIndices;
        
        // Spatial data aligned for SIMD
        alignas(16) Eigen::Vector3d center;
        double extent;
        
        // Node structure info (packed)
        uint8_t childMask{0};  // Bit mask for active children
        bool isLeaf{true};
        
        // Fixed-size array of children pointers
        std::array<std::unique_ptr<Node>, 8> children;

        Node() = default;
        Node(const Eigen::Vector3d& c, double e) : center(c), extent(e) {}

        // Delete copy, allow move
        Node(const Node&) = delete;
        Node& operator=(const Node&) = delete;
        Node(Node&&) = default;
        Node& operator=(Node&&) = default;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        // Fast Morton code computation
        inline uint32_t getMortonCode(const Eigen::Vector3d& point) const {
            return ((point.x() > center.x()) ? 1 : 0) |
                   ((point.y() > center.y()) ? 2 : 0) |
                   ((point.z() > center.z()) ? 4 : 0);
        }
    };

    // Memory management
    std::unique_ptr<Node> root_;
    std::vector<std::unique_ptr<Node>> nodePool;  // For bulk allocation
    static constexpr size_t POOL_BLOCK_SIZE = 1024;
    size_t currentPoolIndex = 0;

    // Tree building helpers
    Node* allocateNode(const Eigen::Vector3d& center, double extent);
    void buildRecursive(Node* node, const std::vector<IndexType>& indices);
    
    // Search helpers
    bool intersectsSphere(const Node* node, const Eigen::Vector3d& query, double sqRadius) const;
    void processPointBlock(const Eigen::Matrix<double, 4, 3>& points,
                         const Eigen::Vector3d& query,
                         double sqRadius,
                         const IndexType* indices,
                         size_t startIdx,
                         RadiusSearchResult& result) const;

    // Member variables
    EigenPointCloudRef cloud_;
    int leafSize_;

    // Constants for optimization
    static constexpr size_t SIMD_BLOCK_SIZE = 4;  // Process 4 points at once
    static constexpr size_t PREFETCH_DISTANCE = 4;  // Prefetch 4 nodes ahead
};

} // namespace py4dgeo