#include "py4dgeo/octree.hpp"
#include <algorithm>
#include <numeric>
#include <stack>

namespace py4dgeo {

Octree::Octree(const EigenPointCloudRef& cloud)
    : cloud_(cloud), leafSize_(32) {
    // Pre-allocate node pool
    nodePool.reserve(POOL_BLOCK_SIZE);
}

Octree Octree::create(const EigenPointCloudRef& cloud) {
    return Octree(cloud);
}

Octree::Node* Octree::allocateNode(const Eigen::Vector3d& center, double extent) {
    if (currentPoolIndex >= nodePool.size()) {
        nodePool.push_back(std::make_unique<Node>(center, extent));
        return nodePool.back().get();
    }
    
    auto& node = nodePool[currentPoolIndex++];
    node = std::make_unique<Node>(center, extent);
    return node.get();
}

void Octree::build_tree(int leaf) {
    leafSize_ = leaf;
    
    // Reset state
    nodePool.clear();
    currentPoolIndex = 0;
    root_.reset();
    
    if (cloud_.rows() == 0) {
        return;
    }
    
    // Use SIMD for bounding box computation
    Eigen::Vector3d min = cloud_.colwise().minCoeff();
    Eigen::Vector3d max = cloud_.colwise().maxCoeff();
    
    Eigen::Vector3d center = (min + max) * 0.5;
    double extent = (max - min).maxCoeff() * 0.5;
    
    if (extent == 0.0) {
        extent = 1.0;
    }
    
    // Pre-allocate for entire tree
    size_t estimatedNodes = std::min(
        size_t(cloud_.rows() / leafSize_ * 1.5),  // Estimate nodes needed
        POOL_BLOCK_SIZE  // Cap at block size
    );
    nodePool.reserve(estimatedNodes);
    
    // Initialize root
    root_ = std::make_unique<Node>(center, extent);
    
    // Create initial indices
    std::vector<IndexType> indices(cloud_.rows());
    std::iota(indices.begin(), indices.end(), 0);
    
    buildRecursive(root_.get(), indices);
}

void Octree::buildRecursive(Node* node, const std::vector<IndexType>& indices) {
    if (indices.empty()) {
        return;
    }

    if (indices.size() <= static_cast<size_t>(leafSize_)) {
        node->isLeaf = true;
        node->pointIndices = indices;
        return;
    }
    
    node->isLeaf = false;
    
    // Pre-allocate child vectors
    std::array<std::vector<IndexType>, 8> childIndices;
    const size_t estimatedSize = indices.size() / 4;
    for (auto& vec : childIndices) {
        vec.reserve(estimatedSize);
    }
    
    // Process points in SIMD blocks
    constexpr size_t BLOCK_SIZE = SIMD_BLOCK_SIZE;
    const size_t numBlocks = indices.size() / BLOCK_SIZE;
    const size_t remainder = indices.size() % BLOCK_SIZE;
    
    Eigen::Matrix<double, BLOCK_SIZE, 3> points;
    
    for (size_t b = 0; b < numBlocks; ++b) {
        // Load block of points
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            points.row(i) = cloud_.row(indices[b * BLOCK_SIZE + i]);
        }
        
        // Compute Morton codes for the block
        for (size_t i = 0; i < BLOCK_SIZE; ++i) {
            uint32_t code = node->getMortonCode(points.row(i));
            childIndices[code].push_back(indices[b * BLOCK_SIZE + i]);
        }
    }
    
    // Handle remaining points
    for (size_t i = 0; i < remainder; ++i) {
        IndexType idx = indices[numBlocks * BLOCK_SIZE + i];
        uint32_t code = node->getMortonCode(cloud_.row(idx));
        childIndices[code].push_back(idx);
    }
    
    // Create children
    double childExtent = node->extent * 0.5;
    for (int i = 0; i < 8; ++i) {
        if (childIndices[i].empty()) continue;
        
        Eigen::Vector3d childCenter = node->center;
        childCenter[0] += ((i & 1) ? childExtent : -childExtent);
        childCenter[1] += ((i & 2) ? childExtent : -childExtent);
        childCenter[2] += ((i & 4) ? childExtent : -childExtent);
        
        node->children[i] = std::make_unique<Node>(childCenter, childExtent);
        node->childMask |= (1 << i);
        buildRecursive(node->children[i].get(), childIndices[i]);
    }
}

bool Octree::intersectsSphere(const Node* node, const Eigen::Vector3d& query, double sqRadius) const {
    // SIMD-optimized distance check
    Eigen::Vector3d diff = (query - node->center).cwiseAbs();
    double maxDist = diff.maxCoeff();
    
    return maxDist <= node->extent + std::sqrt(sqRadius);
}

void Octree::processPointBlock(const Eigen::Matrix<double, 4, 3>& points,
                             const Eigen::Vector3d& query,
                             double sqRadius,
                             const IndexType* indices,
                             size_t startIdx,
                             RadiusSearchResult& result) const {
    Eigen::Vector4d distances = (points.rowwise() - query.transpose()).rowwise().squaredNorm();
    
    for (size_t i = 0; i < SIMD_BLOCK_SIZE; ++i) {
        if (distances[i] <= sqRadius) {
            result.push_back(indices[startIdx + i]);
        }
    }
}

std::size_t Octree::radius_search(const double* querypoint,
                                 double radius,
                                 RadiusSearchResult& result) const {
    result.clear();
    
    if (!root_ || radius < 0.0) {
        return 0;
    }

    const Eigen::Vector3d query = Eigen::Map<const Eigen::Vector3d>(querypoint);
    const double sqRadius = radius * radius;

    // Use custom stack for better performance
    std::vector<const Node*> stack;
    stack.reserve(32);  // Pre-allocate for typical tree depth
    stack.push_back(root_.get());

    while (!stack.empty()) {
        const Node* node = stack.back();
        stack.pop_back();

        if (!intersectsSphere(node, query, sqRadius)) {
            continue;
        }

        if (node->isLeaf) {
            const auto& indices = node->pointIndices;
            const size_t numPoints = indices.size();
            const size_t numBlocks = numPoints / SIMD_BLOCK_SIZE;
            const size_t remainder = numPoints % SIMD_BLOCK_SIZE;

            // Process points in SIMD blocks
            Eigen::Matrix<double, 4, 3> points;
            for (size_t b = 0; b < numBlocks; ++b) {
                const size_t startIdx = b * SIMD_BLOCK_SIZE;
                for (size_t i = 0; i < SIMD_BLOCK_SIZE; ++i) {
                    points.row(i) = cloud_.row(indices[startIdx + i]);
                }
                processPointBlock(points, query, sqRadius, indices.data(), startIdx, result);
            }

            // Handle remaining points
            for (size_t i = 0; i < remainder; ++i) {
                const IndexType idx = indices[numBlocks * SIMD_BLOCK_SIZE + i];
                if ((cloud_.row(idx).transpose() - query).squaredNorm() <= sqRadius) {
                    result.push_back(idx);
                }
            }
        } else {
            // Add children in reverse order (better cache locality)
            uint8_t mask = node->childMask;
            while (mask) {
                int idx = 31 - __builtin_clz(mask);  // Fast bit scan
                if (node->children[idx]) {
                    stack.push_back(node->children[idx].get());
                }
                mask &= ~(1 << idx);
            }
        }
    }

    return result.size();
}

std::size_t Octree::radius_search_with_distances(const double* querypoint,
                                                double radius,
                                                RadiusSearchDistanceResult& result) const {
    result.clear();
    
    if (!root_ || radius < 0.0) {
        return 0;
    }

    const Eigen::Vector3d query = Eigen::Map<const Eigen::Vector3d>(querypoint);
    const double sqRadius = radius * radius;

    std::vector<const Node*> stack;
    stack.reserve(32);
    stack.push_back(root_.get());

    while (!stack.empty()) {
        const Node* node = stack.back();
        stack.pop_back();

        if (!intersectsSphere(node, query, sqRadius)) {
            continue;
        }

        if (node->isLeaf) {
            const auto& indices = node->pointIndices;
            result.reserve(result.size() + indices.size());  // Pre-allocate space

            for (const IndexType idx : indices) {
                Eigen::Vector3d diff = cloud_.row(idx).transpose() - query;
                double sqDist = diff.squaredNorm();
                if (sqDist <= sqRadius) {
                    result.push_back({idx, std::sqrt(sqDist)});
                }
            }
        } else {
            uint8_t mask = node->childMask;
            while (mask) {
                int idx = 31 - __builtin_clz(mask);
                if (node->children[idx]) {
                    stack.push_back(node->children[idx].get());
                }
                mask &= ~(1 << idx);
            }
        }
    }
    
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
              
    return result.size();
}

void Octree::invalidate() {
    // Clean up all allocated nodes
    nodePool.clear();
    currentPoolIndex = 0;
    root_.reset();
}

std::ostream& Octree::saveIndex(std::ostream& stream) const {
    return stream;
}

std::istream& Octree::loadIndex(std::istream& stream) {
    return stream;
}

} // namespace py4dgeo