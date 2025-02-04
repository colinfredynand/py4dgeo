#include "py4dgeo/octree.hpp"
#include <algorithm>
#include <numeric>

namespace py4dgeo {

Octree::Octree(const EigenPointCloudRef& cloud)
    : cloud_(cloud), leafSize_(32) {
    nodes.reserve(INITIAL_NODE_CAPACITY);
}

Octree Octree::create(const EigenPointCloudRef& cloud) {
    return Octree(cloud);
}

Octree::Node* Octree::allocateNode(const Eigen::Vector3d& center, double extent) {
    nodes.push_back(std::make_unique<Node>(center, extent));
    return nodes.back().get();
}

void Octree::build_tree(int leaf) {
    leafSize_ = leaf;
    nodes.clear();
    root_ = nullptr;
    
    if (cloud_.rows() == 0) return;
    
    // Compute bounding box
    Eigen::Vector3d min = cloud_.colwise().minCoeff();
    Eigen::Vector3d max = cloud_.colwise().maxCoeff();
    Eigen::Vector3d center = (min + max) * 0.5;
    double extent = (max - min).maxCoeff() * 0.5;
    if (extent == 0.0) extent = 1.0;
    
    // Initialize point indices
    pointIndices.resize(cloud_.rows());
    std::iota(pointIndices.begin(), pointIndices.end(), 0);
    
    // Create root and build
    root_ = allocateNode(center, extent);
    buildRecursive(root_, pointIndices.data(), cloud_.rows());
}

void Octree::buildRecursive(Node* node, IndexType* indices, size_t size) {
    if (size == 0) return;

    if (size <= static_cast<size_t>(leafSize_)) {
        node->spatial.isLeaf = true;
        node->spatial.start = indices - pointIndices.data();
        node->spatial.size = size;
        return;
    }
    
    node->spatial.isLeaf = false;
    
    // Count points per child
    std::array<size_t, 8> childCounts = {0};
    for (size_t i = 0; i < size; ++i) {
        uint32_t code = node->getMortonCode(cloud_.row(indices[i]));
        ++childCounts[code];
    }
    
    // Calculate child offsets
    std::array<size_t, 8> offsets;
    offsets[0] = 0;
    for (int i = 1; i < 8; ++i) {
        offsets[i] = offsets[i-1] + childCounts[i-1];
    }
    
    // Distribute points
    std::array<size_t, 8> currentOffsets = offsets;
    std::vector<IndexType> tempIndices(size);
    for (size_t i = 0; i < size; ++i) {
        const IndexType idx = indices[i];
        uint32_t code = node->getMortonCode(cloud_.row(idx));
        tempIndices[currentOffsets[code]++] = idx;
    }
    std::copy(tempIndices.begin(), tempIndices.end(), indices);
    
    // Create children
    const double childExtent = node->spatial.extent * 0.5;
    for (int i = 0; i < 8; ++i) {
        if (childCounts[i] == 0) continue;
        
        Eigen::Vector3d childCenter(node->spatial.x, node->spatial.y, node->spatial.z);
        childCenter[0] += ((i & 1) ? childExtent : -childExtent);
        childCenter[1] += ((i & 2) ? childExtent : -childExtent);
        childCenter[2] += ((i & 4) ? childExtent : -childExtent);
        
        node->children[i] = allocateNode(childCenter, childExtent);
        node->spatial.childMask |= (1 << i);
        buildRecursive(node->children[i], indices + offsets[i], childCounts[i]);
    }
}

std::size_t Octree::radius_search(const double* querypoint,
                                 double radius,
                                 RadiusSearchResult& result) const {
    result.clear();
    if (!root_ || radius < 0.0) return 0;

    const Eigen::Vector3d query = Eigen::Map<const Eigen::Vector3d>(querypoint);
    const double sqRadius = radius * radius;
    
    // Fixed-size stack on actual stack
    std::array<const Node*, STACK_SIZE> stack;
    int stackSize = 1;
    stack[0] = root_;
    
    result.reserve(leafSize_ * 2);

    while (stackSize > 0) {
        const Node* node = stack[--stackSize];

        if (!node->intersectsSphere(query, radius)) continue;

        if (node->spatial.isLeaf) {
            const IndexType* indices = pointIndices.data() + node->spatial.start;
            const size_t count = node->spatial.size;
            
            // Process points
            for (size_t i = 0; i < count; ++i) {
                const IndexType idx = indices[i];
                const auto& point = cloud_.row(idx);
                const double dx = point.x() - query.x();
                const double dy = point.y() - query.y();
                const double dz = point.z() - query.z();
                if (dx*dx + dy*dy + dz*dz <= sqRadius) {
                    result.push_back(idx);
                }
            }
        } else {
            // Add children in reverse order
            uint8_t mask = node->spatial.childMask;
            while (mask && stackSize < STACK_SIZE) {
                int idx = 31 - __builtin_clz(mask);
                stack[stackSize++] = node->children[idx];
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
    if (!root_ || radius < 0.0) return 0;

    const Eigen::Vector3d query = Eigen::Map<const Eigen::Vector3d>(querypoint);
    const double sqRadius = radius * radius;
    
    std::array<const Node*, STACK_SIZE> stack;
    int stackSize = 1;
    stack[0] = root_;
    
    result.reserve(leafSize_ * 2);

    while (stackSize > 0) {
        const Node* node = stack[--stackSize];

        if (!node->intersectsSphere(query, radius)) continue;

        if (node->spatial.isLeaf) {
            const IndexType* indices = pointIndices.data() + node->spatial.start;
            const size_t count = node->spatial.size;
            
            for (size_t i = 0; i < count; ++i) {
                const IndexType idx = indices[i];
                const auto& point = cloud_.row(idx);
                const double dx = point.x() - query.x();
                const double dy = point.y() - query.y();
                const double dz = point.z() - query.z();
                const double sqDist = dx*dx + dy*dy + dz*dz;
                if (sqDist <= sqRadius) {
                    result.push_back({idx, std::sqrt(sqDist)});
                }
            }
        } else {
            uint8_t mask = node->spatial.childMask;
            while (mask && stackSize < STACK_SIZE) {
                int idx = 31 - __builtin_clz(mask);
                stack[stackSize++] = node->children[idx];
                mask &= ~(1 << idx);
            }
        }
    }
    
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
              
    return result.size();
}

void Octree::invalidate() {
    nodes.clear();
    pointIndices.clear();
    root_ = nullptr;
}

std::ostream& Octree::saveIndex(std::ostream& stream) const {
    return stream;
}

std::istream& Octree::loadIndex(std::istream& stream) {
    return stream;
}

} // namespace py4dgeo