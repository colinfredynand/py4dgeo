#include "py4dgeo/octree.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <vector>
#include <cstring>

#ifdef PY4DGEO_WITH_OPENMP
#include <omp.h>
#endif

namespace py4dgeo {

// Inline helper to compute the squared Euclidean norm for three doubles.
static inline double squaredNorm3(double dx, double dy, double dz) {
    return dx * dx + dy * dy + dz * dz;
}

Octree::Octree(const EigenPointCloudRef& cloud)
    : cloud_(cloud), leafSize_(32) {
    nodes.reserve(INITIAL_NODES);
    points.reserve(INITIAL_POINTS);
}

Octree Octree::create(const EigenPointCloudRef& cloud) {
    return Octree(cloud);
}

Octree::Node* Octree::allocateNode(double x, double y, double z, double extent) {
    nodes.push_back(std::make_unique<Node>(x, y, z, extent));
    return nodes.back().get();
}

void Octree::build_tree(int leaf) {
    leafSize_ = leaf;
    nodes.clear();
    root_ = nullptr;

    if (cloud_.rows() == 0)
        return;

    // Cache all points and create initial index array.
    points.resize(cloud_.rows());
    std::vector<uint32_t> indices(cloud_.rows());
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < cloud_.rows(); ++i) {
        const auto& pt = cloud_.row(i);
        points[i] = { pt.x(), pt.y(), pt.z(), static_cast<IndexType>(i) };
        indices[i] = i;
    }

    // Compute bounds.
    auto minmax_x = std::minmax_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) { return a.x < b.x; });
    auto minmax_y = std::minmax_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) { return a.y < b.y; });
    auto minmax_z = std::minmax_element(points.begin(), points.end(),
        [](const auto& a, const auto& b) { return a.z < b.z; });

    double min_x = minmax_x.first->x;
    double max_x = minmax_x.second->x;
    double min_y = minmax_y.first->y;
    double max_y = minmax_y.second->y;
    double min_z = minmax_z.first->z;
    double max_z = minmax_z.second->z;

    double cx = (min_x + max_x) * 0.5;
    double cy = (min_y + max_y) * 0.5;
    double cz = (min_z + max_z) * 0.5;
    double extent = std::max({ max_x - min_x, max_y - min_y, max_z - min_z }) * 0.5;
    if (extent == 0.0)
        extent = 1.0;

    root_ = allocateNode(cx, cy, cz, extent);
    buildRecursive(root_, indices.data(), cloud_.rows());
}

void Octree::buildRecursive(Node* node, uint32_t* indices, uint32_t count) {
    if (count <= static_cast<uint32_t>(leafSize_)) {
        node->isLeaf = 1;
        node->pointStart = 0;  // For simplicity; in a complete implementation, record an offset.
        node->pointCount = count;
        return;
    }

    node->isLeaf = 0;

    // Partition points into octants.
    uint32_t offsets[8] = {0};
    uint32_t counts[8] = {0};

    // Count points per child.
    for (uint32_t i = 0; i < count; ++i) {
        const auto& pt = points[indices[i]];
        uint32_t code = node->getMortonCode(pt.x, pt.y, pt.z);
        ++counts[code];
    }

    // Calculate offsets.
    offsets[0] = 0;
    for (int i = 1; i < 8; ++i)
        offsets[i] = offsets[i - 1] + counts[i - 1];

    // Partition points.
    std::vector<uint32_t> temp(count);
    uint32_t curr_offsets[8];
    std::copy(offsets, offsets + 8, curr_offsets);
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t idx = indices[i];
        const auto& pt = points[idx];
        uint32_t code = node->getMortonCode(pt.x, pt.y, pt.z);
        temp[curr_offsets[code]++] = idx;
    }
    std::copy(temp.begin(), temp.end(), indices);

    double half_extent = node->extent * 0.5;
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp parallel for schedule(dynamic, 1) shared(node, indices, offsets, counts, half_extent)
#endif
    for (int i = 0; i < 8; ++i) {
        if (counts[i] == 0)
            continue;
        double cx = node->x + ((i & 1) ? half_extent : -half_extent);
        double cy = node->y + ((i & 2) ? half_extent : -half_extent);
        double cz = node->z + ((i & 4) ? half_extent : -half_extent);
        Node* child = allocateNode(cx, cy, cz, half_extent);
#ifdef PY4DGEO_WITH_OPENMP
#pragma omp critical
#endif
        {
            node->childMask |= (1 << i);
            node->children[i] = child;
        }
        buildRecursive(child, indices + offsets[i], counts[i]);
    }
    // Parallel for loop ensures all iterations complete.
}

std::size_t Octree::radius_search(const double* querypoint,
                                  double radius,
                                  RadiusSearchResult& result) const {
    result.clear();
    if (!root_ || radius < 0.0)
        return 0;

    const double qx = querypoint[0];
    const double qy = querypoint[1];
    const double qz = querypoint[2];
    const double sqRadius = radius * radius;

    Node* stack[STACK_SIZE];
    int stack_size = 1;
    stack[0] = root_;
    result.reserve(leafSize_ * 2);

    while (stack_size > 0) {
        Node* node = stack[--stack_size];
        if (!node->intersectsSphere(qx, qy, qz, radius))
            continue;
        if (node->isLeaf) {
            uint32_t end = node->pointStart + node->pointCount;
            for (uint32_t i = node->pointStart; i < end; ++i) {
                const auto& pt = points[i];
                double dx = pt.x - qx;
                double dy = pt.y - qy;
                double dz = pt.z - qz;
                double sqDist = squaredNorm3(dx, dy, dz);
                if (sqDist <= sqRadius)
                    result.push_back(pt.idx);
            }
        } else {
            uint8_t mask = node->childMask;
            while (mask && stack_size < STACK_SIZE) {
                int idx = highestSetBit(mask);
                stack[stack_size++] = node->children[idx];
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
    if (!root_ || radius < 0.0)
        return 0;

    const double qx = querypoint[0];
    const double qy = querypoint[1];
    const double qz = querypoint[2];
    const double sqRadius = radius * radius;

    Node* stack[STACK_SIZE];
    int stack_size = 1;
    stack[0] = root_;
    result.reserve(leafSize_ * 2);

    while (stack_size > 0) {
        Node* node = stack[--stack_size];
        if (!node->intersectsSphere(qx, qy, qz, radius))
            continue;
        if (node->isLeaf) {
            uint32_t end = node->pointStart + node->pointCount;
            for (uint32_t i = node->pointStart; i < end; ++i) {
                const auto& pt = points[i];
                double dx = pt.x - qx;
                double dy = pt.y - qy;
                double dz = pt.z - qz;
                double sqDist = squaredNorm3(dx, dy, dz);
                if (sqDist <= sqRadius)
                    result.push_back({pt.idx, std::sqrt(sqDist)});
            }
        } else {
            uint8_t mask = node->childMask;
            while (mask && stack_size < STACK_SIZE) {
                int idx = highestSetBit(mask);
                stack[stack_size++] = node->children[idx];
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
    points.clear();
    root_ = nullptr;
}

} // namespace py4dgeo
