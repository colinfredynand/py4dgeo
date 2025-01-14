#include "testsetup.hpp"
#include <py4dgeo/octree.hpp>
#include <benchmark/benchmark.h>
#include <Eigen/Eigen>

using namespace py4dgeo;

static void
octree_radius_search_benchmark(benchmark::State& state)
{
    // Create a point cloud with test data
    auto [cloud, corepoints] = ahk_benchcloud();
    
    // Create and build the octree
    Octree tree = Octree::create(*cloud);
    tree.build_tree(/*leaf=*/10);
    
    // Setup query parameters - using the first corepoint as query point
    const double* query = corepoints->row(0).data();
    double radius = 2.0;  // similar radius as in distances_b.cpp
    
    for (auto _ : state) {
        // Perform radius search
        Octree::RadiusSearchResult result;
        tree.radius_search(query, radius, result);
    }
}

static void
octree_radius_search_with_distances_benchmark(benchmark::State& state)
{
    // Create a point cloud with test data
    auto [cloud, corepoints] = ahk_benchcloud();
    
    // Create and build the octree
    Octree tree = Octree::create(*cloud);
    tree.build_tree(/*leaf=*/10);
    
    // Setup query parameters - using the first corepoint as query point
    const double* query = corepoints->row(0).data();
    double radius = 2.0;  // similar radius as in distances_b.cpp
    
    for (auto _ : state) {
        // Perform radius search with distances
        Octree::RadiusSearchDistanceResult result;
        tree.radius_search_with_distances(query, radius, result);
    }
}

// Register benchmarks
BENCHMARK(octree_radius_search_benchmark)->Unit(benchmark::kMillisecond);
BENCHMARK(octree_radius_search_with_distances_benchmark)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();