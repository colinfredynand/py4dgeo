#include "testsetup.hpp"
#include <py4dgeo/octree.hpp>
#include <benchmark/benchmark.h>
#include <Eigen/Eigen>

using namespace py4dgeo;

static void
octree_radius_search_benchmark(benchmark::State& state)
{
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(/*leaf=*/10);
    
    const double* query = corepoints->row(0).data();
    double radius = 2.0;  // similar radius as in distances_b.cpp
    
    for (auto _ : state) {
        Octree::RadiusSearchResult result;
        tree.radius_search(query, radius, result);
    }
}

static void
octree_radius_search_with_distances_benchmark(benchmark::State& state)
{
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(/*leaf=*/10);
    
    // Setup query parameters - using the first corepoint as query point
    const double* query = corepoints->row(0).data();
    double radius = 2.0;
    
    for (auto _ : state) {
        Octree::RadiusSearchDistanceResult result;
        tree.radius_search_with_distances(query, radius, result);
    }
}

// Multiple query points benchmark
static void octree_multiple_queries_benchmark(benchmark::State& state) {
    // Get the test cloud data
    auto [cloud, corepoints] = ahk_benchcloud();
    
    // Create and build the octree
    Octree tree = Octree::create(*cloud);
    tree.build_tree(10);
    
    // Setup multiple query points (using corepoints)
    double radius = 2.0;
    
    for (auto _ : state) {
        // Perform radius search for each corepoint
        for (IndexType i = 0; i < corepoints->rows(); ++i) {
            Octree::RadiusSearchResult result;
            tree.radius_search(corepoints->row(i).data(), radius, result);
        }
    }
}

BENCHMARK(octree_radius_search_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_radius_search_with_distances_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_multiple_queries_benchmark)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();