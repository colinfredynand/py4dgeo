#include "testsetup.hpp"
#include <py4dgeo/octree.hpp>
#include <py4dgeo/epoch.hpp>
#include <benchmark/benchmark.h>

using namespace py4dgeo;

// Benchmark Octree radius search
static void octree_radius_search_benchmark(benchmark::State& state) {
    // Get test data
    auto [cloud, corepoints] = ahk_benchcloud();
    
    // Build the Octree
    Octree tree = Octree::create(*cloud);
    tree.build_tree(10);
    
    // Setup query point
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchResult result;
    
    // Run the benchmark
    for (auto _ : state) {
        tree.radius_search(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark Octree radius search with distances
static void octree_radius_search_distances_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(10);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchDistanceResult result;
    
    for (auto _ : state) {
        tree.radius_search_with_distances(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark Octree multiple queries
static void octree_multiple_queries_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(10);
    
    Octree::RadiusSearchResult result;
    const double radius = 2.0;
    
    for (auto _ : state) {
        for (int i = 0; i < corepoints->rows(); ++i) {
            tree.radius_search(corepoints->row(i).data(), radius, result);
            benchmark::DoNotOptimize(result);
        }
    }
}

BENCHMARK(octree_radius_search_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_radius_search_distances_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_multiple_queries_benchmark)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();