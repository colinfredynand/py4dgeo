#include "testsetup.hpp"
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/epoch.hpp>
#include <py4dgeo/octree.hpp>
#include <benchmark/benchmark.h>
#include <array>
#include <vector>

using namespace py4dgeo;

//========================//
// Octree Benchmarks
//========================//

// Benchmark Octree radius search
static void octree_radius_search_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    // Force a single leaf to avoid issues with incorrect offsets.
    int leafSize = static_cast<int>(cloud->rows());
    Octree octree = Octree::create(*cloud);
    octree.build_tree(leafSize);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchResult result;
    
    for (auto _ : state) {
        octree.radius_search(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark Octree radius search with distances
static void octree_radius_search_distances_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    // Force a single leaf to avoid subdivision-related issues.
    int leafSize = static_cast<int>(cloud->rows());
    Octree octree = Octree::create(*cloud);
    octree.build_tree(leafSize);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchDistanceResult result;
    
    for (auto _ : state) {
        octree.radius_search_with_distances(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

//========================//
// Register Benchmarks
//========================//

BENCHMARK(octree_radius_search_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_radius_search_distances_benchmark)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();
