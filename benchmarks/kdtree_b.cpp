#include "testsetup.hpp"
#include <py4dgeo/kdtree.hpp>
#include <py4dgeo/epoch.hpp>

#include <benchmark/benchmark.h>

using namespace py4dgeo;

// Benchmark KDTree radius search
static void kdtree_radius_search_benchmark(benchmark::State& state) {
    // Get test data
    auto [cloud, corepoints] = ahk_benchcloud();
    Epoch epoch(*cloud);
    
    // Build the KDTree
    epoch.kdtree.build_tree(10);
    
    // Setup query point at origin
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    KDTree::RadiusSearchResult result;
    
    // Run the benchmark
    for (auto _ : state) {
        epoch.kdtree.radius_search(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark KDTree radius search with distances
static void kdtree_radius_search_distances_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    Epoch epoch(*cloud);
    epoch.kdtree.build_tree(10);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    KDTree::RadiusSearchDistanceResult result;
    
    for (auto _ : state) {
        epoch.kdtree.radius_search_with_distances(querypoint.data(), 2.0, result);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark KDTree nearest neighbors search
static void kdtree_nearest_neighbors_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    Epoch epoch(*cloud);
    epoch.kdtree.build_tree(10);
    
    KDTree::NearestNeighborsResult result;
    const int k = 5;
    
    for (auto _ : state) {
        epoch.kdtree.nearest_neighbors(*corepoints, result, k);
        benchmark::DoNotOptimize(result);
    }
}

// Benchmark KDTree nearest neighbors with distances
static void kdtree_nearest_neighbors_distances_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    Epoch epoch(*cloud);
    epoch.kdtree.build_tree(10);
    
    KDTree::NearestNeighborsDistanceResult result;
    const int k = 5;
    
    for (auto _ : state) {
        epoch.kdtree.nearest_neighbors_with_distances(*corepoints, result, k);
        benchmark::DoNotOptimize(result);
    }
}

BENCHMARK(kdtree_radius_search_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(kdtree_radius_search_distances_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(kdtree_nearest_neighbors_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(kdtree_nearest_neighbors_distances_benchmark)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();