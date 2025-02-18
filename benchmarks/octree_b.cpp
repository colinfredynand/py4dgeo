#include "testsetup.hpp"
#include <py4dgeo/octree.hpp>
#include <py4dgeo/epoch.hpp>
#include <benchmark/benchmark.h>

using namespace py4dgeo;

static void octree_radius_search_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(8);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchResult result;
    result.reserve(100);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(tree.radius_search(querypoint.data(), 2.0, result));
        result.clear();
    }
}

static void octree_radius_search_distances_benchmark(benchmark::State& state) {
    auto [cloud, corepoints] = ahk_benchcloud();
    
    Octree tree = Octree::create(*cloud);
    tree.build_tree(8);
    
    std::array<double, 3> querypoint{0.0, 0.0, 0.0};
    Octree::RadiusSearchDistanceResult result;
    result.reserve(100);
    
    for (auto _ : state) {
        benchmark::DoNotOptimize(tree.radius_search_with_distances(querypoint.data(), 2.0, result));
        result.clear();
    }
}

BENCHMARK(octree_radius_search_benchmark)->Unit(benchmark::kMicrosecond);
BENCHMARK(octree_radius_search_distances_benchmark)->Unit(benchmark::kMicrosecond);

BENCHMARK_MAIN();