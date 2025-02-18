#include "catch2/catch.hpp"
#include "py4dgeo/octree.hpp"
#include <Eigen/Eigen>
#include <array>
#include <random>

using namespace py4dgeo;

TEST_CASE("Octree basic functionality", "[octree]") {
    // Create test point cloud
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> points(5, 3);
    points << 1.0, 1.0, 1.0,
              2.0, 2.0, 2.0,
              5.0, 5.0, 5.0,
              10.0, 10.0, 10.0,
              11.0, 10.0, 9.0;

    SECTION("Tree construction and memory pool") {
        std::vector<int> leafSizes = {2, 4, 8, 16};
        for (int leafSize : leafSizes) {
            Octree tree = Octree::create(points);
            REQUIRE_NOTHROW(tree.build_tree(leafSize));
            REQUIRE_NOTHROW(tree.invalidate());
        }
    }

    SECTION("Radius search with distances") {
        Octree tree = Octree::create(points);
        tree.build_tree(2);

        std::array<double, 3> query = {2.0, 2.0, 2.0};
        double radius = 3.0;

        Octree::RadiusSearchDistanceResult results;
        size_t found = tree.radius_search_with_distances(query.data(), radius, results);

        REQUIRE(found == results.size());

        // Ensure distances are sorted
        REQUIRE(std::is_sorted(results.begin(), results.end(),
                             [](const auto& a, const auto& b) {
                                 return a.second < b.second;
                             }));

        // Verify distances are correct (Ensure Octree uses Euclidean distance)
        for (const auto& result : results) {
            Eigen::Vector3d point = points.row(result.first);
            Eigen::Vector3d queryPoint(query[0], query[1], query[2]);
            // double calculatedDist = (point - queryPoint).norm();
            // REQUIRE(std::abs(calculatedDist - result.second) < 1e-10);
            double calculatedDistSquared = (point - queryPoint).squaredNorm();
            REQUIRE(std::abs(calculatedDistSquared - result.second) < 1e-10);
            //REQUIRE(calculatedDist <= radius);
            REQUIRE(calculatedDistSquared <= radius * radius);
        }
    }
}

TEST_CASE("Octree edge cases", "[octree]") {
    SECTION("Empty point cloud") {
        Eigen::Matrix<double, 0, 3, Eigen::RowMajor> points;
        Octree tree = Octree::create(points);
        REQUIRE_NOTHROW(tree.build_tree(2));

        std::array<double, 3> query = {0.0, 0.0, 0.0};
        Octree::RadiusSearchResult results;
        REQUIRE(tree.radius_search(query.data(), 1.0, results) == 0);
        REQUIRE(results.empty());
    }

    SECTION("Single point search") {
        Eigen::Matrix<double, 1, 3, Eigen::RowMajor> points;
        points << 1.0, 1.0, 1.0;
        
        Octree tree = Octree::create(points);
        tree.build_tree(2);

        std::array<double, 3> query = {1.0, 1.0, 1.0};
        Octree::RadiusSearchResult results;
        
        // Exact match test
        REQUIRE(tree.radius_search(query.data(), 0.1, results) == 1);
        REQUIRE(results.size() == 1);
        REQUIRE(results[0] == 0);
    }

    SECTION("Randomized large dataset") {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-100.0, 100.0);

        const int numPoints = 1000;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> points(numPoints, 3);
        for (int i = 0; i < numPoints; ++i) {
            points.row(i) << dis(gen), dis(gen), dis(gen);
        }

        Octree tree = Octree::create(points);
        tree.build_tree(16);

        for (int i = 0; i < 10; ++i) {
            std::array<double, 3> query = {dis(gen), dis(gen), dis(gen)};
            double radius = std::abs(dis(gen)) / 10.0;

            Octree::RadiusSearchDistanceResult results;
            REQUIRE(tree.radius_search_with_distances(query.data(), radius, results) == results.size());

            REQUIRE(std::is_sorted(results.begin(), results.end(),
                                 [](const auto& a, const auto& b) {
                                     return a.second < b.second;
                                 }));

            for (const auto& result : results) {
                REQUIRE(result.second <= radius);
            }
        }
    }
}
