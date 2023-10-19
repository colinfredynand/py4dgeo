#include "catch2/catch.hpp"
#include "py4dgeo/epoch.hpp"
#include "py4dgeo/registration.hpp"
#include "testsetup.hpp"
#include <iostream>

// DELETE
#include "py4dgeo/compute.hpp"
#include <vector>

using namespace py4dgeo;

TEST_CASE("Affine Transformation", "[compute]")
{
  auto [cloud1, corepoints] = testcloud();
  auto [cloud2, corepoints2] = testcloud();

  SECTION("Perform Transformation: ")
  {
    // Define a transformation
    Transformation t(Transformation::Identity());
    t(0, 3) = 1;

    EigenPointCloud ref(1, 3);
    ref << 1, 2, 3;

    // Apply the transformation
    transform_pointcloud_inplace(*cloud1, t, ref);

    for (IndexType i = 0; i < cloud1->rows(); ++i) {
      if (std::abs((*cloud1)(i, 0) - (*cloud2)(i, 0) - 1.0) >= 1e-8) {
        CAPTURE((*cloud1)(i, 0));
        CAPTURE((*cloud2)(i, 0));
      }
      REQUIRE(std::abs((*cloud1)(i, 0) - (*cloud2)(i, 0) - 1.0) < 1e-8);
      REQUIRE(std::abs((*cloud1)(i, 1) - (*cloud2)(i, 1)) < 1e-8);
      REQUIRE(std::abs((*cloud1)(i, 2) - (*cloud2)(i, 2)) < 1e-8);
    }
  }

  auto [cloud2a, cloud2b] = testcloud_dif_files();
  Epoch epoch_test1(*cloud2a);
  Epoch epoch_test2(*cloud2b);
  DisjointSet set1(epoch_test1.cloud.rows());
  DisjointSet set2(epoch_test2.cloud.rows());

  SECTION("Disjoint Set find: ")
  {
    int label = 10;
    auto set1_find_test = set1.Find(label);
    auto set2_find_test = set2.Find(label);
    REQUIRE(set1_find_test == set2_find_test);
  }

  SECTION("Disjoint Set union: ")
  {
    int label1 = 10;
    int same_label = label1;
    auto same_label_union_test = set1.Union(label1, same_label, true);
    REQUIRE(same_label_union_test == label1);

    int label2 = 11;
    auto nonsize_union_test = set2.Union(label1, label2, false);
    REQUIRE(nonsize_union_test == label2);

    int label3 = 12;
    auto merged_label = set1.Union(label2, label3, false);
    auto size_union_test = set1.Union(label1, merged_label, true);
    REQUIRE(size_union_test == label1);
  }

  SECTION("SupervoxelSegmentation: ")
  {
    auto tree1 = KDTree::create(epoch_test1.cloud);
    tree1.build_tree(10);
    double resolution = 10;
    int k = 10;
    auto n_supervoxels =
      estimate_supervoxel_count(epoch_test1.cloud, resolution);
    std::vector<std::vector<int>> result =
      supervoxel_segmentation(epoch_test1, tree1, resolution, k);
    REQUIRE(result.size() == n_supervoxels);
  }
}
