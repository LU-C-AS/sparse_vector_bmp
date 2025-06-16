#include "bmp_alg.hpp"
#include "sparse_test_util.hpp"
#include "sparse_util.hpp"
#include <gtest/gtest.h>
#include <stdio.h>

using namespace sparse_vector_bmp;
TEST(BMPTest, test1) {
    //arrange
    //act
    //assert
    u32 ncol = 2;
    u32 block_size = 2;

    u32 topk = 5;

    BmpSearchOptions options;
    options.use_lock_ = false;

    Vector<i32> query_idx = {0, 1};
    Vector<f32> query_data = {1.0, 1.0};
    SparseVecRef query(query_idx.size(), query_idx.data(), query_data.data());

    Vector<i32> vec1_idx = {0};
    Vector<f32> vec1_data = {3.0};
    SparseVecRef vec1(vec1_idx.size(), vec1_idx.data(), vec1_data.data());

    Vector<i32> vec2_idx = {1};
    Vector<f32> vec2_data = {1.0};
    SparseVecRef vec2(vec2_idx.size(), vec2_idx.data(), vec2_data.data());

    Vector<i32> vec3_idx = {0, 1};
    Vector<f32> vec3_data = {1.0, 1.0};
    SparseVecRef vec3(vec3_idx.size(), vec3_idx.data(), vec3_data.data());

    BMPAlg<f32, i32> index(ncol, block_size);
    index.AddDoc(vec1, 0);
    index.AddDoc(vec2, 1);
    index.AddDoc(vec3, 2);
    index.AddDoc(vec3, 3);
    index.AddDoc(vec3, 4);
    index.AddDoc(vec3, 5);

    auto [indices, scores] = index.SearchKnn(query, topk, options);
    ASSERT_EQ(indices.size(), topk);
    ASSERT_EQ(indices[0], 0);
    ASSERT_EQ(indices[1], 2);
    ASSERT_EQ(indices[2], 3);
    ASSERT_EQ(indices[3], 4);
    ASSERT_EQ(indices[4], 5);
    ASSERT_EQ(scores.size(), topk);
    ASSERT_EQ(scores[0], 3.0);
    ASSERT_EQ(scores[1], 2.0);
    ASSERT_EQ(scores[2], 2.0);
    ASSERT_EQ(scores[3], 2.0);
    ASSERT_EQ(scores[4], 2.0);

    printf("test complete\n");
}

