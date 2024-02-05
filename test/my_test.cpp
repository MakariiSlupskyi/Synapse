#include <gtest/gtest.h>
#include <vector>
#include "MLL/Linear/Tensor.h"

TEST(TensorTesting, SettingData) {
    ml::Tensor t({2, 3});

    t.fill(1.0d);
    for (size_t i = 0; i < t.getData().size(); ++i) {
        EXPECT_TRUE(t.getData()[i] == 1.0d) << "Setting by constant isn't working!";
    }

    std::vector<double> data = {1, 0, -1, 3};
    t.fill(data);
    for (int i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(t.getData()[i] == data[i]) << "Setting by data isn't working!";
    }
}

TEST(TensorTesting, SimpleIndexing) {
    ml::Tensor t1({2, 3}, {0, 1, 2, 3, 4, 5});
    ml::Tensor t2({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    EXPECT_EQ(t1({0, 0}), 0);
    EXPECT_EQ(t1({1, 1}), 4);

    EXPECT_EQ(t2({1, 2, 1}), 11);
    EXPECT_EQ(t2({0, 1, 1}), 3);
}

TEST(TensorTesting, AggregateFunctions) {
    ml::Tensor t1({2, 3}, {0, 1, 2, 3, 4, 5});

    EXPECT_EQ(t1.sum(), 15);
    EXPECT_EQ(t1.max(), 5);
    EXPECT_EQ(t1.min(), 0);
}

TEST(TensorTesting, Comparising) {
    ml::Tensor t1({2, 2}, {0, 1, 2, 3});
    ml::Tensor t2({2, 2}, {0, 1, 2, 3});

    EXPECT_TRUE(t1 == t2);
}

TEST(TensorTesting, TrivialMathsOperations) {
    ml::Tensor t1({2, 2}, {0, 2, 4, -1});
    ml::Tensor t2({2, 2}, {4, 1, 2, 2});
    ml::Tensor t3({2, 2}, {4, 3, 6, 1});
    ml::Tensor t4({2, 2}, {-4, 1, 2, -3});
    ml::Tensor t5({2, 2}, {0, 3, 8, -2});
    ml::Tensor t6({2, 2}, {0, 2, 2, -0.5d});

    EXPECT_TRUE((t1 + t2) == t3);
    EXPECT_TRUE((t1 - t2) == t4);
    EXPECT_TRUE((t1 * t2) == t5);
    EXPECT_TRUE((t1 / t2) == t6);
}

TEST(TensorTesting, ScalarMathsOperations) {
    ml::Tensor t1({2, 2}, {0, 2, 4, -1});
    ml::Tensor t2({2, 2}, {4, 6, 10, 3});
    ml::Tensor t3({2, 2}, {-2, 0, 2, -3});
    ml::Tensor t4({2, 2}, {0, -2, -4, 1});
    ml::Tensor t5({2, 2}, {0, 1, 2, -0.5});

    EXPECT_TRUE((t1 + 4) == t2);
    EXPECT_TRUE((t1 - 2) == t3);
    EXPECT_TRUE((t1 * -1) == t4);
    EXPECT_TRUE((t1 / 2) == t5);
}