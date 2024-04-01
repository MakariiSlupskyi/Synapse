#include <gtest/gtest.h>
#include <vector>

#include "Synapse/Linear.h"

#include "Synapse/AI/model.h"
#include "Synapse/AI/layers.h"
#include "Synapse/AI/functions/activ_funcs.h"

TEST(TensorTesting, SettingData) {
    syn::Tensor t({2, 3});

    t.fill(1.0);
    for (size_t i = 0; i < t.getData().size(); ++i) {
        EXPECT_TRUE(t.getData()[i] == 1.0);
    }

    std::vector<double> data = {1, 0, -1, 3, 4, 6};
    t.fill(data);
    for (int i = 0; i < data.size(); ++i) {
        EXPECT_TRUE(t.getData()[i] == data[i]);
    }
}

TEST(TensorTesting, SimpleIndexing) {
    syn::Tensor t1({2, 3}, {0, 1, 2, 3, 4, 5});
    syn::Tensor t2({2, 3, 2}, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    EXPECT_EQ(t1({0, 0}), 0);
    EXPECT_EQ(t1({1, 1}), 4);

    EXPECT_EQ(t2({1, 2, 1}), 11);
    EXPECT_EQ(t2({0, 1, 1}), 3);
}

TEST(TensorTesting, AggregateFunctions) {
    syn::Tensor t1({2, 3}, {0, 1, 2, 3, 4, 5});

    EXPECT_EQ(t1.sum(), 15);
    EXPECT_EQ(t1.max(), 5);
    EXPECT_EQ(t1.min(), 0);
}

TEST(TensorTesting, Comparising) {
    syn::Tensor t1({2, 2}, {0, 1, 2, 3});
    syn::Tensor t2({2, 2}, {0, 1, 2, 3});

    EXPECT_TRUE(t1 == t2);
}

TEST(TensorTesting, TrivialMathsOperations) {
    syn::Tensor t1({2, 2}, {0, 2, 4, -1});
    syn::Tensor t2({2, 2}, {4, 1, 2, 2});
    syn::Tensor t3({2, 2}, {4, 3, 6, 1});
    syn::Tensor t4({2, 2}, {-4, 1, 2, -3});
    syn::Tensor t5({2, 2}, {0, 3, 8, -2});
    syn::Tensor t6({2, 2}, {0, 2, 2, -0.5});
    
    EXPECT_TRUE((t1 + t2) == t3);
    EXPECT_TRUE((t1 - t2) == t4);
    EXPECT_TRUE((t1 * t2) == t5);
    EXPECT_TRUE((t1 / t2) == t6);
}

TEST(TensorTesting, ScalarMathsOperations) {
    syn::Tensor t1({2, 2}, {0, 2, 4, -1});
    syn::Tensor t2({2, 2}, {4, 6, 10, 3});
    syn::Tensor t3({2, 2}, {-2, 0, 2, -3});
    syn::Tensor t4({2, 2}, {0, -2, -4, 1});
    syn::Tensor t5({2, 2}, {0, 1, 2, -0.5});

    EXPECT_TRUE((t1 + 4) == t2);
    EXPECT_TRUE((t1 - 2) == t3);
    EXPECT_TRUE((t1 * -1) == t4);
    EXPECT_TRUE((t1 / 2) == t5);
}

TEST(TensorTesting, MatrixMultiplication) {
    syn::Tensor t1({3, 2}, {0, 2, 4, -1, 0, 2});
    syn::Tensor t2({2, 1}, {4, 6});

    auto t3 = t1.matMul(t2);

    EXPECT_EQ(t3.getData()[0], 12);
    EXPECT_EQ(t3.getData()[1], 10);
    EXPECT_EQ(t3.getData()[2], 12);
}

TEST(MachineLearningTesting, TrivialModelCreation) {
    syn::Model model({
        new syn::Dense(2, 4),
        new syn::Dense(4, 3)
    });
}

TEST(MachineLearningTesting, TrivialInfenceWorkflow) {
    syn::Model model({
        new syn::Dense(2, 4),
        new syn::Activation("relu"),
        new syn::Dense(4, 5),
        new syn::Activation("relu"),
        new syn::Dense(5, 3),
        new syn::Activation("sigmoid"),
    });

    auto input = syn::Tensor({2, 1});
    input.randomize();

    auto output = model.predict(input).getData();

    EXPECT_EQ(output.size(), 3);
}

