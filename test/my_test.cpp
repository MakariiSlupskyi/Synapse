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

TEST(TensorTesting, Comparising) {
    ml::Tensor t1({2, 2}, {0, 1, 2, 3});
    ml::Tensor t2({2, 2}, {0, 1, 2, 3});

    EXPECT_TRUE(t1 == t2);
}

TEST(TensorTesting, TrivialAddition) {
    ml::Tensor t1({2, 2}, {1, 0, 1, 1});
    ml::Tensor t2({2, 2}, {0, 1, 2, 3});
    ml::Tensor t3({2, 2}, {1, 1, 3, 4});

    EXPECT_TRUE((t1 + t2) == t3);
}

TEST(TensorTesting, TrivialSubtraction) {
    ml::Tensor t1({2, 2}, {1, 2, 3, 5});
    ml::Tensor t2({2, 2}, {0, 1, 2, 3});
    ml::Tensor t3({2, 2}, {1, 1, 1, 2});

    EXPECT_TRUE((t1 - t2) == t3);
}