#include <iostream>
#include <vector>
#include <chrono>
#include <execution>
#include <algorithm>
#include <numeric>
#include <omp.h>

const long int SIZE = 1 << 25;

void benchmark(void (*func)(void), const char* func_name) {
    using namespace std::chrono;
    auto start = high_resolution_clock::now();
    func();
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);

    std::cout << "Elapsed time (" << func_name << "): " << duration.count() << '\n';
}

void func1() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {6, 7, 8, 9, 10};

    for (int i = 0; i < vec1.size(); ++i) {
        vec1[i] += vec2[i];
    }
}

void func2() {
    std::vector<int> vec1 = {1, 2, 3, 4, 5};
    std::vector<int> vec2 = {6, 7, 8, 9, 10};

    #pragma omp parallel for
    for (int i = 0; i < vec1.size(); ++i) {
        vec1[i] += vec2[i];
    }
}

int main() {
    std::cout << "--- This Is a Benchmark Program ---\n\n";

    benchmark(func1, "test1");
    benchmark(func2, "test2");

    return 0;
}