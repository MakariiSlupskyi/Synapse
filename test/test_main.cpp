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
    std::vector<int> shape = {2, 3};
    std::vector<double> data(10000000);
    std::vector<int> indices = {1, 2};
    double value = 1;

	for (double& elem : data) { elem = value; }
}

void func2() {
    std::vector<int> shape = {2, 3};
    std::vector<double> data(10000000);
    std::vector<int> indices = {1, 2};
    double value = 1;

	for (int i = 0; i < data.size(); ++i) { data[i] = value; }
}

int main() {
    std::cout << "--- This Is a Benchmark Program ---\n\n";

    benchmark(func1, "test1");
    benchmark(func2, "test2");

    return 0;
}