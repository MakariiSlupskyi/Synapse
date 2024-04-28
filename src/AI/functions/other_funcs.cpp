#include "Synapse/AI/functions/other_funcs.h"
#include "Synapse/linear/tensor_funcs.h"
#include <sstream>
#include <cmath>

std::vector<std::string> syn::split(std::string str, char separator) {
    std::vector<std::string> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(item);
    }

    return result;  
}

std::vector<int> syn::splitToI(std::string str, char separator) {
    std::vector<int> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(std::stoi(item));
    }

    return result;  
}

std::vector<double> syn::splitToD(std::string str, char separator) {
    std::vector<double> result;
    std::stringstream ss(str);
    std::string item;

    while (getline(ss, item, separator)) {
        result.push_back(std::stod(item));
    }

    return result;
}