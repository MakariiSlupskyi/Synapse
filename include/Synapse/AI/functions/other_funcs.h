#pragma once

#include "Synapse/linear/tensor.h"
#include <string>
#include <vector>

namespace syn {
	std::vector<std::string> split(std::string str, char separator=' ');
	std::vector<int> splitToI(std::string str, char separator=' ');
	std::vector<double> splitToD(std::string str, char separator=' ');
}