#pragma once

// Cuda Library
// Cuda
#include <curand_kernel.h>

#ifdef _WIN32
#include "Windows.h"
#endif

class Kernel {
public:
	static void initWeightK(dim3 t, dim3 b, double *weight, const int wDim, curandState *states);

	static void initBiasK(dim3 t, dim3 b, double *bias, const int node, curandState *states);
	
	static void outputErrorK(dim3 t, dim3 b, double *output, double *error, const uint8_t label, const int nodes);

	static void actReluK(dim3 t, dim3 b, double *output, const int node);
	static void derivActReluK(dim3 t, dim3 b, double *output, double *error, const int nodes);

	static void actSigmoidK(dim3 t, dim3 b, double *output, const int node);
	static void derivActSigmoidK(dim3 t, dim3 b, double *output, double *error, const int nodes);

	static void actTanhK(dim3 t, dim3 b, double *output, const int node);
	static void derivActTanhK(dim3 t, dim3 b, double *output, double *error, const int nodes);
};
