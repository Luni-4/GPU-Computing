#pragma once

// Cuda Library
#include <curand_kernel.h>

#ifdef _WIN32
#include "Windows.h"
#endif

namespace Kernel {

	 void initWeightK(dim3 t, dim3 b, double *weight, const int &wDim, curandState *states);

	 void initBiasK(dim3 t, dim3 b, double *weight, const int &wDim, curandState *states);
	
	 void outputErrorK(dim3 t, dim3 b, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes);

	 void actReluK(dim3 t, dim3 b, double *output, const int &nodes);
	 void derivActReluK(dim3 t, dim3 b, const double *output, double *error, const int &nodes);

	 void actSigmoidK(dim3 t, dim3 b, double *output, const int &nodes);
	 void derivActSigmoidK(dim3 t, dim3 b, const double *output, double *error, const int &nodes);

	 void actTanhK(dim3 t, dim3 b, double *output, const int &nodes);
	 void derivActTanhK(dim3 t, dim3 b, const double *output, double *error, const int &nodes);
}
