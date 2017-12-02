#pragma once

// Cuda Library
#include <curand_kernel.h>

#ifdef _WIN32
#include "Windows.h"
#endif

// Numero di threads in un blocco
#define THREADS 32

// Converte un numero intero al multiplo più vicino di 32
#define ALIGN_UP(a, b) ((a + (b - 1)) / b) * b

namespace Kernel {
	void initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandState *states);

	void initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandState *states);

	void outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes);

	void actReluK(dim3 b, dim3 t, double *output, double *temp, const int &nodes, const cudaStream_t *streams, const int &nStreams);
	void derivActReluK(dim3 b, dim3 t, double *error, double *temp, const int &nodes, const cudaStream_t *streams, const int &nStreams);

	void actSigmoidK(dim3 b, dim3 t, double *output, const int &nodes, const cudaStream_t *streams, const int &nStreams);
	void derivActSigmoidK(dim3 b, dim3 t, const double *output, double *error, const int &nodes, const cudaStream_t *streams, const int &nStreams);

	void actTanhK(dim3 b, dim3 t, double *output, const int &nodes, const cudaStream_t *streams, const int &nStreams);
	void derivActTanhK(dim3 b, dim3 t, const double *output, double *error, const int &nodes, const cudaStream_t *streams, const int &nStreams);
}
