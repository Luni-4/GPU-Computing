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
	void initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandStateXORWOW_t *states);

	void initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandStateXORWOW_t *states);

	void outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes);

	void actReluK(dim3 b, dim3 t, double *output, double *temp, const int &nodes);
	void derivActReluK(dim3 b, dim3 t, double *error, double *temp, const int &nodes);

	void actSigmoidK(dim3 b, dim3 t, double *output, const int &nodes);
	void derivActSigmoidK(dim3 b, dim3 t, const double *output, double *error, const int &nodes);

	void actTanhK(dim3 b, dim3 t, double *output, const int &nodes);
	void derivActTanhK(dim3 b, dim3 t, const double *output, double *error, const int &nodes);

	void errorPrevOutputK(dim3 b, dim3 t, double *temp, const double *prevOutput, const double *error, const int &nodes, const int &dim, const int &prevDim);

	/*METODI CONVOLUZIONALE*/
	void createSubmatrixBisK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes);
	void createSubmatrixK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes);
	void zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth);
	void zeroPaddingK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth);
	void rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim);
	void rot180K(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim);
}
