// Cuda Library
#include <curand_kernel.h>

#ifdef _WIN32
#include "Windows.h"
#endif

/*  INIZIALIZZAZIONE DEI PESI */

__global__ void initWeight(double *weight, const int wDim, curandStateXORWOW_t *states);

__global__ void initBias(double *bias, const int node, curandStateXORWOW_t *states);


/* CALCOLO DEL DELTA */

__global__ void outputError(const double *output, double *error, const uint8_t *label, const int target, const int node);


/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

/* Funzione di attivazione del Relu e derivata */
__global__ void actRelu(double *output, double *temp, const int node);

__global__ void derivActRelu(double *error, double *temp, const int node);

/* Funzione di attivazione del Sigmoide e derivata */
__global__ void actSigmoid(double *output, const int node);

__global__ void derivActSigmoid(const double *output, double *error, const int node);

/* Funzione di attivazione della Tanh e derivata */
__global__ void actTanh(double *output, const int node);

__global__ void derivActTanh(const double *output, double *error, const int node);


/* AGGIORNAMENTO DEI PESI FULLY CONNECTED*/

__global__ void errorPrevOutput(double *temp, const double *prevOutput, const double *error, const int node, const int prevDim);

/* METODI CONVOLUZIONALE */

__global__ void createSubmatrixBis(double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes);

__global__ void createSubmatrixProduct(double * sub, const double * prevOutput, const double * weightRot, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes);

__global__ void outputFromSub(double * output, double * sub, int filterDim);

__global__ void createSubmatrix(double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes);

__global__ void zeroPaddingBis(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth);

__global__ void zeroPadding(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth);

__global__ void rot180Bis(const double * forwardWeight, double * forwardWeightRot, int filterDim);

__global__ void rot180(const double * forwardWeight, double * forwardWeightRot, int filterDim);
