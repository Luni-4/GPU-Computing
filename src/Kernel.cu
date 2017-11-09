#include "Kernel.h"

#include <stdio.h>

__global__ void initWeight(double *weight, const int wDim, curandState *states) {

	// Gestione degli indici	
	const unsigned  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned  int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterr� il valore casuale
	double r = curand_uniform_double(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < wDim)
		weight[tid] = 0.4 * r;
}

void Kernel::initWeightK(dim3 t, dim3 b, double * weight, const int wDim, curandState * states) {
#ifdef _WIN32
	initWeight NvCUDA2(t, b) (weight, wDim, states);
#else
	initWeight << <t, b >> > (weight, wDim, states);
#endif
}

__global__ void initBias(double *bias, const int node, curandState *states) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterr� il valore casuale
	double r = curand_uniform_double(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < node)
		bias[tid] = 0.4 * r;
}

void Kernel::initBiasK(dim3 t, dim3 b, double * weight, const int wDim, curandState * states) {
#ifdef _WIN32
	initBias NvCUDA2(t, b) (weight, wDim, states);
#else
	initBias << <t, b >> > (weight, wDim, states);
#endif
}

__global__ void outputError(double *output, double *error, const uint8_t label, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	int trueLabel = 0;
	
	/* Il predittore dovrebbe predire con probabilit� 1 solo la label passata alla funzione, quindi la variabile
	trueLabel contiene il valore che ci si aspetterebbe dal predittore, cio� 1 */
	if(tid == label)
	    trueLabel = 1;
	
	// L'errore commesso � dato dalla differenza tra la predizione ottenuta e il valore reale dell'etichetta
	if (tid < node)
		error[tid] = trueLabel - output[tid];
}

void Kernel::outputErrorK(dim3 t, dim3 b, double *output, double *error, const uint8_t label, const int nodes) {
#ifdef _WIN32
	outputError NvCUDA2(t, b) (output, error, label, nodes);
#else
	outputError << <t, b >> > (output, error, label, nodes);
#endif
}




/* Funzione di attivazione del Sigmoide e derivata */

__global__ void actRelu(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = log(1 + exp((output[tid])));
}

__global__ void derivActRelu(double *output, double *error, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (1 / (1 + (exp((-output[tid])))));
}

void Kernel::actReluK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actRelu NvCUDA2(t, b) (output, nodes);
#else
	actRelu << <t, b >> > (output, nodes);
#endif 
}

void Kernel::derivActReluK(dim3 t, dim3 b, double *output, double *error, const int nodes) {
#ifdef _WIN32
	derivActRelu NvCUDA2(t, b) (output, error, nodes);
#else
	derivActRelu << <t, b >> > (output, error, nodes);
#endif 
}



/* Funzione di attivazione del Sigmoide e derivata */

__global__ void actSigmoid(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = 1 / (1 + (exp((-output[tid]))));
}

__global__ void derivActSigmoid(double *output, double *error, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (output[tid] * (1 - output[tid]));
}

void Kernel::actSigmoidK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actSigmoid NvCUDA2(t, b) (output, nodes);
#else
	actSigmoid << <t, b >> > (output, nodes);
#endif 
}

void Kernel::derivActSigmoidK(dim3 t, dim3 b, double *output, double *error, const int nodes) {
#ifdef _WIN32
	derivActSigmoid NvCUDA2(t, b) (output, error, nodes);
#else
	derivActSigmoid << <t, b >> > (output, error, nodes);
#endif 
}




/* Funzione di attivazione della Tanh e derivata */

__global__ void actTanh(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = tanh(output[tid]);
}

__global__ void derivActTanh(double *output, double *error, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (1 - pow(tanh(output[tid]),2));
}

void Kernel::actTanhK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actTanh NvCUDA2(t, b) (output, nodes);
#else
	actTanh << <t, b >> > (output, nodes);
#endif 
}

void Kernel::derivActTanhK(dim3 t, dim3 b, double *output, double *error, const int nodes) {
#ifdef _WIN32
	derivActTanh NvCUDA2(t, b) (output, error, nodes);
#else
	derivActTanh << <t, b >> > (output, error, nodes);
#endif 
}
