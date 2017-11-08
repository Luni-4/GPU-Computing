#pragma once
#include "Kernel.h"

#include <stdio.h>

__global__ void initWeight(double *weight, const int wDim, curandState *states) {

	// Gestione degli indici	
	const unsigned  int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned  int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
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

	// Variabile che conterrà il valore casuale
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

__global__ void actRelu(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = log(1 + exp((output[tid])));
}

void Kernel::actReluK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actRelu NvCUDA2(t, b) (output, nodes);
#else
	actRelu << <t, b >> > (output, nodes);
#endif 
}

__global__ void actSigmoid(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = 1 / (1 + (exp((-output[tid]))));
}

void Kernel::actSigmoidK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actSigmoid NvCUDA2(t, b) (output, nodes);
#else
	actSigmoid << <t, b >> > (output, nodes);
#endif 
}

__global__ void actTanh(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = tanh(output[tid]);
}

void Kernel::actTanhK(dim3 t, dim3 b, double *output, int nodes) {
#ifdef _WIN32
	actTanh NvCUDA2(t, b) (output, nodes);
#else
	actTanh << <t, b >> > (output, nodes);
#endif 
}
