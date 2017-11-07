#pragma once

__global__ void initWeight(double *weight, const int wDim, curandState *states) {

	// Gestione degli indici	
	const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	double r = curand_uniform_double(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < wDim)
		weight[tid] = 0.4 * r;
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

__global__ void actRelu(double *output, const int node) {

    // Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < node)
		output[tid] = log(1 + exp((output[tid])));
}

__global__ void actSigmoid(double *output, const int node) {

    // Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < node)
		output[tid] = 1 / (1 + (exp((-output[tid])) ));
}

__global__ void actTanh(double *output, const int node) {

    // Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < node)
		output[tid] = tanh(output[tid]);
}

