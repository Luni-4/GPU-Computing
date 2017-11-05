#include <iostream>
#include <vector>
#include <algorithm>

// Cuda
#include <curand_kernel.h>

#include "Common.h"
#include "FullyConnected.h"

#ifdef _WIN32
#include "Windows.h"
#endif

FullyConnected::FullyConnected(const int &width, const int &height, const ActFctType &a)
	: LayerDefinition(width, height, 1, FULLY_CONNECTED, a) {
	
	this->_nodes = width * height; 

}

FullyConnected::FullyConnected(const int &width, const ActFctType &a)
	: LayerDefinition(width, 1, 1, FULLY_CONNECTED, a),
      _nodes(width) {

}

FullyConnected::~FullyConnected() {
    CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
    CHECK(cudaFree(error));
}

int FullyConnected::getLayerNodeCount() {
	return _nodes;
}


int FullyConnected::getWeightCount(const int &prevLayerNode) {
	return prevLayerNode * _nodes;
}


std::vector<double> FullyConnected::getWeight() {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wDim * sizeof(double), cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> FullyConnected::getBias() {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}


__global__ void initWeight(double *weight, const int wDim, curandState *states) {

	// Gestione degli indici	
	const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	float r = curand_uniform(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < wDim)
		weight[tid] = 0.4 * r;
}


__global__ void initBias(double *bias, const int node, curandState *states) {

	// Gestione degli indici	
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	float r = curand_uniform(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < node)
		bias[tid] = 0.4 * r;
}

void FullyConnected::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	
	// Dimensione matrice dei pesi
	_wDim = prevLayerWidth * prevLayerHeight * prevLayerDepth * _nodes;

	// Dimensione matrice dei pesi in byte
	const unsigned int wBytes = _wDim * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = _nodes * sizeof(double);
	
	// Impostazione buffer che gestisce il printf in Cuda
	size_t sz = 1048576 * 1000;
    cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

	// Allocare le matrici
	CHECK(cudaMalloc((void**) &weight, wBytes));
	CHECK(cudaMalloc((void**) &bias, Bytes));
	CHECK(cudaMalloc((void**) &output, Bytes));
	CHECK(cudaMalloc((void**) &error, Bytes));

	// Rendere i blocchi multipli di 32
	const int aligned = ALIGN_UP(prevLayerWidth * prevLayerHeight);
	
	// Tanti blocchi quanto sono i nodi e la profondità del layer precedente
	dim3 numBlocks(_nodes, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi del livello precedente
	dim3 threadBlocks(aligned, 1, 1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _nodes * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **) &devStates, numRand * sizeof(curandState))); 

	// Inizializzare i weight del livello
#ifdef _WIN32
	initWeight NvCUDA2(numBlocks, threadBlocks) (weight, _wDim, devStates);
#else
	initWeight <<<numBlocks, threadBlocks>>> (weight, _wDim, devStates);
#endif

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	
	const int b = ALIGN_UP(_nodes);

#ifdef _WIN32
	initBias NvCUDA2(1, b) (bias, b, devStates);
#else
	initBias <<<1, b >>> (bias, b, devStates);
#endif

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected::forward_propagation(const double *prev) {
    
    // Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));
	
	// Dimensione righe matrice, le colonne sono i nodi
	const int r = _wDim / _nodes;
	
	// Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;	
	
    CHECK_CUBLAS(
			cublasDgemv(handle, CUBLAS_OP_N, r, _nodes, &alpha, weight, r, prev, 1, &beta, output, 1));
			
    // Somma con il bias
    CHECK_CUBLAS(
            cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));
     
    // DEBUG
    std::vector<double> outputC(_nodes);
	CHECK(cudaMemcpy(&outputC[0], output, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	
	for (auto t : outputC)
		std::cout << t << std::endl;       
    
}

void FullyConnected::back_propagation() {

}
