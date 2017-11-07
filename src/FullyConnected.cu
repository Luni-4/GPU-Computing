#include <iostream>
#include <vector>
#include <algorithm>


// Cuda Library
// Cuda
#include <curand_kernel.h>

// Cuda Kernel
#include "Kernel.h"

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

}

int FullyConnected::getLayerNodeCount() {
	return _nodes;
}


int FullyConnected::getWeightCount(const int &prevLayerNode) {
	return prevLayerNode * _nodes;
}


std::vector<double> FullyConnected::getWeights() {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wDim * sizeof(double), cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> FullyConnected::getBias() {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
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
	CHECK(cudaMalloc((void**)&weight, wBytes));
	CHECK(cudaMalloc((void**)&bias, Bytes));
	CHECK(cudaMalloc((void**)&output, Bytes));
	CHECK(cudaMalloc((void**)&error, Bytes));

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
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandState)));

	// Inizializzare i weight del livello
#ifdef _WIN32
	initWeight NvCUDA2(numBlocks, threadBlocks) (weight, _wDim, devStates);
#else
	initWeight <<< numBlocks, threadBlocks >>> (weight, _wDim, devStates);
#endif

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
    
    // Convertire il numero di nodi in un multiplo di 32
	const int b = ALIGN_UP(_nodes);

#ifdef _WIN32
	initBias NvCUDA2(1, b) (bias, _nodes, devStates);
#else
	initBias <<<1, b>>> (bias, _nodes, devStates);
#endif    

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	
#ifdef DEBUG
    std::cout << "\n\nValore dei pesi\n\n";
	printFromCuda(weight, _wDim);
	std::cout << "\n\nValore dei bias\n\n";
	printFromCuda(bias, _nodes);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected::forward_propagation(const double *prev) {

	// Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));

	// Dimensione righe matrice, le colonne sono i nodi
	const int r = _wDim / _nodes;
	
	// Convertire il numero di nodi in un multiplo di 32
	const int b = ALIGN_UP(_nodes);

	// Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;

	CHECK_CUBLAS(
		cublasDgemv(handle, CUBLAS_OP_N, r, _nodes, &alpha, weight, r, prev, 1, &beta, output, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());  

	// Somma con il bias
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());  

#ifdef DEBUG
    std::cout << "\n\nOutput dei nodi\n\n";
	printFromCuda(output, _nodes);
#endif

    // Applicare funzione di attivazione
    if(_a == RELU)
    
#ifdef _WIN32
	    actRelu NvCUDA2(1, b) (output, _nodes);
#else
	    actRelu <<<1, b>>> (output, _nodes);
#endif 
    else if(_a == SIGMOID)

#ifdef _WIN32
	    actSigmoid NvCUDA2(1, b) (output, _nodes);
#else
	    actSigmoid <<<1, b>>> (output, _nodes);
#endif

    else

#ifdef _WIN32
	    actTanh NvCUDA2(1, b) (output, _nodes);
#else
	    actTanh <<<1, b>>> (output, _nodes);
#endif 
    
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());    
}

void FullyConnected::back_propagation() {

}

void FullyConnected::deleteCuda() {

	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
}
