#include <iostream>
#include <vector>
#include <algorithm>

// Cuda
#include <curand_kernel.h>

#include "Common.h"
#include "Convolutional.h"

#ifdef _WIN32
#include "Windows.h"
#endif

Convolutional::Convolutional(const int &filterWidth, const int &depth, const int &stride, const ActFctType &a)
	: LayerDefinition(0, 0, depth, CONVOLUTIONAL, a) {
	this->_depth = depth;
	this->_filterWidth = filterWidth;
	this->_stride = stride;
	this->_padding = 0;
}

Convolutional::~Convolutional() {
}

int Convolutional::getLayerNodeCount() {
	return 0;
}

int Convolutional::getWeightCount(const int & prevLayerNode) {
	return 0;
}

std::vector<double> Convolutional::getWeights() {
	return std::vector<double>();
}

std::vector<double> Convolutional::getBias() {
	return std::vector<double>();
}

void Convolutional::forward_propagation(const double * prev) {
}

void Convolutional::back_propagation() {
}

__global__ void initWeightConv(double *weight, const int wDim, curandState *states) {
	// Gestione degli indici	
	const int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterr� il valore casuale
	float r = curand_uniform(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < wDim)
		weight[tid] = 0.4 * r;
}

__global__ void initBiasConv(double *bias, const int node, curandState *states) {

	// Gestione degli indici	
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterr� il valore casuale
	float r = curand_uniform(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < node)
		bias[tid] = 0.4 * r;
}

void Convolutional::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	//numero di nodi dipende da filtro e nodi livello precedente
	//width
	_width = calcOutput(prevLayerWidth, false);
	//height
	_height = calcOutput(prevLayerHeight, false);
	//depth = numero di filtri

	this->_nodes = _width * _height * _depth;

	std::cout << _width << " - " << _height << " - " << _depth << std::endl;

	// Dimensione matrice dei pesi
	_wDim = _filterWidth * _filterWidth * prevLayerDepth * _depth;

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
	const int aligned = ALIGN_UP(_filterWidth * _filterWidth);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi compongono i filtri
	dim3 threadBlocks(aligned, 1, 1);

	// Tanti blocchi quanto sono i filtri e la profondit� del layer precedente
	dim3 numBlocks(_depth, prevLayerDepth, 1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _depth * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandState)));

	// Inizializzare i weight del livello
#ifdef _WIN32
	initWeightConv NvCUDA2(numBlocks, threadBlocks) (weight, _wDim, devStates);
#else
	initWeightConv << <numBlocks, threadBlocks >> > (weight, _wDim, devStates);
#endif

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	const int b = ALIGN_UP(_nodes);

#ifdef _WIN32
	initBiasConv NvCUDA2(1, b) (bias, b, devStates);
#else
	initBiasConv << <1, b >> > (bias, b, devStates);
#endif

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}

void Convolutional::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
}

int Convolutional::calcOutput(int prevLayerWidth, bool withPadding) {
	//PER ORA NON CONSIDERATO CASO IN CUI SI GENERANO ERRORI (padding numero non intero, filtro pi� grande dell'input, stride che non combacia, ecc)
	if (withPadding) {
		_padding = (_filterWidth - 1) / 2;
		return prevLayerWidth;
	}
	return ((prevLayerWidth - _filterWidth + 0) / _stride) + 1;
}