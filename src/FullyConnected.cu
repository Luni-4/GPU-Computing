#include <iostream>
#include <vector>
#include <algorithm>

#include "Common.h"
#include "FullyConnected.h"

FullyConnected::FullyConnected(const int &width, const int &height, const ActFctType &a)
: LayerDefinition(width, height, 1, FULLY_CONNECTED, a),
  _isCuda(false)
{

}

FullyConnected::FullyConnected(const int &width, const ActFctType &a)
: LayerDefinition(width, 1, 1, FULLY_CONNECTED, a),
  _isCuda(false)
{

}

FullyConnected::~FullyConnected()
{
	if (_isCuda){
		CHECK(cudaFree(weight));
		CHECK(cudaFree(bias));
		CHECK(cudaFree(output));
		CHECK(cudaFree(error));
	}
}

int FullyConnected::getLayerNodeCount()
{
    return _width * _height;
}


int FullyConnected::getWeightCount(const int &prevLayerNode)
{
    return prevLayerNode * this->getLayerNodeCount();
}


std::vector<double> FullyConnected::getWeight()
{
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wDim * sizeof(double), cudaMemcpyDeviceToHost));
	return wCPU;
}


__global__ void initWeight(double *weight, const int wDim, curandState *states) {	
	
	// Gestione degli indici	
	const int blockId   = blockIdx.y * gridDim.x + blockIdx.x;				
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

void FullyConnected::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth)
{
	// Abilita Cuda
	_isCuda = true; 

	// Numero di nodi livello fully connected
	const int node = this->getLayerNodeCount();
	
	// Dimensione matrice dei pesi
	_wDim = prevLayerWidth * prevLayerHeight * prevLayerDepth * node;

	// Dimensione matrice dei pesi in byte
	const unsigned int wBytes = _wDim * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = node * sizeof(double);
	
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
	dim3 numBlocks(node, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi del livello precedente
	dim3 threadBlocks(aligned, 1, 1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = node * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **) &devStates, numRand * sizeof(curandState))); 

	// Inizializzare i weight del livello
	initWeight<<<numBlocks, threadBlocks>>>(weight, _wDim, devStates);	

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	
	const int b = ALIGN_UP(node);
	
	initBias<<<1, b>>>(bias, b, devStates);
	
	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected::forward_propagation()
{

}

void FullyConnected::back_propagation()
{

}
