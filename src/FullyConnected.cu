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
	std::vector<double> wCPU(_wBytes);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}


__global__ void initWeight(double *weight, const unsigned int rDim, const unsigned int cDim, curandState *states) {

	// Gestione degli indici
	const unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
	const unsigned int idy = blockDim.y * blockIdx.y + threadIdx.y;
	const unsigned int tid = idy * cDim  + idx;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	float r;

	// Evitare warp divergence
	int wid = tid / warpSize;

	if (!(wid % 2))
		r = -(curand_uniform(&states[tid]));	// Branch1: thread tid = 0-31, 64-95, ...
	else
		r = curand_uniform(&states[tid]);		// Branch2: thread tid = 32-63, 96-127, ...

	// ricostruzione indice corretto del vettore c
	int i = 2 * (tid % warpSize) + tid/(2*warpSize) + wid % 2;

	if (i < rDim * cDim)
 		weight[i] = 0.4 * r;
 }

void FullyConnected::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth)
{
	// Abilita Cuda
	_isCuda = true;

	// Numero di nodi livello fully connected
	const unsigned int node = this->getLayerNodeCount();

	// Dimensione matrice dei pesi in byte
	_wBytes = prevLayerWidth * prevLayerHeight * node * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = node * sizeof(double);

	// Allocare le matrici
	CHECK(cudaMalloc((void**) &weight, _wBytes));
	CHECK(cudaMalloc((void**) &bias, Bytes));
	CHECK(cudaMalloc((void**) &output, Bytes));
	CHECK(cudaMalloc((void**) &error, Bytes));

	// Rendere i blocchi multipli di 32
	const unsigned int a = ALIGN_UP(prevLayerWidth);
	const unsigned int b = ALIGN_UP(prevLayerHeight);

	// Tanti blocchi quanto sono i nodi e la profondità del layer precedente
	dim3 numBlocks(node,prevLayerDepth,1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi del livello precedente
	dim3 threadBlocks(a,b,1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = node * prevLayerDepth * a * b;

	// Alloca la memoria
	CHECK(cudaMalloc((void **) &devStates, numRand * sizeof(curandState)));

	// Inizializzare i weight
	initWeight<<<numBlocks, threadBlocks>>>(weight, (prevLayerHeight * prevLayerDepth), (prevLayerWidth * node), devStates);

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
