#include <cstdio>
#include <vector>


// Cuda
#include <cuda_runtime.h>

#include "Common.h"
#include "Network.h"

Network::Network(const std::vector<LayerDefinition*>& layers)
	: _layers(layers) {

}

Network::~Network() {

}

void Network::train(Data &data, const int &epoch, const double &eta, const double &lambda) {
	//Leggere i dati dal training set
	data.readTrainData();

	// Caricare i dati in Cuda
	cudaDataLoad(data);

	// Inizializzare le strutture della rete
	cudaInitStruct();



}


void Network::cudaDataLoad(Data &data) {
	const int dBytes = data.getDataSize() * sizeof(double);
	const int lBytes = data.getLabelSize() * sizeof(uint8_t);

	// Allocare le matrici
	CHECK(cudaMalloc((void**)&cudaData, dBytes));
	CHECK(cudaMalloc((void**)&cudaLabels, lBytes));

	// Passare i dati
	CHECK(cudaMemcpy(cudaData, data.getCudaData(), dBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(cudaLabels, data.getCudaLabels(), lBytes, cudaMemcpyHostToDevice));

	// Liberare i dati dalla CPU
	data.clearDataCPU();
}

void Network::cudaInitStruct() {

}



