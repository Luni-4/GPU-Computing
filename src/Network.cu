// Cuda
#include <cuda_runtime.h>

#include <iostream>
#include <iterator>

#include "Common.h"
#include "Network.h"

Network::Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers) {
	for (auto& l : layers)
		_layers.push_back(l.get());
}

Network::~Network() {

}

void Network::train(Data *data, const int &epoch, const double &learningRate) {
	
    // Definire la rete
    setNetwork(data);
    
    // Numero di esempi nel training set
	const int nImages = 1;// data->getLabelSize();

	// Dimensione della singola immagine
#ifdef DEBUG
    const int imgDim = 6;
#else
	const int imgDim = data->getImgDimension();
#endif

	// Quantità di dati da allocare e copiare
	const int iBytes = imgDim * sizeof(double);

	// Allocare il buffer di input della singola coppia (etichetta,immagine)
	CHECK(cudaMalloc((void**)&inputImg, iBytes));
	
	// Elabora ogni immagine
	for(int i = 0; i < nImages; i++) {	
        int imgIndex = i * imgDim;
        
	    // Copia dell'immagine corrente nel buffer
	    CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), iBytes, cudaMemcpyDeviceToDevice));

	    for(int j = 0; j < 1; j++) {
	        forwardPropagation();

            backPropagation(i, learningRate);
        }
    }

	// cancellare i dati di train dal device
	CHECK(cudaFree(cudaData));
	CHECK(cudaFree(cudaLabels));

	// DA SPOSTARE NELLA FUNZIONE CHE FA IL TEST
	cudaClearAll();

}

void Network::setNetwork(Data *data) {
    //Leggere i dati dal training set
	data->readTrainData();

	// Caricare i dati in Cuda
	cudaDataLoad(data);

	// Inizializzare le strutture della rete
	cudaInitStruct(data);
}


void Network::cudaDataLoad(Data *data) {
	const int dBytes = data->getDataSize() * sizeof(double);
	const int lBytes = data->getLabelSize() * sizeof(uint8_t);

	// Allocare le matrici
	CHECK(cudaMalloc((void**)&cudaData, dBytes));
	CHECK(cudaMalloc((void**)&cudaLabels, lBytes));

	// Passare i dati
	CHECK(cudaMemcpy(cudaData, data->getCudaData(), dBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(cudaLabels, data->getCudaLabels(), lBytes, cudaMemcpyHostToDevice));

	// Liberare i dati dalla CPU
	data->clearDataCPU();
}

void Network::cudaInitStruct(Data *data) {

#ifdef DEBUG
    _layers.front()->defineCuda(2, 1, 3);
#else
	_layers.front()->defineCuda(data->getImgWidth(), data->getImgHeight(), data->getImgDepth());
#endif

	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
		const int prevWidth = (*it - 1)->getWidth();
		const int prevHeight = (*it - 1)->getHeight();
		const int prevDepth = (*it - 1)->getDepth();

		(*it)->defineCuda(prevWidth, prevHeight, prevDepth);
	}
}

void Network::forwardPropagation() {

	_layers.front()->forward_propagation(inputImg);
	
	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
        const double *outputPointer = (*it - 1)->getCudaOutputPointer();
        (*it)->forward_propagation(outputPointer);
	}
}

void Network::backPropagation(const int &target, const double &learningRate) {
    
    // Caso in cui ci sia solo un livello
    if(_layers.size() == 1) {    
	    _layers.back()->back_propagation_output(inputImg, cudaLabels, target, learningRate);
	    return;
	}
	
	// Caso in cui i livelli > 1	    
    auto prevOutput = (*_layers.end() - 1)->getCudaOutputPointer(); 
    _layers.back()->back_propagation_output(prevOutput, cudaLabels, target, learningRate);
    
    // Back Propagation sui livelli intermedi
    for (auto it = _layers.rbegin() + 1; it != _layers.rend() -1; ++it) {
        auto prev = (*it + 1)->getCudaOutputPointer();
        auto forwardWeight = (*it - 1)->getCudaWeightPointer(); 
        auto forwardError = (*it - 1)->getCudaErrorPointer();
        auto forwardNodes = (*it - 1)->getNodeCount();
        (*it)->back_propagation(prev, forwardWeight, forwardError, forwardNodes, learningRate);
	}
	
	// Back Propagation al primp livello (solo input precedente a lui)
	 auto forwardWeight = (*_layers.begin() + 1)->getCudaWeightPointer(); 
     auto forwardError = (*_layers.begin() + 1)->getCudaErrorPointer();
     auto forwardNodes = (*_layers.begin() + 1)->getNodeCount();
	_layers.front()->back_propagation(inputImg, forwardWeight, forwardError, forwardNodes, learningRate);
	    
}

void Network::cudaClearAll() {

	// Liberare il buffer contenente le immagini in input alla rete
	CHECK(cudaFree(inputImg));

	// Liberare la memoria cuda associata ad ogni layer
	for (auto l : _layers)
		l->deleteCuda();

	// Liberare la memoria del device
	CHECK(cudaDeviceReset());

}
