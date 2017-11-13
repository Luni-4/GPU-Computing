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
<<<<<<< HEAD
	
    // Definire la rete
    setNetwork(data);
    
    // Numero di esempi nel training set
	const int nImages = 1;// data->getLabelSize();
=======
	//Leggere i dati dal training set
	data->readTrainData();

	// Caricare i dati in Cuda
	cudaDataLoad(data);

	// Inizializzare le strutture della rete
	cudaInitStruct(data);

	// Numero di esempi nel training set
	const int nImages = data->getLabelSize();
>>>>>>> convolutional network with incomplete forward propagation

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
<<<<<<< HEAD
	for(int i = 0; i < nImages; i++) {	
        int imgIndex = i * imgDim;
        
	    // Copia dell'immagine corrente nel buffer
	    CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), iBytes, cudaMemcpyDeviceToDevice));

	    for(int j = 0; j < 1; j++) {
	        forwardPropagation();

            backPropagation(i, learningRate);
        }
    }
=======
	//for(int i = 0; i < nImages; i++)
	//{
	int imgIndex = i * imgDim;

	// Copia dell'immagine corrente nel buffer
	CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), iBytes, cudaMemcpyDeviceToDevice));

	//for(int j = 0; j < epoch; j++)
	//{
		// Forward_propagation per ogni livello
	forwardPropagation();
	return;

	// Calcolo dell'errore per ogni livello
	//error();

		// Backward_propagation per ogni livello
	backPropagation(i, learningRate);
	//}

	//}
>>>>>>> convolutional network with incomplete forward propagation

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
	    auto pv = std::prev(it, 1);
		auto prevWidth = (*pv)->getWidth();
		auto prevHeight = (*pv)->getHeight();
		auto prevDepth = (*pv)->getDepth();
		(*it)->defineCuda(prevWidth, prevHeight, prevDepth);
	}
}

void Network::forwardPropagation() {

	_layers.front()->forward_propagation(inputImg);
	
	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
	    auto pv = std::prev(it, 1);
        auto *outputPointer = (*pv)->getCudaOutputPointer();
        (*it)->forward_propagation(outputPointer);
	}
}

void Network::backPropagation(const int &target, const double &learningRate) {
<<<<<<< HEAD
    
    // Caso in cui ci sia solo un livello
    if(_layers.size() == 1) {    
	    _layers.back()->back_propagation_output(inputImg, cudaLabels, target, learningRate);
	    return;
	}
	
	// Caso in cui i livelli > 1 (tornare indietro di 2 livelli)	    
    auto prevOutput = (*std::prev(_layers.end(), 2))->getCudaOutputPointer();     
    _layers.back()->back_propagation_output(prevOutput, cudaLabels, target, learningRate);
    
    // Back Propagation sui livelli intermedi
    for (auto it = _layers.rbegin() + 1; it != _layers.rend() -1; ++it) {
        
        auto pv = std::next(it, 1);
        auto fw = std::prev(it, 1);
        
        auto prev = (*pv)->getCudaOutputPointer();
        auto forwardWeight = (*fw)->getCudaWeightPointer(); 
        auto forwardError = (*fw)->getCudaErrorPointer();
        auto forwardNodes = (*fw)->getNodeCount();
        (*it)->back_propagation(prev, forwardWeight, forwardError, forwardNodes, learningRate);
	}
	
	// Back Propagation al primo livello (solo input precedente a lui)
	auto fw = std::next(_layers.begin(), 1);
	auto forwardWeight = (*fw)->getCudaWeightPointer(); 
    auto forwardError = (*fw)->getCudaErrorPointer();
    auto forwardNodes = (*fw)->getNodeCount();
	_layers.front()->back_propagation(inputImg, forwardWeight, forwardError, forwardNodes, learningRate);
	    
=======


	// TODO Non è inputImg ma output livello i - 1
	_layers.back()->back_propagation_output(inputImg, cudaLabels, target, learningRate);


	/*for (auto it = _layers.rbegin() - 1; it != _layers.rend(); ++it) {
		(*it)->backward_propagation();
	}*/
>>>>>>> convolutional network with incomplete forward propagation
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
