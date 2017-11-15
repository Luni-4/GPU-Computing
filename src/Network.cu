// Cuda
#include <cuda_runtime.h>

#include <iostream>
#include <iterator>

#include "Common.h"
#include "Network.h"

Network::Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers)
    : _imgDim(0),
	  _iBytes(0),
	  _testError(0),
	  _isPredict(false) {
	  
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
	_imgDim = 6;
#else
	_imgDim = data->getImgDimension();
#endif

	// Quantità di dati da allocare e copiare
	_iBytes = _imgDim * sizeof(double);

	// Allocare il buffer di input della singola coppia (etichetta,immagine)
	CHECK(cudaMalloc((void**)&inputImg, _iBytes));

	// Elabora ogni immagine
	for (int i = 0; i < nImages; i++) {
		int imgIndex = i * _imgDim;

		// Copia dell'immagine corrente nel buffer
		CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), _iBytes, cudaMemcpyDeviceToDevice));

		for (int j = 0; j < 1; j++) {
			forwardPropagation();

			backPropagation(i, learningRate);
		}
	}

	// Cancellare i dati di train dal device
	CHECK(cudaFree(cudaData));
	CHECK(cudaFree(cudaLabels));

}

void Network::predict(Data *data) {

    _isPredict = true;    
    
    //Leggere i dati dal test set
    data->readTestData();
    
    // Caricare i dati in Cuda
	cudaDataLoad(data);
	
	// Numero di esempi nel test set
	const int nImages = data->getLabelSize();
	
	// Ottenere array contenente le labels
	const uint8_t *labels = data->getLabels();
	
	// Definire dimensione dell'array delle predizioni
	_predictions.resize(nImages); 

	// Elabora ogni immagine
	for (int i = 0; i < nImages; i++) {
		int imgIndex = i * _imgDim;

		// Copia dell'immagine corrente nel buffer
		CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), _iBytes, cudaMemcpyDeviceToDevice));
				
		forwardPropagation();
		
		predictLabel(i, labels[i]);		
	}
	
	// Cancellare il vettore contenente le labels
	data->clearLabels();

	// Cancellare i dati di test dal device
	CHECK(cudaFree(cudaData));
	CHECK(cudaFree(cudaLabels));

	// Pulizia della memoria Cuda e reset del device 
	cudaClearAll();
}

inline void Network::predictLabel(const int &index, const uint8_t &label) {
    
    // Calcolare predizione al livello di output
    uint8_t prediction = 0;// _layers.back()->getPredictionIndex();
    
    // Salvare la predizione nell'array
    _predictions[index] = prediction;
    
    // Verificare che la predizione sia corretta
    if(prediction != label)
        _testError++;
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
	CHECK(cudaMemcpy(cudaData, data->getData(), dBytes, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy(cudaLabels, data->getLabels(), lBytes, cudaMemcpyHostToDevice));
    
    // Liberare le label dalla CPU (solo in fase di train) 
	if (!_isPredict)
	    data->clearLabels();
    
    // Liberare le immagini dalla CPU
    data->clearData();
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

void Network::forwardPropagation(void) {

	_layers.front()->forward_propagation(inputImg);

	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
		auto pv = std::prev(it, 1);
		auto *outputPointer = (*pv)->getCudaOutputPointer();
		(*it)->forward_propagation(outputPointer);
	}
}

void Network::backPropagation(const int &target, const double &learningRate) {
	// Caso in cui ci sia solo un livello
	if (_layers.size() == 1) {
		_layers.back()->back_propagation_output(inputImg, cudaLabels, target, learningRate);
		return;
	}

	// Caso in cui i livelli > 1 (tornare indietro di 2 livelli)	    
	auto prevOutput = (*std::prev(_layers.end(), 2))->getCudaOutputPointer();
	_layers.back()->back_propagation_output(prevOutput, cudaLabels, target, learningRate);

	// Back Propagation sui livelli intermedi
	for (auto it = _layers.rbegin() + 1; it != _layers.rend() - 1; ++it) {

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
