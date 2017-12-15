#ifdef _WIN32
#include "Windows.h"
#endif

#include <iostream>
#include <fstream>

// Cuda
#include <cuda_runtime.h>

#include "Common.h"
#include "Network.h"

Network::Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers)
	: _nImages(0),
	_imgDim(0),
	_iBytes(0),
	_testRight(0),
	_isPredict(false) {

	for (auto& l : layers)
		_layers.push_back(l.get());
}

Network::~Network() {

}

void Network::train(Data *data, const int &epoch, const double &learningRate) {

	// Definire la rete
	setNetwork(data);

	// Dimensione della singola immagine
	_imgDim = data->getImgDimension();

	// Indice che reperisce la giusta immagine da mandare in input alla rete
	_imgIndex = 0;

	for (int i = 0; i < _nImages; i++) {
		std::cout << i << " of " << _nImages << "\r";

		// Copia dell'immagine corrente nel buffer
		//CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), _iBytes, cudaMemcpyDeviceToDevice));

		forwardPropagation();

		backPropagation(i, learningRate);

		// Incrementare l'indice
		_imgIndex += _imgDim;
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

	// Ottenere array contenente le labels
	const uint8_t *labels = data->getLabels();

	// Definire dimensione dell'array delle predizioni
	_predictions.reserve(_nImages);

	// Indice che reperisce la giusta immagine da mandare in input alla rete
	_imgIndex = 0;

	// Elabora ogni immagine
	for (int i = 0; i < _nImages; i++) {
		std::cout << i << " of " << _nImages << "\r";

		// Copia dell'immagine corrente nel buffer
		//CHECK(cudaMemcpy(inputImg, (cudaData + imgIndex), _iBytes, cudaMemcpyDeviceToDevice));

		forwardPropagation();

		predictLabel(i, labels[i]);

		// Incrementare l'indice
		_imgIndex += _imgDim;
	}

	// Stampare risultati ottenuti in fase di test
	printNetworkError(data->getLabelSize());

	// Cancellare il vettore contenente le labels
	data->clearLabels();

	// Cancellare i dati di test dal device
	CHECK(cudaFree(cudaData));
	CHECK(cudaFree(cudaLabels));

	// Pulizia della memoria Cuda e reset del device 
	cudaClearAll();
}


void Network::cudaDataLoad(Data *data) {

	// Impone a Null i puntatori
	cudaData = NULL;
	cudaLabels = NULL;

	// Numero di esempi presenti
	_nImages = data->getLabelSize();

	const int dBytes = data->getDataSize() * sizeof(double);
	const int lBytes = _nImages * sizeof(uint8_t);

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

	_layers.front()->defineCuda(data->getImgWidth(), data->getImgHeight(), data->getImgDepth());

	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
		auto pv = std::prev(it, 1);
		auto prevWidth = (*pv)->getWidth();
		auto prevHeight = (*pv)->getHeight();
		auto prevDepth = (*pv)->getDepth();
		(*it)->defineCuda(prevWidth, prevHeight, prevDepth);
	}
}

void Network::forwardPropagation(void) {

	_layers.front()->forward_propagation(cudaData + _imgIndex);

	for (auto it = _layers.begin() + 1; it != _layers.end(); ++it) {
		auto pv = std::prev(it, 1);
		auto *outputPointer = (*pv)->getCudaOutputPointer();
		(*it)->forward_propagation(outputPointer);
	}
}

void Network::backPropagation(const int &target, const double &learningRate) {
	// Caso in cui ci sia solo un livello
	if (_layers.size() == 1) {
		_layers.back()->back_propagation_output(cudaData + _imgIndex, cudaLabels, target, learningRate);
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
		auto currentError = (*it)->getCudaErrorPointer();
		auto currentNodes = (*it)->getNodeCount();

		(*fw)->calcError(currentError, currentNodes);

		(*it)->back_propagation(prev, learningRate);
	}

	// Back Propagation al primo livello (solo input precedente a lui)
	auto fw = std::next(_layers.begin(), 1);
	auto currentError = _layers.front()->getCudaErrorPointer();
	auto currentNodes = _layers.front()->getNodeCount();

	(*fw)->calcError(currentError, currentNodes);
	_layers.front()->back_propagation(cudaData + _imgIndex, learningRate);
}

void Network::printWeightsOnFile(const std::string &filename) {

	std::ofstream ofs(filename.c_str(), std::ios::out | std::ios::binary);

	if (!ofs.is_open()) {
		std::cerr << "Errore nell'apertura del file per i pesi!!" << std::endl;
		cudaClearAll();
		exit(1);
	}

	for (auto l : _layers) {
		auto weights = l->getWeights();
		auto weightSize = l->getWeightCount();
		auto bias = l->getBias();
		auto biaSize = l->getNodeCount();

		auto prev = weightSize / biaSize;

		ofs << "Livello" << std::endl << std::endl;

		ofs << "Weights Matrix" << std::endl << std::endl;

		printOnFile(weights, prev, ofs);

		ofs << std::endl << std::endl;

		ofs << "Bias Matrix" << std::endl << std::endl;

		printOnFile(bias, biaSize, ofs);

		ofs << std::endl << std::endl << std::endl;
	}

	ofs.close();
}


inline void Network::setNetwork(Data *data) {
	//Leggere i dati dal training set
	data->readTrainData();

	// Caricare i dati in Cuda
	cudaDataLoad(data);

	// Inizializzare le strutture della rete
	cudaInitStruct(data);
}

inline void Network::predictLabel(const int &index, const uint8_t &label) {

	// Calcolare predizione al livello di output
	int prediction = _layers.back()->getPredictionIndex();

#ifdef DEBUG	
	std::cout << "\n\nPredizione: " << prediction << std::endl;
	std::cout << "Etichetta: " << unsigned(label) << std::endl << std::endl << std::endl;
#endif

	// Salvare la predizione nell'array
	_predictions[index] = prediction;

	// Verificare che la predizione sia corretta
	if (prediction == label)
		_testRight++;
}

inline void Network::printNetworkError(const int &nImages) {

	// Calcolare accuratezza
	double accuracy = (static_cast<double>(_testRight) / nImages) * 100;

	// Stampare numero di errori commessi
	std::cout << "Immagini classificate correttamente: " << _testRight << std::endl;
	std::cout << "Immagini classificate scorrettamente: " << nImages - _testRight << std::endl;
	std::cout << "Accuratezza della rete: " << accuracy << std::endl;
}

inline void Network::cudaClearAll(void) {

	// Liberare la memoria cuda associata ad ogni layer
	for (auto l : _layers)
		l->deleteCuda();

	// Liberare la memoria del device
	CHECK(cudaDeviceReset());

}

void Network::printW() {
	for (auto l : _layers)
		l->printW();
}
