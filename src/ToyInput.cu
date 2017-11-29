#include "Common.h"

#include <iostream>

#include "ToyInput.h"

ToyInput::ToyInput()
	: _imgWidth(2),
	  _imgHeight(1),
	  _imgDepth(3),
	  _isTrain(false),
	  _isTest(false) {
    
    _imgDim = _imgWidth * _imgHeight * _imgDepth;
}

ToyInput::~ToyInput() {

}

void ToyInput::readTrainData(void) {
	if (_isTrain)
		return;
		
	// Pulire i vettori e impostare i dati
    cleanSetData(1);

	std::fill(data.begin(), data.end(), 1.0);

	std::cout << "\n\nVettore contenente una sola immagine\n\n";
	printVector<double>(data, _imgDepth);

	std::fill(labels.begin(), labels.end(), 1);

	std::cout << "\n\nVettore contenente l'etichetta dell'immagine\n\n";
	printVector<uint8_t>(labels, 1);    
	
	// Lette le immagini di train ed il test deve essere zero
	_isTrain = true;
	_isTest = false;
}



void ToyInput::readTestData(void) {
	if (_isTest)
		return;
	
	// Pulire i vettori e impostare i dati
	cleanSetData(1);

	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}

inline void ToyInput::cleanSetData(const uint32_t &nImages) {

	// Ripulire i dati
	clearData();

	// Ripulire le labels
	clearLabels();

	// Dimensione dei dati (training o test)
	data.resize(_imgDim * nImages);
	
	// Dimensione delle labels (training o test)
	labels.resize(nImages);
}
