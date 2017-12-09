#include "Common.h"

#include <iostream>

#include "ToyInput.h"

ToyInput::ToyInput()
	: _imgWidth(2),
	  _imgHeight(2),
	  _imgDepth(1),
	  _isTrain(false),
	  _isTest(false) {
    
    _imgDim = _imgWidth * _imgHeight * _imgDepth;
}

ToyInput::~ToyInput() {

}

void ToyInput::readTrainData(void) {
	if (_isTrain)
		return;
		
	process();
	
	// Lette le immagini di train ed il test deve essere zero
	_isTrain = true;
	_isTest = false;
}



void ToyInput::readTestData(void) {
	if (_isTest)
		return;
	
    process();

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

inline void ToyInput::process(void) {

    // Pulire i vettori e impostare i dati
    cleanSetData(4);
    
    for(std::size_t i = 0; i < labels.size(); i++)
        for(uint32_t j = 0; j < _imgDim; j++)
	        data[(i * _imgDim) + j] = ((i * _imgDim) + j) + 1;

	std::cout << "\n\nVettore contenente una sola immagine\n\n";
	printVector<double>(data, _imgDepth);
    
    for(std::size_t i = 0; i < labels.size(); i++)
	    labels[i] = i;	

	std::cout << "\n\nVettore contenente l'etichetta dell'immagine\n\n";
	for(std::size_t i = 0; i < labels.size(); i++)
	    std::cout << unsigned(labels[i]) << std::endl;
}
