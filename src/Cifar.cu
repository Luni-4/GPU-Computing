#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef DEBUG
#include "Common.h"
#endif

#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

#include "Cifar.h"

Cifar::Cifar(const std::string &filePath, const bool &isCifar10)
	: _filePath(filePath),
	  _isCifar10(isCifar10),
	  _isTrain(false),
	  _isTest(false) {
	  
	  _imgDim = imgWidth_c * imgHeight_c * imgDepth_c;
}


Cifar::~Cifar() {

}

inline void Cifar::cleanSetData(const int &set) {

	// Ripulire i dati
	clearData();

	// Ripulire le labels
	clearLabels();

	// Dimensione dei dati (training o test)
	data.reserve(_imgDim * set);
	
	// Dimensione delle labels (training o test)
	labels.reserve(set);
}

void Cifar::readTrainData(void) {
	if (_isTrain)
		return;

	cleanSetData(cifarTrainDim);

	if (_isCifar10)
		// Leggere le immagini e le etichette di train di Cifar 10
		readCifarTrain10(train_cifar10);
	else
	    // Leggere le immagini e le etichette di train di Cifar 100
	    readCifar100(train_cifar100, cifarTrainDim);  


	// Lette le immagini di train ed il test deve essere zero
	_isTrain = true;
	_isTest = false;
}



void Cifar::readTestData(void) {
	if (_isTest)
		return;

	cleanSetData(cifarTestDim);

	if (_isCifar10)
		// Leggere le immagini e le etichette di test di Cifar 10
		readCifar10(test_cifar10);
	else
	    // Leggere le immagini e le etichette di test di Cifar 100
	    readCifar100(test_cifar100, cifarTestDim);
	    
	    
	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}


inline void Cifar::readCifarTrain10(const std::vector<std::string> &fileName) {

	// Lettura dei 5 file binari
	for (auto f : fileName)
		// Lettura di un file per volta
		readCifar10(f);
}


void Cifar::readCifar10(const std::string &fileName) {
    
    int index = 0;
    
    std::ifstream ifs((_filePath + fileName).c_str(), std::ios::in | std::ios::binary); 

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura del file di Cifar 10 " << fileName << "!!" << std::endl;
		exit(1);
	}

	// Leggere il file
	std::vector<uint8_t> dataFile((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	
	// Effettuare le varie operazioni
	for(uint32_t i = 0; i < cifarTestDim; i++) {
	    
	    std::copy_n(dataFile.begin() + index, 1, std::back_inserter(labels));
	    
	    index += 1 + _imgDim;
	   
	    // Lettura delle immagini
	    std::transform(dataFile.begin() + (index - _imgDim), dataFile.begin() + index, std::back_inserter(data),
	               [](const uint8_t &d) -> double { return static_cast<double>(d) / 255.0; }); //(d - 127) / 128; });    
    }
    
    ifs.close(); 

}

void Cifar::readCifar100(const std::string &fileName, const int &iterations) {

    int index = 1;
    
    std::ifstream ifs((_filePath + fileName).c_str(), std::ios::in | std::ios::binary); 

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura del file di Cifar 10 " << fileName << "!!" << std::endl;
		exit(1);
	}

	// Leggere il file
	std::vector<uint8_t> dataFile((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
	
	// Suddividere il file in labels e data
	for(int i = 0; i < iterations; i++) {
	    
	    std::copy_n(dataFile.begin() + index, 1, std::back_inserter(labels));
	    
	    index += (2 + _imgDim);
	   
	    // Lettura delle immagini
	    std::transform(dataFile.begin() + (index - _imgDim - 1), dataFile.begin() + (index - 1), std::back_inserter(data),
	               [](const uint8_t &d) -> double { return static_cast<double>(d) / 255.0; }); //(d - 127) / 128; }); 
	       
    }
    
    ifs.close();   
} 
