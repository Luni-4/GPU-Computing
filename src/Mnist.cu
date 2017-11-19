#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef DEBUG
#include "Common.h"
#endif

#include <iostream>
#include <fstream>
#include <algorithm>

#include "Mnist.h"

Mnist::Mnist(const std::string &filename)
	: _filename(filename),
	_isTrain(false),
	_isTest(false) {
    
    _imgDim = imgHeight * imgWidth;
}

Mnist::~Mnist() {

}

void Mnist::readTrainData(void) {
	if (_isTrain)
		return;

#ifdef TOYINPUT
	data.reserve(6);
	std::fill(data.begin(), data.end(), 1.0);

	std::cout << "\n\nVettore contenente una sola immagine\n\n";
	printVector<double>(data, 3);

	labels.reserve(1);
	std::fill(labels.begin(), labels.end(), 1);

	std::cout << "\n\nVettore contenente l'etichetta dell'immagine\n\n";
	printVector<uint8_t>(labels, 1);
#else
    // Pulire i vettori e impostare i dati
    cleanSetData();
    
	// Leggere le immagini di train
	readImages(train_image_file_mnist);

	// Leggere le etichette di train
	readLabels(train_label_file_mnist);
#endif

	// Lette le immagini di train ed il test deve essere zero
	_isTrain = true;
	_isTest = false;
}



void Mnist::readTestData(void) {
	if (_isTest)
		return;
	
	// Pulire i vettori e impostare i dati
	cleanSetData();

	// Leggere le immagini di test
	readImages(test_image_file_mnist);

	// Leggere le etichette di test
	readLabels(test_label_file_mnist);

	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}


void Mnist::readImages(const std::string &datafile) {
	
	// Vettore contenente i pixel
	std::vector<uint8_t> pixel;
	
	// Numero di immagini
	const int nSize = nImages * _imgDim;

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura dei dati!!" << std::endl;
		exit(1);
	}

	/* 
	   Leggere Magic Number (4 bytes)
	   Leggere numero massimo di immagini (4 bytes)
	   Leggere larghezza immagini (4 bytes)
	   Leggere altezza immagini (4 bytes)	
	*/	
	ifs.ignore(16);	
	
	// Lettura dell'immagine
	std::copy_n(std::istreambuf_iterator<char>(ifs), nSize, std::back_inserter(pixel)); 
	
	// Assegnare i valori a data
	for (std::size_t i = 0; i < pixel.size(); i++)
	    data[i] = (static_cast<double>(pixel[i]) - 127) / 128;
	
	// Chiudere il file
	ifs.close();
}

void Mnist::readLabels(const std::string &datafile) {

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura delle labels!!" << std::endl;
		exit(1);
	}
	
	/* 
	   Leggere Magic Number (4 bytes)
	   Leggere numero massimo di labels (4 bytes)
	*/
	ifs.ignore(8);
	
	// Lettura delle labels
	std::copy_n(std::istreambuf_iterator<char>(ifs), nImages, std::back_inserter(labels)); 

	ifs.close();
}

inline void Mnist::cleanSetData(void) {

	// Ripulire i dati
	clearData();

	// Ripulire le labels
	clearLabels();

	// Dimensione dei dati (training o test)
	data.resize(_imgDim * nImages);
	
	// Dimensione delle labels (training o test)
	labels.reserve(nImages);
}
