#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef TOYINPUT
#include "Common.h"
#endif

#include <iostream>
#include <fstream>
#include <iterator>
#include <algorithm>

#include "Mnist.h"

Mnist::Mnist(const std::string &filePath)
	: _filePath(filePath),
	_isTrain(false),
	_isTest(false) {

	_imgDim = imgHeight_m * imgWidth_m;
}

Mnist::~Mnist() {

}

void Mnist::readTrainData(void) {
	if (_isTrain)
		return;

	// Pulire i vettori e impostare i dati
	cleanSetData(nTrain_m);

	// Leggere le immagini di train
	readImages(train_image_file_mnist, nTrain_m);

	// Leggere le etichette di train
	readLabels(train_label_file_mnist, nTrain_m);

	// Lette le immagini di train ed il test deve essere zero
	_isTrain = true;
	_isTest = false;
}



void Mnist::readTestData(void) {
	if (_isTest)
		return;

	// Pulire i vettori e impostare i dati
	cleanSetData(nTest_m);

	// Leggere le immagini di test
	readImages(test_image_file_mnist, nTest_m);

	// Leggere le etichette di test
	readLabels(test_label_file_mnist, nTest_m);

	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}


void Mnist::readImages(const std::string &fileName, const uint32_t &nImages) {
	// Vettore contenente i pixel
	std::vector<uint8_t> pixel;

	// Numero di immagini
	const int nSize = nImages * _imgDim;

	std::ifstream ifs((_filePath + fileName).c_str(), std::ios::in | std::ios::binary);

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
		data[i] = static_cast<double>(pixel[i]) / 255.0; //(static_cast<double>(pixel[i]) - 127) / 128;

	// Chiudere il file
	ifs.close();
}

void Mnist::readLabels(const std::string &fileName, const uint32_t &nImages) {

	std::ifstream ifs((_filePath + fileName).c_str(), std::ios::in | std::ios::binary);

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

inline void Mnist::cleanSetData(const uint32_t &nImages) {

	// Ripulire i dati
	clearData();

	// Ripulire le labels
	clearLabels();

	// Dimensione dei dati (training o test)
	data.resize(_imgDim * nImages);

	// Dimensione delle labels (training o test)
	labels.reserve(nImages);
}
