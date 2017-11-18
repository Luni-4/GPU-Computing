#ifdef _WIN32
#include "Windows.h"
#endif

#ifdef DEBUG
#include "Common.h"
#endif

#include <cstdint>
#include <iostream>
#include <fstream>
#include <algorithm>

#include "Cifar.h"

Cifar::Cifar(const std::string &filename, const bool &isCifar10)
	: _filename(filename),
	_imgWidth(32),
	_imgHeight(32),
	_imgDepth(3),
	_isCifar10(isCifar10),
	_isTrain(false),
	_isTest(false) {

	_imgDim = _imgWidth * _imgHeight * _imgDepth;
	_pixel.resize(_imgDim);

}


Cifar::~Cifar() {

}

inline void Cifar::cleanSetData(const int &set) {

	// Ripulire i dati
	clearData();

	// Ripulire le labels
	clearLabels();

	// Dimensione dei dati (training o test)
	data.resize(_imgDim * set);

	// Dimensione delle labels (training o test)
	labels.resize(set);
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

	// Svuotare vettore temporaneo
	_pixel.clear();

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

	// Svuotare vettore temporaneo
	_pixel.clear();

	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}


inline void Cifar::readCifarTrain10(const std::vector<std::string> &datafile) {

	// Lettura dei 5 file binari
	for (auto s : datafile)
		// Lettura di un file per volta
		readCifar10(s);
}


void Cifar::readCifar10(const std::string &datafile) {

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura del file di Cifar 10" << datafile << "!!" << std::endl;
		exit(1);
	}

	for (int i = 0; i < cifarTestDim; i++) {

		// Leggere label fine
		ifs.read(reinterpret_cast<char *>(&labels[i]), sizeof(labels[0]));

		// Lettura dell'immagine    
		ifs.read(reinterpret_cast<char *>(&_pixel), _imgDim);

		// Usare una lambda per convertire i _pixel nell'intervallo richiesto
		auto convert = [](const uint8_t &d) -> double { return (static_cast<double>(d) - 127) / 128; };
		std::transform(_pixel.begin(), _pixel.end(), data.end(), convert);
	}

	ifs.close();
}

void Cifar::readCifar100(const std::string &datafile, const int &iterations) {

	uint8_t temp;

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura del file di Cifar 100" << datafile << "!!" << std::endl;
		exit(1);
	}

	for (int i = 0; i < iterations; i++) {

		// Leggere label coarse
		ifs.read(reinterpret_cast<char *>(&temp), sizeof(temp));

		// Leggere label fine
		ifs.read(reinterpret_cast<char *>(&labels[i]), sizeof(labels[0]));

		// Lettura dell'immagine    
		ifs.read(reinterpret_cast<char *>(&_pixel), _imgDim);

		// Usare una lambda per convertire i _pixel nell'intervallo richiesto
		auto convert = [](const uint8_t &d) -> double { return (static_cast<double>(d) - 127) / 128; };
		std::transform(_pixel.begin(), _pixel.end(), data.end(), convert);
	}

	ifs.close();
}
