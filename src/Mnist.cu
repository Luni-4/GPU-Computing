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
	_imgWidth(0),
	_imgHeight(0),
	_isTrain(false),
	_isTest(false) {
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

	// Leggere le immagini di test
	readImages(test_image_file_mnist);

	// Leggere le etichette di test
	readLabels(test_label_file_mnist);

	// Lette le immagini di test ed il train deve essere zero
	_isTest = true;
	_isTrain = false;
}


void Mnist::readImages(const std::string &datafile) {
	// Eliminare il contenuto di data
	data.clear();

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura dei dati!!" << std::endl;
		exit(1);
	}

	uint32_t temp;

	// Leggere Magic Number
	ifs.read(reinterpret_cast<char *>(&temp), sizeof(temp));

	// Leggere numero massimo di immagini
	ifs.read(reinterpret_cast<char *>(&temp), sizeof(temp));

	// Leggere larghezza immagini
	ifs.read(reinterpret_cast<char *>(&_imgWidth), sizeof(_imgWidth));
	_imgWidth = flipBytes(_imgWidth);

	// Leggere altezza immagini
	ifs.read(reinterpret_cast<char *>(&_imgHeight), sizeof(_imgHeight));
	_imgHeight = flipBytes(_imgHeight);

	// Lettura delle immagini
	std::transform(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>(), std::back_inserter(data),
		[](const uint8_t &d) -> double { return static_cast<uint8_t>(d); });

	// Conversione delle immagini
	std::for_each(data.begin(), data.end(), [](double &d) { d = ((d - 127) / 128); });

	ifs.close();
}

void Mnist::readLabels(const std::string &datafile) {
	// Eliminare il contenuto di labels
	labels.clear();

	std::ifstream ifs((_filename + datafile).c_str(), std::ios::in | std::ios::binary);

	if (!ifs.is_open()) {
		std::cerr << "Errore nell'apertura delle labels!!" << std::endl;
		exit(1);
	}


	uint32_t temp;

	// Leggere Magic Number
	ifs.read(reinterpret_cast<char *>(&temp), sizeof(temp));

	// Leggere numero massimo di labels
	ifs.read(reinterpret_cast<char *>(&temp), sizeof(temp));

	// Lettura delle labels
	std::copy((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>(), std::back_inserter(labels));

	ifs.close();
}


inline uint32_t Mnist::flipBytes(const uint32_t &n) {

	uint32_t b0, b1, b2, b3;

	b0 = (n & 0x000000ff) << 24u;
	b1 = (n & 0x0000ff00) << 8u;
	b2 = (n & 0x00ff0000) >> 8u;
	b3 = (n & 0xff000000) >> 24u;

	return (b0 | b1 | b2 | b3);
}
