//#ifdef DEBUG
#include "Common.h"
//#endif

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iterator>

#include "Mnist.h"

Mnist::Mnist(const std::string& filename)
	: _filename(filename),
	imgWidth(0),
	imgHeight(0),
	isTrain(false),
	isTest(false) {
}

Mnist::~Mnist() {

}

void Mnist::readTrainData() {
	if (isTrain)
		return;

#ifdef DEBUG
	data.resize(6);
	std::fill(data.begin(), data.end(), 1.0);

	std::cout << "\n\nVettore contenente una sola immagine\n\n";
	printVector<double>(data, 3);

	labels.resize(1);
	std::fill(labels.begin(), labels.end(), 1);

	std::cout << "\n\nVettore contenente l'etichetta dell'immagine\n\n";
	printVector<uint8_t>(labels, 1);
#else
	// Leggere le immagini di train
	readImages(train_image_file_mnist);

	// Leggere le etichette di train
	readLabels(train_label_file_mnist);

	// Lette le immagini di train ed il test deve essere zero
	isTrain = true;
	isTest = false;
#endif
}



void Mnist::readTestData() {
	if (isTest)
		return;

	// Leggere le immagini di test
	readImages(test_image_file_mnist);

	// Leggere le etichette di test
	readLabels(test_label_file_mnist);

	// Lette le immagini di test ed il train deve essere zero
	isTest = true;
	isTrain = false;
}


void Mnist::readImages(const std::string& datafile) {
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
	ifs.read(reinterpret_cast<char *>(&imgWidth), sizeof(imgWidth));
	imgWidth = flipBytes(imgWidth);

	// Leggere altezza immagini
	ifs.read(reinterpret_cast<char *>(&imgHeight), sizeof(imgHeight));
	imgHeight = flipBytes(imgHeight);

	// Lettura dell'immagine    
	std::vector<uint8_t> pixel((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

	// Impostare data della stessa dell'input
	data.resize(pixel.size());

	// Usare una lambda per convertire i pixel nell'intervallo richiesto
	auto convert = [](const uint8_t &d) -> double { return (static_cast<double>(d) - 127) / 128; };
	std::transform(pixel.begin(), pixel.end(), data.begin(), convert);

	ifs.close();
}

void Mnist::readLabels(const std::string& datafile) {
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


inline uint32_t Mnist::flipBytes(const uint32_t& n) {

	uint32_t b0, b1, b2, b3;

	b0 = (n & 0x000000ff) << 24u;
	b1 = (n & 0x0000ff00) << 8u;
	b2 = (n & 0x00ff0000) >> 8u;
	b3 = (n & 0xff000000) >> 24u;

	return (b0 | b1 | b2 | b3);
}


