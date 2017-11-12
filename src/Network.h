#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>
#include <memory>

#include "Data.h"
#include "LayerDefinition.h"
 

class Network {

public:
	Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers);
	~Network();

	void train(Data *data, const int &epoch, const double &learningRate);
	std::vector<uint8_t> predict(Data *data);

private:
	void cudaDataLoad(Data *data);
	void cudaInitStruct(Data *data);
	void setNetwork(Data *data);

	void forwardPropagation();
	void backPropagation(const int &target, const double &learningRate);
	
    void cudaClearAll();

private:
	std::vector<LayerDefinition*> _layers;
	double *cudaData;
    uint8_t *cudaLabels;
    double *inputImg;

};


