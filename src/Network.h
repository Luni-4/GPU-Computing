#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

#include "Data.h"
#include "LayerDefinition.h"
 

class Network {

public:
	Network(const std::vector<LayerDefinition*> &layers);
	~Network();

	void train(Data *data, const int &epoch, const double &eta, const double &lambda);
	std::vector<uint8_t> predict(Data &data);

private:
	void cudaDataLoad(Data *data);
	void cudaInitStruct(Data *data);

	void forwardPropagation();
	void predictionError();
	void backwardPropagation();

private:
	std::vector<LayerDefinition*> _layers;
	double *cudaData;
    uint8_t *cudaLabels;
    double *inputImg;

};


