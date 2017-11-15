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
	void predict(Data *data);
	
	inline std::vector<uint8_t> getPredictions(void) const { return _predictions; }
	inline int getTestError(void) const { return _testError; } 

private:
	void cudaDataLoad(Data *data);
	void cudaInitStruct(Data *data);
	void setNetwork(Data *data);
	void predictLabel(const int &index, const uint8_t &label);

	void forwardPropagation(void);
	void backPropagation(const int &target, const double &learningRate);
	
    void cudaClearAll();

private:
	std::vector<LayerDefinition*> _layers;
	std::vector<uint8_t> _predictions;
	int _imgDim;
	int _iBytes;
	int _testError;
	bool _isPredict;
	
	// Cuda
	double *cudaData;
    uint8_t *cudaLabels;
    double *inputImg;

};


