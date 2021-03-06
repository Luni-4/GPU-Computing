#pragma once

#include <memory>

#include "Data.h"
#include "LayerDefinition.h" 

class Network {

public:
	Network(const std::vector<std::unique_ptr<LayerDefinition>> &layers);
	~Network();

	void train(Data *data, const int &epoch, const double &learningRate);
	void predict(Data *data);

	void printWeightsOnFile(const std::string &filename, Data *data);

	inline std::vector<uint8_t> getPredictions(void) const { return _predictions; }
	inline int getTestRight(void) const { return _testRight; }

	void cudaClearAll(Data *data);

	void printW();

private:
	void cudaDataLoad(Data *data);
	void cudaInitStruct(Data *data);

	void forwardPropagation(void);
	void backPropagation(const int &target, const double &learningRate);

	inline void setNetwork(Data *data);
	inline void predictLabel(const int &index, const uint8_t &label);
	inline void printNetworkError(const int &nImages);

private:
	std::vector<LayerDefinition*> _layers;
	std::vector<uint8_t> _predictions;
	int _nImages;
	int _imgDim;
	int _iBytes;
	int _testRight;
	bool _isPredict;

	unsigned int _imgIndex;

	// Cuda
	double *cudaData;
	uint8_t *cudaLabels;
	double *inputImg;

};


