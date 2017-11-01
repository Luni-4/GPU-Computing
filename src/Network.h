/*#pragma once

#include <cstdio>
#include <string>
 

class Network {

public:
	Network(const std::vector<layerDefinition*>& layers);
	~Network();

	void train(const Data& data, const int epoch, const double eta, const double lambda);
	std::vector<uint8_t> predict(const Data& data);

private:
	void cudaDataLoad(const Data& data);
	void cudaInitWeight();
	void cudaGetWeight(); // nel caso in cui si volessero salvare
	void cudaCleanAll();

	void forward_propagation();
	void predictionError();
	void backward_propagation();

private:
	std::vector<layerDefinition*> _layers;

};*/


