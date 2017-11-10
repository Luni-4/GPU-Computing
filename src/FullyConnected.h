#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "LayerDefinition.h"

class FullyConnected : public LayerDefinition {

public:

	FullyConnected(const int &width, const int &height, const ActFctType &a);
	FullyConnected(const int &width, const ActFctType &a);
	~FullyConnected();

	int getLayerNodeCount() override;
	int getWeightCount(const int &prevLayerNode) override;
	std::vector<double> getWeights() override;
	std::vector<double> getBias() override;

	void forward_propagation(const double *prev) override;

	void back_propagation() override;
	void back_propagation_output(const double *prev, const uint8_t *labels, const int &target, const double &learningRate) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda() override;


private:
	int _wDim;
	int _nodes;
	int _prevLayerDim;
	int _alignedNodes;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori

	// Handle per cuBlas
	cublasHandle_t handle;

};
