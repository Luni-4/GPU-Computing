#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "LayerDefinition.h"

class Convolutional : public LayerDefinition {
public:
	Convolutional(const int &filterWidth, const int &filterDepth, const int &stride, const ActFctType &a);
	~Convolutional();

	int getLayerNodeCount() override;
	int getWeightCount(const int &prevLayerNode) override;
	std::vector<double> getWeights() override;
	std::vector<double> getBias() override;

	void forward_propagation(const double *prev) override;

	void back_propagation() override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda() override;

private:
	int _wDim;
	int _nodes;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori

	int _filterWidth;
	int _stride;
	int _padding;

	// Handle per cuBlas
	cublasHandle_t handle;

	int calcOutput(int prevLayerWidth, bool withPadding);
};

