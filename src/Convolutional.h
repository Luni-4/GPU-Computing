#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "LayerDefinition.h"

class Convolutional : public LayerDefinition {
public:
	Convolutional(const int &filterWidth, const int &filterDepth, const int &stride, const ActFctType &a);
	~Convolutional();

	int getNodeCount() const override { return _nodes; }
	int getWeightCount() const override { return _wDim; }
	std::vector<double> getWeights() override;
	std::vector<double> getBias() override;

	void forward_propagation(const double *prev) override;

	void back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) override;
	void back_propagation_output(const double *prev, const uint8_t *labels, const int &target, const double &learningRate) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda() override;

	double* getCudaOutputPointer() const override { return output; }
	double* getCudaWeightPointer() const override { return weight; }
	double* getCudaErrorPointer() const override { return error; }

private:
	int _wDim;
	int _nodes;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori

	int _prevLayerWidth;

	int _filterWidth;
	int _stride;
	int _padding;

	int _alignedNodes;

	// Handle per cuBlas
	cublasHandle_t handle;

	int _calcOutput(int prevLayerWidth, bool withPadding);
};

