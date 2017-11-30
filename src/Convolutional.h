#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "LayerDefinition.h"

class Convolutional : public LayerDefinition {
public:
	Convolutional(const int &filterWidth, const int &filterDepth, const int &stride, const ActFctType &a);
	~Convolutional();

	int getNodeCount(void) const override { return _nodes; }
	int getWeightCount(void) const override { return _wDim; }
	std::vector<double> getWeights(void) override;
	std::vector<double> getBias(void) override;
	uint8_t getPredictionIndex(void) override;

	void forward_propagation(const double *prevOutput) override;

	void back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) override;
	void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda(void) override;

	double* getCudaOutputPointer() const override { return output; }
	double* getCudaWeightPointer() const override { return weight; }
	double* getCudaErrorPointer() const override { return error; }

private:
	int _wDim;
	int _wBytes;
	int _nodes;
	int _prevLayerWidth;
	int _prevLayerDepth;
	int _alignedNodes;

	//Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori
	double *temp; // Matrice temporanea usata per l'aggiornamento dei pesi
	double *tempOutput; // Matrice temporanea usata per l'aggiornamento dei pesi

	int _filterWidth;
	int _filterDim;
	int _stride;
	int _padding;

	// Handle per cuBlas
	cublasHandle_t handle;

private:
	void updateWeights(const double *prevOutput, const double &learningRate);
	void calcBackPropagation(const double *prevOutput, const double &learningRate);
	int _calcOutput(bool withPadding);
};

