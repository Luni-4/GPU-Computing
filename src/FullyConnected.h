#pragma once

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "LayerDefinition.h"

class FullyConnected : public LayerDefinition {

public:
	FullyConnected(const int &width, const int &height, const ActFctType &a);
	FullyConnected(const int &width, const ActFctType &a);
	~FullyConnected();

	int getNodeCount() const override { return _nodes; }
	int getWeightCount(const int &prevLayerNode) const override { return prevLayerNode * _nodes; }
	std::vector<double> getWeights() override;
	std::vector<double> getBias() override;

	void forward_propagation(const double *prevOutput) override;

	void back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) override;
	void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda() override;
	
	double* getCudaOutputPointer() const override { return output;}
	double* getCudaWeightPointer() const override { return weight; }
	double* getCudaErrorPointer() const override { return error; }
	
private:
    void updateWeights(const double *prevOutput, const double &learningRate);
    void calcBackPropagation(const double *prevOutput, const double &learningRate);

private:
	int _wDim;
	int _wBytes;
	int _nodes;
	int _prevLayerDim;
	int _alignedNodes;	
	
	// Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori
	double *temp; // Matrice temporanea usata per l'aggiornamento dei pesi

	// Handle per cuBlas
	cublasHandle_t handle;

};
