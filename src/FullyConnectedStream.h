#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "LayerDefinition.h"

class FullyConnectedStream : public LayerDefinition {

public:
	FullyConnectedStream(const int &width, const int &height, const ActFctType &a);
	FullyConnectedStream(const int &width, const ActFctType &a);
	~FullyConnectedStream();

	int getNodeCount(void) const override { return _nodes; }
	int getWeightCount(void) const override { return _wDim; }
	std::vector<double> getWeights(void) override;
	std::vector<double> getBias(void) override;
	int getPredictionIndex(void) override;

	void forward_propagation(const double *prevOutput) override;

	//void calcError(double *prevError, const int &prevNodes) override;

	void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) override;
	void back_propagation(const double *prevOutput, double *prevErr, const double &learningRate, const bool notFirst) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda(void) override;

	double* getCudaOutputPointer(void) const override { return output; }
	double* getCudaWeightPointer(void) const override { return weight; }
	//double* getCudaErrorPointer(void) const override { return error; }
	double* getCudaPrevErrorPointer(void) const override { return prevError; }

	void printW() override;

private:
	void updateWeights(const double *prevOutput, const double &learningRate);

	inline void initStreams(void);

private:
	int _wDim;
	int _wBytes;
	int _nodes;
	int _prevLayerDim;
	int _alignedNodes;

	int _nStreams;
	int _matrix;
	int _alignedMatrix;

	// Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;

	double *weight; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori
	double *prevError; // Matrice contenente gli errori da passare al livello precedente in backpropagation
	double *temp; // Matrice temporanea usata per l'aggiornamento dei pesi

	// Handle per cuBlas
	cublasHandle_t handle;

	// Array di streams
	cudaStream_t *streams;

};
