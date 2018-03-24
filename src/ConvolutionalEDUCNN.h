#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "LayerDefinition.h"

class ConvolutionalEDUCNN : public LayerDefinition {
public:
	ConvolutionalEDUCNN(const int &filterWidth, const int &filterDepth, const int &stride);
	~ConvolutionalEDUCNN();

	int getNodeCount(void) const override { return _nodes; }
	int getWeightCount(void) const override { return _wDim; }
	std::vector<double> getWeights(void) override;
	std::vector<double> getBias(void) override;
	int getPredictionIndex(void) override;

	void forward_propagation(const double *prevOutput) override;

	void back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) override;
	void back_propagation(const double *prevOutput, double *prevError, const double &learningRate, const bool notFirst) override;

	void defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) override;
	void deleteCuda(void) override;

	double* getCudaOutputPointer(void) const override { return output; }
	double* getCudaWeightPointer(void) const override { return weight; }
	//double* getCudaErrorPointer(void) const override { return error; }
	double* getCudaPrevErrorPointer(void) const override { return prevError; }

	void printW() override;

private:
	void calcError();
	void updateWeights(const double *prevOutput, const double &learningRate);
	int _calcOutput(bool withPadding);

private:
	int _wDim;
	int _wBytes;
	int _nodes;
	int _uniqueNodes;
	int _prevLayerWidth;
	int _prevLayerDepth;
	int _alignedNodes;

	//Fattori dei prodotti
	const double alpha = 1.0f;
	double beta = 0.0f;

	double *weight; // Matrice dei pesi in Cuda
	double *weightRot; // Matrice dei pesi in Cuda
	double *bias; // Matrice per i bias in Cuda
	double *output; // Matrice dell'output in Cuda
	double *error; // Matrice degli errori
	double *prevError; // Matrice degli errori
	double *tempWeight; // Matrice temporanea usata per l'aggiornamento dei pesi

	int _filterWidth;
	int _filterDim;
	int _stride;
	int _padding;

	double *subForward; // Submatrici
	double *subCalcError; // Submatrici
	double *subBack; // Submatrici

	int paddingWidth;
	int paddingSize;
	double *padding;

	// Handle per cuBlas
	cublasHandle_t handle;
};

