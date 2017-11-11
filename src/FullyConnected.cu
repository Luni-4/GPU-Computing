#include <iostream>
#include <vector>
#include <algorithm>

#ifdef _WIN32
#include "Windows.h"
#endif

// Funzioni comuni
#include "Common.h"

// Cuda Kernel
#include "Kernel.h"

// Classi
#include "FullyConnected.h"

FullyConnected::FullyConnected(const int &width, const int &height, const ActFctType &a)
	: LayerDefinition(width, height, 1, FULLY_CONNECTED, a) {

	this->_nodes = width * height;		
	this->_alignedNodes = ALIGN_UP(_nodes);

}

FullyConnected::FullyConnected(const int &width, const ActFctType &a)
	: LayerDefinition(width, 1, 1, FULLY_CONNECTED, a),
	_nodes(width) {
	
	this->_alignedNodes = ALIGN_UP(_nodes);

}

FullyConnected::~FullyConnected() {

}


std::vector<double> FullyConnected::getWeights() {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wDim * sizeof(double), cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> FullyConnected::getBias() {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}


void FullyConnected::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {

	// Dimensione matrice dei pesi
	_wDim = prevLayerWidth * prevLayerHeight * prevLayerDepth * _nodes;
	
	// Salvare dimensione del livello precedente
	_prevLayerDim = prevLayerWidth * prevLayerHeight * prevLayerDepth;  

	// Dimensione matrice dei pesi in byte
	_wBytes = _wDim * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = _nodes * sizeof(double);
	
	// Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));
	
#ifdef DEBUG
	// Impostazione buffer che gestisce il printf in Cuda
	size_t sz = 1048576 * 1000;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);
#endif

	// Allocare le matrici
	CHECK(cudaMalloc((void**)&weight, _wBytes));
	CHECK(cudaMalloc((void**)&bias, Bytes));
	CHECK(cudaMalloc((void**)&output, Bytes));
	CHECK(cudaMalloc((void**)&error, Bytes));
	CHECK(cudaMalloc((void**)&temp, _wBytes));

	// Rendere i blocchi multipli di 32
	const int aligned = ALIGN_UP(prevLayerWidth * prevLayerHeight);

	// Tanti blocchi quanto sono i nodi e la profondità del layer precedente
	dim3 numBlocks(_nodes, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi del livello precedente
	dim3 threadBlocks(aligned, 1, 1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _nodes * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandState)));

	// Inizializzare i weight del livello
	Kernel::initWeightK(numBlocks, threadBlocks, weight, _wDim, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	// Inizializzare i bias del livello
	Kernel::initBiasK(1, _alignedNodes, bias, _nodes, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore dei pesi\n\n";
	printFromCudaFormatted(weight, _wDim, prevLayerWidth);
	std::cout << "\n\nValore dei bias\n\n";
	printFromCudaFormatted(bias, _nodes, 1);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected::forward_propagation(const double *prevOutput) {

	CHECK_CUBLAS(
		cublasDgemv(handle, CUBLAS_OP_N, _prevLayerDim, _nodes, &alpha, weight, _prevLayerDim, prevOutput, 1, &beta, output, 1));

    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());

	// Somma con il bias
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));

    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nOutput dei nodi\n\n";
	printFromCuda(output, _nodes);
#endif

	// Applicare funzione di attivazione
	if (_a == RELU)
		Kernel::actReluK(1, _alignedNodes, output, _nodes);
	else if (_a == SIGMOID)
		Kernel::actSigmoidK(1, _alignedNodes, output, _nodes);
	else if (_a == TANH)
		Kernel::actTanhK(1, _alignedNodes, output, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
}

void FullyConnected::back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) {
    
    // Propagazione dell'errore dal livello successivo
    CHECK_CUBLAS(
		cublasDgemv(handle, CUBLAS_OP_T, _nodes ,forwardNodes, &alpha, forwardWeight, _nodes, forwardError, 1, &beta, error, 1));
		
	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);   

}

void FullyConnected::back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) {
    
    // Calcolo dell'errore per ogni nodo
    Kernel::outputErrorK(1, _alignedNodes, output, error, labels, target, _nodes);
    
    // CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
    
    // Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);  

}

void FullyConnected::deleteCuda() {

	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(temp));
}

void FullyConnected::calcBackPropagation(const double *prevOutput, const double &learningRate) {

    // Applicare derivata della funzione di attivazione
	if (_a == RELU)
		Kernel::derivActReluK(1, _alignedNodes, output, error, _nodes);
	else if (_a == SIGMOID)
		Kernel::derivActSigmoidK(1, _alignedNodes, output, error, _nodes);
	else if (_a == TANH)
		Kernel::derivActTanhK(1, _alignedNodes, output, error, _nodes);
		
	// CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());
    
    // Aggiornare i pesi (da mettere in funzione)    
    updateWeights(prevOutput, learningRate);
}

void FullyConnected::updateWeights(const double *prevOutput, const double &learningRate) {
	
	// Riempire la matrice temporanea di 0
	CHECK(cudaMemset(temp, 0, _wBytes));
	
	for (int i = 0; i < _nodes; i++)
	    CHECK_CUBLAS(
		    cublasDaxpy(handle, _prevLayerDim, &output[i], prevOutput, 1, temp + i, 1));
    
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());
    
    // Aggiornamento effettivo dei pesi 
    CHECK_CUBLAS(
		cublasDaxpy(handle, _wDim, &learningRate, temp, 1, weight, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());
	
	// Aggiornamento del bias 
    CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &learningRate, error, 1, bias, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());
}
