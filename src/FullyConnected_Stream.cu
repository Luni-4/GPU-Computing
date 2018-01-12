#include <iostream>

#ifdef _WIN32
#include "Windows.h"
#endif

// Funzioni comuni
#include "Common.h"

// Cuda Kernel
#include "Kernel_Stream.h"

// Classi
#include "FullyConnected_Stream.h"

FullyConnected_Stream::FullyConnected_Stream(const int &width, const int &height, const ActFctType &a)
	: LayerDefinition(width, height, 1, FULLY_CONNECTED, a) {

	this->_nodes = width * height;

	initStreams();

}

FullyConnected_Stream::FullyConnected_Stream(const int &width, const ActFctType &a)
	: LayerDefinition(width, 1, 1, FULLY_CONNECTED, a),
	_nodes(width) {

	initStreams();

}

FullyConnected_Stream::~FullyConnected_Stream() {

}


std::vector<double> FullyConnected_Stream::getWeights(void) {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> FullyConnected_Stream::getBias(void) {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}

int FullyConnected_Stream::getPredictionIndex(void) {
	int maxIndex;

	// Individuare indice (classe) che corrisponde al valore massimo di output
	CHECK_CUBLAS(
		cublasIdamax(handle, _nodes, output, 1, &maxIndex));

	return maxIndex - 1;
}

void FullyConnected_Stream::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {

	// Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));

	// Impostazioni della cache
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	// Creazione degli stream
	streams = (cudaStream_t *)malloc(_nStreams * sizeof(cudaStream_t));

	for (int i = 0; i < _nStreams; i++) {
		CHECK(cudaStreamCreate(&(streams[i])));
	}

	// Dimensione matrice dei pesi
	_wDim = prevLayerWidth * prevLayerHeight * prevLayerDepth * _nodes;

	// Salvare dimensione del livello precedente
	_prevLayerDim = prevLayerWidth * prevLayerHeight * prevLayerDepth;

	// Dimensione matrice dei pesi in byte
	_wBytes = _wDim * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = _nodes * sizeof(double);

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
	const int aligned = ALIGN_UP(prevLayerWidth * prevLayerHeight, THREADS);

	// Tanti blocchi quanto sono i nodi e la profondità del layer precedente
	dim3 numBlocks(_nodes, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi del livello precedente
	dim3 threadBlocks(aligned, 1, 1);

	// Inizializza array per numeri casuali
	curandStateXORWOW_t *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _nodes * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandStateXORWOW_t)));

	// Inizializzare i weight del livello
	Kernel_Stream::initWeightK(numBlocks, threadBlocks, weight, _wDim, devStates);

	// Inizializzare i bias del livello
	Kernel_Stream::initBiasK(_alignedNodes / THREADS, THREADS, bias, _nodes, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore dei pesi\n\n";
	pettyPrintCuda(weight, _wDim, _prevLayerDim);
	std::cout << "\n\nValore dei bias\n\n";
	pettyPrintCuda(bias, _nodes, 1);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected_Stream::forward_propagation(const double *prevOutput) {

	for (int i = 0; i < _nStreams; i++) {
		int indexW = i * _alignedMatrix * _prevLayerDim;
		int indexO = i * _alignedMatrix;
		CHECK_CUBLAS(cublasSetStream(handle, streams[i]));
		CHECK_CUBLAS(
			cublasDgemv(handle, CUBLAS_OP_T, _prevLayerDim, _alignedMatrix, &alpha, weight + indexW, _prevLayerDim, prevOutput, 1, &beta, output + indexO, 1));
	}

#ifdef DEBUG
	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nOutput dei nodi senza bias\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif

	for (int i = 0; i < _nStreams; i++) {
		int indexO = i * _alignedMatrix;
		CHECK_CUBLAS(cublasSetStream(handle, streams[i]));
		CHECK_CUBLAS(
			cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _alignedMatrix, &alpha, bias + indexO, 1, &alpha, output + indexO, 1, output + indexO, 1));
	}

#ifdef DEBUG
	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nOutput dei nodi con bias sommato\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif    

	// Applicare funzione di attivazione
	if (_a == RELU)
		Kernel_Stream::actReluK(1, _alignedMatrix, streams, _nStreams, output, temp, _nodes);
	else if (_a == SIGMOID)
		Kernel_Stream::actSigmoidK(1, _alignedMatrix, streams, _nStreams, output, _nodes);
	else if (_a == TANH)
		Kernel_Stream::actTanhK(1, _alignedMatrix, streams, _nStreams, output, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nOutput dei nodi con funzione di attivazione\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif
}

void FullyConnected_Stream::calcError(double *prevError, const int &prevNodes) {

	int m = ALIGN_UP(prevNodes, _nStreams) / _nStreams;

	// Propagazione dell'errore dal livello successivo
	for (int i = 0; i < _nStreams; i++) {
		int indexW = i * m * _nodes;
		int indexO = i * m;
		CHECK_CUBLAS(cublasSetStream(handle, streams[i]));
		CHECK_CUBLAS(
			cublasDgemv(handle, CUBLAS_OP_N, m, _nodes, &alpha, weight + indexW, m, error, 1, &beta, prevError + indexO, 1));
	}

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nForward weight\n\n";
	pettyPrintCuda(weight, _wDim, prevNodes);
	std::cout << "\n\nForward error\n\n";
	pettyPrintCuda(error, _nodes, 1);
	std::cout << "\n\nErrore commesso sui nodi back propagation\n\n";
	pettyPrintCuda(prevError, prevNodes, 1);
#endif
}


void FullyConnected_Stream::back_propagation(const double *prevOutput, const double &learningRate) {

	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);

}

void FullyConnected_Stream::back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) {

	// Calcolo dell'errore per ogni nodo
	Kernel_Stream::outputErrorK(_alignedNodes / THREADS, THREADS, output, error, labels, target, _nodes);

	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi back propagation output\n\n";
	pettyPrintCuda(error, _nodes, 1);
#endif

	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);

}

inline void FullyConnected_Stream::calcBackPropagation(const double *prevOutput, const double &learningRate) {

	// Applicare derivata della funzione di attivazione
	if (_a == RELU)
		Kernel_Stream::derivActReluK(1, _alignedMatrix, streams, _nStreams, error, temp, _nodes);
	else if (_a == SIGMOID)
		Kernel_Stream::derivActSigmoidK(1, _alignedMatrix, streams, _nStreams, output, error, _nodes);
	else if (_a == TANH)
		Kernel_Stream::derivActTanhK(1, _alignedMatrix, streams, _nStreams, output, error, _nodes);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi con relativa derivata\n\n";
	pettyPrintCuda(error, _nodes, 1);
#endif

	// Aggiornare i pesi (da mettere in funzione)    
	updateWeights(prevOutput, learningRate);
}

void FullyConnected_Stream::updateWeights(const double *prevOutput, const double &learningRate) {

	/*for (int i = 0; i < _nodes; i++){
		CHECK_CUBLAS(
			cublasDaxpy(handle, _prevLayerDim, &error[i], prevOutput, 1, temp + (i * _prevLayerDim), 1));

		// CPU deve attendere che esecuzione della funzione finisca
		CHECK(cudaDeviceSynchronize());
	}*/

	int dim = ALIGN_UP(_alignedMatrix * _prevLayerDim, THREADS);

	Kernel_Stream::errorPrevOutputK(dim / THREADS, THREADS, streams, _nStreams, temp, prevOutput, error, _alignedMatrix, _alignedMatrix * _prevLayerDim, _prevLayerDim);

	//CHECK_CUBLAS(
			 //cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _alignedMatrix, &alpha, bias + indexO, 1, &alpha, output + indexO, 1, output + indexO, 1)); 

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nMatrice temporanea per aggiornamento pesi\n\n";
	pettyPrintCuda(temp, _wDim, _prevLayerDim);
#endif

	// Deve ricevere lo scalare dall'host
	//cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);    

	// Aggiornamento effettivo dei pesi 
	//CHECK_CUBLAS(
		//+cublasDaxpy(handle, _wDim, &learningRate, temp, 1, weight, 1));

	for (int i = 0; i < _nStreams; i++) {
		int indexW = i * _alignedMatrix * _prevLayerDim;
		CHECK_CUBLAS(cublasSetStream(handle, streams[i]));
		CHECK_CUBLAS(
			cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, _alignedMatrix, _prevLayerDim, &learningRate, temp + indexW, _alignedMatrix, &alpha, weight + indexW, _alignedMatrix, weight + indexW, _alignedMatrix));
	}

	// CPU deve attendere che esecuzione della funzione finisca
	//CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nMatrice dei pesi aggiornata\n\n";
	pettyPrintCuda(weight, _wDim, _prevLayerDim);
#endif

	// Aggiornamento del bias 
	//CHECK_CUBLAS(
		//cublasDaxpy(handle, _nodes, &learningRate, error, 1, bias, 1));

	for (int i = 0; i < _nStreams; i++) {
		int indexO = i * _alignedMatrix;
		CHECK_CUBLAS(cublasSetStream(handle, streams[i]));
		CHECK_CUBLAS(
			cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _alignedMatrix, &learningRate, error + indexO, 1, &alpha, bias + indexO, 1, bias + indexO, 1));
	}

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nVettore del bias aggiornato\n\n";
	pettyPrintCuda(bias, _nodes, 1);
#endif
}

inline void FullyConnected_Stream::initStreams(void) {

	this->_alignedNodes = ALIGN_UP(_nodes, THREADS);

	// Numero degli stream
	this->_nStreams = 2;

	// Numero di elementi che uno stream deve elaborare
	this->_alignedMatrix = _nodes / _nStreams;
}

void FullyConnected_Stream::deleteCuda(void) {

	CHECK_CUBLAS(cublasDestroy(handle));

	for (int i = 0; i < _nStreams; i++) {
		CHECK(cudaStreamDestroy(streams[i]));
	}

	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(temp));

	free(streams);
}

void FullyConnected_Stream::printW() {
	printFromCudaFormatted(weight, _wDim, _prevLayerDim);
}
