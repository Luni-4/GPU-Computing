#ifdef _WIN32
#include "Windows.h"
#endif

#include <iostream>
#include <vector>
#include <algorithm>

// Cuda Kernel
#include "Kernel.h"

#include "Common.h"
#include "Convolutional.h"

__global__ void createSubmatrix(double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {

	// Gestione degli indici	
	const unsigned int blockId = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const unsigned int tid = blockId + threadIdx.x;

	//tid rappresenta il nodo di output
	//blockIdx.x rappresenta la colonna da cui inizia la submatrice
	//blockIdx.y rappresenta la riga da cui inizia la submatrice

	//Estraggo submatrici
	if (tid < uniqueNodes) {
		for (int i = 0; i < filterWidth; i++) {
			memcpy((sub + i * filterWidth + tid * filterWidth * filterWidth), (prevOutput + blockIdx.x * stride + (blockIdx.y * stride + i) * prevLayerWidth), filterWidth * sizeof(double));
		}
	}
}

void createSubmatrixK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrix NvCUDA2(b, t) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrix << <b, t >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

Convolutional::Convolutional(const int &filterWidth, const int &depth, const int &stride, const ActFctType &a)
	: LayerDefinition(0, 0, depth, CONVOLUTIONAL, a) {
	this->_filterWidth = filterWidth;
	this->_filterDim = filterWidth * filterWidth;
	this->_depth = depth;
	this->_stride = stride;
	this->_padding = 0;
}

Convolutional::~Convolutional() {
}

std::vector<double> Convolutional::getWeights() {
	return std::vector<double>();
}

std::vector<double> Convolutional::getBias() {
	return std::vector<double>();
}

uint8_t Convolutional::getPredictionIndex(void) {
	return uint8_t();
}

void Convolutional::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	_prevLayerWidth = prevLayerWidth;
	_prevLayerDepth = prevLayerDepth;

	//numero di nodi dipende da filtro e nodi livello precedente
	//width
	_width = _calcOutput(prevLayerWidth, false);
	//height
	_height = _calcOutput(prevLayerHeight, false);
	//depth = numero di filtri

	this->_nodes = _width * _height * _depth;
	_alignedNodes = ALIGN_UP(_nodes);

#ifdef DEBUG
	std::cout << "dimensioni output del livello: " << _width << " - " << _height << " - " << _depth << std::endl;
#endif

	// Dimensione matrice dei pesi
	_wDim = _filterDim * prevLayerDepth * _depth;

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
	const int aligned = ALIGN_UP(_filterDim);

	// Tanti blocchi quanto sono i filtri e la profondità del layer precedente
	dim3 numBlocks(_depth, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi che compongono i filtri
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
	Kernel::initBiasK((_alignedNodes / THREADS), THREADS, bias, _nodes, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore dei pesi\n\n";
	printFromCudaFormatted(weight, _wDim, _filterWidth);
	std::cout << "\n\nValore dei bias\n\n";
	printFromCudaFormatted(bias, _nodes, _width);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}

void Convolutional::forward_propagation(const double * prevOutput) {
#ifdef DEBUG
	std::cout << "\n\nValore dell'input\n\n";
	printFromCudaFormatted(prevOutput, _prevLayerWidth * _prevLayerWidth, _prevLayerWidth);
#endif

	double *sub; // Submatrici

				 // Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output
	int uniqueNodes = _width * _height;
	const unsigned int subBytes = uniqueNodes * _filterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

	// Blocchi bidimensionali contenenti tanti thread quanti il depth del livello precedente
	dim3 threadBlocks(_prevLayerDepth, 1, 1);

	// Tanti blocchi quanti sono i nodi in output (width * height), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	dim3 numBlocks(_width, _height, 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _filterWidth, _stride, uniqueNodes);
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore submatrici\n\n";
	printFromCudaFormatted(sub, uniqueNodes * _filterDim, _filterWidth);
#endif

	//Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));

	//ora sono in una situazione simile al fully connected
	for (int i = 0; i < _depth; i++) {
		CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _filterDim, uniqueNodes, &alpha, sub, _filterDim, weight + (i * _filterDim), 1, &beta, output + (i * uniqueNodes), 1));
	}

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore output senza bias\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	// Somma con il bias
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));

#ifdef DEBUG
	std::cout << "\n\nValore output prima di funzione di attivazione\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	// Applicare funzione di attivazione
	if (_a == RELU)
		Kernel::actReluK((_alignedNodes / THREADS), THREADS, output, _nodes);
	else if (_a == SIGMOID)
		Kernel::actSigmoidK((_alignedNodes / THREADS), THREADS, output, _nodes);
	else if (_a == TANH)
		Kernel::actTanhK((_alignedNodes / THREADS), THREADS, output, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore output\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	CHECK(cudaFree(sub));
}

void Convolutional::back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) {
}

void Convolutional::back_propagation_output(const double * prev, const uint8_t * labels, const int & target, const double & learningRate) {
}

void Convolutional::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(temp));
}

int Convolutional::_calcOutput(int prevLayerWidth, bool withPadding) {
	//PER ORA NON CONSIDERATO CASO IN CUI SI GENERANO ERRORI (padding numero non intero, filtro più grande dell'input, stride che non combacia, ecc)
	if (_filterWidth > prevLayerWidth) {
		std::cerr << "Le dimensioni del filtro superano le dimensioni del livello precedente!!" << std::endl;
		exit(1);
	}

	if (withPadding) {
		_padding = (_filterWidth - 1) / 2;
		return prevLayerWidth;
	}
	return ((prevLayerWidth - _filterWidth + 0) / _stride) + 1;
}
