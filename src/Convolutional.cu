#include <iostream>
#include <vector>
#include <algorithm>

// Cuda Kernel
#include "Kernel.h"

#include "Common.h"
#include "Convolutional.h"

#ifdef _WIN32
#include "Windows.h"
#endif

__global__ void createSubmatrix(double * sub, const double * prev, const int prevLayerWidth, const int filterWidth, const int nodes) {

	// Gestione degli indici	
	const unsigned int blockId = blockIdx.x * blockDim.x + blockIdx.y * blockDim.x * gridDim.x;
	const unsigned int tid = blockId + threadIdx.x;

	//tid rappresenta il nodo di output
	//blockIdx.x rappresenta la colonna da cui inizia la submatrice
	//blockIdx.y rappresenta la riga da cui inizia la submatrice

	//Estraggo submatrici
	if (tid < nodes) {
		for (int i = 0; i < filterWidth; i++) {
			memcpy((sub + i * filterWidth + tid * filterWidth * filterWidth), (prev + blockIdx.x + (blockIdx.y + i) * prevLayerWidth), filterWidth * sizeof(double));
		}
	}
}

void createSubmatrixK(dim3 t, dim3 b, double * sub, const double * prev, const int prevLayerWidth, const int filterWidth, const int nodes) {
#ifdef _WIN32
	createSubmatrix NvCUDA2(t, b) (sub, prev, prevLayerWidth, filterWidth, nodes);
#else
	createSubmatrix << <t, b >> > (sub, prev, prevLayerWidth, filterWidth, nodes);
#endif
}

Convolutional::Convolutional(const int &filterWidth, const int &depth, const int &stride, const ActFctType &a)
	: LayerDefinition(0, 0, depth, CONVOLUTIONAL, a) {
	this->_depth = depth;
	this->_filterWidth = filterWidth;
	this->_stride = stride;
	this->_padding = 0;
}

Convolutional::~Convolutional() {
}

int Convolutional::getLayerNodeCount() {
	return 0;
}

int Convolutional::getWeightCount(const int & prevLayerNode) {
	return 0;
}

std::vector<double> Convolutional::getWeights() {
	return std::vector<double>();
}

std::vector<double> Convolutional::getBias() {
	return std::vector<double>();
}

void Convolutional::forward_propagation(const double * prev) {
#ifdef DEBUG
	std::cout << "\n\nValore dell'input\n\n";
	printFromCudaFormatted(prev, _prevLayerWidth * _prevLayerWidth, _prevLayerWidth);
#endif

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output
	const unsigned int subBytes = _nodes * _filterWidth * _filterWidth * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

	// Blocchi bidimensionali contenenti tanti thread quanti i numeri di filtri
	dim3 threadBlocks(_depth, 1, 1);

	// Tanti blocchi quanti sono i nodi in output (width * height), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	dim3 numBlocks(_width, _height, 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, prev, _prevLayerWidth, _filterWidth, _nodes);
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore submatrici\n\n";
	printFromCudaFormatted(sub, _nodes * _filterWidth * _filterWidth, _filterWidth);
	//printFromCudaFormatted(sub, _filterWidth * _filterWidth, _filterWidth);
#endif

	//Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));

	//Fattori dei prodotti
	const double alpha = 1.0f;
	const double beta = 0.0f;

	//ora sono in una situazione simile al fully connected
	CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, 24, 25, &alpha, sub + 00, 24, weight, 1, &beta, output + 00, 1));

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	std::cout << "\n\nValore output\n\n";
	printFromCudaFormatted(output, _nodes, 24);

	{
		//test correttezza
		double tot = 0;
		std::vector<double> test(25);
		std::vector<double> testW(25);
		CHECK(cudaMemcpy(&test[0], sub + 25 * 0, 25 * sizeof(double), cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy(&testW[0], weight, 25 * sizeof(double), cudaMemcpyDeviceToHost));
		for (int i = 0; i < 25; i++) {
			std::cout << test[i] << " ";
		}
		std::cout << std::endl;
		for (int i = 0; i < 25; i++) {
			std::cout << testW[i] << " ";
		}
		std::cout << std::endl;
		for (int i = 0; i < 25; i++) {
			tot += test[i] * testW[i];
		}
		std::cout << std::endl;
		std::cout << tot << std::endl;

		double data1[] = {
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99,
			-0.99, -0.99, -0.99, -0.99, -0.99
		};
		double data2[] = {
			-0.17, 0.28, -0.16, 0.21, -0.07,
			0.1, -0.34, 0.38, -0.24, 0.01,
			-0.14, 0.18, -0.03, 0.07, -0.17,
			0.39, -0.29, 0.08, -0.19, 0.25,
			-0.08, 0.13, -0.34, 0.1, -0.26
		};
		int m_row = 25;
		int m_col = 25;
		std::vector<double> matrix(data1, data1 + sizeof(data1) / sizeof(double));
		std::vector<double> vector(data2, data2 + sizeof(data2) / sizeof(double));

		double * d_matrix, *d_vector, *d_res;
		CHECK(cudaMalloc((void**)&d_matrix, m_row * m_col * sizeof(double)));
		CHECK(cudaMalloc((void**)&d_vector, m_col * sizeof(double)));
		CHECK(cudaMalloc((void**)&d_res, m_row * sizeof(double)));

		CHECK(cudaMemcpy(d_matrix, &matrix[0], m_row * m_col * sizeof(double), cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_vector, &vector[0], m_col * sizeof(double), cudaMemcpyHostToDevice));

		CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_N, m_row, m_col, &alpha, sub, m_row, weight, 1, &beta, d_res, 1));

		CHECK(cudaDeviceSynchronize());

		CHECK(cudaMemcpy(&matrix[0], d_res, m_row * sizeof(double), cudaMemcpyDeviceToHost));
		std::cout << std::endl;
		for (int i = 0; i < m_row; i++) {
			std::cout << matrix[i] << " ";
		}
		std::cout << std::endl;
	}
}

void Convolutional::back_propagation() {
}

void Convolutional::back_propagation_output(const double * prev, const uint8_t * labels, const int & target, const double & learningRate) {
}

void Convolutional::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	_prevLayerWidth = prevLayerWidth;

	//numero di nodi dipende da filtro e nodi livello precedente
	//width
	_width = _calcOutput(prevLayerWidth, false);
	//height
	_height = _calcOutput(prevLayerHeight, false);
	//depth = numero di filtri

	this->_nodes = _width * _height * _depth;

#ifdef DEBUG
	std::cout << _width << " - " << _height << " - " << _depth << std::endl;
#endif

	// Dimensione matrice dei pesi
	_wDim = _filterWidth * _filterWidth * prevLayerDepth * _depth;

	// Dimensione matrice dei pesi in byte
	const unsigned int wBytes = _wDim * sizeof(double);

	// Dimensione bias, output, error
	const unsigned int Bytes = _nodes * sizeof(double);

	// Impostazione buffer che gestisce il printf in Cuda
	size_t sz = 1048576 * 1000;
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, sz);

	// Allocare le matrici
	CHECK(cudaMalloc((void**)&weight, wBytes));
	CHECK(cudaMalloc((void**)&bias, Bytes));
	CHECK(cudaMalloc((void**)&output, Bytes));
	CHECK(cudaMalloc((void**)&error, Bytes));

	// Rendere i blocchi multipli di 32
	const int aligned = ALIGN_UP(_filterWidth * _filterWidth);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi che compongono i filtri
	dim3 threadBlocks(aligned, 1, 1);

	// Tanti blocchi quanto sono i filtri e la profondità del layer precedente
	dim3 numBlocks(_depth, prevLayerDepth, 1);

	// Inizializza array per numeri casuali
	curandState *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _depth * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandState)));

	// Inizializzare i weight del livello
	Kernel::initWeightK(numBlocks, threadBlocks, weight, _wDim, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

	// Convertire il numero di filtri in un multiplo di 32
	//datascience.stackexchange.com/questions/11853/question-about-bias-in-convolutional-networks
	const int b = ALIGN_UP(_depth);

	// Inizializzare i bias del livello
	Kernel::initBiasK(1, b, bias, _depth, devStates);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore dei pesi\n\n";
	printFromCudaFormatted(weight, _wDim, _filterWidth);
	std::cout << "\n\nValore dei bias\n\n";
	printFromCudaFormatted(bias, _depth, 1);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}

void Convolutional::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
}

int Convolutional::_calcOutput(int prevLayerWidth, bool withPadding) {
	//PER ORA NON CONSIDERATO CASO IN CUI SI GENERANO ERRORI (padding numero non intero, filtro più grande dell'input, stride che non combacia, ecc)
	if (withPadding) {
		_padding = (_filterWidth - 1) / 2;
		return prevLayerWidth;
	}
	return ((prevLayerWidth - _filterWidth + 0) / _stride) + 1;
}
