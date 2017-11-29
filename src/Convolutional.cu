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

__global__ void zeroPadding(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
	//blockIdx.y rappresenta la riga 
	const unsigned int p = forwardFilterWidth - 1;
	const unsigned int d = forwardErrorWidth + (p * 2);
	const unsigned int tid = ((blockIdx.y + p) * d) + p;

	memcpy((error + tid), (forwardError + blockIdx.y * forwardErrorWidth), (forwardErrorWidth * sizeof(double)));
}

void zeroPaddingK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPadding NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPadding << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
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

// TEST
std::vector<double> Convolutional::getWeights() {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}

// TEST
std::vector<double> Convolutional::getBias() {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}

// TEST
uint8_t Convolutional::getPredictionIndex(void) {
	int maxIndex;

	// Individuare indice (classe) che corrisponde al valore massimo di output
	CHECK_CUBLAS(
		cublasIdamax(handle, _nodes, output, 1, &maxIndex));

	return maxIndex;
}

void Convolutional::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	_prevLayerWidth = prevLayerWidth;
	_prevLayerDepth = prevLayerDepth;

	//numero di nodi dipende da filtro e nodi livello precedente
	//width
	_width = _calcOutput(false);
	//height
	_height = _calcOutput(false);
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

	const int forwardErrorWidth = 20;
	const int forwardFilterWidth = 5;
	const int forwardStride = 1;
	const int forwardDepth = 1;

#ifdef DEBUG
	std::cout << "\n\nForward error\n\n";
	printFromCudaFormatted(forwardError, forwardErrorWidth * forwardErrorWidth, forwardErrorWidth);
#endif

	// matrice temporanea inizializzata a 0 per zero padding
	double *padding;
	const int pBytes = (forwardFilterWidth - 1) * 2 + forwardErrorWidth;
	CHECK(cudaMalloc((void**)&padding, pBytes * pBytes * sizeof(double)));
	CHECK(cudaMemset(padding, 0, pBytes * pBytes * sizeof(double)));

	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	dim3 threadBlocks(_depth, 1, 1);//????????????????????????????

	// Tanti blocchi quante sono le righe di forwardError, in questo modo nel kernel sfrutto gli id.y per righe
	dim3 numBlocks(1, forwardErrorWidth, 1);

	zeroPaddingK(numBlocks, threadBlocks, padding, forwardError, forwardErrorWidth, forwardFilterWidth);

#ifdef DEBUG
	std::cout << "\n\nForward error con zero padding\n\n";
	printFromCudaFormatted(padding, pBytes * pBytes, pBytes);
#endif

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output
	const int uniqueNodes = _width * _height;
	const int forwardFilterDim = forwardFilterWidth * forwardFilterWidth;
	const unsigned int subBytes = uniqueNodes * forwardFilterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

	// Blocchi bidimensionali contenenti tanti thread quanti il depth del livello precedente
	threadBlocks = dim3(_prevLayerDepth, 1, 1);

	// Tanti blocchi quanti sono i nodi in output (width * height), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	numBlocks = dim3(_width, _height, 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, padding, pBytes, forwardFilterWidth, forwardStride, uniqueNodes);

	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore submatrici zero padding\n\n";
	printFromCudaFormatted(sub, uniqueNodes * forwardFilterDim, forwardFilterWidth);
#endif

	//ora sono in una situazione simile alla convoluzione
	for (int i = 0; i < _depth; i++) { //?????????????????????????????
		CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, forwardFilterDim, uniqueNodes, &alpha, sub, forwardFilterDim, forwardWeight + (i * forwardFilterDim), 1, &beta, error + (i * uniqueNodes), 1));
	}

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi back propagation\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	CHECK(cudaFree(sub));
	CHECK(cudaFree(padding));

	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);
}

void Convolutional::back_propagation_output(const double * prevOutput, const uint8_t * labels, const int & target, const double & learningRate) {
	// Calcolo dell'errore per ogni nodo
	Kernel::outputErrorK((_alignedNodes / THREADS), THREADS, output, error, labels, target, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi back propagation output\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);
}

void Convolutional::calcBackPropagation(const double *prevOutput, const double &learningRate) {

	// Applicare derivata della funzione di attivazione
	if (_a == RELU)
		Kernel::derivActReluK(1, _alignedNodes, output, error, _nodes);
	else if (_a == SIGMOID)
		Kernel::derivActSigmoidK(1, _alignedNodes, output, error, _nodes);
	else if (_a == TANH)
		Kernel::derivActTanhK(1, _alignedNodes, output, error, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi con relativa derivata\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	// Aggiornare i pesi (da mettere in funzione)    
	updateWeights(prevOutput, learningRate);
}

void Convolutional::updateWeights(const double *prevOutput, const double &learningRate) {

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di filtro
	//(prima genero sottomatrici grandi quanto _filterDim e ne genero tante quante uniqueNodes,
	// ora genero sottomatrici grandi quanto uniqueNodes e ne genero tante quante _filterDim)
	int uniqueNodes = _width * _height;
	const unsigned int subBytes = uniqueNodes * _filterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	dim3 threadBlocks(_depth, 1, 1);

	// Tanti blocchi quanti sono i nodi dei filtri (_filterWidth * _filterWidth), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	dim3 numBlocks(_filterWidth, _filterWidth, 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _width, _stride, _filterDim);
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nValore submatrici backpropagation\n\n";
	printFromCudaFormatted(sub, uniqueNodes * _filterDim, _width);
#endif

	// Riempire la matrice temporanea di 0
	CHECK(cudaMemset(temp, 0, _wBytes));

	//ora sono in una situazione simile al fully connected
	for (int i = 0; i < _depth; i++) {
		CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, uniqueNodes, _filterDim, &alpha, sub, uniqueNodes, error + (i * uniqueNodes), 1, &beta, temp + (i * _filterDim), 1));
	}

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nMatrice temporanea per aggiornamento pesi\n\n";
	printFromCudaFormatted(temp, _wDim, _filterWidth);
#endif

	// Aggiornamento effettivo dei pesi 
	CHECK_CUBLAS(
		cublasDaxpy(handle, _wDim, &learningRate, temp, 1, weight, 1));

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nMatrice dei pesi aggiornata\n\n";
	printFromCudaFormatted(weight, _wDim, _filterWidth);
#endif

	// Aggiornamento del bias 
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &learningRate, error, 1, bias, 1));

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nVettore del bias aggiornato\n\n";
	printFromCudaFormatted(bias, _nodes, _width);
#endif

	CHECK(cudaFree(sub));
}


void Convolutional::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(temp));
}

int Convolutional::_calcOutput(bool withPadding) {
	//PER ORA NON CONSIDERATO CASO IN CUI SI GENERANO ERRORI (padding numero non intero, filtro più grande dell'input, stride che non combacia, ecc)
	if (_filterWidth > _prevLayerWidth) {
		std::cerr << "Le dimensioni del filtro superano le dimensioni del livello precedente!!" << std::endl;
		exit(1);
	}

	if (withPadding) {
		_padding = (_filterWidth - 1) / 2;
		return _prevLayerWidth;
	}

	//+(_stride - 1)) serve per aggiornare per eccesso
	return ((_prevLayerWidth - _filterWidth + (_stride - 1)) / _stride) + 1;
}
