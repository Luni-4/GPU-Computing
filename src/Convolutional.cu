#ifdef _WIN32
#include "Windows.h"
#endif

#include <iostream>
#include "Common.h"

// Cuda Kernel
#include "Kernel.h"

#include "Convolutional.h"

__global__ void createSubmatrixBis(double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {

	// es 24x24 * 1 sottomatrici di 5x5 (ho input di 28x28) 
	// lancio thread di grandezza 5x5 e blocchi di grandezza 24x24
	// tid va da 0 a 24*24*5*5 = 14400
	// blockIdx.x e blockIdx.y rappresentano la sottomatrice
	// blockIdx.z rappresenta la profondità del livello precedente
	// blockDim.x è il numero di thread nel blocco in orizzontale, 5
	// gridDim.x è il numero di blocchi, 24
	// ad ogni tid corrisponde una posizione dell'input, pTid
	//printf("tid %d, blockId %d, blockIdx.x %d, blockIdx.y %d, gridDim.x %d\n", tid, blockId, blockIdx.x, blockIdx.y, gridDim.x);

	const unsigned int gDim = (gridDim.x * gridDim.y);
	const unsigned int bDim = (blockDim.x * blockDim.y);
	const unsigned int depth = blockIdx.z * gDim * bDim;
	const unsigned int blockId = depth + (blockIdx.y * gridDim.x + blockIdx.x) * bDim;
	const unsigned int tid = blockId + threadIdx.y * blockDim.x + threadIdx.x;

	const unsigned int pDepth = blockIdx.z * prevLayerWidth * prevLayerWidth;
	const unsigned int pBlockId = pDepth + (blockIdx.y * stride) * prevLayerWidth + blockIdx.x * stride;
	const unsigned int pTid = pBlockId + threadIdx.y * prevLayerWidth + threadIdx.x;

	sub[tid] = prevOutput[pTid];
}

void createSubmatrixBisK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrixBis NvCUDA2(b, t) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrixBis << <b, t >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

__global__ void createSubmatrix(double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {

	// es 20x20 * 2 sottomatrici di 5x5 (ho due input di 24x24) 
	// lancio thread di grandezza 2 e blocchi di grandezza 20x20
	// tid va da 0 a 20*20*2 = 800
	// blockIdx.x rappresenta la colonna da cui inizia la submatrice, va da 0 a 20
	// blockIdx.y rappresenta la riga da cui inizia la submatrice, va da 0 a 20
	// blockDim.x è il numero di thread nel blocco, 2
	// gridDim.x è il numero di blocchi, 20
	// printf("tid %d, blockIdx.x %d, blockDim.x %d, blockIdx.y %d, gridDim.x %d\n", tid, blockIdx.x, blockDim.x, blockIdx.y, gridDim.x);

	// Gestione degli indici	
	const unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	const unsigned int tid = blockId + threadIdx.x * uniqueNodes;

	//blockIdx.x rappresenta la colonna da cui inizia la submatrice
	//blockIdx.y rappresenta la riga da cui inizia la submatrice

	//Estraggo submatrici
	if (tid < uniqueNodes * blockDim.x) {
		for (int i = 0; i < filterWidth; i++) {
			memcpy((sub + tid * filterWidth * filterWidth + i * filterWidth), (prevOutput + (threadIdx.x * prevLayerWidth * prevLayerWidth) + (blockIdx.y * stride + i) * prevLayerWidth + blockIdx.x * stride), filterWidth * sizeof(double));
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

__global__ void zeroPaddingBis(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
	//threadIdx.y rappresenta la riga 
	//threadIdx.x rappresenta la colonna
	const unsigned int paddingLeft = forwardFilterWidth - 1;
	const unsigned int widthWithPadding = forwardErrorWidth + (paddingLeft * 2);
	const unsigned int paddingTop = blockIdx.z * widthWithPadding * widthWithPadding + paddingLeft * widthWithPadding;

	const unsigned int tid = paddingTop + threadIdx.y * widthWithPadding + paddingLeft + threadIdx.x;
	const unsigned int pTid = blockIdx.z * forwardErrorWidth * forwardErrorWidth + threadIdx.y * forwardErrorWidth + threadIdx.x;

	error[tid] = forwardError[pTid];
}

void zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPaddingBis NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPaddingBis << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#endif
}

__global__ void zeroPadding(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
	//blockIdx.y rappresenta la riga 
	const unsigned int paddingLeft = forwardFilterWidth - 1;
	const unsigned int widthWithPadding = forwardErrorWidth + (paddingLeft * 2);
	const unsigned int tid = ((blockIdx.y + paddingLeft) * widthWithPadding) + (widthWithPadding * widthWithPadding * threadIdx.x) + paddingLeft;

	memcpy((error + tid), (forwardError + blockIdx.y * forwardErrorWidth + forwardErrorWidth * forwardErrorWidth * threadIdx.x), (forwardErrorWidth * sizeof(double)));
}

void zeroPaddingK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPadding NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPadding << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#endif
}

__global__ void rot180Bis(const double * forwardWeight, double * forwardWeightRot, int filterDim) {

	// es 2 filtri di 5x5
	// per ora lancio thread di grandezza 2 e blocchi di grandezza 5x5
	// tid va da 0 a 5*5*2 = 50
	// blockIdx.x rappresenta la colonna da cui inizia la submatrice, va da 0 a 5
	// blockIdx.y rappresenta la riga da cui inizia la submatrice, va da 0 a 5
	// blockDim.x è il numero di thread nel blocco, 2
	// gridDim.x è il numero di blocchi, 5
	//printf("tid %d, threadIdx.x %d, threadIdx.x %d, blockIdx.x %d, blockDim.x %d, blockIdx.y %d, blockDim.y %d, gridDim.x %d\n", tid, threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x, blockIdx.y, blockDim.y, gridDim.x);

	// Gestione degli indici
	const unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	const unsigned int tid = blockId + (threadIdx.x + threadIdx.y * blockDim.x) * filterDim;

	const int plus = filterDim + (threadIdx.x + threadIdx.y *  blockDim.x) * filterDim - 1;
	forwardWeightRot[tid] = forwardWeight[plus - blockId];
}

void rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
	rot180Bis NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
	rot180Bis << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
#endif
}

__global__ void rot180(const double * forwardWeight, double * forwardWeightRot, int filterDim) {

	// es 2 filtri di 5x5
	// per ora lancio thread di grandezza 2 e blocchi di grandezza 5x5
	// tid va da 0 a 5*5*2 = 50
	// blockIdx.x rappresenta la colonna da cui inizia la submatrice, va da 0 a 5
	// blockIdx.y rappresenta la riga da cui inizia la submatrice, va da 0 a 5
	// blockDim.x è il numero di thread nel blocco, 2
	// gridDim.x è il numero di blocchi, 5
	//printf("tid %d, threadIdx.x %d, threadIdx.x %d, blockIdx.x %d, blockDim.x %d, blockIdx.y %d, blockDim.y %d, gridDim.x %d\n", tid, threadIdx.x, threadIdx.y, blockIdx.x, blockDim.x, blockIdx.y, blockDim.y, gridDim.x);

	// Gestione degli indici
	const unsigned int blockId = blockIdx.x + blockIdx.y * gridDim.x;
	const unsigned int tid = blockId + (threadIdx.x + threadIdx.y * blockDim.x) * filterDim;


	const int plus = filterDim + (threadIdx.x + threadIdx.y *  blockDim.x) * filterDim - 1;
	memcpy((forwardWeightRot + tid), (forwardWeight + plus - blockId), (sizeof(double)));
}

void rot180K(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
	rot180 NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
	rot180 << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
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

std::vector<double> Convolutional::getWeights(void) {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> Convolutional::getBias(void) {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}

int Convolutional::getPredictionIndex(void) {
	int maxIndex;

	// Individuare indice (classe) che corrisponde al valore massimo di output
	CHECK_CUBLAS(
		cublasIdamax(handle, _nodes, output, 1, &maxIndex));

	return maxIndex - 1;
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
	_alignedNodes = ALIGN_UP(_nodes, THREADS);

	_uniqueNodes = _width * _height;

#ifdef DEBUG
	std::cout << "dimensioni output del livello: " << _width << " - " << _height << " - " << _depth << std::endl;
#endif

	//Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));

	// Impostazioni della cache
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

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
	CHECK(cudaMalloc((void**)&weightRot, _wBytes));
	CHECK(cudaMalloc((void**)&bias, Bytes));
	CHECK(cudaMalloc((void**)&output, Bytes));
	CHECK(cudaMalloc((void**)&error, Bytes));
	CHECK(cudaMalloc((void**)&errorRot, Bytes));
	CHECK(cudaMalloc((void**)&tempWeight, _wBytes));
	CHECK(cudaMalloc((void**)&tempOutput, Bytes));

	// Rendere i blocchi multipli di 32
	const int aligned = ALIGN_UP(_filterDim, THREADS);

	// Tanti blocchi quanto sono i filtri e la profondità del layer precedente
	dim3 numBlocks(_depth, prevLayerDepth, 1);

	// Blocchi bidimensionali contenenti tanti thread quanti i nodi che compongono i filtri
	dim3 threadBlocks(aligned, 1, 1);

	// Inizializza array per numeri casuali
	curandStateXORWOW_t *devStates;

	// Numero di sequenze diverse per il rand
	const int numRand = _nodes * prevLayerDepth * aligned;

	// Alloca la memoria
	CHECK(cudaMalloc((void **)&devStates, numRand * sizeof(curandStateXORWOW_t)));

	// Inizializzare i weight del livello
	Kernel::initWeightK(numBlocks, threadBlocks, weight, _wDim, devStates);

	// Inizializzare i bias del livello
	Kernel::initBiasK((_alignedNodes / THREADS), THREADS, bias, _nodes, devStates);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore dei pesi\n\n";
	printFromCudaFormatted(weight, _wDim, _filterWidth);
	std::cout << "\n\nValore dei bias\n\n";
	printFromCudaFormatted(bias, _nodes, _width);
	std::cout << "\n\n\n\n";
#endif

	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	threadBlocks = dim3(_depth, _prevLayerDepth, 1);

	// Tanti blocchi quante sono le righe e le colonne di forwardError
	numBlocks = dim3(_filterWidth, _filterWidth, 1);

#ifdef BIS
	rot180BisK(numBlocks, threadBlocks, weight, weightRot, _filterDim);
#else
	rot180K(numBlocks, threadBlocks, weight, weightRot, _filterDim);
#endif

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore dei pesi ruotati\n\n";
	printFromCudaFormatted(weightRot, _wDim, _filterWidth);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}

void Convolutional::forward_propagation(const double * prevOutput) {

#ifdef DEBUG
	std::cout << "\n\nValore dell'input\n\n";
	printFromCudaFormatted(prevOutput, _prevLayerWidth * _prevLayerWidth * _prevLayerDepth, _prevLayerWidth);
#endif

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo che compone un blocco di output * la profondità del livello precedente e grande quanto un filtro
	const unsigned int subBytes = _uniqueNodes * _prevLayerDepth * _filterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

#ifdef BIS
	// Blocchi tridimensionali contenenti tanti thread quanti la grandezza dei filtri
	dim3 threadBlocks(_filterWidth, _filterWidth, 1);

	// Tanti blocchi quanti sono i nodi in output e il depth del livello precedente
	dim3 numBlocks(_width, _height, _prevLayerDepth);

	createSubmatrixBisK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _filterWidth, _stride, _uniqueNodes);
#else
	// Blocchi bidimensionali contenenti tanti thread quanti il depth del livello precedente
	dim3 threadBlocks(_prevLayerDepth, 1, 1);

	// Tanti blocchi quanti sono i nodi in output (width * height), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	dim3 numBlocks(_width, _height, 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _filterWidth, _stride, _uniqueNodes);
#endif

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici\n\n";
	printFromCudaFormatted(sub, _uniqueNodes * _prevLayerDepth * _filterDim, _filterWidth);
#endif

	//ora sono in una situazione simile al fully connected
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _filterDim, _uniqueNodes, &alpha, sub + (j * _uniqueNodes), _filterDim, weightRot + (i * _filterDim * _prevLayerDepth) + (j * _filterDim), 1, &beta, output + (i * _uniqueNodes), 1));
		}
	}

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore output senza bias\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	// Somma con il bias
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));
	//CHECK_CUBLAS(
		//cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _nodes, &alpha, bias, 1, &alpha, output, 1, output, 1));

#ifdef DEBUG
	std::cout << "\n\nValore output prima di funzione di attivazione\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	// Applicare funzione di attivazione
	if (_a == RELU)
		Kernel::actReluK((_alignedNodes / THREADS), THREADS, output, tempOutput, _nodes);
	else if (_a == SIGMOID)
		Kernel::actSigmoidK((_alignedNodes / THREADS), THREADS, output, _nodes);
	else if (_a == TANH)
		Kernel::actTanhK((_alignedNodes / THREADS), THREADS, output, _nodes);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore output *************************************************\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

	CHECK(cudaFree(sub));
}

void Convolutional::calcError(double *prevError, const int &prevNodes) {

	//prev error è l'errore del livello precedente che devo riempire, 
	//error è l'errore che ho usato al passo precedente (non ruotato) quando sono passato da questo livello

#ifdef DEBUG
	std::cout << "\n\error in calc error\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	// matrice temporanea inizializzata a 0 per zero padding
	double *padding;
	const int paddingWidth = (_filterWidth - 1) * 2 + _width;
	const int uniquePadding = paddingWidth * paddingWidth;
	const int paddingSize = uniquePadding * _depth; //come output 
	CHECK(cudaMalloc((void**)&padding, paddingSize * sizeof(double)));
	CHECK(cudaMemset(padding, 0, paddingSize * sizeof(double)));

#ifdef BIS
	// Blocchi bidimensionali contenenti tanti thread quanti sono i nodi in output
	dim3 threadBlocks(_height, _width, 1);

	// Tanti blocchi quanto il numero di filtri
	dim3 numBlocks(1, 1, _depth);

	zeroPaddingBisK(numBlocks, threadBlocks, padding, error, _width, _filterWidth);
#else
	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	dim3 threadBlocks(_depth, 1, 1);

	// Tanti blocchi quante sono le righe di forwardError, in questo modo nel kernel sfrutto gli id.y per righe
	dim3 numBlocks(1, _width, 1);

	zeroPaddingK(numBlocks, threadBlocks, padding, error, _width, _filterWidth);
#endif

#ifdef DEBUG
	std::cout << "\n\nerror con zero padding\n\n";
	printFromCudaFormatted(padding, paddingSize, paddingWidth);
#endif

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output di L-1
	const int prevUniqueNodes = prevNodes / _prevLayerDepth;
	const unsigned int subBytes = prevUniqueNodes * _depth * _filterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

#ifdef BIS
	// Blocchi tridimensionali contenenti tanti thread quanti la grandezza dei filtri
	threadBlocks = dim3(_filterWidth, _filterWidth, 1);

	// Tanti blocchi quanti sono i nodi in input e il depth del livello precedente
	numBlocks = dim3(sqrt(prevUniqueNodes), sqrt(prevUniqueNodes), _prevLayerDepth);

	createSubmatrixBisK(numBlocks, threadBlocks, sub, padding, paddingWidth, _filterWidth, _stride, prevUniqueNodes);
#else
	// Blocchi bidimensionali contenenti tanti thread quanti il depth del livello precedente
	threadBlocks = dim3(_depth, 1, 1);

	// Tanti blocchi quanti sono i nodi in output (width * height), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	numBlocks = dim3(sqrt(prevUniqueNodes), sqrt(prevUniqueNodes), 1);

	createSubmatrixK(numBlocks, threadBlocks, sub, padding, paddingWidth, _filterWidth, _stride, prevUniqueNodes);
#endif

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici zero padding\n\n";
	printFromCudaFormatted(sub, prevUniqueNodes * _depth * _filterDim, _filterWidth);
#endif

	//ora sono in una situazione simile alla convoluzione
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _filterDim, prevUniqueNodes, &alpha, sub + (i * prevUniqueNodes), _filterDim, weightRot + ((i + j * _depth) * _filterDim), 1, &beta, prevError + (j * prevUniqueNodes), 1));
		}
	}

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi back propagation\n\n";
	printFromCudaFormatted(prevError, prevNodes, sqrt(prevUniqueNodes));
#endif

	CHECK(cudaFree(sub));
	CHECK(cudaFree(padding));
}

void Convolutional::back_propagation(const double *prevOutput, const double &learningRate) {
	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);
}

void Convolutional::back_propagation_output(const double * prevOutput, const uint8_t * labels, const int & target, const double & learningRate) {
	// Calcolo dell'errore per ogni nodo
	Kernel::outputErrorK((_alignedNodes / THREADS), THREADS, output, error, labels, target, _nodes);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi back propagation output\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);
}

void Convolutional::calcBackPropagation(const double *prevOutput, const double &learningRate) {

	// Blocchi bidimensionali contenenti tanti thread quanti il numero di depth in output
	dim3 threadBlocks(_depth, 1, 1);

	// Tanti blocchi quante sono le righe e le colonne di error
	dim3 numBlocks(_width, _height, 1);

#ifdef BIS
	rot180BisK(numBlocks, threadBlocks, error, errorRot, _uniqueNodes);
#else
	rot180K(numBlocks, threadBlocks, error, errorRot, _uniqueNodes);
#endif

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore error ruotato\n\n";
	printFromCudaFormatted(errorRot, _nodes, _width);
#endif

	// Applicare derivata della funzione di attivazione
	if (_a == RELU)
		Kernel::derivActReluK((_alignedNodes / THREADS), THREADS, errorRot, tempOutput, _nodes);
	else if (_a == SIGMOID)
		Kernel::derivActSigmoidK((_alignedNodes / THREADS), THREADS, output, errorRot, _nodes);
	else if (_a == TANH)
		Kernel::derivActTanhK((_alignedNodes / THREADS), THREADS, output, errorRot, _nodes);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi con relativa derivata\n\n";
	printFromCudaFormatted(errorRot, _nodes, _width);
#endif

	updateWeights(prevOutput, learningRate);
}

void Convolutional::updateWeights(const double *prevOutput, const double &learningRate) {

	double *sub; // Submatrici

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di filtro
	//(prima genero sottomatrici grandi quanto _filterDim e ne genero tante quante uniqueNodes,
	// ora genero sottomatrici grandi quanto uniqueNodes e ne genero tante quante _filterDim)
	const unsigned int subBytes = _uniqueNodes * _prevLayerDepth * _filterDim * sizeof(double);

	// Alloco submatrice
	CHECK(cudaMalloc((void**)&sub, subBytes));

#ifdef BIS
	// Blocchi tridimensionali contenenti tanti thread quanti sono i nodi in output
	dim3 threadBlocks(_width, _height, 1);

	// Tanti blocchi quanti la grandezza dei filtri e il depth del livello precedente
	dim3 numBlocks(_filterWidth, _filterWidth, _prevLayerDepth);

	createSubmatrixBisK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _width, _stride, _filterDim);
#else
	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	dim3 threadBlocks(_prevLayerDepth, 1, 1);

	// Tanti blocchi quanti sono i nodi dei filtri (_filterWidth * _filterWidth), in questo modo nel kernel sfrutto gli id per righe e colonne delle submatrici
	dim3 numBlocks(_filterWidth, _filterWidth, 1);

	// come in forward ma sostituendo filterWidth con width e uniquenodes con filterdim
	createSubmatrixK(numBlocks, threadBlocks, sub, prevOutput, _prevLayerWidth, _width, _stride, _filterDim);
#endif

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici backpropagation\n\n";
	printFromCudaFormatted(sub, _uniqueNodes * _filterDim, _width);
#endif

	// Riempire la matrice temporanea di 0
	CHECK(cudaMemset(tempWeight, 0, _wBytes));

	//ora sono in una situazione simile al fully connected
	double backAlpha = 1.0 / _uniqueNodes;
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _uniqueNodes, _filterDim, &backAlpha, sub + (j * _filterDim), _uniqueNodes, errorRot + (i * _uniqueNodes), 1, &beta, tempWeight + ((i + j * _depth) * _filterDim), 1));
		}
	}

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nMatrice temporanea per aggiornamento pesi\n\n";
	printFromCudaFormatted(tempWeight, _wDim, _filterWidth);
#endif

	// Aggiornamento effettivo dei pesi 
	CHECK_CUBLAS(
		cublasDaxpy(handle, _wDim, &learningRate, tempWeight, 1, weight, 1));
	//CHECK_CUBLAS(
		//cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, _wDim, _depth, &learningRate, tempWeight, _wDim, &alpha, weight, _wDim, weight, _wDim));

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nMatrice dei pesi aggiornata\n\n";
	printFromCudaFormatted(weight, _wDim, _filterWidth);
#endif

	// Ruoto subito i pesi aggiornati per poi usarli nella backpropagation al livello L-1
	// Blocchi bidimensionali contenenti tanti thread quanti il numero di filtri
	threadBlocks = dim3(_depth, _prevLayerDepth, 1);

	// Tanti blocchi quante sono le righe e le colonne di forwardError
	numBlocks = dim3(_filterWidth, _filterWidth, 1);

#ifdef BIS
	rot180BisK(numBlocks, threadBlocks, weight, weightRot, _filterDim);
#else
	rot180K(numBlocks, threadBlocks, weight, weightRot, _filterDim);
#endif

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore dei pesi ruotati\n\n";
	printFromCudaFormatted(weightRot, _wDim, _filterWidth);
#endif

	// Aggiornamento del bias 
	CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &learningRate, errorRot, 1, bias, 1));
	//CHECK_CUBLAS(
		//cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _nodes, &learningRate, errorRot, 1, &alpha, bias, 1, bias, 1));

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nVettore del bias aggiornato\n\n";
	printFromCudaFormatted(bias, _nodes, _width);
#endif

	CHECK(cudaFree(sub));
}

void Convolutional::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));
	CHECK(cudaFree(weight));
	CHECK(cudaFree(weightRot));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(errorRot));
	CHECK(cudaFree(tempWeight));
	CHECK(cudaFree(tempOutput));
}

void Convolutional::printW() {
	printFromCudaFormatted(weight, _wDim, _filterWidth);
	//printFromCudaFormatted(bias, _nodes, _width);
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
