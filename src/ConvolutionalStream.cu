#ifdef _WIN32
#include "Windows.h"
#endif

#include <iostream>
#include "Common.h"

// Cuda Kernel
#include "KernelCPU.h"
#include "KernelStreamCPU.h"

#include "ConvolutionalStream.h"

ConvolutionalStream::ConvolutionalStream(const int &filterWidth, const int &depth, const int &stride, const ActFctType &a)
	: LayerDefinition(0, 0, depth, CONVOLUTIONAL, a) {
	this->_filterWidth = filterWidth;
	this->_filterDim = filterWidth * filterWidth;
	this->_depth = depth;
	this->_stride = stride;
	this->_padding = 0;
}

ConvolutionalStream::~ConvolutionalStream() {
}

std::vector<double> ConvolutionalStream::getWeights(void) {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> ConvolutionalStream::getBias(void) {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}

int ConvolutionalStream::getPredictionIndex(void) {
	int maxIndex;

	// Individuare indice (classe) che corrisponde al valore massimo di output
	CHECK_CUBLAS(
		cublasIdamax(handle, _nodes, output, 1, &maxIndex));

	return maxIndex - 1;
}

void ConvolutionalStream::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {
	_prevLayerWidth = prevLayerWidth;
	_prevLayerDepth = prevLayerDepth;

	//numero di nodi dipende da filtro e nodi livello precedente
	//width
	_width = _calcOutput(false);
	//height
	_height = _calcOutput(false);
	//depth = numero di filtri

	_uniqueNodes = _width * _height;
	this->_nodes = _width * _height * _depth;

#ifdef DEBUG
	std::cout << "dimensioni output del livello: " << _width << " - " << _height << " - " << _depth << std::endl;
#endif

	initStreams();

	// Creazione degli stream
	streams = (cudaStream_t *)malloc(_nStreams * sizeof(cudaStream_t));

	for (int i = 0; i < _nStreams; i++) {
		CHECK(cudaStreamCreate(&(streams[i])));
	}

	// Creazione degli stream
	streamsBack = (cudaStream_t *)malloc(_nStreamsBack * sizeof(cudaStream_t));

	for (int i = 0; i < _nStreamsBack; i++) {
		CHECK(cudaStreamCreate(&(streamsBack[i])));
	}

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

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo che compone un blocco di output * la profondità del livello precedente e grande quanto un filtro
	unsigned int subBytes = _uniqueNodes * _prevLayerDepth * _filterDim * sizeof(double);
	CHECK(cudaMalloc((void**)&subForward, subBytes));

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output di L-1
	const int prevUniqueNodes = _prevLayerWidth * _prevLayerWidth;
	subBytes = prevUniqueNodes * _depth * _filterDim * sizeof(double);
	CHECK(cudaMalloc((void**)&subCalcError, subBytes));

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di filtro
	//(prima genero sottomatrici grandi quanto _filterDim e ne genero tante quante uniqueNodes,
	// ora genero sottomatrici grandi quanto uniqueNodes e ne genero tante quante _filterDim)
	subBytes = _uniqueNodes * _prevLayerDepth * _filterDim * sizeof(double);
	CHECK(cudaMalloc((void**)&subBack, subBytes));

	// matrice temporanea inizializzata a 0 per zero padding
	paddingWidth = (_filterWidth - 1) * 2 + _width;
	const int uniquePadding = paddingWidth * paddingWidth;
	paddingSize = uniquePadding * _depth; //come output 
	CHECK(cudaMalloc((void**)&padding, paddingSize * sizeof(double)));
	CHECK(cudaMemset(padding, 0, paddingSize * sizeof(double)));

#ifdef DEBUG
	std::cout << "Memoria allocata \n" << std::endl;
#endif

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

	Kernel::rot180BisK(numBlocks, threadBlocks, weight, weightRot, _filterDim);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore dei pesi ruotati\n\n";
	printFromCudaFormatted(weightRot, _wDim, _filterWidth);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}

void ConvolutionalStream::forward_propagation(const double * prevOutput) {

#ifdef DEBUG
	std::cout << "\n\nValore dell'input\n\n";
	printFromCudaFormatted(prevOutput, _prevLayerWidth * _prevLayerWidth * _prevLayerDepth, _prevLayerWidth);
#endif

	// Blocchi tridimensionali contenenti tanti thread quanti la grandezza dei filtri
	dim3 threadBlocks(_filterWidth, _filterWidth, 1);

	// Tanti blocchi quanti sono i nodi in output e il depth del livello precedente
	dim3 numBlocks(_width, _height, _prevLayerDepth);

	Kernel::createSubmatrixBisK(numBlocks, threadBlocks, subForward, prevOutput, _prevLayerWidth, _filterWidth, _stride, _uniqueNodes);

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici\n\n";
	printFromCudaFormatted(subForward, _uniqueNodes * _prevLayerDepth * _filterDim, _filterWidth);
#endif

	//ora sono in una situazione simile al fully connected
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
#ifdef CUBLAS_STREAM
			for (int k = 0; k < _nStreams; k++) {
				int indexW = k * _alignedMatrix * _filterDim;
				int indexO = k * _alignedMatrix;
				CHECK_CUBLAS(cublasSetStream(handle, streams[k]));
				CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _alignedMatrix, _uniqueNodes, &alpha, subForward + (j * _uniqueNodes), _alignedMatrix, weightRot + (i * _filterDim * _prevLayerDepth) + (j * _filterDim) + indexW, 1, &beta, output + (i * _uniqueNodes) + indexO, 1));
			}
#else
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _filterDim, _uniqueNodes, &alpha, subForward + (j * _uniqueNodes), _filterDim, weightRot + (i * _filterDim * _prevLayerDepth) + (j * _filterDim), 1, &beta, output + (i * _uniqueNodes), 1));
#endif 
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
#ifdef STREAM
	if (_a == RELU)
		KernelStream::actReluK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, output, tempOutput, _nodes);
	else if (_a == SIGMOID)
		KernelStream::actSigmoidK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, output, _nodes);
	else if (_a == TANH)
		KernelStream::actTanhK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, output, _nodes);
#else
	if (_a == RELU)
		Kernel::actReluK((_alignedNodes / THREADS), THREADS, output, tempOutput, _nodes);
	else if (_a == SIGMOID)
		Kernel::actSigmoidK((_alignedNodes / THREADS), THREADS, output, _nodes);
	else if (_a == TANH)
		Kernel::actTanhK((_alignedNodes / THREADS), THREADS, output, _nodes);
#endif

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore output *************************************************\n\n";
	printFromCudaFormatted(output, _nodes, _width);
#endif

}

void ConvolutionalStream::calcError(double *prevError, const int &prevNodes) {

	//prev error è l'errore del livello precedente che devo riempire, 
	//error è l'errore che ho usato al passo precedente (non ruotato) quando sono passato da questo livello

#ifdef DEBUG
	std::cout << "\n\error in calc error\n\n";
	printFromCudaFormatted(error, _nodes, _width);
#endif

	// Blocchi bidimensionali contenenti tanti thread quanti sono i nodi in output
	dim3 threadBlocks(_height, _width, 1);

	// Tanti blocchi quanto il numero di filtri
	dim3 numBlocks(1, 1, _depth);

	Kernel::zeroPaddingBisK(numBlocks, threadBlocks, padding, error, _width, _filterWidth);

#ifdef DEBUG
	std::cout << "\n\nerror con zero padding\n\n";
	printFromCudaFormatted(padding, paddingSize, paddingWidth);
#endif

	// Dimensione insieme submatrici in byte = creo una submatrice per ogni nodo di output di L-1
	const int prevUniqueNodes = prevNodes / _prevLayerDepth;

	// Blocchi tridimensionali contenenti tanti thread quanti la grandezza dei filtri
	threadBlocks = dim3(_filterWidth, _filterWidth, 1);

	// Tanti blocchi quanti sono i nodi in input e il depth del livello precedente
	numBlocks = dim3(sqrt(prevUniqueNodes), sqrt(prevUniqueNodes), _prevLayerDepth);

	Kernel::createSubmatrixBisK(numBlocks, threadBlocks, subCalcError, padding, paddingWidth, _filterWidth, _stride, prevUniqueNodes);

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici zero padding\n\n";
	printFromCudaFormatted(subCalcError, prevUniqueNodes * _depth * _filterDim, _filterWidth);
#endif

	//ora sono in una situazione simile alla convoluzione
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
#ifdef CUBLAS_STREAM
			for (int k = 0; k < _nStreams; k++) {
				int indexW = k * _alignedMatrix * _filterDim;
				int indexO = k * _alignedMatrix;
				CHECK_CUBLAS(cublasSetStream(handle, streams[k]));
				CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _alignedMatrix, prevUniqueNodes, &alpha, subCalcError + (i * prevUniqueNodes), _alignedMatrix, weightRot + ((i + j * _depth) * _filterDim) + indexW, 1, &beta, prevError + (j * prevUniqueNodes) + indexO, 1));
			}
#else
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _filterDim, prevUniqueNodes, &alpha, subCalcError + (i * prevUniqueNodes), _filterDim, weightRot + ((i + j * _depth) * _filterDim), 1, &beta, prevError + (j * prevUniqueNodes), 1));
#endif 
		}
	}

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi back propagation\n\n";
	printFromCudaFormatted(prevError, prevNodes, sqrt(prevUniqueNodes));
#endif

}

void ConvolutionalStream::back_propagation(const double *prevOutput, const double &learningRate) {
	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);
}

void ConvolutionalStream::back_propagation_output(const double * prevOutput, const uint8_t * labels, const int & target, const double & learningRate) {
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

void ConvolutionalStream::calcBackPropagation(const double *prevOutput, const double &learningRate) {

	// Blocchi bidimensionali contenenti tanti thread quanti il numero di depth in output
	dim3 threadBlocks(_depth, 1, 1);

	// Tanti blocchi quante sono le righe e le colonne di error
	dim3 numBlocks(_width, _height, 1);

	Kernel::rot180BisK(numBlocks, threadBlocks, error, errorRot, _uniqueNodes);

#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore error ruotato\n\n";
	printFromCudaFormatted(errorRot, _nodes, _width);
#endif

	// Applicare derivata della funzione di attivazione
#ifdef STREAM
	if (_a == RELU)
		KernelStream::derivActReluK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, errorRot, tempOutput, _nodes);
	else if (_a == SIGMOID)
		KernelStream::derivActSigmoidK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, output, errorRot, _nodes);
	else if (_a == TANH)
		KernelStream::derivActTanhK(1, _alignedMatrixKernel, streamsBack, _nStreamsBack, output, errorRot, _nodes);
#else
	if (_a == RELU)
		Kernel::derivActReluK((_alignedNodes / THREADS), THREADS, errorRot, tempOutput, _nodes);
	else if (_a == SIGMOID)
		Kernel::derivActSigmoidK((_alignedNodes / THREADS), THREADS, output, errorRot, _nodes);
	else if (_a == TANH)
		Kernel::derivActTanhK((_alignedNodes / THREADS), THREADS, output, errorRot, _nodes);
#endif


#ifdef DEBUG
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nErrore commesso sui nodi con relativa derivata\n\n";
	printFromCudaFormatted(errorRot, _nodes, _width);
#endif

	updateWeights(prevOutput, learningRate);
}

void ConvolutionalStream::updateWeights(const double *prevOutput, const double &learningRate) {

	// Blocchi tridimensionali contenenti tanti thread quanti sono i nodi in output
	dim3 threadBlocks(_width, _height, 1);

	// Tanti blocchi quanti la grandezza dei filtri e il depth del livello precedente
	dim3 numBlocks(_filterWidth, _filterWidth, _prevLayerDepth);

	Kernel::createSubmatrixBisK(numBlocks, threadBlocks, subBack, prevOutput, _prevLayerWidth, _width, _stride, _filterDim);

#ifdef DEBUG_SUB
	CHECK(cudaDeviceSynchronize());
	std::cout << "\n\nValore submatrici backpropagation\n\n";
	printFromCudaFormatted(subBack, _uniqueNodes * _filterDim, _width);
#endif

	// Riempire la matrice temporanea di 0
	CHECK(cudaMemset(tempWeight, 0, _wBytes));

	//ora sono in una situazione simile al fully connected
	for (int i = 0; i < _depth; i++) {
		for (int j = 0; j < _prevLayerDepth; j++) {
#ifdef CUBLAS_STREAM
			for (int k = 0; k < _nStreamsBack; k++) {
				int indexW = k * _alignedMatrixBack* _uniqueNodes;
				int indexO = k * _alignedMatrixBack;
				CHECK_CUBLAS(cublasSetStream(handle, streams[k]));
				CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _alignedMatrixBack, _filterDim, &alpha, subBack + (j * _filterDim), _alignedMatrixBack, errorRot + (i * _uniqueNodes) + indexW, 1, &beta, tempWeight + ((i + j * _depth) * _filterDim) + indexO, 1));
			}
#else
			CHECK_CUBLAS(cublasDgemv(handle, CUBLAS_OP_T, _uniqueNodes, _filterDim, &alpha, subBack + (j * _filterDim), _uniqueNodes, errorRot + (i * _uniqueNodes), 1, &beta, tempWeight + ((i + j * _depth) * _filterDim), 1));
#endif
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

	Kernel::rot180BisK(numBlocks, threadBlocks, weight, weightRot, _filterDim);

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

}

int ConvolutionalStream::_calcOutput(bool withPadding) {
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


inline void ConvolutionalStream::initStreams(void) {
	this->_alignedNodes = ALIGN_UP(_nodes, THREADS);

	// Numero degli stream
	this->_nStreams = 5;
	this->_nStreamsBack = 4;


	// Numero di elementi che uno stream deve elaborare
	this->_alignedMatrix = _filterDim / _nStreams;
	this->_alignedMatrixBack = _uniqueNodes / _nStreamsBack;
	this->_alignedMatrixKernel = _nodes / _nStreamsBack;

	std::cout << _filterDim << "/" << _nStreams << "=>" << _alignedMatrix << " ";
	std::cout << _uniqueNodes << "/" << _nStreamsBack << "=>" << _alignedMatrixBack << " ";
	std::cout << _nodes << "/" << _nStreamsBack << "=>" << _alignedMatrixKernel << std::endl;
}

void ConvolutionalStream::deleteCuda() {
	CHECK_CUBLAS(cublasDestroy(handle));

	for (int i = 0; i < _nStreams; i++) {
		CHECK(cudaStreamDestroy(streams[i]));
	}

	CHECK(cudaFree(weight));
	CHECK(cudaFree(weightRot));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(errorRot));
	CHECK(cudaFree(tempWeight));
	CHECK(cudaFree(tempOutput));
	CHECK(cudaFree(subForward));
	CHECK(cudaFree(subCalcError));
	CHECK(cudaFree(subBack));
	CHECK(cudaFree(padding));

	free(streams);
}

void ConvolutionalStream::printW() {
	printFromCudaFormatted(weight, _wDim, _filterWidth);
	//printFromCudaFormatted(bias, _nodes, _width);
}