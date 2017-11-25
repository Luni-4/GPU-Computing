#include <iostream>

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


std::vector<double> FullyConnected::getWeights(void) {
	std::vector<double> wCPU(_wDim);
	CHECK(cudaMemcpy(&wCPU[0], weight, _wBytes, cudaMemcpyDeviceToHost));
	return wCPU;
}

std::vector<double> FullyConnected::getBias(void) {
	std::vector<double> bCPU(_nodes);
	CHECK(cudaMemcpy(&bCPU[0], bias, _nodes * sizeof(double), cudaMemcpyDeviceToHost));
	return bCPU;
}

uint8_t FullyConnected::getPredictionIndex(void) {
	int maxIndex;
	
	// Individuare indice (classe) che corrisponde al valore massimo di output
	CHECK_CUBLAS(
		cublasIdamax(handle, _nodes, output, 1, &maxIndex));
	
	return maxIndex;
}

void FullyConnected::defineCuda(const int &prevLayerWidth, const int &prevLayerHeight, const int &prevLayerDepth) {

    // Creare l'handle di cuBLAS
	CHECK_CUBLAS(cublasCreate(&handle));
	
	// Impostazioni della cache
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	
	// Numero degli stream
	//_nStreams = 15;
	
	// Creazione degli stream
	//streams = (cudaStream_t *)malloc(_nStreams * sizeof(cudaStream_t));
	
	/*for(int i = 0; i < _nStreams; i++) {
		CHECK(cudaStreamCreate(&(streams[i])));
	}*/

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
	pettyPrintCuda(weight, _wDim, _prevLayerDim);
	std::cout << "\n\nValore dei bias\n\n";
	pettyPrintCuda(bias, _nodes, 1);
	std::cout << "\n\n\n\n";
#endif

	// Distrugge gli stati
	CHECK(cudaFree(devStates));
}


void FullyConnected::forward_propagation(const double *prevOutput) {
    
   /* for(int i = 0; i < _nStreams; i++) {
        CHECK_CUBLAS(cublasSetStream(handle,streams[i]));    
        CHECK_CUBLAS(
               cublasDgemv(handle, CUBLAS_OP_T, _prevLayerDim, _nStreams, &alpha, weight + (i * _nStreams), _prevLayerDim, prevOutput, 1, &beta, output + (i * _nStreams), 1));
    //}*/
    
    CHECK_CUBLAS(
        cublasDgemv(handle, CUBLAS_OP_T, _prevLayerDim, _nodes, &alpha, weight, _prevLayerDim, prevOutput, 1, &beta, output, 1));
        
    // CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
    
    /*for(int i = 0; i < _nStreams; i++) {
        CHECK(cudaStreamSynchronize(streams[i]));    
    }*/
               
	/*CHECK_CUBLAS(
			cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _nodes, _prevLayerDim,
		            &alpha, prevOutput, 1,
		            weight, _prevLayerDim, &beta,
		            output, 1));*/
    
    /*CHECK_CUBLAS(
            cublasDgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, 30, _prevLayerDim,
		            &alpha, prevOutput, 1, 0,
		            weight, _prevLayerDim, _prevLayerDim * 30, &beta,
		            output, 1, 30,
		            10));*/
	
#ifdef DEBUG
	std::cout << "\n\nOutput dei nodi senza bias\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif

    CHECK_CUBLAS(
               cublasDaxpy(handle, _nodes, &alpha, bias, 1, output, 1));

	// Somma con il bias
	//CHECK_CUBLAS(
		//cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, _nodes, &alpha, bias, 1, &alpha, output, 1, output, 1));
		

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nOutput dei nodi con bias sommato\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif

	// Applicare funzione di attivazione
	if (_a == RELU)
		Kernel::actReluK(1, _alignedNodes, output, temp, _nodes);
	else if (_a == SIGMOID)
		Kernel::actSigmoidK(1, _alignedNodes, output, _nodes);
	else if (_a == TANH)
		Kernel::actTanhK(1, _alignedNodes, output, _nodes);

	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nOutput dei nodi con funzione di attivazione\n\n";
	pettyPrintCuda(output, _nodes, 1);
#endif
}

void FullyConnected::back_propagation(const double *prevOutput, const double *forwardWeight, const double *forwardError, const int &forwardNodes, const double &learningRate) {
    
    // Propagazione dell'errore dal livello successivo
    CHECK_CUBLAS(
		cublasDgemv(handle, CUBLAS_OP_N, _nodes ,forwardNodes, &alpha, forwardWeight, _nodes, forwardError, 1, &beta, error, 1));
	
	// CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nForward weight\n\n";
	pettyPrintCuda(forwardWeight, _nodes * forwardNodes, forwardNodes);
	std::cout << "\n\nForward error\n\n";
	pettyPrintCuda(forwardError, forwardNodes, 1);
	std::cout << "\n\nErrore commesso sui nodi back propagation\n\n";
	pettyPrintCuda(error, _nodes, 1);
#endif
		
	// Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);   

}

void FullyConnected::back_propagation_output(const double *prevOutput, const uint8_t *labels, const int &target, const double &learningRate) {
    
    // Calcolo dell'errore per ogni nodo
    Kernel::outputErrorK(_alignedNodes / THREADS, _alignedNodes, output, error, labels, target, _nodes);
    
    // CPU deve attendere che esecuzione della funzione finisca
	CHECK(cudaDeviceSynchronize());
	
#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi back propagation output\n\n";
	pettyPrintCuda(error, _nodes, 1);
#endif
    
    // Calcolo della Back Propagation
	calcBackPropagation(prevOutput, learningRate);  

}

void FullyConnected::deleteCuda(void) {

	CHECK_CUBLAS(cublasDestroy(handle));
	
	/*for(int i = 0; i < _nStreams; i++){
		CHECK(cudaStreamDestroy(streams[i]));
	}*/
	
	CHECK(cudaFree(weight));
	CHECK(cudaFree(bias));
	CHECK(cudaFree(output));
	CHECK(cudaFree(error));
	CHECK(cudaFree(temp));
	
	//free(streams);
	
}

void FullyConnected::calcBackPropagation(const double *prevOutput, const double &learningRate) {

    // Applicare derivata della funzione di attivazione
	if (_a == RELU)
		Kernel::derivActReluK(_alignedNodes / THREADS, _alignedNodes, output, error, temp, _nodes);
	else if (_a == SIGMOID)
		Kernel::derivActSigmoidK(_alignedNodes / THREADS, _alignedNodes, output, error, _nodes);
	else if (_a == TANH)
		Kernel::derivActTanhK(_alignedNodes / THREADS, _alignedNodes, output, error, _nodes);
		
	// CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());
    
#ifdef DEBUG
	std::cout << "\n\nErrore commesso sui nodi con relativa derivata\n\n";
	pettyPrintCuda(error, _nodes, 1);
#endif
    
    // Aggiornare i pesi (da mettere in funzione)    
    updateWeights(prevOutput, learningRate);
}

void FullyConnected::updateWeights(const double *prevOutput, const double &learningRate) {
	
	// Riempire la matrice temporanea di 0
	CHECK(cudaMemset(temp, 0, _wBytes));

#ifdef DEBUG
	std::cout << "\n\nMatrice temporanea valore iniziale\n\n";
	pettyPrintCuda(temp, _wDim, _prevLayerDim);
#endif
	
	// Deve ricevere lo scalare dal device
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);		
	
	for (int i = 0; i < _nodes; i++){
	    CHECK_CUBLAS(
		    cublasDaxpy(handle, _prevLayerDim, &error[i], prevOutput, 1, temp + (i * _prevLayerDim), 1));   
        
        // CPU deve attendere che esecuzione della funzione finisca
        CHECK(cudaDeviceSynchronize());
    }

#ifdef DEBUG
	std::cout << "\n\nMatrice temporanea per aggiornamento pesi\n\n";
	pettyPrintCuda(temp, _wDim, _prevLayerDim);
#endif

    // Deve ricevere lo scalare dall'host
	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);    
    
    // Aggiornamento effettivo dei pesi 
    CHECK_CUBLAS(
		cublasDaxpy(handle, _wDim, &learningRate, temp, 1, weight, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nMatrice dei pesi aggiornata\n\n";
	pettyPrintCuda(weight, _wDim, _prevLayerDim);
#endif
	
	// Aggiornamento del bias 
    CHECK_CUBLAS(
		cublasDaxpy(handle, _nodes, &learningRate, error, 1, bias, 1));
		
    // CPU deve attendere che esecuzione della funzione finisca
    CHECK(cudaDeviceSynchronize());

#ifdef DEBUG
	std::cout << "\n\nVettore del bias aggiornato\n\n";
	pettyPrintCuda(bias, _nodes, 1);
#endif
}
