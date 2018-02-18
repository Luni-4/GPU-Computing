#include <math_constants.h>
#include <cstdio>

#include "Kernel.h"

/*  INIZIALIZZAZIONE DEI PESI */

__global__ void initWeight(double *weight, const int wDim, curandStateXORWOW_t *states) {
	// Gestione degli indici	
	const unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
	const unsigned int tid = blockId * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	double r = curand_normal_double(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < wDim)
#ifdef TOYINPUT
		weight[tid] = 1.0;
#else
		//weight[tid] = 0.01 * r;
		weight[tid] = 0.13;
#endif
}

__global__ void initBias(double *bias, const int node, curandStateXORWOW_t *states) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// Sequenza di rand diversa per ogni thread
	curand_init(tid, 0, 0, &states[tid]);

	// Variabile che conterrà il valore casuale
	double r = curand_normal_double(&states[tid]);

	if (tid % 2 == 0)
		r = -r;

	if (tid < node)
#ifdef TOYINPUT
		bias[tid] = 1.0;
#else
		//bias[tid] = r;
		bias[tid] = 0.0;
#endif
}

/* CALCOLO DEL DELTA */

__global__ void outputError(const double *output, double *error, const uint8_t *label, const int target, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	double trueLabel = 0.0;

	/* Il predittore dovrebbe predire con probabilità 1 solo la label passata alla funzione, quindi la variabile
	trueLabel contiene il valore che ci si aspetterebbe dal predittore, cioè 1 */
	if (tid == label[target])
		trueLabel = 1.0;

	// L'errore commesso è dato dalla differenza tra la predizione ottenuta e il valore reale dell'etichetta
	if (tid < node)
		error[tid] = trueLabel - output[tid];
}

/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

/* Funzione di attivazione del Relu e derivata */
__global__ void actRelu(double *output, double *temp, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node) {
		temp[tid] = output[tid];
		output[tid] = log(1 + exp((output[tid])));
	}
}

__global__ void derivActRelu(double *error, double *temp, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (1 / (1 + (exp((-temp[tid])))));
}

/* Funzione di attivazione del Sigmoide e derivata */
__global__ void actSigmoid(double *output, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = 1 / (1 + (exp((-output[tid]))));
}

__global__ void derivActSigmoid(const double *output, double *error, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	double r;

	if (tid < node) {
		r = output[tid] * (1 - output[tid]);
		error[tid] = error[tid] * r;
	}
}

/* Funzione di attivazione della Tanh e derivata */
__global__ void actTanh(double *output, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = tanh(output[tid]);
}

__global__ void derivActTanh(const double *output, double *error, const int node) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (1 - pow(output[tid], 2));
}

/* AGGIORNAMENTO DEI PESI FULLY CONNECTED*/

__global__ void errorPrevOutput(double *temp, const double *prevOutput, const double *error, const int node, const int prevDim) {
	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	const unsigned int column = tid % prevDim;
	const unsigned int row = (tid - column) / prevDim;

	if (tid < node)
		temp[tid] = error[row] * prevOutput[column];
}

/* METODI CONVOLUZIONALE */

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

__global__ void createSubmatrixProduct(double * sub, const double * prevOutput, const double * weightRot, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
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

	const unsigned int wTid = blockDim.x * threadIdx.y + threadIdx.x;

	sub[tid] = prevOutput[pTid] * weightRot[wTid];
}

__global__ void outputFromSub(double * output, double * sub, int filterDim) {
	const unsigned int wTid = blockDim.x * threadIdx.y + threadIdx.x;
	const unsigned int sTid = wTid * filterDim;

	double result = 0;
	for (int i = 0; i < filterDim; i++) {
		result += sub[sTid + i];
	}

	output[wTid] = result;
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

__global__ void zeroPadding(double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
	//blockIdx.y rappresenta la riga 
	const unsigned int paddingLeft = forwardFilterWidth - 1;
	const unsigned int widthWithPadding = forwardErrorWidth + (paddingLeft * 2);
	const unsigned int tid = ((blockIdx.y + paddingLeft) * widthWithPadding) + (widthWithPadding * widthWithPadding * threadIdx.x) + paddingLeft;

	memcpy((error + tid), (forwardError + blockIdx.y * forwardErrorWidth + forwardErrorWidth * forwardErrorWidth * threadIdx.x), (forwardErrorWidth * sizeof(double)));
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
