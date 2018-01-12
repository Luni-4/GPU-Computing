#include "Kernel_Stream.h"

#include <cstdio>

namespace Kernel_Stream {

	/*  INIZIALIZZAZIONE DEI PESI */

	__global__ void initWeight(double *weight, const int wDim, curandState *states) {
		// Gestione degli indici	
		const unsigned int blockId = blockIdx.y * gridDim.x + blockIdx.x;
		const unsigned int tid = blockId * blockDim.x + threadIdx.x;

		// Sequenza di rand diversa per ogni thread
		curand_init(tid, 0, 0, &states[tid]);

		// Variabile che conterrà il valore casuale
		double r = curand_uniform_double(&states[tid]);

		if (tid % 2 == 0)
			r = -r;

		if (tid < wDim)
#ifdef TOYINPUT
			weight[tid] = 1.0f;
#else
			weight[tid] = 0.4 * r;
#endif
	}

	void Kernel_Stream::initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandState *states) {
#ifdef _WIN32
		initWeight NvCUDA2(b, t) (weight, wDim, states);
#else
		initWeight << <b, t >> > (weight, wDim, states);
#endif
	}

	__global__ void initBias(double *bias, const int node, curandState *states) {
		// Gestione degli indici	
		const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		// Sequenza di rand diversa per ogni thread
		curand_init(tid, 0, 0, &states[tid]);

		// Variabile che conterrà il valore casuale
		double r = curand_uniform_double(&states[tid]);

		if (tid % 2 == 0)
			r = -r;

		if (tid < node)
#ifdef TOYINPUT
			bias[tid] = 1.0f;
#else
			bias[tid] = 0.4 * r;
#endif
	}

	void Kernel_Stream::initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandState *states) {
#ifdef _WIN32
		initBias NvCUDA2(b, t) (bias, wDim, states);
#else
		initBias << <b, t >> > (bias, wDim, states);
#endif
	}






	/* CALCOLO DEL DELTA */

	__global__ void outputError(const double *output, double *error, const uint8_t *label, const int target, const int node) {
		// Gestione degli indici	
		const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		int trueLabel = 0;

		/* Il predittore dovrebbe predire con probabilità 1 solo la label passata alla funzione, quindi la variabile
		trueLabel contiene il valore che ci si aspetterebbe dal predittore, cioè 1 */
		if (tid == label[target])
			trueLabel = 1;

		// L'errore commesso è dato dalla differenza tra la predizione ottenuta e il valore reale dell'etichetta
		if (tid < node)
			error[tid] = trueLabel - output[tid];
	}

	void Kernel_Stream::outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes) {
#ifdef _WIN32
		outputError NvCUDA2(b, t) (output, error, label, target, nodes);
#else
		outputError << <b, t >> > (output, error, label, target, nodes);
#endif
	}




	/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */


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

	void Kernel_Stream::actReluK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, double *temp, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			actRelu NvCUDA4(b, t, 0, streams[i]) (output + indexO, temp + indexO, nodes);
#else
			actRelu << <b, t, 0, streams[i] >> > (output + indexO, temp + indexO, nodes);
#endif
		}
	}

	void Kernel_Stream::derivActReluK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *error, double *temp, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			derivActRelu NvCUDA4(b, t, 0, streams[i]) (error + indexO, temp + indexO, nodes);
#else
			derivActRelu << <b, t, 0, streams[i] >> > (error + indexO, temp + indexO, nodes);
#endif
		}
	}

	__global__ void actSigmoid(double *output, const int node) {
		// Gestione degli indici	
		const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < node)
			output[tid] = 1 / (1 + (exp((-output[tid]))));
	}

	__global__ void derivActSigmoid(const double *output, double *error, const int node) {
		// Gestione degli indici	
		const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

		if (tid < node)
			error[tid] = error[tid] * (output[tid] * (1 - output[tid]));
	}

	void Kernel_Stream::actSigmoidK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			actSigmoid NvCUDA4(b, t, 0, streams[i]) (output + indexO, nodes);
#else
			actSigmoid << <b, t, 0, streams[i] >> > (output + indexO, nodes);
#endif
		}
	}

	void Kernel_Stream::derivActSigmoidK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, const double *output, double *error, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			derivActSigmoid NvCUDA4(b, t, 0, streams[i]) (output + indexO, error + indexO, nodes);
#else
			derivActSigmoid << <b, t, 0, streams[i] >> > (output + indexO, error + indexO, nodes);
#endif
		}
	}

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

	void Kernel_Stream::actTanhK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			actTanh NvCUDA4(b, t, 0, streams[i]) (output + indexO, nodes);
#else
			actTanh << <b, t, 0, streams[i] >> > (output + indexO, nodes);
#endif
		}
	}

	void Kernel_Stream::derivActTanhK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, const double *output, double *error, const int &nodes) {
		for (int i = 0; i < nStreams; i++) {
			int indexO = i * t.x;
#ifdef _WIN32
			derivActTanh NvCUDA4(b, t, 0, streams[i]) (output + indexO, error + indexO, nodes);
#else
			derivActTanh << <b, t, 0, streams[i] >> > (output + indexO, error + indexO, nodes);
#endif
		}
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

	void Kernel_Stream::errorPrevOutputK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *temp, const double *prevOutput, const double *error, const int &nodes, const int &dim, const int &prevDim) {
		for (int i = 0; i < nStreams; i++) {
			int indexW = i * dim;
			int indexO = i * nodes;
#ifdef _WIN32
			errorPrevOutput NvCUDA4(b, t, 0, streams[i]) (temp + indexW, prevOutput, error + indexO, dim, prevDim);
#else
			errorPrevOutput << <b, t, 0, streams[i] >> > (temp + indexW, prevOutput, error + indexO, dim, prevDim);
#endif
		}
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

	void Kernel_Stream::createSubmatrixBisK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
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

	void Kernel_Stream::createSubmatrixK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
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

	void Kernel_Stream::zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
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

	void Kernel_Stream::zeroPaddingK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
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

	void Kernel_Stream::rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
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

	void Kernel_Stream::rot180K(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
		rot180 NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
		rot180 << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
#endif
	}
}