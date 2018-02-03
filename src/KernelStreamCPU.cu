#include "KernelStreamCPU.h"
#include "Kernel.h"

/*  INIZIALIZZAZIONE DEI PESI */

void KernelStream::initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandState *states) {
#ifdef _WIN32
	initWeight NvCUDA2(b, t) (weight, wDim, states);
#else
	initWeight << <b, t >> > (weight, wDim, states);
#endif
}

void KernelStream::initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandState *states) {
#ifdef _WIN32
	initBias NvCUDA2(b, t) (bias, wDim, states);
#else
	initBias << <b, t >> > (bias, wDim, states);
#endif
}

/* CALCOLO DEL DELTA */

void KernelStream::outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes) {
#ifdef _WIN32
	outputError NvCUDA2(b, t) (output, error, label, target, nodes);
#else
	outputError << <b, t >> > (output, error, label, target, nodes);
#endif
}

/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

void KernelStream::actReluK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, double *temp, const int &nodes) {
	for (int i = 0; i < nStreams; i++) {
		int indexO = i * t.x;
#ifdef _WIN32
		actRelu NvCUDA4(b, t, 0, streams[i]) (output + indexO, temp + indexO, nodes);
#else
		actRelu << <b, t, 0, streams[i] >> > (output + indexO, temp + indexO, nodes);
#endif
	}
}

void KernelStream::derivActReluK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *error, double *temp, const int &nodes) {
	for (int i = 0; i < nStreams; i++) {
		int indexO = i * t.x;
#ifdef _WIN32
		derivActRelu NvCUDA4(b, t, 0, streams[i]) (error + indexO, temp + indexO, nodes);
#else
		derivActRelu << <b, t, 0, streams[i] >> > (error + indexO, temp + indexO, nodes);
#endif
	}
}

void KernelStream::actSigmoidK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, const int &nodes) {
	for (int i = 0; i < nStreams; i++) {
		int indexO = i * t.x;
#ifdef _WIN32
		actSigmoid NvCUDA4(b, t, 0, streams[i]) (output + indexO, nodes);
#else
		actSigmoid << <b, t, 0, streams[i] >> > (output + indexO, nodes);
#endif
	}
}

void KernelStream::derivActSigmoidK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, const double *output, double *error, const int &nodes) {
	for (int i = 0; i < nStreams; i++) {
		int indexO = i * t.x;
#ifdef _WIN32
		derivActSigmoid NvCUDA4(b, t, 0, streams[i]) (output + indexO, error + indexO, nodes);
#else
		derivActSigmoid << <b, t, 0, streams[i] >> > (output + indexO, error + indexO, nodes);
#endif
	}
}

void KernelStream::actTanhK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *output, const int &nodes) {
	for (int i = 0; i < nStreams; i++) {
		int indexO = i * t.x;
#ifdef _WIN32
		actTanh NvCUDA4(b, t, 0, streams[i]) (output + indexO, nodes);
#else
		actTanh << <b, t, 0, streams[i] >> > (output + indexO, nodes);
#endif
	}
}

void KernelStream::derivActTanhK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, const double *output, double *error, const int &nodes) {
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

void KernelStream::errorPrevOutputK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double *temp, const double *prevOutput, const double *error, const int &nodes, const int &dim, const int &prevDim) {
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

void KernelStream::createSubmatrixBisK(dim3 b, dim3 t, const cudaStream_t *streams, const int &nStreams, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
	for (int i = 0; i < nStreams; i++) {
#ifdef _WIN32
		createSubmatrixBis NvCUDA4(b, t, 0, streams[i])  (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
		createSubmatrixBis << <b, t, 0, streams[i] >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
	}
}

void KernelStream::zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPaddingBis NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPaddingBis << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#endif
}

void KernelStream::rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
	rot180Bis NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
	rot180Bis << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
#endif
}