#include "KernelCPU.h"
#include "Kernel.h"

/*  INIZIALIZZAZIONE DEI PESI */

void Kernel::initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandStateXORWOW_t *states) {
#ifdef _WIN32
	initWeight NvCUDA2(b, t) (weight, wDim, states);
#else
	initWeight << <b, t >> > (weight, wDim, states);
#endif
}

void Kernel::initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandStateXORWOW_t *states) {
#ifdef _WIN32
	initBias NvCUDA2(b, t) (bias, wDim, states);
#else
	initBias << <b, t >> > (bias, wDim, states);
#endif
}

/* CALCOLO DEL DELTA */

void Kernel::outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes) {
#ifdef _WIN32
	outputError NvCUDA2(b, t) (output, error, label, target, nodes);
#else
	outputError << <b, t >> > (output, error, label, target, nodes);
#endif
}

/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

void Kernel::actReluK(dim3 b, dim3 t, double *output, double *temp, const int &nodes) {
#ifdef _WIN32
	actRelu NvCUDA2(b, t) (output, temp, nodes);
#else
	actRelu << <b, t >> > (output, temp, nodes);
#endif
}

void Kernel::derivActReluK(dim3 b, dim3 t, double *error, double *temp, const int &nodes) {
#ifdef _WIN32
	derivActRelu NvCUDA2(b, t) (error, temp, nodes);
#else
	derivActRelu << <b, t >> > (error, temp, nodes);
#endif 
}

void Kernel::actSigmoidK(dim3 b, dim3 t, double *output, const int &nodes) {
#ifdef _WIN32
	actSigmoid NvCUDA2(b, t) (output, nodes);
#else
	actSigmoid << <b, t >> > (output, nodes);
#endif
}

void Kernel::derivActSigmoidK(dim3 b, dim3 t, const double *output, double *error, const int &nodes) {
#ifdef _WIN32
	derivActSigmoid NvCUDA2(b, t) (output, error, nodes);
#else
	derivActSigmoid << <b, t >> > (output, error, nodes);
#endif
}

void Kernel::actTanhK(dim3 b, dim3 t, double *output, const int &nodes) {
#ifdef _WIN32
	actTanh NvCUDA2(b, t) (output, nodes);
#else
	actTanh << <b, t >> > (output, nodes);
#endif
}

void Kernel::derivActTanhK(dim3 b, dim3 t, const double *output, double *error, const int &nodes) {
#ifdef _WIN32
	derivActTanh NvCUDA2(b, t) (output, error, nodes);
#else
	derivActTanh << <b, t >> > (output, error, nodes);
#endif
}

/* AGGIORNAMENTO DEI PESI FULLY CONNECTED*/

void Kernel::errorPrevOutputK(dim3 b, dim3 t, double *temp, const double *prevOutput, const double *error, const int &nodes, const int &dim, const int &prevDim) {
#ifdef _WIN32
	errorPrevOutput NvCUDA2(b, t) (temp, prevOutput, error, dim, prevDim);
#else
	errorPrevOutput << <b, t >> > (temp, prevOutput, error, dim, prevDim);
#endif
}

/* METODI CONVOLUZIONALE */

void Kernel::createSubmatrixBisK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrixBis NvCUDA2(b, t) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrixBis << <b, t >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

void Kernel::createSubmatrixProductK(dim3 b, dim3 t, double * sub, const double * prevOutput, const double * weightRot, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrixProduct NvCUDA2(b, t) (sub, prevOutput, weightRot, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrixProduct << <b, t >> > (sub, prevOutput, weightRot, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

void Kernel::outputFromSubK(dim3 b, dim3 t, double * output, double * sub, int filterDim) {
#ifdef _WIN32
	outputFromSub NvCUDA2(b, t) (output, sub, filterDim);
#else
	outputFromSub << <b, t >> > (output, sub, filterDim);
#endif
}


void Kernel::createSubmatrixK(dim3 b, dim3 t, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrix NvCUDA2(b, t) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrix << <b, t >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

void Kernel::zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPaddingBis NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPaddingBis << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#endif
}

void Kernel::zeroPaddingK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
#ifdef _WIN32
	zeroPadding NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#else
	zeroPadding << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
#endif
}

void Kernel::rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
	rot180Bis NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
	rot180Bis << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
#endif
}

void Kernel::rot180K(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
#ifdef _WIN32
	rot180 NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
#else
	rot180 << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
#endif
}
