#include "KernelStreamsBisCPU.h"
#include "Kernel.h"

/*  INIZIALIZZAZIONE DEI PESI */

//void KernelStreamsBis::initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandStateXORWOW_t *states) {
//#ifdef _WIN32
//	initWeight NvCUDA2(b, t) (weight, wDim, states);
//#else
//	initWeight << <b, t >> > (weight, wDim, states);
//#endif
//}
//
//void KernelStreamsBis::initBiasK(dim3 b, dim3 t, double *bias, const int &wDim, curandStateXORWOW_t *states) {
//#ifdef _WIN32
//	initBias NvCUDA2(b, t) (bias, wDim, states);
//#else
//	initBias << <b, t >> > (bias, wDim, states);
//#endif
//}

/* CALCOLO DEL DELTA */

//void KernelStreamsBis::outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes) {
//#ifdef _WIN32
//	outputError NvCUDA2(b, t) (output, error, label, target, nodes);
//#else
//	outputError << <b, t >> > (output, error, label, target, nodes);
//#endif
//}

/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

void KernelStreamsBis::actReluK(dim3 b, dim3 t, cudaStream_t stream, double *output, double *temp, const int &nodes) {
#ifdef _WIN32
	actRelu NvCUDA4(b, t, 0, stream) (output, temp, nodes);
#else
	actRelu << <b, t, 0, stream >> > (output, temp, nodes);
#endif
}

//void KernelStreamsBis::derivActReluK(dim3 b, dim3 t, double *error, double *temp, const int &nodes) {
//#ifdef _WIN32
//	derivActRelu NvCUDA2(b, t) (error, temp, nodes);
//#else
//	derivActRelu << <b, t >> > (error, temp, nodes);
//#endif 
//}

void KernelStreamsBis::actSigmoidK(dim3 b, dim3 t, cudaStream_t stream, double *output, const int &nodes) {
#ifdef _WIN32
	actSigmoid NvCUDA4(b, t, 0, stream) (output, nodes);
#else
	actSigmoid << <b, t, 0, stream >> > (output, nodes);
#endif
}

//void KernelStreamsBis::derivActSigmoidK(dim3 b, dim3 t, const double *output, double *error, const int &nodes) {
//#ifdef _WIN32
//	derivActSigmoid NvCUDA2(b, t) (output, error, nodes);
//#else
//	derivActSigmoid << <b, t >> > (output, error, nodes);
//#endif
//}

void KernelStreamsBis::actTanhK(dim3 b, dim3 t, cudaStream_t stream, double *output, const int &nodes) {
#ifdef _WIN32
	actTanh NvCUDA4(b, t, 0, stream) (output, nodes);
#else
	actTanh << <b, t, 0, stream >> > (output, nodes);
#endif
}

//void KernelStreamsBis::derivActTanhK(dim3 b, dim3 t, const double *output, double *error, const int &nodes) {
//#ifdef _WIN32
//	derivActTanh NvCUDA2(b, t) (output, error, nodes);
//#else
//	derivActTanh << <b, t >> > (output, error, nodes);
//#endif
//}

/* AGGIORNAMENTO DEI PESI FULLY CONNECTED*/

//void KernelStreamsBis::errorPrevOutputK(dim3 b, dim3 t, double *temp, const double *prevOutput, const double *error, const int &nodes, const int &dim, const int &prevDim) {
//#ifdef _WIN32
//	errorPrevOutput NvCUDA2(b, t) (temp, prevOutput, error, dim, prevDim);
//#else
//	errorPrevOutput << <b, t >> > (temp, prevOutput, error, dim, prevDim);
//#endif
//}

/* METODI CONVOLUZIONALE */

void KernelStreamsBis::createSubmatrixBisK(dim3 b, dim3 t, cudaStream_t stream, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrixBis NvCUDA4(b, t, 0, stream) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrixBis << <b, t, 0, stream >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}

//void KernelStreamsBis::zeroPaddingBisK(dim3 b, dim3 t, double * error, const double * forwardError, const int forwardErrorWidth, const int forwardFilterWidth) {
//#ifdef _WIN32
//	zeroPaddingBis NvCUDA2(b, t) (error, forwardError, forwardErrorWidth, forwardFilterWidth);
//#else
//	zeroPaddingBis << <b, t >> > (error, forwardError, forwardErrorWidth, forwardFilterWidth);
//#endif
//}
//
//void KernelStreamsBis::rot180BisK(dim3 b, dim3 t, const double * forwardWeight, double * forwardWeightRot, int filterDim) {
//#ifdef _WIN32
//	rot180Bis NvCUDA2(b, t) (forwardWeight, forwardWeightRot, filterDim);
//#else
//	rot180Bis << <b, t >> > (forwardWeight, forwardWeightRot, filterDim);
//#endif
//}
