#include "KernelStreamsBisCPU.h"
#include "Kernel.h"

/* FUNZIONI DI ATTIVAZIONE E RELATIVE DERIVATE */

void KernelStreamsBis::actReluK(dim3 b, dim3 t, cudaStream_t stream, double *output, double *temp, const int &nodes) {
#ifdef _WIN32
	actRelu NvCUDA4(b, t, 0, stream) (output, temp, nodes);
#else
	actRelu << <b, t, 0, stream >> > (output, temp, nodes);
#endif
}

void KernelStreamsBis::actSigmoidK(dim3 b, dim3 t, cudaStream_t stream, double *output, const int &nodes) {
#ifdef _WIN32
	actSigmoid NvCUDA4(b, t, 0, stream) (output, nodes);
#else
	actSigmoid << <b, t, 0, stream >> > (output, nodes);
#endif
}

void KernelStreamsBis::actTanhK(dim3 b, dim3 t, cudaStream_t stream, double *output, const int &nodes) {
#ifdef _WIN32
	actTanh NvCUDA4(b, t, 0, stream) (output, nodes);
#else
	actTanh << <b, t, 0, stream >> > (output, nodes);
#endif
}

/* METODI CONVOLUZIONALE */

void KernelStreamsBis::createSubmatrixBisK(dim3 b, dim3 t, cudaStream_t stream, double * sub, const double * prevOutput, const int prevLayerWidth, const int filterWidth, const int stride, const int uniqueNodes) {
#ifdef _WIN32
	createSubmatrixBis NvCUDA4(b, t, 0, stream) (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#else
	createSubmatrixBis << <b, t, 0, stream >> > (sub, prevOutput, prevLayerWidth, filterWidth, stride, uniqueNodes);
#endif
}
