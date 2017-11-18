#include "Kernel.h"

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
		weight[tid] = 1.0;
#else
		weight[tid] = 0.4 * r;
#endif
}

void Kernel::initWeightK(dim3 b, dim3 t, double *weight, const int &wDim, curandState *states) {
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
		bias[tid] = 1.0;
#else
		bias[tid] = 0.4 * r;
#endif
}

void Kernel::initBiasK(dim3 b, dim3 t, double *weight, const int &wDim, curandState *states) {
#ifdef _WIN32
	initBias NvCUDA2(b, t) (weight, wDim, states);
#else
	initBias << <b, t >> > (weight, wDim, states);
#endif
}

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

void Kernel::outputErrorK(dim3 b, dim3 t, const double *output, double *error, const uint8_t *label, const int &target, const int &nodes) {
#ifdef _WIN32
	outputError NvCUDA2(b, t) (output, error, label, target, nodes);
#else
	outputError << <b, t >> > (output, error, label, target, nodes);
#endif
}




/* Funzione di attivazione del Sigmoide e derivata */

__global__ void actRelu(double *output, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		output[tid] = log(1 + exp((output[tid])));
}

__global__ void derivActRelu(const double *output, double *error, const int node) {

	// Gestione degli indici	
	const unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < node)
		error[tid] = error[tid] * (1 / (1 + (exp((-output[tid])))));
}

void Kernel::actReluK(dim3 b, dim3 t, double *output, const int &nodes) {
#ifdef _WIN32
	actRelu NvCUDA2(b, t) (output, nodes);
#else
	actRelu << <b, t >> > (output, nodes);
#endif 
}

void Kernel::derivActReluK(dim3 b, dim3 t, const double *output, double *error, const int &nodes) {
#ifdef _WIN32
	derivActRelu NvCUDA2(b, t) (output, error, nodes);
#else
	derivActRelu << <b, t >> > (output, error, nodes);
#endif 
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

	if (tid < node)
		error[tid] = error[tid] * (output[tid] * (1 - output[tid]));
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
		error[tid] = error[tid] * (1 - pow(tanh(output[tid]), 2));
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
