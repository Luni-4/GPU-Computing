#pragma once

#include <vector>

// Converte un numero intero al multiplo più vicino di 32
#define ALIGN_UP(a) ((a + 31) / 32) * 32


#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t err = call;                                              \
    if (err != cudaSuccess)                                                    \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", err,                         \
                cudaGetErrorString(err));                                      \
    }                                                                          \
}

#define CHECK_CUBLAS(call)                                                     \
{                                                                              \
    cublasStatus_t err;                                                        \
    if ((err = (call)) != CUBLAS_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CUBLAS error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CURAND(call)                                                     \
{                                                                              \
    curandStatus_t err;                                                        \
    if ((err = (call)) != CURAND_STATUS_SUCCESS)                               \
    {                                                                          \
        fprintf(stderr, "Got CURAND error %d at %s:%d\n", err, __FILE__,       \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUFFT(call)                                                      \
{                                                                              \
    cufftResult err;                                                           \
    if ( (err = (call)) != CUFFT_SUCCESS)                                      \
    {                                                                          \
        fprintf(stderr, "Got CUFFT error %d at %s:%d\n", err, __FILE__,        \
                __LINE__);                                                     \
        exit(1);                                                               \
    }                                                                          \
}

#define CHECK_CUSPARSE(call)                                                   \
{                                                                              \
    cusparseStatus_t err;                                                      \
    if ((err = (call)) != CUSPARSE_STATUS_SUCCESS)                             \
    {                                                                          \
        fprintf(stderr, "Got error %d at %s:%d\n", err, __FILE__, __LINE__);   \
        cudaError_t cuda_err = cudaGetLastError();                             \
        if (cuda_err != cudaSuccess)                                           \
        {                                                                      \
            fprintf(stderr, "  CUDA error \"%s\" also detected\n",             \
                    cudaGetErrorString(cuda_err));                             \
        }                                                                      \
        exit(1);                                                               \
    }                                                                          \
}

template<typename T>
inline void printVector(std::vector<T> &a, const int &dim) {	
    
	for (std::size_t i = 0; i < a.size(); i++) {
		std::cout << a[i] << " ";
		if ((i + 1) % dim == 0)
			std::cout << " :" << i + 1 << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;
	}
}


inline void printFromCuda(const double *deb, const int &dim) {

	// DEBUG
	std::vector<double> outputC(dim);
	CHECK(cudaMemcpy(&outputC[0], deb, dim * sizeof(double), cudaMemcpyDeviceToHost));

	for (auto t : outputC)
		std::cout << t << std::endl;
}


inline void printFromCudaFormatted(const double *deb, const int wdim, const int &dim) {

	// DEBUG
	std::vector<double> outputC(wdim);
	CHECK(cudaMemcpy(&outputC[0], deb, wdim * sizeof(double), cudaMemcpyDeviceToHost));

	for (std::size_t i = 0; i < outputC.size(); i++) {
		int cut = (outputC[i] * 100);
		double o = (static_cast<double>(cut)) / 100;
		std::cout << o << " ";
		if ((i + 1) % dim == 0)
			std::cout << " :" << i << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;
	}
}

inline void pettyPrintCuda(const double *deb, const int wdim, const int &dim) {

	// DEBUG
	std::vector<double> outputC(wdim);
	CHECK(cudaMemcpy(&outputC[0], deb, wdim * sizeof(double), cudaMemcpyDeviceToHost));

	for (std::size_t i = 0; i < outputC.size(); i++) {
		std::cout << outputC[i] << " ";
		if ((i + 1) % dim == 0)
			std::cout << " :" << i + 1 << std::endl;
		if ((i + 1) % (dim * dim) == 0)
			std::cout << std::endl;
	}
}
